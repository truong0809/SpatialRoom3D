"""So sánh các chế độ huấn luyện (vd. default và mcmc) trên cùng các cảnh, kèm kiểm tra điều kiện so sánh.

    python -m src.evaluation.compare_modes --root /workspace/outputs/benchmarks/deep_blending/gsplat \\
        --baseline default --modes default mcmc \\
        --reference-config /workspace/outputs/reports/configs/deep_blending_<thời gian>.json \\
        --b0-root /workspace/outputs/benchmarks/deep_blending/inria_b0 --out report.json --markdown report.md

- Tìm mọi lượt chạy (thư mục có run_meta.json) dưới --root; mỗi thư mục chỉ tính một lần.
- Nhóm theo (cảnh, biến thể). Biến thể = chế độ + tham số riêng (`--set`), kèm đánh dấu khi ngân sách
  Gaussian đặt tay (`--cap-max`) hoặc tập test là danh sách riêng. Ví dụ "mcmc[opacity-reg=0.001]" là biến thể
  khác "mcmc". Các seed của cùng biến thể được lấy trung bình. Nếu một cảnh có lượt chạy trên nhiều model
  COLMAP khác nhau (cảnh được dựng lại), mỗi model thành một dòng "cảnh@<mã model>" riêng.
- Lượt chạy có sai khác protocol bị loại khỏi bảng (liệt kê riêng), trừ khi --include-deviating.
  Khi đó chúng thành biến thể riêng, không bao giờ trộn với lượt đúng protocol.
- Baseline (--baseline) là một biến thể; mặc định "default" (chế độ default, không tham số riêng, đúng protocol).
  Với lượt chạy thử QUICK, baseline là "default{max_steps=3000}".
- Chênh lệch giữa hai biến thể chỉ tính trên các seed chung của hai bên; trung bình trên các cảnh chỉ tính
  trên các seed chung của mọi biến thể trong cảnh đó.
- Đối chiếu với số liệu Inria (công bố hoặc B0 tự chạy) chỉ thực hiện khi mọi lượt baseline dùng cùng điều kiện
  với Inria (đúng số vòng, LPIPS-VGG, mỗi ảnh thứ 8 làm test, không sai khác protocol); nếu không thì ghi N/A.

Báo cáo chỉ ghi số liệu, chênh lệch và các mục kiểm tra; không tự kết luận chế độ nào tốt hơn.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

from src.evaluation.compare_reference import METRICS, SceneResult, compare, load_scene
from src.evaluation.gsplat_results import RunResult, load_summary

BUDGET_TOLERANCE = 0.01  # chênh lệch tương đối của số Gaussian cuối vẫn coi là "cùng ngân sách"
MIN_SEEDS = 3  # docs/PROJECT_PLAN.md mục 4.2
GOOD_STATUSES = ("OK", "EQUAL")
INRIA_TEST_EVERY = 8  # Inria: llffhold=8, gán cứng


def discover(roots: list[Path]) -> list[RunResult]:
    runs, seen = [], set()
    for root in roots:
        for meta in sorted(Path(root).rglob("run_meta.json")):
            folder = meta.parent.resolve()
            if folder in seen or any(".prev_" in part for part in meta.parent.parts):
                continue  # đã tính (các --root chồng nhau) hoặc thư mục cũ do --force / --rerun-incomplete đổi tên
            seen.add(folder)
            runs.append(load_summary(meta.parent))
    return runs


def variant_of(run: RunResult) -> str:
    label = run.mode or "?"
    if run.args:
        label += "[" + ", ".join(f"{key}={value}" for key, value in sorted(run.args.items())) + "]"
    if run.cap_policy == "cli" and run.cap_max:
        label += f"(cap_max={run.cap_max})"
    if run.split_rule == "test_list":
        label += f"<test:{(run.split_sha256 or '?')[:6]}>"
    if run.deviations:
        label += "{" + ", ".join(f"{key}={item.get('used')}" for key, item in sorted(run.deviations.items())) + "}"
    return label


def mode_of(variant: str) -> str:
    match = re.match(r"[A-Za-z0-9_]+", variant)
    return match.group(0) if match else variant


def _finite(values) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def _mean(values):
    values = _finite(values)
    return statistics.fmean(values) if values else None


def _std(values):
    values = _finite(values)
    return statistics.stdev(values) if len(values) > 1 else None


def _ratio(a, b):
    return a / b if a and b else None


def aggregate(runs: list[RunResult]) -> dict:
    seeds = [run.seed for run in runs if run.seed is not None]
    return {
        "n": len(runs),
        "seeds": sorted(seeds),
        "duplicate_seeds": sorted({seed for seed in seeds if seeds.count(seed) > 1}),
        "metrics": {name: _mean(run.metrics.get(name) for run in runs) for name in METRICS},
        "metrics_std": {name: _std([run.metrics.get(name) for run in runs]) for name in METRICS},
        "num_gaussians": _mean(run.num_gaussians for run in runs),
        "train_minutes": _mean(None if run.train_seconds is None else run.train_seconds / 60 for run in runs),
        "train_loop_minutes": _mean(None if run.train_loop_seconds is None else run.train_loop_seconds / 60
                                    for run in runs),
        "peak_vram_gib": _mean(None if run.peak_vram_mib is None else run.peak_vram_mib / 1024 for run in runs),
        "torch_peak_gib": _mean(run.torch_peak_gib for run in runs),
        "render_ms_per_image": _mean(run.render_ms_per_image for run in runs),
        "ply_size_mb": _mean(run.ply_size_mb for run in runs),
        "run_dirs": [run.run_dir for run in runs],
    }


def _same(values) -> bool:
    return len({json.dumps(value, sort_keys=True) for value in values}) <= 1


def _seed_set(runs: list[RunResult]) -> set:
    return {run.seed for run in runs if run.seed is not None}


def _with_seeds(runs: list[RunResult], seeds: set) -> list[RunResult]:
    return [run for run in runs if run.seed in seeds]


def fairness(baseline_runs: list[RunResult], runs: list[RunResult]) -> list[dict]:
    """Các điều kiện phải giống nhau để so sánh một biến thể với baseline trên cùng cảnh."""
    both = baseline_runs + runs
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    deviating = [run.run_dir for run in both if run.deviations]
    add("protocol", "OK" if _same([run.protocol for run in both]) and not deviating else "DIFF",
        {"deviating_runs": deviating} if deviating else "same protocol, no deviations")
    for name, attr in (("split", "split_sha256"), ("gsplat_commit", "gsplat_commit"), ("gpu", "gpu")):
        values = [getattr(run, attr) for run in both]
        add(name, "OK" if _same(values) else "DIFF", sorted({str(value)[:12] if name == "split" else str(value)
                                                              for value in values}))
    applied = [run.split_applied_ok for run in both]
    if any(value is False for value in applied):
        add("split_applied", "DIFF", "gsplat used other images")
    elif any(value is None for value in applied):
        add("split_applied", "UNKNOWN", "split_applied file missing")
    else:
        add("split_applied", "OK", "images used == split.json")
    duplicates = sorted({seed for group in (baseline_runs, runs) for seed in aggregate(group)["duplicate_seeds"]})
    base_seeds, seeds = sorted(_seed_set(baseline_runs)), sorted(_seed_set(runs))
    if duplicates:
        add("seeds", "DUP", {"duplicate_seeds": duplicates, "baseline": base_seeds, "mode": seeds})
    elif base_seeds != seeds:
        add("seeds", "DIFF", {"baseline": base_seeds, "mode": seeds})
    else:
        add("seeds", "OK" if len(seeds) >= MIN_SEEDS else "FEW", {"seeds": seeds, "min": MIN_SEEDS})
    common = _seed_set(baseline_runs) & _seed_set(runs)
    base_count = _mean(run.num_gaussians for run in _with_seeds(baseline_runs, common))
    count = _mean(run.num_gaussians for run in _with_seeds(runs, common))
    ratio = _ratio(count, base_count)
    if ratio is not None:
        add("gaussian_budget", "EQUAL" if abs(ratio - 1) <= BUDGET_TOLERANCE else "UNEQUAL",
            {"baseline": round(base_count), "mode": round(count), "ratio": round(ratio, 4)})
    return checks


def baseline_runs_of(runs: list[RunResult], baseline: str, scenes: list[str]) -> list[RunResult]:
    """Các lượt OK của biến thể baseline trên các cảnh cho trước."""
    wanted = set(scenes)
    return [run for run in runs if run.status == "OK" and run.scene in wanted and variant_of(run) == baseline]


def inria_comparability(runs: list[RunResult], iteration: int) -> list[str]:
    """Lý do (nếu có) khiến các lượt baseline gsplat không so được với Inria (công bố hoặc B0)."""
    reasons = []
    for run in runs:
        name = Path(run.run_dir).name
        if run.deviations:
            reasons.append(f"{name}: protocol deviations {sorted(run.deviations)}")
        if run.protocol.get("max_steps") != iteration:
            reasons.append(f"{name}: max_steps {run.protocol.get('max_steps')} != {iteration}")
        if run.lpips_net != "vgg":
            reasons.append(f"{name}: LPIPS {run.lpips_net} (Inria uses VGG)")
        if run.protocol.get("test_every") != INRIA_TEST_EVERY or run.split_rule not in (None, "every_n"):
            reasons.append(f"{name}: test views are not every {INRIA_TEST_EVERY}th image")
        if run.split_applied_ok is not True:
            reasons.append(f"{name}: images used by gsplat not verified against split.json")
    return reasons


def b0_check(report: dict, b0_root: Path, scenes: list[str], method: str, iteration: int,
             tolerance: dict, baseline_runs: list[RunResult] | None = None) -> dict:
    """Đối chiếu baseline gsplat với lượt Inria (B0) tự chạy, từng cảnh, cùng quy tắc chia test."""
    result = {"b0_root": str(b0_root), "tolerance": tolerance, "scenes": [], "reasons": []}
    if baseline_runs is not None:
        result["reasons"] = inria_comparability(baseline_runs, iteration)
        if result["reasons"]:
            result["verdict"] = "N/A"
            return result
    statuses = []
    for scene in scenes:
        b0 = load_scene(scene, Path(b0_root) / scene, method, iteration)
        b1 = report["scenes"].get(scene, {}).get(report["baseline"])
        if b0.status != "OK" or b1 is None:
            result["scenes"].append({"scene": scene, "status": "MISSING", "b0": b0.metrics,
                                     "b1": b1 and b1["metrics"]})
            statuses.append("MISSING")
            continue
        delta = {name: b1["metrics"][name] - b0.metrics[name] for name in METRICS}
        within = {name: abs(delta[name]) <= tolerance[name] for name in METRICS}
        status = "OK" if all(within.values()) else "FAIL"
        result["scenes"].append({"scene": scene, "status": status, "b0": b0.metrics, "b1": b1["metrics"],
                                 "delta": delta, "within_tolerance": within})
        statuses.append(status)
    if not statuses:
        result["verdict"] = "INCOMPLETE"
    elif "FAIL" in statuses:  # một cảnh lệch là FAIL, kể cả khi cảnh khác còn thiếu
        result["verdict"] = "FAIL"
    elif "MISSING" in statuses:
        result["verdict"] = "INCOMPLETE"
    else:
        result["verdict"] = "PASS"
    return result


def _scene_keys(runs: list[RunResult]) -> dict[str, str]:
    """run_dir -> khóa cảnh; cảnh có nhiều model COLMAP thì thêm mã model để không trộn lẫn."""
    models: dict[str, set] = {}
    for run in runs:
        models.setdefault(str(run.scene), set()).add(run.scene_fingerprint)
    keys = {}
    for run in runs:
        scene = str(run.scene)
        if len(models[scene]) > 1:
            keys[run.run_dir] = f"{scene}@{(run.scene_fingerprint or 'unknown')[:8]}"
        else:
            keys[run.run_dir] = scene
    return keys


def build_report(runs: list[RunResult], baseline: str, modes: list[str],
                 reference: dict | None = None, tolerance: dict | None = None,
                 reference_scenes: list[str] | None = None, include_deviating: bool = False,
                 iteration: int = 30000) -> dict:
    base_mode = mode_of(baseline)
    if base_mode not in modes:
        modes = [base_mode, *modes]
    grouped: dict[str, dict[str, list[RunResult]]] = {}
    report = {"baseline": baseline, "modes": modes, "variants": [], "scenes": {}, "deltas": {}, "fairness": {},
              "incomplete": [], "excluded": [], "mean_over_scenes": {}, "mean_over_scenes_seeds": {},
              "complete_scenes": [], "warnings": []}
    selected = [run for run in runs if run.mode in modes]
    keys = _scene_keys(selected)
    for scene in sorted({key for key in keys.values() if "@" in key}):
        report["warnings"].append(f"{scene.split('@')[0]}: runs on several COLMAP models; shown separately")
    for run in selected:
        if run.status != "OK":
            report["incomplete"].append({"run_dir": run.run_dir, "scene": run.scene, "mode": run.mode,
                                         "status": run.status, **({"problem": run.problem} if run.problem else {})})
            continue
        if run.deviations and not include_deviating:
            report["excluded"].append({"run_dir": run.run_dir, "scene": run.scene, "variant": variant_of(run),
                                       "reason": "protocol deviation"})
            continue
        grouped.setdefault(keys[run.run_dir], {}).setdefault(variant_of(run), []).append(run)

    variants = sorted({variant for by_variant in grouped.values() for variant in by_variant},
                      key=lambda v: (v != baseline, modes.index(mode_of(v)), v))
    report["variants"] = variants
    for scene in sorted(grouped):
        by_variant = grouped[scene]
        report["scenes"][scene] = {variant: aggregate(by_variant[variant]) for variant in variants
                                   if variant in by_variant}
        missing = [variant for variant in variants if variant not in by_variant]
        # chế độ được yêu cầu nhưng không có lượt OK nào (ở bất kỳ biến thể nào) cũng là thiếu
        missing += [mode for mode in modes
                    if mode not in {mode_of(v) for v in by_variant} and mode not in {mode_of(m) for m in missing}]
        if baseline not in by_variant and baseline not in missing:
            missing.append(baseline)
        if missing:
            report["incomplete"].append({"scene": scene, "missing_variants": missing})
        if baseline not in by_variant:
            continue
        for variant in variants:
            if variant == baseline or variant not in by_variant:
                continue
            report["fairness"].setdefault(scene, {})[variant] = fairness(by_variant[baseline], by_variant[variant])
            common = _seed_set(by_variant[baseline]) & _seed_set(by_variant[variant])
            if not common:
                report["incomplete"].append({"scene": scene, "variant": variant,
                                             "reason": "no seed in common with the baseline"})
                continue
            base = aggregate(_with_seeds(by_variant[baseline], common))
            current = aggregate(_with_seeds(by_variant[variant], common))
            report["deltas"].setdefault(scene, {})[variant] = {
                **{name: current["metrics"][name] - base["metrics"][name] for name in METRICS},
                "seeds": sorted(common),
                "num_gaussians_ratio": _ratio(current["num_gaussians"], base["num_gaussians"]),
                "train_time_ratio": _ratio(current["train_minutes"], base["train_minutes"]),
            }
        scene_common = set.intersection(*(_seed_set(by_variant[v]) for v in by_variant))
        if not missing and scene_common:
            report["complete_scenes"].append(scene)
            report["mean_over_scenes_seeds"][scene] = sorted(scene_common)

    for variant in variants:
        values = [aggregate(_with_seeds(grouped[scene][variant], set(report["mean_over_scenes_seeds"][scene])))
                  ["metrics"] for scene in report["complete_scenes"]]
        if values:
            report["mean_over_scenes"][variant] = {name: statistics.fmean(v[name] for v in values)
                                                   for name in METRICS}

    if reference is not None:
        scenes = reference_scenes or sorted(grouped)
        reasons = inria_comparability(baseline_runs_of(selected, baseline, scenes), iteration)
        if reasons:
            report["reference_check"] = {"mode": baseline, "scenes": scenes, "verdict": "N/A", "reasons": reasons}
        else:
            results = []
            for scene in scenes:
                agg = report["scenes"].get(scene, {}).get(baseline)
                if agg is None:
                    results.append(SceneResult(scene=scene, model_dir=""))
                else:
                    results.append(SceneResult(scene=scene, model_dir=";".join(agg["run_dirs"]), status="OK",
                                               metrics=dict(agg["metrics"])))
            report["reference_check"] = {"mode": baseline, "scenes": scenes,
                                         **compare(results, reference, tolerance or {})}
    return report


def _f(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def _metric(agg: dict, name: str, digits: int) -> str:
    value, std = agg["metrics"][name], agg["metrics_std"][name]
    return _f(value, digits) + ("" if std is None else f" ± {std:.{digits}f}")


def _conditions_cell(checks: list[dict]) -> str:
    bad = [f"{c['check']}={c['status']}" for c in checks if c["status"] not in GOOD_STATUSES]
    return "đủ điều kiện" if not bad else "chưa đủ điều kiện: " + ", ".join(bad)


def format_markdown(report: dict) -> str:
    baseline = report["baseline"]
    lines = [f"Baseline: `{baseline}`. Số liệu là trung bình ± độ lệch chuẩn theo seed (n = số lượt chạy).", "",
             "| Cảnh | Biến thể | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Số Gaussian | Train (phút) | VRAM đỉnh (GiB) "
             "| Render (ms/ảnh) | PLY (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for scene, by_variant in report["scenes"].items():
        for variant, agg in by_variant.items():
            count = "-" if agg["num_gaussians"] is None else f"{agg['num_gaussians']:,.0f}"
            lines.append(f"| {scene} | {variant} | {agg['n']} | {_metric(agg, 'PSNR', 3)} | {_metric(agg, 'SSIM', 4)} "
                         f"| {_metric(agg, 'LPIPS', 4)} | {count} | {_f(agg['train_minutes'], 1)} "
                         f"| {_f(agg['peak_vram_gib'], 1)} | {_f(agg['render_ms_per_image'], 2)} "
                         f"| {_f(agg['ply_size_mb'], 0)} |")
    if report["deltas"]:
        lines += ["", f"Chênh lệch so với `{baseline}`, tính trên các seed chung "
                      "(dương ở PSNR/SSIM và âm ở LPIPS là tốt hơn):", "",
                  "| Cảnh | Biến thể | Seed | ΔPSNR | ΔSSIM | ΔLPIPS | Tỉ lệ số Gaussian | Tỉ lệ thời gian | Điều kiện |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for scene, by_variant in report["deltas"].items():
            for variant, delta in by_variant.items():
                cell = _conditions_cell(report["fairness"][scene][variant])
                seeds = ",".join(str(seed) for seed in delta["seeds"])
                lines.append(f"| {scene} | {variant} | {seeds} | {delta['PSNR']:+.3f} | {delta['SSIM']:+.4f} "
                             f"| {delta['LPIPS']:+.4f} | {_f(delta['num_gaussians_ratio'], 3)} "
                             f"| {_f(delta['train_time_ratio'], 2)} | {cell} |")
    if report["mean_over_scenes"]:
        lines += ["", "Trung bình trên các cảnh có đủ mọi biến thể (chỉ các seed chung của cảnh): "
                  + ", ".join(f"{scene} (seed {','.join(map(str, report['mean_over_scenes_seeds'][scene]))})"
                              for scene in report["complete_scenes"]), ""]
        for variant, m in report["mean_over_scenes"].items():
            lines.append(f"- `{variant}`: PSNR {_f(m['PSNR'])}, SSIM {_f(m['SSIM'])}, LPIPS {_f(m['LPIPS'])}")
    if report.get("reference_check"):
        ref = report["reference_check"]
        lines += ["", f"Đối chiếu `{ref['mode']}` với số liệu công bố ({', '.join(ref['scenes'])}): "
                      f"**{ref['verdict']}**"]
        if ref.get("reasons"):
            lines += [f"- không so được: {reason}" for reason in ref["reasons"]]
        elif ref.get("mean"):
            lines.append("- chênh lệch trung bình: " + ", ".join(f"{k} {v:+.3f}" for k, v in ref["delta"].items()))
    if report.get("b0_check"):
        b0 = report["b0_check"]
        lines += ["", f"Đối chiếu `{baseline}` (gsplat) với lượt Inria tự chạy (B0), từng cảnh: **{b0['verdict']}**"]
        lines += [f"- không so được: {reason}" for reason in b0.get("reasons", [])]
        for row in b0["scenes"]:
            if row["status"] == "MISSING":
                lines.append(f"- {row['scene']}: thiếu kết quả")
            else:
                lines.append(f"- {row['scene']}: " + ", ".join(f"Δ{k} {v:+.3f}" for k, v in row["delta"].items())
                             + f" ({row['status']})")
    for title, key in (("Cảnh báo", "warnings"), ("Bị loại (sai khác protocol)", "excluded"),
                       ("Chưa đủ / lỗi", "incomplete")):
        if report[key]:
            lines += ["", f"{title}:"] + [f"- {item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}"
                                         for item in report[key]]
    lines += ["", "Ghi chú: chỉ nói \"cải thiện\" khi mọi mục điều kiện là OK, ngân sách Gaussian EQUAL, "
                  f"cùng bộ ≥ {MIN_SEEDS} seed và kết quả nhất quán trên các cảnh "
                  "(docs/design/training_modes.md, mục 4.3)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path, action="append",
                        help="thư mục chứa các lượt chạy (lặp lại được)")
    parser.add_argument("--baseline", default="default",
                        help="biến thể baseline (mặc định default; lượt chạy thử: 'default{max_steps=3000}')")
    parser.add_argument("--modes", nargs="+", default=["default", "mcmc"])
    parser.add_argument("--include-deviating", action="store_true",
                        help="giữ lượt sai khác protocol thành biến thể riêng thay vì loại bỏ")
    parser.add_argument("--reference-config", type=Path,
                        help="bản JSON của mục benchmark trong configs/<benchmark>.yaml (reference/tolerance/scenes; "
                             "scripts/run_evaluate.sh ghi ra trong outputs/reports/configs/)")
    parser.add_argument("--b0-root", type=Path,
                        help="thư mục kết quả Inria tự chạy (mỗi cảnh một thư mục có results.json) để đối chiếu từng cảnh")
    parser.add_argument("--out", type=Path, help="ghi báo cáo JSON")
    parser.add_argument("--markdown", type=Path, help="ghi bảng Markdown")
    parser.add_argument("--strict", action="store_true",
                        help="trả mã 1 nếu một phép đối chiếu được yêu cầu (công bố / B0) không PASS")
    args = parser.parse_args(argv)

    config = json.loads(args.reference_config.read_text(encoding="utf-8")) if args.reference_config else None
    reference = config["reference"]["metrics"] if config else None
    tolerance = config["tolerance"] if config else None
    scenes = config.get("scenes") if config else None
    iteration = int(config.get("iteration", 30000)) if config else 30000
    runs = discover(args.root)
    report = build_report(runs, args.baseline, args.modes, reference, tolerance, scenes,
                          args.include_deviating, iteration)
    if args.b0_root:
        if config is None:
            parser.error("--b0-root needs --reference-config (scenes, method, iteration, tolerance)")
        b0_scenes = scenes or sorted(report["scenes"])
        report["b0_check"] = b0_check(report, args.b0_root, b0_scenes, config.get("method", "ours_30000"),
                                      iteration, tolerance, baseline_runs_of(runs, args.baseline, b0_scenes))
    text = format_markdown(report)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nĐã ghi: {args.out}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8")
    if args.strict:
        verdicts = [report[key]["verdict"] for key in ("reference_check", "b0_check") if report.get(key)]
        if any(verdict != "PASS" for verdict in verdicts):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
