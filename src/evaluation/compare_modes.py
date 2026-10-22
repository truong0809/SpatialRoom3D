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
  Với lượt chạy thử QUICK, baseline là "default{max_steps=3000}". Tên một chế độ chỉ có một biến thể cũng được
  (vd. --baseline mcmc --modes mcmc X: so chế độ X với mcmc; khi đó không đưa --reference-config, vì
  đối chiếu với Inria chỉ dành cho baseline default). Chế độ có thêm các biến thể của một kế hoạch thực nghiệm (lượt có
  tag, mục `variants`): tên chế độ chọn các lượt không có tag của nó.
- Lượt có tag (biến thể của một kế hoạch thực nghiệm, vd. các lượt dò tham số của tune_mcmc nằm cùng thư mục với thực
  nghiệm chính) chỉ vào bảng khi tag nằm trong `--tags`; mặc định bị để ngoài (một dòng cảnh báo ghi số lượt và các
  tag), để bảng của một kế hoạch khác không lẫn chúng.
- Lượt đã lọc Gaussian (`<lượt>/filtered/`, src.training.prune): thêm bảng chỉ số sau khi lọc, chênh lệch so với chính
  lượt đó trước khi lọc, số Gaussian giữ lại và dung lượng PLY. Mọi bảng khác là mô hình chưa lọc.
- Ngoài chỉ số: số cặp (cảnh, seed) và số cảnh mà mỗi biến thể hơn baseline, kèm chênh lệch trung bình ± độ lệch chuẩn
  theo cặp và các cảnh chưa đủ điều kiện; chất lượng theo dung lượng (mỗi bản của mô hình, gồm bản đã lọc, xếp theo dung
  lượng PLY, kèm byte mỗi Gaussian; đánh dấu bản không bị bản nào nhẹ hơn mà tốt hơn vượt); cấu trúc mô hình từ
  `<lượt>/splat_stats.json` (src.evaluation.splat_stats: tỉ lệ Gaussian gần trong suốt, rất to, bậc SH) khi đã tính. Hai
  bảng sau tính trên các seed chung của mọi dòng trong cảnh.
- `Render (ms/ảnh)` là thời gian gsplat đo khi chấm, trong tiến trình chấm riêng, gồm cả lần gọi đầu (nạp CUDA): không
  phải tốc độ render của mô hình, không đổi ra FPS.
- Chênh lệch giữa hai biến thể chỉ tính trên các seed chung của hai bên; trung bình trên các cảnh chỉ tính
  trên các seed chung của mọi biến thể trong cảnh đó.
- Đối chiếu với số liệu Inria (công bố hoặc B0 tự chạy) chỉ thực hiện khi mọi lượt baseline dùng cùng điều kiện
  với Inria (đúng số vòng, LPIPS-VGG, mỗi ảnh thứ 8 làm test, không sai khác protocol); nếu không thì ghi N/A.
- `--modes`: các chế độ đưa vào bảng; mặc định mọi chế độ có lượt chạy dưới --root.
- `--scenes`: chỉ các cảnh này (bảng và đối chiếu B0 từng cảnh). Khi đó chỉ là một phần các cảnh của benchmark thì
  phần đối chiếu với số công bố (trung bình mọi cảnh) không làm được và ghi SKIPPED; --strict không tính mục SKIPPED.

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
STRICT_OK = ("PASS", "SKIPPED")  # --strict: mục bị bỏ qua có chủ đích (chạy một phần cảnh) không tính là lỗi


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


def select_tags(runs: list[RunResult], tags: list[str] | None) -> tuple[list[RunResult], list[RunResult]]:
    """(lượt đưa vào bảng, lượt có tag bị để ngoài): lượt không tag luôn được giữ; lượt có tag chỉ khi tag nằm trong
    `tags`."""
    wanted = set(tags or [])
    kept = [run for run in runs if not run.tag or run.tag in wanted]
    return kept, [run for run in runs if run.tag and run.tag not in wanted]


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


def _bytes_per_gaussian(run: RunResult):
    if run.splat_stats and run.splat_stats.get("bytes_per_gaussian"):
        return run.splat_stats["bytes_per_gaussian"]
    if run.ply_size_mb and run.num_gaussians:
        return run.ply_size_mb * 2**20 / run.num_gaussians  # gần đúng: dung lượng đã làm tròn, gồm cả header
    return None


def _nested(data, *keys):
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def aggregate_splat(runs: list[RunResult]) -> dict | None:
    """Thống kê PLY cuối (splat_stats.json) trung bình theo seed; None khi chưa lượt nào có. Một mục chỉ có giá trị khi
    mọi lượt có thống kê đều có mục đó (vd. tỉ lệ theo scene_scale cần scene_scale của mọi lượt)."""
    done = [run.splat_stats for run in runs if run.splat_stats]
    if not done:
        return None

    def mean(*keys):
        values = [_nested(stats, *keys) for stats in done]
        return _mean(values) if all(value is not None for value in values) else None
    return {
        "n": len(done),
        "unchecked": sum(stats.get("ply_checked") is False for stats in done),
        "sh_degree": sorted({stats["sh_degree"] for stats in done if stats.get("sh_degree") is not None}),
        "opacity_below": {t: mean("opacity", "below", t) for t in ("0.005", "0.02", "0.1")},
        "opacity_p50": mean("opacity", "p50"),
        "largest_rel_above": {t: mean("largest_axis_relative", "above", t) for t in ("0.01", "0.1")},
        "largest_rel_p50": mean("largest_axis_relative", "p50"),
    }


def filtered_ok(run: RunResult) -> bool:
    return (run.filtered or {}).get("status") == "OK"


def aggregate_filtered(runs: list[RunResult]) -> dict | None:
    """Bản đã lọc Gaussian (src.training.prune) của các lượt đã lọc xong: chỉ số sau khi lọc và chênh lệch so với chính
    lượt đó trước khi lọc (cùng ảnh test), trung bình theo seed; None khi chưa lượt nào được lọc."""
    done = [run for run in runs if filtered_ok(run)]
    if not done:
        return None

    def change(name):
        return _mean(None if run.metrics.get(name) is None else run.filtered["metrics"][name] - run.metrics[name]
                     for run in done)
    thresholds = sorted({json.dumps(run.filtered.get("thresholds"), sort_keys=True) for run in done})
    return {
        "n": len(done),
        "seeds": sorted(run.seed for run in done if run.seed is not None),
        "metrics": {name: _mean(run.filtered["metrics"][name] for run in done) for name in METRICS},
        "delta": {name: change(name) for name in METRICS},
        "num_gaussians": _mean(run.filtered.get("num_gaussians") for run in done),
        "num_gaussians_before": _mean(run.num_gaussians for run in done),
        "kept_fraction": _mean(run.filtered.get("kept_fraction") for run in done),
        "ply_size_mb": _mean(run.filtered.get("ply_size_mb") for run in done),
        "ply_size_mb_before": _mean(run.ply_size_mb for run in done),
        "render_ms_per_image": _mean(run.filtered.get("render_ms_per_image") for run in done),
        "thresholds": [json.loads(text) for text in thresholds],
    }


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
        "bytes_per_gaussian": _mean(_bytes_per_gaussian(run) for run in runs),
        "run_dirs": [run.run_dir for run in runs],
        "filtered": aggregate_filtered(runs),
    }


def _per_seed(runs: list[RunResult]) -> dict:
    groups: dict = {}
    for run in runs:
        groups.setdefault(run.seed, []).append(run)
    return {seed: {name: _mean(run.metrics.get(name) for run in group) for name in METRICS}
            for seed, group in groups.items()}


def seed_pairs(scene: str, base_runs: list[RunResult], runs: list[RunResult], seeds: set) -> list[dict]:
    """Chênh lệch chỉ số theo từng seed chung (biến thể trừ baseline); seed lặp lại thì lấy trung bình."""
    base, current = _per_seed(base_runs), _per_seed(runs)
    pairs = []
    for seed in sorted(seeds):
        pairs.append({"scene": scene, "seed": seed,
                      **{name: (None if current[seed][name] is None or base[seed][name] is None
                                else current[seed][name] - base[seed][name]) for name in METRICS}})
    return pairs


def _better(name: str, delta) -> bool:
    return delta is not None and (delta < 0 if name == "LPIPS" else delta > 0)


def wins(pairs: list[dict], scene_deltas: dict, fairness_checks: dict) -> dict:
    """Cặp (cảnh, seed) và cảnh (chênh lệch trung bình trên các seed chung của cảnh, như bảng chênh lệch) mà biến thể
    tốt hơn baseline; bằng nhau không tính. `not_fair`: các cảnh có mục điều kiện chưa đạt."""
    not_fair = {scene: [f"{c['check']}={c['status']}" for c in checks if c["status"] not in GOOD_STATUSES]
                for scene, checks in fairness_checks.items()}
    return {"pairs": len(pairs), "scenes": sorted({pair["scene"] for pair in pairs}),
            "better": {name: sum(_better(name, pair[name]) for pair in pairs) for name in METRICS},
            "scene_count": len(scene_deltas),
            "scenes_better": {name: sum(_better(name, delta[name]) for delta in scene_deltas.values())
                              for name in METRICS},
            "mean": {name: _mean(pair[name] for pair in pairs) for name in METRICS},
            "std": {name: _std([pair[name] for pair in pairs]) for name in METRICS},
            "not_fair": {scene: bad for scene, bad in sorted(not_fair.items()) if bad}}


def _dominates(a: dict, b: dict) -> bool:
    """a nhẹ hơn hoặc bằng b, PSNR và LPIPS tốt hơn hoặc bằng, và hơn hẳn ở ít nhất một mục."""
    values = (a["ply_size_mb"], b["ply_size_mb"], a["metrics"]["PSNR"], b["metrics"]["PSNR"], a["metrics"]["LPIPS"],
              b["metrics"]["LPIPS"])
    if any(value is None for value in values):
        return False
    size_a, size_b, psnr_a, psnr_b, lpips_a, lpips_b = values
    return (size_a <= size_b and psnr_a >= psnr_b and lpips_a <= lpips_b
            and (size_a < size_b or psnr_a > psnr_b or lpips_a < lpips_b))


def size_quality(by_variant: dict[str, list[RunResult]], seeds: set) -> dict:
    """Mỗi bản của mô hình trong một cảnh (mỗi biến thể, và bản đã lọc của nó), xếp theo dung lượng PLY. Mọi dòng tính
    trên cùng `seeds` (các seed chung của cảnh); bản đã lọc chỉ có khi mọi lượt trên các seed đó đã lọc xong.
    `not_dominated` là None khi dòng thiếu dung lượng hoặc chỉ số."""
    rows, left_out = [], []
    for variant, runs in by_variant.items():
        chosen = _with_seeds(runs, seeds)
        agg = aggregate(chosen)
        rows.append({"version": variant, "variant": variant, "filtered": False, "n": agg["n"],
                     "ply_size_mb": agg["ply_size_mb"], "num_gaussians": agg["num_gaussians"],
                     "bytes_per_gaussian": agg["bytes_per_gaussian"], "metrics": dict(agg["metrics"])})
        ok = sum(filtered_ok(run) for run in chosen)
        if ok < len(chosen):  # chưa lọc hết, hoặc lọc lỗi, trên các seed chung: không so được trên cùng seed
            failed = sum(run.filtered is not None and not filtered_ok(run) for run in chosen)
            outside = sum(filtered_ok(run) for run in runs if run.seed not in seeds)
            if ok or failed or outside:
                left_out.append({"variant": variant, "total": len(chosen), "ok": ok, "failed": failed,
                                 "outside": outside})
        elif agg["filtered"]:
            f = agg["filtered"]
            rows.append({"version": f"{variant} + đã lọc", "variant": variant, "filtered": True, "n": f["n"],
                         "ply_size_mb": f["ply_size_mb"], "num_gaussians": f["num_gaussians"],
                         "bytes_per_gaussian": (f["ply_size_mb"] * 2**20 / f["num_gaussians"]
                                                if f["ply_size_mb"] and f["num_gaussians"] else None),
                         "metrics": dict(f["metrics"])})
    for row in rows:
        if row["ply_size_mb"] is None or row["metrics"]["PSNR"] is None or row["metrics"]["LPIPS"] is None:
            row["not_dominated"] = None
        else:
            row["not_dominated"] = not any(_dominates(other, row) for other in rows if other is not row)
    rows.sort(key=lambda row: (row["ply_size_mb"] is None, row["ply_size_mb"] or 0, row["version"]))
    return {"seeds": sorted(seeds), "rows": rows, "filtered_left_out": left_out}


def structure(by_variant: dict[str, list[RunResult]]) -> dict | None:
    """Cấu trúc mô hình (splat_stats.json) của các biến thể có thống kê, trên các seed mà mọi biến thể đó đều có thống
    kê; None khi không biến thể nào có, hoặc không có seed chung."""
    with_stats = {variant: [run for run in runs if run.splat_stats] for variant, runs in by_variant.items()}
    with_stats = {variant: runs for variant, runs in with_stats.items() if runs}
    if not with_stats:
        return None
    seeds = set.intersection(*(_seed_set(runs) for runs in with_stats.values()))
    return {"seeds": sorted(seeds), "variants": {variant: aggregate_splat(_with_seeds(runs, seeds))
                                                 for variant, runs in with_stats.items()} if seeds else {},
            "with_stats": sorted(with_stats)}


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


def resolve_baseline(runs: list[RunResult], baseline: str, include_deviating: bool = False) -> str:
    """`--baseline` là tên một biến thể. Nếu là tên một chế độ mà chế độ đó chỉ có đúng một biến thể (vd. "mcmc" khi
    mọi lượt mcmc của Deep Blending chạy với opacity_reg 0.001: biến thể "mcmc[opacity-reg=0.001]"), dùng biến thể đó.
    Chế độ có thêm các biến thể của một kế hoạch thực nghiệm (lượt có tag, mục variants): mốc là các lượt không có tag
    của chế độ đó, khi chúng chỉ thuộc một biến thể."""
    usable = [run for run in runs if run.status == "OK" and (include_deviating or not run.deviations)]
    variants = {variant_of(run) for run in usable}
    if baseline in variants or not re.fullmatch(r"[A-Za-z0-9_]+", baseline):
        return baseline
    candidates = sorted(variant for variant in variants if mode_of(variant) == baseline)
    if len(candidates) > 1:
        plain = sorted({variant_of(run) for run in usable if run.mode == baseline and not run.tag})
        if len(plain) == 1:
            return plain[0]
        print(f"WARN: mode {baseline} has several variants {candidates}; pass one of them as --baseline",
              file=sys.stderr)
    return candidates[0] if len(candidates) == 1 else baseline


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


def build_report(runs: list[RunResult], baseline: str, modes: list[str] | None,
                 reference: dict | None = None, tolerance: dict | None = None,
                 reference_scenes: list[str] | None = None, include_deviating: bool = False,
                 iteration: int = 30000, skip_reference: str | None = None) -> dict:
    """`modes` None: mọi chế độ có lượt chạy. `skip_reference`: lý do không đối chiếu với số công bố (vd. chỉ chạy một
    phần cảnh của benchmark); khi có, mục đối chiếu ghi SKIPPED."""
    base_mode = mode_of(baseline)
    if modes is None:
        modes = sorted({run.mode for run in runs if run.mode and run.mode != base_mode})
    if base_mode not in modes:
        modes = [base_mode, *modes]
    grouped: dict[str, dict[str, list[RunResult]]] = {}
    report = {"baseline": baseline, "modes": modes, "variants": [], "scenes": {}, "deltas": {}, "fairness": {},
              "incomplete": [], "excluded": [], "mean_over_scenes": {}, "mean_over_scenes_seeds": {},
              "complete_scenes": [], "warnings": [], "pairs": {}, "wins": {}, "size_quality": {}, "structure": {}}
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
        if run.filtered is not None and not filtered_ok(run):
            report["incomplete"].append({"run_dir": run.run_dir, "scene": run.scene, "variant": variant_of(run),
                                         "filtered": run.filtered.get("status"),
                                         **({"problem": run.filtered["problem"]} if run.filtered.get("problem") else {})})
        grouped.setdefault(keys[run.run_dir], {}).setdefault(variant_of(run), []).append(run)

    variants = sorted({variant for by_variant in grouped.values() for variant in by_variant},
                      key=lambda v: (v != baseline, modes.index(mode_of(v)), v))
    report["variants"] = variants
    for scene in sorted(grouped):
        by_variant = grouped[scene]
        report["scenes"][scene] = {variant: aggregate(by_variant[variant]) for variant in variants
                                   if variant in by_variant}
        ordered = {variant: by_variant[variant] for variant in variants if variant in by_variant}
        scene_seeds = set.intersection(*(_seed_set(runs) for runs in ordered.values()))
        report["size_quality"][scene] = (size_quality(ordered, scene_seeds) if scene_seeds
                                         else {"seeds": [], "rows": [], "filtered_left_out": []})
        shape = structure(ordered)
        if shape:
            report["structure"][scene] = shape
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
            report["pairs"].setdefault(variant, []).extend(
                seed_pairs(scene, by_variant[baseline], by_variant[variant], common))
        scene_common = set.intersection(*(_seed_set(by_variant[v]) for v in by_variant))
        if not missing and scene_common:
            report["complete_scenes"].append(scene)
            report["mean_over_scenes_seeds"][scene] = sorted(scene_common)

    for variant, pairs in report["pairs"].items():
        scene_deltas = {scene: deltas[variant] for scene, deltas in report["deltas"].items() if variant in deltas}
        report["wins"][variant] = wins(pairs, scene_deltas,
                                       {scene: report["fairness"][scene][variant] for scene in scene_deltas})
    for scene, by_variant in report["scenes"].items():
        for variant, agg in by_variant.items():
            if agg["filtered"] and len(agg["filtered"]["thresholds"]) > 1:
                report["warnings"].append(f"{scene} / {variant}: runs filtered with different thresholds "
                                          f"{agg['filtered']['thresholds']}; their mean mixes them")

    for variant in variants:
        values = [aggregate(_with_seeds(grouped[scene][variant], set(report["mean_over_scenes_seeds"][scene])))
                  ["metrics"] for scene in report["complete_scenes"]]
        if values:
            report["mean_over_scenes"][variant] = {name: statistics.fmean(v[name] for v in values)
                                                   for name in METRICS}

    if reference is not None and skip_reference:
        report["reference_check"] = {"mode": baseline, "scenes": reference_scenes or sorted(grouped),
                                     "verdict": "SKIPPED", "reasons": [skip_reference]}
    elif reference is not None:
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


def _signed(value, digits=3):
    return "-" if value is None else f"{value:+.{digits}f}"


def _threshold_text(thresholds: dict | None) -> str:
    if not thresholds:
        return "ngưỡng không rõ"
    parts = []
    if thresholds.get("min_opacity"):
        parts.append(f"opacity < {thresholds['min_opacity']}")
    if thresholds.get("max_scale"):
        parts.append(f"trục lớn nhất > {thresholds['max_scale']} × scene_scale")
    return "bỏ " + " hoặc ".join(parts) if parts else "không ngưỡng nào"


def filtered_lines(report: dict) -> list[str]:
    """Bảng trước / sau khi lọc Gaussian, chỉ khi có lượt đã lọc."""
    rows = [(scene, variant, agg["filtered"]) for scene, by_variant in report["scenes"].items()
            for variant, agg in by_variant.items() if agg.get("filtered")]
    if not rows:
        return []
    thresholds = sorted({json.dumps(t, sort_keys=True) for _, _, f in rows for t in f["thresholds"]})
    mixed = len(thresholds) > 1  # ngưỡng khác nhau giữa các dòng: thêm cột ngưỡng cho từng dòng
    lines = ["", "Sau khi lọc Gaussian (src.training.prune: " + "; ".join(_threshold_text(json.loads(t)) for t in thresholds)
             + "), chấm lại trên cùng ảnh test. Δ là chênh lệch so với chính các lượt đó trước khi lọc (âm ở PSNR/SSIM "
             "và dương ở LPIPS là kém đi); n = số lượt đã lọc. Bảng chính ở trên là mô hình chưa lọc.", "",
             "| Cảnh | Biến thể | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ΔPSNR | ΔSSIM | ΔLPIPS | Số Gaussian giữ lại "
             "| PLY (MB) trước → sau | Render (ms/ảnh) |" + (" Ngưỡng |" if mixed else ""),
             "|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if mixed else "")]
    for scene, variant, f in rows:
        kept = "-" if f["num_gaussians"] is None else f"{f['num_gaussians']:,.0f}"
        if f["kept_fraction"] is not None:
            kept += f" ({f['kept_fraction'] * 100:.1f}%)"
        row = (f"| {scene} | {variant} | {f['n']} | {_f(f['metrics']['PSNR'], 3)} | {_f(f['metrics']['SSIM'], 4)} "
               f"| {_f(f['metrics']['LPIPS'], 4)} | {_signed(f['delta']['PSNR'], 3)} "
               f"| {_signed(f['delta']['SSIM'], 4)} | {_signed(f['delta']['LPIPS'], 4)} | {kept} "
               f"| {_f(f['ply_size_mb_before'], 0)} → {_f(f['ply_size_mb'], 0)} "
               f"| {_f(f['render_ms_per_image'], 2)} |")
        if mixed:
            row += " " + "; ".join(_threshold_text(t) for t in f["thresholds"]) + " |"
        lines.append(row)
    return lines


def _pct(value, digits=1):
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _plus_minus(mean, std, digits):
    return _signed(mean, digits) + ("" if std is None else f" ± {std:.{digits}f}")


def wins_lines(report: dict) -> list[str]:
    """Số cặp (cảnh, seed) và số cảnh mà mỗi biến thể tốt hơn baseline, kèm điều kiện so sánh."""
    if not report["wins"]:
        return []
    lines = ["", f"Số cặp (cảnh, seed) và số cảnh tốt hơn `{report['baseline']}` (PSNR / SSIM cao hơn, LPIPS thấp hơn; "
                 "bằng nhau không tính). Cặp: cùng cảnh, cùng seed. Cảnh: chênh lệch trung bình trên các seed chung của "
                 "cảnh, như bảng chênh lệch ở trên. Δ: trung bình ± độ lệch chuẩn theo cặp. Đọc cùng độ lệch chuẩn và cột "
                 "điều kiện: nhiều cặp tốt hơn với chênh lệch nhỏ hơn độ lệch chuẩn chưa phải \"cải thiện\".", "",
             "| Biến thể | Cặp tốt hơn: PSNR · SSIM · LPIPS | Cảnh tốt hơn: PSNR · SSIM · LPIPS | ΔPSNR | ΔSSIM | ΔLPIPS "
             "| Điều kiện |",
             "|---|---|---|---|---|---|---|"]
    for variant, w in report["wins"].items():
        n, better = w["pairs"], w["better"]
        k, scenes = w["scene_count"], w["scenes_better"]
        fair = ("đủ điều kiện" if not w["not_fair"] else "chưa đủ: " + "; ".join(
            f"{scene} ({', '.join(bad)})" for scene, bad in w["not_fair"].items()))
        lines.append(f"| {variant} | {better['PSNR']}/{n} · {better['SSIM']}/{n} · {better['LPIPS']}/{n} "
                     f"| {scenes['PSNR']}/{k} · {scenes['SSIM']}/{k} · {scenes['LPIPS']}/{k} "
                     f"| {_plus_minus(w['mean']['PSNR'], w['std']['PSNR'], 3)} "
                     f"| {_plus_minus(w['mean']['SSIM'], w['std']['SSIM'], 4)} "
                     f"| {_plus_minus(w['mean']['LPIPS'], w['std']['LPIPS'], 4)} | {fair} |")
    return lines


def size_quality_lines(report: dict) -> list[str]:
    """Chất lượng theo dung lượng: mọi bản của mô hình trong từng cảnh, trên các seed chung, xếp theo dung lượng PLY."""
    items = report["size_quality"]
    has_rows = any(row["ply_size_mb"] is not None for table in items.values() for row in table["rows"])
    if not has_rows and all(table["seeds"] for table in items.values()):
        return []
    lines = ["", "Chất lượng theo dung lượng: mỗi bản của mô hình trong từng cảnh (gồm bản đã lọc), xếp theo dung lượng PLY; "
                 "mọi dòng của một cảnh tính trên cùng các seed chung (cột Seed). Bản đã lọc chỉ có khi mọi lượt trên các "
                 "seed đó đã lọc xong. Không bị vượt: ✓ khi không có bản nào nhẹ hơn hoặc bằng mà PSNR và LPIPS đều tốt hơn "
                 "hoặc bằng (hơn hẳn ở ít nhất một mục); - khi dòng thiếu dung lượng hoặc chỉ số.", "",
             "| Cảnh | Bản | Seed | PLY (MB) | Byte/Gaussian | Số Gaussian | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Không bị vượt |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    if not has_rows:
        lines = lines[:2]  # không dòng nào có dung lượng: chỉ còn ghi chú
    notes = []
    for scene, table in items.items():
        seeds = ",".join(str(seed) for seed in table["seeds"])
        if not table["seeds"]:
            notes.append(f"- {scene}: các biến thể không có seed chung, không có dòng nào")
        for row in table["rows"] if has_rows else []:
            count = "-" if row["num_gaussians"] is None else f"{row['num_gaussians']:,.0f}"
            mark = "-" if row["not_dominated"] is None else ("✓" if row["not_dominated"] else "")
            metrics = row["metrics"]
            lines.append(f"| {scene} | {row['version']} | {seeds} | {_f(row['ply_size_mb'], 0)} "
                         f"| {_f(row['bytes_per_gaussian'], 0)} | {count} | {_f(metrics['PSNR'], 3)} "
                         f"| {_f(metrics['SSIM'], 4)} | {_f(metrics['LPIPS'], 4)} | {mark} |")
        for item in table["filtered_left_out"]:
            reason = f"{item['ok']}/{item['total']} lượt trên các seed {seeds} đã lọc xong"
            if item["failed"]:
                reason += f", {item['failed']} lượt lọc lỗi"
            if item["outside"]:
                reason += f", {item['outside']} lượt đã lọc ở seed ngoài các seed chung"
            notes.append(f"- {scene}: bản đã lọc của {item['variant']} không đưa vào bảng này ({reason})")
    return lines + ([""] + notes if notes else [])


def structure_lines(report: dict) -> list[str]:
    """Cấu trúc mô hình từ splat_stats.json, chỉ khi có lượt đã tính."""
    if not report["structure"]:
        return []
    rows, notes, unchecked = [], [], 0
    for scene, shape in report["structure"].items():
        if not shape["seeds"]:
            notes.append(f"- {scene}: các biến thể có thống kê ({', '.join(shape['with_stats'])}) không có seed chung, "
                         "không có dòng nào")
        seeds = ",".join(str(seed) for seed in shape["seeds"])
        for variant, s in shape["variants"].items():
            degree = ",".join(str(d) for d in s["sh_degree"]) or "-"
            below, above = s["opacity_below"], s["largest_rel_above"]
            rows.append(f"| {scene} | {variant} | {seeds} | {degree} | {_pct(below['0.005'])} | {_pct(below['0.02'])} "
                        f"| {_pct(below['0.1'])} | {_f(s['opacity_p50'], 3)} | {_pct(above['0.1'], 2)} "
                        f"| {_f(s['largest_rel_p50'], 4)} |")
            unchecked += s["unchecked"]
    lines = ["", "Cấu trúc mô hình, đọc từ PLY cuối (src.evaluation.splat_stats), trung bình trên các seed mà mọi biến thể "
                 "có thống kê của cảnh đều có (cột Seed). Đây là thước đo gián tiếp: Gaussian gần trong suốt hay rất to chưa "
                 "chắc là floater; floater thật xem ở ảnh so sánh và video độ sâu (scripts/run_panels.sh)."]
    if rows:
        lines += ["", "| Cảnh | Biến thể | Seed | Bậc SH | opacity < 0,005 | opacity < 0,02 | opacity < 0,1 | opacity trung vị "
                      "| trục lớn nhất > 0,1 × scene_scale | trục lớn nhất trung vị (× scene_scale) |",
                  "|---|---|---|---|---|---|---|---|---|---|"] + rows
    if unchecked:
        notes.append(f"- {unchecked} lượt có thống kê nhưng không còn file PLY để kiểm lại (vd. gói kết quả nhẹ): thống kê "
                     "là của PLY lúc tính.")
    return lines + ([""] + notes if notes else [])


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
    lines += ["", "`Render (ms/ảnh)` ở mọi bảng: thời gian gsplat đo khi chấm ảnh test, trong tiến trình chấm riêng, gồm cả "
                  "lần gọi đầu (nạp CUDA). Không phải tốc độ render của mô hình; không đổi ra FPS."]
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
    lines += wins_lines(report)
    lines += filtered_lines(report)
    lines += size_quality_lines(report)
    lines += structure_lines(report)
    if report.get("reference_check"):
        ref = report["reference_check"]
        lines += ["", f"Đối chiếu `{ref['mode']}` với số liệu công bố ({', '.join(ref['scenes'])}): "
                      f"**{ref['verdict']}**"]
        if ref.get("reasons"):
            label = "bỏ qua" if ref["verdict"] == "SKIPPED" else "không so được"
            lines += [f"- {label}: {reason}" for reason in ref["reasons"]]
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
                        help="biến thể baseline (mặc định default; lượt chạy thử: 'default{max_steps=3000}'); tên một "
                             "chế độ chỉ có một biến thể cũng được, vd. mcmc để so một chế độ khác với mcmc")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="các chế độ đưa vào bảng (mặc định: mọi chế độ có lượt chạy dưới --root)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="chỉ các cảnh này; khi chỉ là một phần các cảnh của benchmark, phần đối chiếu với số "
                             "công bố (trung bình mọi cảnh) ghi SKIPPED, đối chiếu B0 chỉ trên các cảnh này")
    parser.add_argument("--include-deviating", action="store_true",
                        help="giữ lượt sai khác protocol thành biến thể riêng thay vì loại bỏ")
    parser.add_argument("--tags", nargs="+", default=[], metavar="TAG",
                        help="đưa cả các lượt có tag này vào bảng (biến thể của một kế hoạch thực nghiệm, mục variants); "
                             "mặc định lượt có tag bị để ngoài")
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
    skip_reference = None
    if args.scenes:
        wanted = set(args.scenes)
        runs = [run for run in runs if run.scene in wanted]
        if scenes and set(scenes) != wanted:
            skip_reference = (f"chỉ chạy {len(wanted)} cảnh ({', '.join(args.scenes)}), số liệu công bố là trung bình "
                              f"của {len(scenes)} cảnh ({', '.join(scenes)})")
        scenes = list(args.scenes)
    runs, left_out = select_tags(runs, args.tags)
    baseline = resolve_baseline(runs, args.baseline, args.include_deviating)
    report = build_report(runs, baseline, args.modes, reference, tolerance, scenes,
                          args.include_deviating, iteration, skip_reference)
    if left_out:
        report["warnings"].append(
            f"left out {len(left_out)} run(s) with a tag ({', '.join(sorted({run.tag for run in left_out}))}): variants "
            "of another experiment plan; pass their tags with --tags (TAGS=... for scripts/run_evaluate.sh) to include them")
    if args.b0_root:
        if config is None:
            parser.error("--b0-root needs --reference-config (scenes, method, iteration, tolerance)")
        b0_scenes = scenes or sorted(report["scenes"])
        report["b0_check"] = b0_check(report, args.b0_root, b0_scenes, config.get("method", "ours_30000"),
                                      iteration, tolerance, baseline_runs_of(runs, baseline, b0_scenes))
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
        if any(verdict not in STRICT_OK for verdict in verdicts):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
