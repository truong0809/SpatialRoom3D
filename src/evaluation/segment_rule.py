"""Kiểm tra đủ lượt và áp luật ghi trước ở docs/design/experiments.md mục 3.8 cho lượt đánh giá theo đoạn trên phòng tự
quay (scripts/pod/run_segments.sh): 3 phòng × (default, mcmc, 3 biến thể của mcmc) × 3 seed.

Chỉ đọc summary.json (lượt chưa xong: run_meta.json) của <root>/<phòng>/<cấu hình>_s<seed>/, không ghi gì vào đó; thư mục
*.prev_* (lượt cũ đã được thay khi chạy lại) không được đọc. Báo cáo gồm:
1. độ đầy đủ: mỗi phòng × cấu hình × seed là OK, FAILED, RUNNING hay chưa có;
2. điều kiện so sánh trong từng phòng: mọi lượt cùng tập chia, cùng commit gsplat, cùng GPU; mcmc và mọi biến thể có số
   Gaussian bằng default cùng seed (ngân sách EQUAL);
3. mức dao động của từng phòng, chỉ từ 6 lượt default, mcmc: σ_r = √((σ_default² + σ_mcmc²) / 2) (độ lệch chuẩn mẫu trên 3
   seed), d_r = √2·σ_r; chênh lệch trung bình theo cặp cùng seed (mcmc − default; biến thể − mcmc);
4. kết luận: biến thể là cải thiện so với mcmc khi (1) mọi điều kiện ở mục 2 đạt ở cả 3 phòng, (2) ở ít nhất 2 phòng nó tốt
   hơn quá d_r ở ít nhất một chỉ số (PSNR, SSIM cao hơn; LPIPS thấp hơn), (3) không phòng nào nó kém quá d_r ở chỉ số nào.
   Thiếu số ở phòng nào thì kết luận là CHƯA ĐỦ SỐ.
Các mục kiểm tra khác (protocol, mô hình COLMAP, ...) nằm trong báo cáo của src.evaluation.compare_modes
(scripts/run_evaluate.sh); kết luận ở đây không thay cho việc đọc các mục đó.

    python -m src.evaluation.segment_rule [--root /workspace/outputs/custom_segments] [--markdown <file.md>] [--json <file.json>]

Exit 0 khi đủ mọi lượt, 1 khi còn thiếu lượt (báo cáo vẫn được in và ghi).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOMS = ("video8", "video9", "video10")
SEEDS = (42, 43, 44)
BASELINE, REFERENCE = "default", "mcmc"
VARIANTS = ("rbg", "mono", "kr")
CONFIGS = (BASELINE, REFERENCE) + tuple(f"{REFERENCE}_{variant}" for variant in VARIANTS)
METRICS = (("PSNR", 1, 3), ("SSIM", 1, 4), ("LPIPS", -1, 4))  # (tên, +1 cao hơn là tốt / -1 thấp hơn là tốt, số lẻ)
MIN_ROOMS_BETTER = 2
SAME = (("split", "split_sha256"), ("gsplat_commit", "gsplat_commit"), ("gpu", "gpu"))
WORDS = {"better": "tốt hơn", "worse": "KÉM", "within": "dao động"}


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_run(folder: Path) -> dict:
    """{"status": OK | FAILED | RUNNING | MISSING | NO_METRICS | ..., "summary": summary.json khi OK, không thì None}."""
    summary = _read(folder / "summary.json")
    if summary is None:
        meta = _read(folder / "run_meta.json")
        return {"status": str(meta.get("status") or "?") if meta else "MISSING", "summary": None}
    metrics = summary.get("metrics") or {}
    status = str(summary.get("status") or "?")
    if status == "OK" and not all(isinstance(metrics.get(name), (int, float)) and math.isfinite(metrics[name])
                                  for name, _, _ in METRICS):
        status = "NO_METRICS"
    return {"status": status, "summary": summary if status == "OK" else None}


def _metric(run: dict, name: str) -> float:
    return float(run["summary"]["metrics"][name])


def spread(runs: dict, seeds=SEEDS) -> dict | None:
    """d_r của mỗi chỉ số từ các lượt default, mcmc; None khi thiếu lượt nào trong số đó."""
    if not all(runs[config][seed]["summary"] for config in (BASELINE, REFERENCE) for seed in seeds):
        return None
    result = {}
    for name, _, _ in METRICS:
        variances = [statistics.stdev(_metric(runs[config][seed], name) for seed in seeds) ** 2
                     for config in (BASELINE, REFERENCE)]
        result[name] = math.sqrt(2) * math.sqrt(sum(variances) / 2)
    return result


def paired(runs: dict, first: str, second: str, d: dict | None, seeds=SEEDS) -> dict | None:
    """Trung bình chênh lệch cùng seed (first − second) và nhãn so với d_r; None khi thiếu lượt."""
    if not all(runs[first][seed]["summary"] and runs[second][seed]["summary"] for seed in seeds):
        return None
    deltas, flags = {}, {}
    for name, sign, _ in METRICS:
        delta = statistics.fmean(_metric(runs[first][seed], name) - _metric(runs[second][seed], name) for seed in seeds)
        deltas[name] = delta
        if d is not None:
            flags[name] = "better" if sign * delta > d[name] else "worse" if sign * delta < -d[name] else "within"
    return {"delta": deltas, "flags": flags}


def room_report(runs: dict, seeds=SEEDS) -> dict:
    ok = [run["summary"] for by_seed in runs.values() for run in by_seed.values() if run["summary"]]
    same = {}
    for name, key in SAME:
        values = sorted({str(summary.get(key)) for summary in ok})
        same[name] = {"ok": len(values) <= 1 and "None" not in values, "values": values}
    budget = {}
    for config in CONFIGS[1:]:
        mismatches = []
        for seed in seeds:
            run, base = runs[config][seed]["summary"], runs[BASELINE][seed]["summary"]
            if run and base and run.get("num_gaussians") != base.get("num_gaussians"):
                mismatches.append({"seed": seed, "gaussians": run.get("num_gaussians"), "default": base.get("num_gaussians")})
        budget[config] = mismatches
    stats = {}
    for config in CONFIGS:
        found = [run for run in runs[config].values() if run["summary"]]
        stats[config] = {"n": len(found), **{name: {
            "mean": statistics.fmean(_metric(run, name) for run in found) if found else None,
            "std": statistics.stdev(_metric(run, name) for run in found) if len(found) > 1 else None}
            for name, _, _ in METRICS}}
    d = spread(runs, seeds)
    comparisons = {f"{REFERENCE} - {BASELINE}": paired(runs, REFERENCE, BASELINE, d, seeds)}
    for variant in VARIANTS:
        comparisons[f"{variant} - {REFERENCE}"] = paired(runs, f"{REFERENCE}_{variant}", REFERENCE, d, seeds)
    return {"status": {config: {str(seed): runs[config][seed]["status"] for seed in seeds} for config in CONFIGS},
            "same": same, "budget": budget, "stats": stats, "d": d, "comparisons": comparisons}


def verdict(rooms: dict, variant: str) -> dict:
    config = f"{REFERENCE}_{variant}"
    better, worse, unfair, missing = {}, {}, [], []
    for room, info in rooms.items():
        comparison = info["comparisons"][f"{variant} - {REFERENCE}"]
        if info["d"] is None or comparison is None:
            missing.append(room)
            continue
        flags = comparison["flags"]
        if [name for name, flag in flags.items() if flag == "better"]:
            better[room] = [name for name, flag in flags.items() if flag == "better"]
        if [name for name, flag in flags.items() if flag == "worse"]:
            worse[room] = [name for name, flag in flags.items() if flag == "worse"]
        if not all(check["ok"] for check in info["same"].values()) or info["budget"][config] or info["budget"][REFERENCE]:
            unfair.append(room)
    reasons = []
    if unfair:
        reasons.append(f"điều kiện so sánh không đạt ở {', '.join(unfair)}")
    if worse:
        reasons.append("kém quá d_r ở " + "; ".join(f"{room} ({', '.join(names)})" for room, names in worse.items()))
    if len(better) < MIN_ROOMS_BETTER and not missing:
        reasons.append(f"chỉ tốt hơn quá d_r ở {len(better)} phòng (cần {MIN_ROOMS_BETTER})")
    if missing:
        status = "CHƯA ĐỦ SỐ"
        reasons.insert(0, f"thiếu số ở {', '.join(missing)}")
    else:
        status = "KHÔNG ĐẠT" if reasons else "ĐẠT"
    return {"status": status, "better": better, "worse": worse, "unfair": unfair, "missing": missing,
            "reasons": reasons}


def build_report(root: Path, rooms=ROOMS, seeds=SEEDS) -> dict:
    runs = {room: {config: {seed: load_run(Path(root) / room / f"{config}_s{seed}") for seed in seeds}
                   for config in CONFIGS} for room in rooms}
    per_room = {room: room_report(runs[room], seeds) for room in rooms}
    not_ok = [f"{room}/{config}_s{seed} ({run['status']})" for room in rooms for config in CONFIGS
              for seed, run in runs[room][config].items() if not run["summary"]]
    total = len(rooms) * len(CONFIGS) * len(seeds)
    return {"root": str(root), "rooms": per_room, "seeds": list(seeds), "total": total, "ok": total - len(not_ok),
            "not_ok": not_ok, "verdicts": {variant: verdict(per_room, variant) for variant in VARIANTS}}


def _num(value, digits) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _signed(value, digits) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def markdown(report: dict) -> str:
    rooms, seeds = report["rooms"], report["seeds"]
    lines = ["# Đánh giá theo đoạn: kiểm tra đủ lượt và luật mục 3.8", "",
             f"Thư mục: `{report['root']}` | phòng: {', '.join(rooms)} | seed: {', '.join(map(str, seeds))}. Luật ghi trước "
             "khi chạy: `docs/design/experiments.md` mục 3.8 (D26). Các mục kiểm tra khác của bảng so sánh "
             "(`src.evaluation.compare_modes`) vẫn phải đọc riêng.", "",
             "## 1. Độ đầy đủ (số seed OK)", "",
             "| Phòng | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for room, info in rooms.items():
        lines.append(f"| {room} | " + " | ".join(f"{info['stats'][config]['n']}/{len(seeds)}" for config in CONFIGS) + " |")
    lines += ["", f"Tổng: {report['ok']}/{report['total']} lượt OK."]
    if report["not_ok"]:
        lines.append("Chưa OK: " + ", ".join(report["not_ok"]) + ".")
    lines += ["", "## 2. Điều kiện so sánh trong từng phòng", "",
              "| Phòng | Tập chia | Commit gsplat | GPU | Số Gaussian bằng default cùng seed |", "|---|---|---|---|---|"]
    for room, info in rooms.items():
        cells = ["OK" if info["same"][name]["ok"] else "KHÁC: " + ", ".join(info["same"][name]["values"]) for name, _ in SAME]
        wrong = [f"{config} s{item['seed']} ({item['gaussians']} / {item['default']})"
                 for config, items in info["budget"].items() for item in items]
        lines.append(f"| {room} | " + " | ".join(cells) + f" | {'EQUAL' if not wrong else 'KHÁC: ' + ', '.join(wrong)} |")
    lines += ["", "## 3. Chỉ số và chênh lệch", "",
              "Trung bình ± độ lệch chuẩn trên các seed OK; chênh lệch là trung bình theo cặp cùng seed, nhãn so với d_r của "
              "phòng (tốt hơn / dao động / KÉM)."]
    for room, info in rooms.items():
        d = info["d"]
        lines += ["", f"### {room}", "",
                  "d_r: " + (" | ".join(f"{name} {_num(d[name], digits)}" for name, _, digits in METRICS) if d
                             else "chưa tính được (thiếu lượt default / mcmc)"), "",
                  "| Cấu hình | Seed OK | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
        for config in CONFIGS:
            stats = info["stats"][config]
            cells = [_num(stats[name]["mean"], digits) + ("" if stats[name]["std"] is None else
                                                          f" ± {stats[name]['std']:.{digits}f}")
                     for name, _, digits in METRICS]
            lines.append(f"| {config} | {stats['n']}/{len(seeds)} | " + " | ".join(cells) + " |")
        lines += ["", "| So sánh | ΔPSNR | ΔSSIM | ΔLPIPS |", "|---|---|---|---|"]
        for label, comparison in info["comparisons"].items():
            if comparison is None:
                lines.append(f"| {label} | chưa đủ seed | | |")
                continue
            cells = [_signed(comparison["delta"][name], digits)
                     + (f" ({WORDS[comparison['flags'][name]]})" if comparison["flags"] else "")
                     for name, _, digits in METRICS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## 4. Kết luận theo luật mục 3.8 (so với mcmc)", "",
              "| Biến thể | Phòng tốt hơn quá d_r | Phòng kém quá d_r | Kết luận |", "|---|---|---|---|"]
    for variant, result in report["verdicts"].items():
        better = "; ".join(f"{room} ({', '.join(names)})" for room, names in result["better"].items()) or "không"
        worse = "; ".join(f"{room} ({', '.join(names)})" for room, names in result["worse"].items()) or "không"
        lines.append(f"| {variant} | {better} | {worse} | **{result['status']}**"
                     + (f": {'; '.join(result['reasons'])}" if result["reasons"] else "") + " |")
    lines += ["", "ĐẠT: đủ cả ba điều của luật. Tên gọi khi báo cáo theo mục 3.8: `rbg` là chọn cấu hình, `mono` là tích "
                  "hợp kỹ thuật đã công bố, `kr` là heuristic của nhóm; không gọi là thuật toán mới."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("/workspace/outputs/custom_segments"))
    parser.add_argument("--markdown", type=Path, help="ghi báo cáo Markdown")
    parser.add_argument("--json", type=Path, help="ghi mọi số đã tính (JSON)")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"ERROR: {args.root} not found", file=sys.stderr)
        return 2
    report = build_report(args.root)
    text = markdown(report)
    print(text, end="")
    for path, content in ((args.markdown, text), (args.json, json.dumps(report, indent=2, ensure_ascii=False) + "\n")):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return 0 if not report["not_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
