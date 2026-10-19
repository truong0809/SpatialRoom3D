"""Kế hoạch một thực nghiệm (configs/experiments/<tên>.yaml): các chế độ huấn luyện, seed, benchmark và cảnh, có chạy
B0 không, ước lượng. scripts/pod/run_experiment.sh đọc kế hoạch qua module này; module chỉ đọc và kiểm tra cấu hình,
không chạy gì.

    python3 -m src.training.experiment --plan main                        # các dòng "khóa|giá trị" cho bash
    python3 -m src.training.experiment --export main --out kế_hoạch.json  # bản chụp kế hoạch đã kiểm tra
    python3 -m src.training.experiment --space main                       # chỗ còn cần cho các lượt chưa xong

Kiểm tra:
  * `modes` có trong train_modes của configs/base.yaml và gồm chế độ baseline (default); `also_against` (tùy chọn):
    các chế độ khác trong `modes` dùng làm mốc cho thêm một bảng so sánh (vd. một chế độ mới so với mcmc);
  * `seeds` là số nguyên không âm, không trùng;
  * mỗi `benchmarks[].dataset` là một luồng `kind: benchmark` có mục download (src/datasets/fetch.py), `scenes` thuộc
    danh sách cảnh của luồng, `b0` là true / false;
  * `estimate.gpu_hours`, `estimate.disk_gb`: số dương (chỉ để in và kiểm tra chỗ trống).
Chỗ còn cần (`--space`, xem `space_needs`): mỗi lượt gsplat chưa xong của một seed tính bằng dung lượng thật của lượt cùng
cảnh, cùng chế độ đã xong ở seed khác (lớn nhất); chưa có thì bằng phần chia đều của `estimate.disk_gb` cho mọi lượt.
Cần PyYAML (python3 của template Runpod thường có, ô 1 của notebook kiểm tra; hoặc env gs-gsplat).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

from src.datasets.registry import CONFIG_DIR, load_registry, read_yaml

EXPERIMENTS = "experiments"  # configs/experiments/<tên>.yaml
NAME = re.compile(r"[a-z0-9_]+")


def experiment_path(config_dir: Path, name: str) -> Path:
    if not NAME.fullmatch(name):
        raise ValueError(f"invalid experiment name {name!r} (lowercase letters, digits and '_')")
    return Path(config_dir) / EXPERIMENTS / f"{name}.yaml"


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} must be a positive number")
    return value


def load_experiment(name: str, config_dir: Path = CONFIG_DIR, ws: str = "/workspace") -> dict:
    """Đọc và kiểm tra một kế hoạch; lỗi là ValueError có tên file."""
    path = experiment_path(config_dir, name)
    if not path.is_file():
        available = sorted(p.stem for p in (Path(config_dir) / EXPERIMENTS).glob("*.yaml"))
        raise ValueError(f"no experiment {name!r}: {path} not found (available: {available})")
    raw = read_yaml(path)
    where = f"{path.name}"
    registry = load_registry(config_dir, ws)
    train_modes = registry.get("train_modes") or {}
    known_modes = train_modes.get("modes") or {}
    baseline = train_modes.get("baseline_mode", "default")

    modes = raw.get("modes")
    if not isinstance(modes, list) or not modes or not all(isinstance(m, str) for m in modes):
        raise ValueError(f"{where}: modes must be a non-empty list of mode names")
    unknown = [mode for mode in modes if mode not in known_modes]
    if unknown:
        raise ValueError(f"{where}: unknown mode(s) {unknown}; configs/base.yaml has {sorted(known_modes)}")
    if baseline not in modes:
        raise ValueError(f"{where}: modes must include the baseline mode {baseline!r}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"{where}: modes are listed twice")
    modes = [baseline] + [mode for mode in modes if mode != baseline]  # baseline chạy trước, các chế độ khác lấy ngân sách từ nó
    also = raw.get("also_against") or []
    if (not isinstance(also, list) or any(mode not in modes or mode == baseline for mode in also)
            or len(set(also)) != len(also)):
        raise ValueError(f"{where}: also_against must list distinct modes of the plan other than {baseline!r}")

    seeds = raw.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)):
        raise ValueError(f"{where}: seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{where}: a seed is listed twice")

    benches = raw.get("benchmarks")
    if not isinstance(benches, list) or not benches:
        raise ValueError(f"{where}: benchmarks must be a non-empty list")
    datasets = registry.get("datasets") or {}
    plan_benches = []
    for index, bench in enumerate(benches):
        here = f"{where}: benchmarks[{index}]"
        if not isinstance(bench, dict):
            raise ValueError(f"{here} must be a mapping with dataset, scenes, b0")
        dataset = bench.get("dataset")
        entry = datasets.get(dataset)
        if entry is None or entry.get("kind") != "benchmark":
            benchmarks = sorted(n for n, e in datasets.items() if e.get("kind") == "benchmark")
            raise ValueError(f"{here}: dataset must be one of the benchmarks {benchmarks}, not {dataset!r}")
        if not entry.get("download"):
            raise ValueError(f"{here}: configs/{dataset}.yaml has no download section")
        known = [str(scene) for scene in entry.get("scenes") or []]
        scenes = bench.get("scenes") or known
        if not isinstance(scenes, list) or not all(isinstance(s, str) for s in scenes):
            raise ValueError(f"{here}: scenes must be a list of scene names")
        if [s for s in scenes if s not in known] or len(set(scenes)) != len(scenes):
            raise ValueError(f"{here}: scenes must be distinct scenes of {dataset} ({known}), got {scenes}")
        b0 = bench.get("b0", False)
        if not isinstance(b0, bool):
            raise ValueError(f"{here}: b0 must be true or false")
        if any(other["dataset"] == dataset for other in plan_benches):
            raise ValueError(f"{here}: {dataset} is listed twice")
        plan_benches.append({"dataset": dataset, "scenes": scenes, "b0": b0,
                             "all_scenes": sorted(scenes) == sorted(known)})

    estimate = raw.get("estimate") or {}
    if not isinstance(estimate, dict):
        raise ValueError(f"{where}: estimate must be a mapping")
    gpu_hours = _number(estimate.get("gpu_hours"), f"{where}: estimate.gpu_hours")
    disk_gb = _number(estimate.get("disk_gb"), f"{where}: estimate.disk_gb")
    return {
        "name": name,
        "description": str(raw.get("description") or ""),
        "config_file": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "modes": modes,
        "also_against": also,
        "seeds": seeds,
        "benchmarks": plan_benches,
        "estimate": {"gpu_hours": gpu_hours, "disk_gb": disk_gb},
    }


def tree_bytes(path: Path) -> int:
    """Tổng kích thước khai báo của một cây thư mục, như `du -sb`: file, thư mục, liên kết (không đi theo liên kết);
    liên kết cứng đếm một lần."""
    total, seen, stack = 0, set(), [str(path)]
    while stack:
        current = stack.pop()
        try:
            info = os.lstat(current)
        except OSError:
            continue
        if info.st_nlink > 1 and not stat.S_ISDIR(info.st_mode):
            if (info.st_dev, info.st_ino) in seen:
                continue
            seen.add((info.st_dev, info.st_ino))
        total += info.st_size
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(current) as entries:
                    stack.extend(entry.path for entry in entries)
            except OSError:
                pass
    return total


def run_finished(run_dir: Path) -> bool:
    """Lượt gsplat đã xong: summary.json (src/training/run.py ghi sau cùng) có status OK."""
    try:
        return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8")).get("status") == "OK"
    except (OSError, ValueError, AttributeError):
        return False


def space_needs(plan: dict, config_dir: Path = CONFIG_DIR, ws: str = "/workspace",
                datasets: list[str] | None = None) -> dict:
    """Byte còn cần ghi cho từng seed: tổng của các lượt gsplat chưa xong (benchmark × cảnh × chế độ) của seed đó;
    `datasets`: chỉ tính các benchmark này (vd. bỏ benchmark không tải được), mặc định mọi benchmark của kế hoạch.

    Một lượt chưa xong được tính bằng dung lượng thật lớn nhất của lượt cùng benchmark, cùng cảnh, cùng chế độ đã xong
    ở seed khác (kích thước gần như không đổi giữa các seed); chưa có lượt nào như vậy thì bằng `estimate.disk_gb` chia
    đều cho mọi lượt của kế hoạch (phần này gồm cả dữ liệu, B0 và ảnh PNG thu nhỏ, chỉ ghi một lần). Lượt đã xong không
    cần thêm gì, kể cả khi chạy lại sau khi pod dừng giữa chừng."""
    registry = load_registry(config_dir, ws)
    runs = []
    for bench in plan["benchmarks"]:
        root = Path(registry["datasets"][bench["dataset"]]["results"]["gsplat"])
        for scene in bench["scenes"]:
            for mode in plan["modes"]:
                for seed in plan["seeds"]:
                    runs.append(((bench["dataset"], scene, mode), seed, root / scene / f"{mode}_s{seed}"))
    estimate = round(plan["estimate"]["disk_gb"] * 1e9 / len(runs))
    if datasets is not None:
        unknown = sorted(set(datasets) - {bench["dataset"] for bench in plan["benchmarks"]})
        if unknown:
            raise ValueError(f"benchmark(s) {unknown} are not in the plan {plan['name']}")
        runs = [run for run in runs if run[0][0] in datasets]
    sizes = {(key, seed): tree_bytes(path) for key, seed, path in runs if run_finished(path)}
    seeds = []
    for seed in plan["seeds"]:
        need = finished = measured = total = 0
        for key, run_seed, _ in runs:
            if run_seed != seed:
                continue
            total += 1
            if (key, seed) in sizes:
                finished += 1
                continue
            others = [size for (other_key, other_seed), size in sizes.items() if other_key == key]
            if others:
                need += max(others)
                measured += 1
            else:
                need += estimate
        seeds.append({"seed": seed, "need_bytes": need, "finished": finished, "runs": total, "measured": measured})
    return {"run_estimate_bytes": estimate, "seeds": seeds}


def plan_lines(plan: dict) -> list[str]:
    """Kế hoạch dạng "khóa|giá trị" (danh sách nối bằng dấu cách) cho scripts/pod/run_experiment.sh."""
    def words(values) -> str:
        return " ".join(str(value) for value in values)
    lines = [f"name|{plan['name']}", f"description|{plan['description'].replace('|', '/')}",
             f"modes|{words(plan['modes'])}", f"also_against|{words(plan['also_against'])}",
             f"seeds|{words(plan['seeds'])}",
             f"gpu_hours|{plan['estimate']['gpu_hours']:g}", f"disk_gb|{plan['estimate']['disk_gb']:g}"]
    for bench in plan["benchmarks"]:
        lines.append(f"bench|{bench['dataset']}|{1 if bench['b0'] else 0}|{words(bench['scenes'])}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=CONFIG_DIR, help="thư mục cấu hình (mặc định configs/ của repo)")
    parser.add_argument("--ws", default=os.environ.get("WS") or "/workspace", help="thư mục gốc trên pod (WS)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", metavar="TÊN", help="in kế hoạch dạng khóa|giá trị")
    action.add_argument("--export", metavar="TÊN", help="ghi kế hoạch đã kiểm tra ra JSON (--out)")
    action.add_argument("--space", metavar="TÊN", help="byte còn cần cho từng seed: dòng seed|<seed>|<byte>|<xong>|<lượt>|<đo>")
    parser.add_argument("--datasets", nargs="*", help="với --space: chỉ tính các benchmark này")
    parser.add_argument("--out", type=Path, help="file JSON cho --export")
    args = parser.parse_args(argv)
    ws = args.ws.rstrip("/") or "/"
    try:
        plan = load_experiment(args.plan or args.export or args.space, args.configs, ws)
        space = space_needs(plan, args.configs, ws, args.datasets) if args.space else None
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if space is not None:
        print(f"run_estimate|{space['run_estimate_bytes']}")
        for row in space["seeds"]:
            print(f"seed|{row['seed']}|{row['need_bytes']}|{row['finished']}|{row['runs']}|{row['measured']}")
        return 0
    if args.plan:
        print("\n".join(plan_lines(plan)))
        return 0
    if args.out is None:
        print("ERROR: --export needs --out <file.json>", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
