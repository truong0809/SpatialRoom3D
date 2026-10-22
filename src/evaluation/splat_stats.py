"""Thống kê cấu trúc của mô hình Gaussian đã train, đọc từ file PLY cuối của lượt (gsplat `export_splats`).

Dùng để so các chế độ ngoài PSNR / SSIM / LPIPS (docs/design/experiments.md mục 2.5):
- dung lượng: số Gaussian, byte mỗi Gaussian, bậc SH (theo số hệ số `f_rest`);
- thước đo gián tiếp của lớp mờ và floater: tỉ lệ Gaussian gần trong suốt (opacity < 0,005 / 0,02 / 0,1; 0,02 là ngưỡng
  mặc định của bộ lọc, src/training/prune.py), tỉ lệ Gaussian rất to (trục lớn nhất > 0,01 / 0,1 × scene_scale; 0,1 là
  ngưỡng của bộ lọc), opacity và trục lớn nhất trung vị.
Đây là thống kê của mô hình, không phải phép đo floater: Gaussian gần trong suốt chưa chắc là floater, và floater thật chỉ
thấy được qua ảnh render và video độ sâu (src/evaluation/panels.py).

    python -m src.evaluation.splat_stats --root /workspace/outputs/benchmarks/deep_blending/gsplat [--root ...] [--force]

Ghi `<lượt>/splat_stats.json` cạnh `run_meta.json`. Đã có, file PLY không đổi (cùng kích thước, cùng thời điểm sửa) và
cùng `scene_scale` thì bỏ qua. `scene_scale` lấy từ bản đã lọc (`filtered/filter_meta.json`) nếu có, nếu không từ dòng "Scene scale:" mà
simple_trainer.py in ra trong `train.log`; không có thì các tỉ lệ theo `scene_scale` để trống.
Cần numpy (env gs-gsplat có sẵn). Mã thoát: 0 khi mọi lượt đọc được hoặc được bỏ qua; 1 khi có PLY đọc lỗi.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from src.evaluation.gsplat_results import (FILTER_DIR, FILTER_META, SPLAT_STATS, SPLAT_STATS_VERSION, load_splat_stats,
                                           read_json_or_none)
from src.training.prune import logged_scene_scale

OPACITY_BELOW = (0.005, 0.02, 0.1)  # tỉ lệ Gaussian có opacity (sau sigmoid) nhỏ hơn các ngưỡng này
LARGEST_ABOVE = (0.01, 0.1)  # tỉ lệ Gaussian có trục lớn nhất > ngưỡng × scene_scale
CHUNK = 1 << 20  # số Gaussian mỗi lần đọc (giới hạn bộ nhớ khi PLY lớn)


class PlyError(ValueError):
    pass


def final_ply(run_dir: Path) -> Path | None:
    """PLY ở bước cuối của lượt (như gsplat_results.load_run): ply/point_cloud_<max_steps - 1>.ply."""
    meta = read_json_or_none(Path(run_dir) / "run_meta.json") or {}
    steps = ((meta.get("spec") or {}).get("protocol") or {}).get("max_steps")
    if steps is None:
        return None
    return Path(run_dir) / "ply" / f"point_cloud_{int(steps) - 1}.ply"


def read_header(path: Path) -> tuple[int, list[str], int]:
    """(số Gaussian, tên các thuộc tính, độ dài header) của một PLY nhị phân little-endian, mọi thuộc tính là float."""
    names, count, fmt = [], None, None
    with open(path, "rb") as handle:
        for _ in range(4096):
            raw = handle.readline()
            if not raw:
                raise PlyError("no end_header")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts or parts[0] in ("ply", "comment", "obj_info"):
                continue
            if parts[0] == "format":
                fmt = " ".join(parts[1:])
            elif parts[0] == "element":
                if parts[1] != "vertex" or count is not None:
                    raise PlyError(f"unexpected element: {line}")
                count = int(parts[2])
            elif parts[0] == "property":
                if len(parts) != 3 or parts[1] not in ("float", "float32"):
                    raise PlyError(f"only float properties are supported: {line}")
                names.append(parts[2])
        else:
            raise PlyError("header too long")
        header = handle.tell()
    if fmt != "binary_little_endian 1.0":
        raise PlyError(f"unsupported format: {fmt}")
    if count is None:
        raise PlyError("no vertex element")
    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    missing = [name for name in required if name not in names]
    if missing:
        raise PlyError(f"missing properties: {missing}")
    return count, names, header


def sh_degree(names: list[str]) -> int | None:
    rest = sum(name.startswith("f_rest_") for name in names)
    degree = math.isqrt(rest // 3 + 1) - 1
    return degree if rest % 3 == 0 and (degree + 1) ** 2 - 1 == rest // 3 else None


def scene_scale_of(run_dir: Path) -> tuple[float | None, str | None]:
    meta = read_json_or_none(Path(run_dir) / FILTER_DIR / FILTER_META) or {}
    if isinstance(meta.get("scene_scale"), (int, float)) and meta["scene_scale"] > 0:
        return float(meta["scene_scale"]), "filter_meta"
    logged = logged_scene_scale(Path(run_dir))
    if logged is not None and logged > 0:
        return float(logged), "train_log"
    return None, None


def ply_stats(path: Path, scene_scale: float | None = None) -> dict:
    """Thống kê một PLY của gsplat (opacity lưu dạng logit, scale dạng log)."""
    import numpy as np

    count, names, header = read_header(path)
    size = path.stat().st_size
    itemsize = 4 * len(names)
    if size != header + count * itemsize:
        raise PlyError(f"file size {size} != header {header} + {count} × {itemsize} bytes (truncated?)")
    if count == 0:
        raise PlyError("no Gaussians")
    data = np.memmap(path, dtype=np.dtype([(name, "<f4") for name in names]), mode="r", offset=header, shape=(count,))
    opacity = np.empty(count, dtype=np.float32)
    largest = np.empty(count, dtype=np.float32)
    for start in range(0, count, CHUNK):
        part = data[start:start + CHUNK]
        logits = part["opacity"].astype(np.float64)
        opacity[start:start + len(part)] = 1.0 / (1.0 + np.exp(-logits))
        scales = np.stack([part["scale_0"], part["scale_1"], part["scale_2"]], axis=1).astype(np.float64)
        largest[start:start + len(part)] = np.exp(scales.max(axis=1))
    del data
    finite = np.isfinite(opacity) & np.isfinite(largest)
    opacity, largest = opacity[finite], largest[finite]
    if opacity.size == 0:
        raise PlyError("no finite Gaussian")

    def quantiles(values):
        return {f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90, 99)}

    stats = {
        "num_gaussians": count,
        "non_finite": int(count - opacity.size),
        "bytes_per_gaussian": itemsize,
        "sh_degree": sh_degree(names),
        "opacity": {**quantiles(opacity), "below": {str(t): float(np.mean(opacity < t)) for t in OPACITY_BELOW}},
        "largest_axis": quantiles(largest),
        "scene_scale": scene_scale,
    }
    if scene_scale:
        relative = largest / scene_scale
        stats["largest_axis_relative"] = {**quantiles(relative),
                                          "above": {str(t): float(np.mean(relative > t)) for t in LARGEST_ABOVE}}
    return stats


def update_run(run_dir: Path, force: bool = False) -> tuple[str, str]:
    """(trạng thái, mô tả): OK (vừa tính), CACHED (đã có, còn khớp), SKIP (không có PLY cuối), FAILED."""
    run_dir = Path(run_dir)
    ply = final_ply(run_dir)
    if ply is None or not ply.is_file():
        return "SKIP", "no final PLY"
    scale, source = scene_scale_of(run_dir)
    cached = None if force else load_splat_stats(run_dir)
    if cached is not None and cached.get("ply_checked") and cached.get("scene_scale") == scale:
        return "CACHED", ""  # cùng PLY, cùng scene_scale (vd. lượt vừa được lọc thì scene_scale có thể mới có)
    st = ply.stat()
    record = {"version": SPLAT_STATS_VERSION, "ply": str(ply.relative_to(run_dir)), "ply_bytes": st.st_size,
              "ply_mtime_ns": st.st_mtime_ns, "scene_scale_source": source}
    try:
        record.update(ply_stats(ply, scale), status="OK")
    except (PlyError, OSError, ValueError) as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    tmp = run_dir / (SPLAT_STATS + ".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / SPLAT_STATS)
    if record["status"] != "OK":
        return "FAILED", record["error"]
    below = record["opacity"]["below"]["0.02"] * 100
    above = record.get("largest_axis_relative", {}).get("above", {}).get("0.1")
    text = (f"{record['num_gaussians']:,} Gaussian, {record['bytes_per_gaussian']} B/Gaussian, SH {record['sh_degree']}, "
            f"opacity < 0.02: {below:.1f}%")
    if above is not None:
        text += f", trục > 0.1 × scene_scale: {above * 100:.2f}%"
    return "OK", text


def runs_under(roots: list[Path]) -> list[Path]:
    found, seen = [], set()
    for root in roots:
        for meta in sorted(Path(root).rglob("run_meta.json")):
            folder = meta.parent.resolve()
            if folder in seen or any(".prev_" in part for part in meta.parent.parts):
                continue
            seen.add(folder)
            found.append(meta.parent)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path, action="append", help="thư mục chứa các lượt (lặp lại được)")
    parser.add_argument("--force", action="store_true", help="tính lại cả khi splat_stats.json còn khớp PLY")
    args = parser.parse_args(argv)
    failed = 0
    for run in runs_under(args.root):
        status, text = update_run(run, args.force)
        if status != "CACHED":
            print(f"{status:6} {run}" + (f": {text}" if text else ""), file=sys.stderr if status == "FAILED" else sys.stdout)
        failed += status == "FAILED"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
