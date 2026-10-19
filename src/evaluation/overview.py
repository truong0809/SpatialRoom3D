"""Tổng hợp nhanh kết quả trên pod: lượt cài môi trường (run_all) gần nhất, B0 (3DGS gốc của Inria) và hai chế độ
gsplat trên Deep Blending, các cảnh tự chụp / tự quay (ảnh đầu vào, COLMAP, các lượt train, file PLY để xem 3D).

Thư viện chuẩn + PyYAML (đọc configs/*.yaml), không cần env conda; chỉ đọc file. Chạy từ thư mục gốc của repo:
    python3 -m src.evaluation.overview [--ws /workspace] [--out-root DIR] [--scene phong1 ...]
Thường gọi qua: bash scripts/pod/results.sh [cảnh ...] (tự chọn python có PyYAML)

Nguồn đọc (mới nhất theo thời gian sửa file):
    outputs/reports/run_all_*.tsv               tóm tắt từng bước của scripts/pod/run_all.sh
    outputs/reports/baseline_db_summary_*.json  B0 so với số liệu Inria công bố (src.evaluation.compare_reference)
    outputs/reports/modes_db_<thời gian>.md     gsplat default/mcmc đầy đủ (src.evaluation.compare_modes)
    outputs/reports/modes_db_quick_*.json       lượt chạy thử 3000 vòng (chỉ kiểm tra luồng chạy)
    data/custom_scenes/<cảnh>/               nguồn gốc (ảnh / video) của cảnh tự chụp
    <out-root>/<cảnh>/colmap/{frames_report,sfm_report}.json, <out-root>/<cảnh>/<chế độ>_s<seed>/
    (hai thư mục gốc lấy từ configs/custom_scene.yaml: mặc định data/custom_scenes và outputs/custom_scenes, giống
    scripts/pod/run_scene.sh)
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
from pathlib import Path

from src.datasets.registry import CONFIG_DIR, PREPARED_SUBDIR, load_registry
from src.evaluation.gsplat_results import load_summary, read_json_or_none


def latest(folder: Path, pattern: str) -> Path | None:
    files = [path for path in folder.glob(pattern) if path.is_file()]
    return max(files, key=lambda path: (path.stat().st_mtime, path.name)) if files else None


def _f(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def _minutes(seconds) -> str:
    return "-" if seconds is None else f"{seconds / 60:.1f}"


def _count(value) -> str:
    return "-" if value is None else f"{value:,}"


def _hms(seconds: str) -> str:
    if not seconds.isdigit():
        return ""
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------------------------------- run_all
STATUS_LABEL = {"OK": "OK", "FAIL": "LỖI", "SKIP": "BỎ QUA"}


def run_all_section(ws: Path) -> list[str]:
    path = latest(ws / "outputs" / "reports", "run_all_*.tsv")
    if path is None:
        return ["== run_all: chưa có tóm tắt (chưa có lượt nào chạy xong)"]
    lines = [f"== run_all ({path.name})"]
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if row.startswith("# run_all"):
            lines.append("   " + row[2:])
        if not row.strip() or row.startswith("#"):
            continue
        step, status, code, seconds, note = (row.split("\t") + [""] * 5)[:5]
        label = STATUS_LABEL.get(status, status)
        if status == "FAIL":
            label += f" (exit {code})"
        detail = note if status == "SKIP" else _hms(seconds)
        lines.append(f"   {step:<13} {label:<14} {detail}")
    return lines


# ------------------------------------------------------------------------------------------------- B0
def b0_section(ws: Path) -> list[str]:
    path = latest(ws / "outputs" / "reports", "baseline_db_summary_*.json")
    data = read_json_or_none(path) if path else None
    if not data:
        return ["== B0 (3DGS gốc của Inria, Deep Blending): chưa có (khi cần: bash scripts/run_train.sh deep_blending b0)"]
    lines = [f"== B0 (3DGS gốc của Inria, Deep Blending) - {path.name}",
             f"   {'cảnh':<14} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'train(ph)':>9} {'Gaussian':>11}"]
    for scene in data.get("scenes") or []:
        metrics = scene.get("metrics") or {}
        if scene.get("status") == "OK":
            values = f"{_f(metrics.get('PSNR')):>7} {_f(metrics.get('SSIM')):>7} {_f(metrics.get('LPIPS')):>7}"
        else:
            values = f"{'THIẾU KẾT QUẢ':>23}"
        lines.append(f"   {scene.get('scene', '?'):<14} {values} {_minutes(scene.get('train_seconds')):>9} "
                     f"{_count(scene.get('num_gaussians')):>11}")
    mean, ref, delta = data.get("mean") or {}, data.get("reference") or {}, data.get("delta") or {}
    if mean:
        lines.append(f"   {'trung bình':<14} {_f(mean.get('PSNR')):>7} {_f(mean.get('SSIM')):>7} "
                     f"{_f(mean.get('LPIPS')):>7}")
        lines.append(f"   {'Inria công bố':<14} {_f(ref.get('PSNR')):>7} {_f(ref.get('SSIM')):>7} "
                     f"{_f(ref.get('LPIPS')):>7}")
        if delta:
            lines.append(f"   {'chênh lệch':<14} {delta.get('PSNR', 0):>+7.3f} {delta.get('SSIM', 0):>+7.3f} "
                         f"{delta.get('LPIPS', 0):>+7.3f}")
    tolerance = ", ".join(f"{name} ±{value}" for name, value in (data.get("tolerance") or {}).items())
    lines.append(f"   Kết luận: {data.get('verdict', '?')}" + (f" (ngưỡng {tolerance})" if tolerance else ""))
    return lines


# --------------------------------------------------------------------------------- gsplat Deep Blending
def modes_section(ws: Path) -> list[str]:
    reports = ws / "outputs" / "reports"
    lines = []
    quick = latest(reports, "modes_db_quick_*.json")
    data = read_json_or_none(quick) if quick else None
    if data is None:
        lines.append("== Chạy thử gsplat 3000 vòng: chưa có (khi cần: QUICK=1 bash scripts/run_train.sh deep_blending gsplat)")
    else:
        runs = sum(agg.get("n", 0) for by_variant in (data.get("scenes") or {}).values()
                   for agg in by_variant.values())
        bad = [item.get("run_dir") or item.get("scene") for item in data.get("incomplete") or []]
        state = "OK" if not bad else "CHƯA ĐỦ / LỖI: " + ", ".join(str(item) for item in bad)
        lines.append(f"== Chạy thử gsplat 3000 vòng ({quick.name}): {runs} lượt OK | {state}")
        lines.append("   (chỉ kiểm tra luồng chạy; số liệu 3000 vòng không dùng để so sánh)")
    full = latest(reports, "modes_db_2*.md")
    if full is None:
        lines.append("== gsplat default / mcmc đầy đủ trên Deep Blending: chưa có "
                     "(khi cần: bash scripts/run_train.sh deep_blending gsplat)")
    else:
        lines.append(f"== gsplat default / mcmc đầy đủ trên Deep Blending ({full.name}):")
        lines += ["   " + row for row in full.read_text(encoding="utf-8", errors="replace").splitlines()]
    return lines


# -------------------------------------------------------------------------------------- cảnh tự quay
def _owner_alive(run_dir: Path) -> bool:
    """Tiến trình đã ghi run_meta.json (lượt RUNNING) còn chạy trên máy này không."""
    meta = read_json_or_none(run_dir / "run_meta.json") or {}
    owner = meta.get("owner") or {}
    pid = owner.get("pid")
    if owner.get("host") != socket.gethostname() or not isinstance(pid, int):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"src.training.run" in cmdline


PREPARED = PREPARED_SUBDIR  # <out-root>/<cảnh>/colmap: ảnh đã chuẩn hóa, COLMAP, báo cáo (scripts/pipeline/prepare_scene.sh)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def custom_roots(ws: Path) -> tuple[Path, Path]:
    """(thư mục nguồn, thư mục kết quả) của cảnh tự chụp theo registry; registry hỏng thì dùng bố cục mặc định."""
    try:
        entry = load_registry(CONFIG_DIR, str(ws))["datasets"]["custom_scene"]
        return Path(entry["root"]), Path(entry["results"]["gsplat"])
    except (ValueError, KeyError) as error:
        print(f"WARNING: registry: {error}; using the default layout under {ws}", file=sys.stderr)
        return ws / "data" / "custom_scenes", ws / "outputs" / "custom_scenes"


def scene_names(ws: Path, out_root: Path) -> list[str]:
    names = set()
    for root in (custom_roots(ws)[0], out_root):
        if root.is_dir():
            names.update(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    return sorted(names)


def run_dirs(out_root: Path, scene: str) -> list[Path]:
    root = out_root / scene
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name != PREPARED
                  and ".prev_" not in path.name and not path.name.startswith("."))


def source_line(folder: Path) -> str | None:
    """Nguồn gốc trong data/custom_scenes/<cảnh>/: số ảnh nằm thẳng trong thư mục, các file khác (video...)."""
    if not folder.is_dir():
        return None
    files = [path for path in folder.iterdir() if path.is_file() and not path.name.startswith(".")]
    photos = sum(1 for path in files if path.suffix.lower() in IMAGE_EXTS)
    others = sorted(path.name for path in files if path.suffix.lower() not in IMAGE_EXTS)
    text = f"   nguồn: {folder} | {photos} ảnh JPG/JPEG/PNG"
    if others:
        text += " | file khác: " + ", ".join(others[:5]) + (" ..." if len(others) > 5 else "")
    return text


def scene_section(ws: Path, scene: str, out_root: Path) -> list[str]:
    folder = out_root / scene / PREPARED
    lines = [f"== Cảnh {scene} ({folder})"]
    source = source_line(custom_roots(ws)[0] / scene)
    if source:
        lines.append(source)
    frames = read_json_or_none(folder / "frames_report.json")
    if frames and frames.get("source_type") == "images":
        text = f"   ảnh: dùng {frames.get('selected', '?')} / {frames.get('photos_found', '?')} ảnh JPG/JPEG/PNG"
        if frames.get("duplicates_dropped"):
            text += f", bỏ {frames['duplicates_dropped']} ảnh trùng"
        if frames.get("unreadable"):
            text += f", bỏ {len(frames['unreadable'])} file hỏng"
        if frames.get("blurry_kept"):
            text += f" | {len(frames['blurry_kept'])} ảnh có độ nét thấp (vẫn giữ)"
        lines.append(text)
    elif frames:
        text = f"   frame: chọn {frames.get('selected', '?')} / {frames.get('candidates', '?')} ứng viên"
        if frames.get("rejected_blurry") is not None:
            text += f", loại vì mờ {frames['rejected_blurry']}"
        if frames.get("source_type") == "video":
            text += f" | fps {frames.get('fps')}" + (" (tự chọn)" if frames.get("fps_auto") else "")
            if frames.get("duplicates_dropped"):
                text += f" | bỏ {frames['duplicates_dropped']} frame trùng"
        lines.append(text)
    elif folder.is_dir():
        lines.append("   frame: chưa có frames_report.json")
    else:
        lines.append(f"   chưa chuẩn bị (chưa có {folder}): bash scripts/pod/run_scene.sh <nguồn> {scene} --background")
    sfm = read_json_or_none(folder / "sfm_report.json")
    if sfm:
        model = sfm.get("model") or {}
        ratio = sfm.get("registration_ratio")
        text = f"   COLMAP: {sfm.get('verdict', '?')}"
        if model:
            text += (f" | đăng ký {model.get('registered_images', '?')}/{sfm.get('input_images', '?')} ảnh"
                     + (f" ({ratio:.0%})" if isinstance(ratio, (int, float)) else "")
                     + f" | {_count(model.get('points3d'))} điểm"
                     + f" | sai số chiếu lại {_f(model.get('mean_reprojection_error_px'), 2)} px")
        groups = sfm.get("split_groups", len(sfm.get("distorted_models") or []))  # báo cáo cũ: không có split_groups
        if groups > 1:
            text += f" | COLMAP tách {groups} nhóm ảnh, chỉ dùng nhóm lớn nhất"
        if sfm.get("error"):
            text += f" | lỗi: {sfm['error']}"
        lines.append(text)
        lines += [f"     - {warning}" for warning in sfm.get("warnings") or []]
        coverage = sfm.get("coverage") or {}
        if coverage.get("status") in ("OK", "LOW"):
            spread = coverage.get("camera_extent_over_depth")
            lines.append(f"   góc nhìn: {'đủ' if coverage['status'] == 'OK' else 'THIẾU'} | phủ "
                         f"{coverage.get('azimuth_coverage_deg')}/360 độ theo phương ngang, khoảng trống lớn nhất "
                         f"{coverage.get('largest_gap_deg')} độ | camera di chuyển "
                         + ("-" if spread is None else f"{spread:.2f}") + " x khoảng cách tới cảnh")
            lines += [f"     - {warning}" for warning in coverage.get("warnings") or []]
        lines += [f"     - gợi ý: {hint}" for hint in sfm.get("hints") or []]
    elif folder.is_dir():
        lines.append("   COLMAP: chưa có sfm_report.json")

    runs = run_dirs(out_root, scene)
    if not runs:
        lines.append(f"   train: chưa có lượt nào ({out_root})")
        return lines
    lines.append(f"   {'lượt':<16} {'trạng thái':<12} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'Gaussian':>11} "
                 f"{'train(ph)':>9}")
    plys, videos = [], []
    for run_dir in runs:
        result = load_summary(run_dir)
        status = result.status
        if status == "RUNNING" and not _owner_alive(run_dir):
            status = "DỞ DANG"  # pod tắt hoặc tiến trình bị dừng giữa chừng
        metrics = result.metrics or {}
        lines.append(f"   {run_dir.name:<16} {status:<12} {_f(metrics.get('PSNR'), 2):>7} "
                     f"{_f(metrics.get('SSIM')):>7} {_f(metrics.get('LPIPS')):>7} {_count(result.num_gaussians):>11} "
                     f"{_minutes(result.train_seconds):>9}")
        if result.problem:
            lines.append(f"     - {result.problem}")
        plys += sorted((run_dir / "ply").glob("*.ply"))
        # video quỹ đạo của bước đánh giá, và của các lần render lại (scripts/run_render.sh)
        videos += sorted((run_dir / "videos").glob("*.mp4")) + sorted(run_dir.glob("render_*/videos/*.mp4"))
    if any(path.parent.parent.name.startswith("default") for path in plys):
        lines.append("   Ghi chú: ảnh test là mỗi ảnh thứ 8 của cùng video (nội suy giữa các góc rất gần nhau),"
                     " nên PSNR cao hơn khi xem từ góc mới.")
    for path in plys:
        lines.append(f"   PLY (mở bằng SuperSplat / trình xem 3DGS): {path} ({path.stat().st_size / 2**20:.0f} MB)")
    for path in videos:
        lines.append(f"   video: {path} ({path.stat().st_size / 2**20:.1f} MB)")
    return lines


def package_line(ws: Path) -> list[str]:
    path = latest(ws, "ket_qua_*.tar.gz")
    if path is None:
        return ["== Gói kết quả: chưa có (bash scripts/pod/pack_results.sh)"]
    return [f"== Gói kết quả mới nhất để tải về: {path} ({path.stat().st_size / 2**20:.1f} MB)"]


def build(ws: Path, scenes: list[str] | None, out_root: Path | None = None) -> list[str]:
    out_root = out_root or custom_roots(ws)[1]
    lines: list[str] = []
    if not scenes:
        lines += run_all_section(ws) + [""] + b0_section(ws) + [""] + modes_section(ws) + [""]
        scenes = scene_names(ws, out_root)
        if not scenes:
            lines.append("== Cảnh tự chụp: chưa có (ảnh vào data/custom_scenes/<tên_cảnh>/, rồi "
                         "bash scripts/pod/run_scene.sh <thư_mục_ảnh> <tên_cảnh> --background)")
            lines.append("")
    for scene in scenes:
        lines += scene_section(ws, scene, out_root) + [""]
    return lines + package_line(ws)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ws", type=Path, default=Path("/workspace"), help="thư mục gốc trên network volume")
    parser.add_argument("--out-root", type=Path,
                        help="thư mục kết quả của các cảnh tự chụp (mặc định theo registry: <ws>/outputs/custom_scenes)")
    parser.add_argument("--scene", action="append", default=[], help="chỉ xem cảnh này (lặp lại được)")
    args = parser.parse_args(argv)
    if not args.ws.is_dir():
        print(f"ERROR: {args.ws} not found", file=sys.stderr)
        return 2
    print("\n".join(build(args.ws, args.scene, args.out_root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
