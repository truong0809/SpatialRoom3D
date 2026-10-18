"""Báo cáo chất lượng ước lượng camera (COLMAP) cho một cảnh, đọc trước khi huấn luyện.

Đọc:
  <scene>/input/                 ảnh đầu vào
  <scene>/frames_report.json     nguồn (video / thư mục ảnh) và tên file gốc của từng ảnh
  <scene>/distorted/sparse/*/    các model COLMAP trước khi khử méo (để phát hiện bị tách mảnh)
  <scene>/sparse/0/              model đã khử méo, dùng cho 3DGS

Kết luận (`verdict`, quyết định có train hay không, `scripts/pod/run_scene.sh` dừng khi FAIL) chỉ dựa trên tỉ lệ ảnh
được đăng ký, việc ảnh bị tách thành nhiều nhóm (model phụ chỉ lặp lại ảnh của model lớn nhất không tính, xem
`split_groups`), sai số chiếu lại, và model có điểm 3D hay không (không có thì FAIL: 3DGS khởi tạo từ điểm SfM).
Thêm vào đó:
  * `hints`: giải thích cho người dùng khi có ảnh không ghép được hoặc bị tách nhóm (thường do ảnh không cùng một
    căn phòng hoặc thiếu ảnh chồng lấp), kèm tên file gốc của ảnh;
  * `coverage`: độ phủ hướng nhìn theo phương ngang (hợp các góc nhìn ngang của camera) và mức camera di chuyển so
    với khoảng cách tới cảnh. Chỉ cảnh báo (LOW), không đổi `verdict`: chụp một phần căn phòng vẫn hợp lệ.

Ngưỡng đánh giá là kinh nghiệm ban đầu, sẽ hiệu chỉnh sau các lần quay / chụp thử.

Cách dùng:
  python -m src.reconstruction.report --scene /workspace/outputs/custom_scenes/phong/colmap
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.reconstruction.colmap_model import (
    model_exists,
    read_cameras_binary,
    read_images_binary,
    read_points3d_summary,
    read_points3d_xyz,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
REGISTRATION_PASS = 0.95
REGISTRATION_WARN = 0.80
REPROJECTION_WARN_PX = 1.5
COVERAGE_GAP_WARN_DEG = 90.0   # khoảng hướng ngang liền không camera nào nhìn tới
STEEP_PITCH_DEG = 70.0         # camera nhìn gần thẳng lên trần / xuống sàn: không tính vào hướng ngang
SPREAD_WARN = 0.05             # camera dịch chuyển < 5 % khoảng cách tới cảnh: gần như đứng một chỗ
MAX_POINTS_FOR_DEPTH = 20000
CAPTURE_ADVICE = ("chụp thêm ảnh chồng lấp (mỗi ảnh trùng khoảng 60-80 % với ảnh trước), di chuyển giữa các lần chụp, "
                  "cùng một căn phòng, cùng ống kính và không zoom; hoặc quay video theo docs/data/capture_protocol.md")
VIDEO_ADVICE = ("quay lại theo docs/data/capture_protocol.md: đi chậm và luôn có dịch chuyển (không đứng một chỗ xoay "
                "máy), xoay người từ từ, tránh quay sát mảng tường trơn, khóa phơi sáng và lấy nét; hoặc chụp ảnh")


def capture_advice(source_type: str | None) -> str:
    """Cách chụp / quay lại, theo loại nguồn (không rõ nguồn thì dùng lời khuyên cho ảnh, có nhắc cả video)."""
    return VIDEO_ADVICE if source_type == "video" else CAPTURE_ADVICE


def list_models(sparse_dir: Path) -> list[dict]:
    """Các model con trong một thư mục sparse/, kèm số ảnh đã đăng ký, xếp giảm dần."""
    models = []
    if sparse_dir.is_dir():
        for child in sorted(sparse_dir.iterdir()):
            if child.is_dir() and model_exists(child):
                models.append({"path": str(child), "registered": len(read_images_binary(child / "images.bin"))})
    return sorted(models, key=lambda m: m["registered"], reverse=True)


def split_groups(models: list[dict]) -> tuple[list[list[str]], int]:
    """Nhóm ảnh nằm ngoài model lớn nhất (`models` xếp giảm dần như `list_models`), và số model phụ thừa.

    COLMAP (`multiple_models`) có thể khởi tạo lại từ một cặp ảnh đã có trong model lớn nhất và để lại một model nhỏ
    chỉ gồm các ảnh đó (đã gặp với COLMAP 3.9.1: model phụ 2 ảnh, cả 2 đều nằm trong model 36/36 ảnh). Model như vậy
    không làm mất ảnh nào nên không tính là bị tách; chỉ ảnh không có trong model lớn nhất mới tạo thành nhóm riêng.
    """
    if len(models) < 2:
        return [], 0
    seen = {image.name for image in read_images_binary(Path(models[0]["path"]) / "images.bin").values()}
    groups, redundant = [], 0
    for sub in models[1:]:
        names = sorted(image.name for image in read_images_binary(Path(sub["path"]) / "images.bin").values())
        extra = [name for name in names if name not in seen]
        seen.update(extra)
        if extra:
            groups.append(extra)
        else:
            redundant += 1
    return groups, redundant


def analyze_model(model_dir: Path) -> dict:
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    points = read_points3d_summary(model_dir / "points3D.bin")
    observations = [image.num_observations for image in images.values()]
    return {
        "path": str(model_dir),
        "registered_images": len(images),
        "registered_names": sorted(image.name for image in images.values()),
        "points3d": points.count,
        "mean_reprojection_error_px": points.mean_error,
        "median_reprojection_error_px": points.median_error,
        "mean_track_length": points.mean_track_length,
        "mean_observations_per_image": (sum(observations) / len(observations)) if observations else None,
        "cameras": [
            {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params}
            for cam in cameras.values()
        ],
    }


# ------------------------------------------------------------------------------------------------- độ phủ

def _rotation(qvec) -> np.ndarray:
    """Quaternion (w, x, y, z) -> ma trận quay (cam_from_world)."""
    w, x, y, z = np.asarray(qvec, dtype=float) / np.linalg.norm(qvec)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _horizontal_fov_deg(camera) -> float:
    """Góc nhìn ngang (theo chiều rộng ảnh) từ tiêu cự; mọi model COLMAP đều có fx là tham số đầu tiên."""
    return math.degrees(2 * math.atan(camera.width / (2 * camera.params[0])))


def analyze_coverage(model_dir: Path, labels: dict | None = None, source_type: str | None = None) -> dict:
    """Độ phủ hướng nhìn và mức di chuyển của camera trong một model COLMAP.

    Hướng lên của cảnh = trung bình hướng lên (-y) của các camera (ảnh đã xoay theo EXIF nên đứng thẳng). Mỗi camera
    phủ [phương vị - FOV/2, phương vị + FOV/2] theo phương ngang; độ phủ là phần hợp của các khoảng đó trên 360 độ
    (bước 1 độ). Camera nhìn gần thẳng lên / xuống (> STEEP_PITCH_DEG) không tính. Mức di chuyển = bán kính chứa 90 %
    vị trí camera chia cho khoảng cách trung vị từ camera tới các điểm 3D (cùng đơn vị của model).
    """
    labels = labels or {}
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = sorted(read_images_binary(model_dir / "images.bin").values(), key=lambda image: image.name)
    result: dict = {"registered_images": len(images), "status": "UNKNOWN", "warnings": []}
    if len(images) < 3:
        result["reason"] = "fewer than 3 registered images"
        return result

    rotations = np.array([_rotation(image.qvec) for image in images])
    centers = -np.einsum("nji,nj->ni", rotations, np.array([image.tvec for image in images], dtype=float))
    forward, down = rotations[:, 2, :], rotations[:, 1, :]  # trục z (nhìn) và y (xuống) của camera trong hệ thế giới
    up = -down.mean(axis=0)
    if np.linalg.norm(up) < 0.3:  # camera xoay đủ mọi hướng: không suy ra được hướng lên
        result["reason"] = "camera orientations do not agree on an up direction"
        return result
    up /= np.linalg.norm(up)
    pitch = np.degrees(np.arcsin(np.clip(forward @ up, -1.0, 1.0)))
    level = np.abs(pitch) <= STEEP_PITCH_DEG
    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    azimuth = np.degrees(np.arctan2(forward @ e2, forward @ e1)) % 360.0
    fov = np.array([_horizontal_fov_deg(cameras[image.camera_id]) for image in images])

    covered = np.zeros(360, dtype=bool)
    for angle, width in zip(azimuth[level], fov[level]):
        bins = np.arange(math.floor(angle - width / 2), math.ceil(angle + width / 2)) % 360
        covered[bins] = True
    gap, gap_start = 0, None
    if covered.any() and not covered.all():
        start = int(np.argmax(covered))  # quay vòng từ một ô đã phủ để khoảng trống không bị cắt đôi
        run, run_start = 0, None
        for step in range(1, 361):
            index = (start + step) % 360
            if not covered[index]:
                run_start = index if run == 0 else run_start
                run += 1
                if run > gap:
                    gap, gap_start = run, run_start
            else:
                run = 0
    elif not covered.any():
        gap = 360

    points = read_points3d_xyz(model_dir / "points3D.bin")
    if len(points) > MAX_POINTS_FOR_DEPTH:
        points = points[np.random.default_rng(0).choice(len(points), MAX_POINTS_FOR_DEPTH, replace=False)]
    depth = None
    if len(points):
        depth = float(np.median([np.median(np.linalg.norm(points - center, axis=1)) for center in centers]))
    extent = float(np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 90))
    spread = extent / depth if depth else None

    result.update({
        "up_axis": up.round(4).tolist(),
        "level_cameras": int(level.sum()), "steep_cameras": int((~level).sum()),
        "horizontal_fov_deg": round(float(np.median(fov)), 1),
        "azimuth_coverage_deg": int(covered.sum()),
        "largest_gap_deg": int(gap),
        "camera_extent": round(extent, 4), "median_scene_depth": None if depth is None else round(depth, 4),
        "camera_extent_over_depth": None if spread is None else round(spread, 3),
    })
    video = source_type == "video"
    what, capture = ("frame", "quay") if video else ("ảnh", "chụp")
    if gap >= COVERAGE_GAP_WARN_DEG:
        message = (f"có khoảng hướng ngang khoảng {gap} độ không {what} nào nhìn tới (độ phủ {int(covered.sum())}/360 "
                   "độ): phần phòng phía đó sẽ thiếu hoặc mờ.")
        if gap_start is not None and level.any():
            gap_end = (gap_start + gap) % 360
            idx = np.where(level)[0]
            before = idx[np.argmin((gap_start - (azimuth[idx] + fov[idx] / 2)) % 360)]
            after = idx[np.argmin(((azimuth[idx] - fov[idx] / 2) - gap_end) % 360)]
            name = [labels.get(images[i].name, images[i].name) for i in (before, after)]
            message += (f" Khoảng trống nằm giữa hướng nhìn của {what} {name[0]} và {what} {name[1]}: "
                        f"{capture} thêm về phía đó.")
        result["warnings"].append(message + " Bỏ qua nếu chỉ cần dựng một phần căn phòng.")
    if spread is not None and spread < SPREAD_WARN:
        result["warnings"].append(
            f"camera gần như đứng một chỗ (dịch chuyển khoảng {spread:.0%} khoảng cách tới cảnh): chỉ xoay máy tại chỗ "
            "thì thiếu thị sai để dựng hình 3D, cần "
            + ("đi lại trong phòng khi quay." if video else "di chuyển vài bước giữa các lần chụp."))
    result["status"] = "LOW" if result["warnings"] else "OK"
    return result


# ------------------------------------------------------------------------------------------------ báo cáo

def read_frames_report(scene_dir: Path) -> dict:
    """frames_report.json của cảnh (rỗng nếu chưa có hoặc hỏng)."""
    try:
        return json.loads((scene_dir / "frames_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _labels(frames: dict) -> dict:
    """Tên trong input/ -> 'tên (tên file gốc)' cho ảnh chụp; video giữ nguyên tên frame."""
    return {record["name"]: f"{record['name']} ({record['source_name']})"
            for record in frames.get("frames") or [] if record.get("source_name")}


def _examples(names: list[str], labels: dict, limit: int = 8) -> str:
    shown = [labels.get(name, name) for name in names[:limit]]
    return ", ".join(shown) + (f" ... (+{len(names) - limit})" if len(names) > limit else "")


def failure_hints(error: str | None, source_type: str | None) -> list[str]:
    """Gợi ý khi COLMAP không dựng được mô hình nào (lỗi giữa chừng hoặc không đăng ký được ảnh)."""
    what = "các ảnh" if source_type == "images" else "các frame"
    capture = "quay" if source_type == "video" else "chụp"
    hints = []
    if error and "initial image pair" in error:
        hints.append(f"COLMAP không tìm được cặp ảnh nào đủ chồng lấp và đủ dịch chuyển để bắt đầu dựng: {what} "
                     f"ít trùng nhau, {capture} từ cùng một chỗ, hoặc nhìn vào bề mặt trơn ít chi tiết.")
    hints.append(f"COLMAP không ước lượng được vị trí camera cho {what}, nên không train. "
                 f"Cách xử lý: {capture_advice(source_type)}.")
    return hints


def analyze_scene(scene_dir: Path) -> dict:
    input_dir = scene_dir / "input"
    input_names = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) \
        if input_dir.is_dir() else []
    distorted = list_models(scene_dir / "distorted" / "sparse")
    final_dir = scene_dir / "sparse" / "0"
    frames = read_frames_report(scene_dir)
    labels = _labels(frames)

    report: dict = {"scene": str(scene_dir), "input_images": len(input_names), "source_type": frames.get("source_type"),
                    "distorted_models": distorted, "warnings": [], "hints": []}
    if model_exists(final_dir):
        model_dir = final_dir
        model = analyze_model(final_dir)
    elif distorted:
        model_dir = Path(distorted[0]["path"])
        model = analyze_model(model_dir)
        report["warnings"].append("undistorted model sparse/0 not found; statistics use the largest distorted model")
    else:
        report["model"] = None
        report["verdict"] = "FAIL"
        report["warnings"].append("no COLMAP reconstruction found")
        report["hints"] = failure_hints(None, frames.get("source_type"))
        return report

    registered_names = set(model.pop("registered_names"))
    report["model"] = model
    report["unregistered_images"] = [name for name in input_names if name not in registered_names]
    ratio = model["registered_images"] / len(input_names) if input_names else None
    report["registration_ratio"] = ratio

    if ratio is None:
        verdict = "WARN"
        report["warnings"].append("input/ folder not found: registration ratio unknown")
    elif ratio >= REGISTRATION_PASS:
        verdict = "PASS"
    elif ratio >= REGISTRATION_WARN:
        verdict = "WARN"
        report["warnings"].append(f"only {ratio:.0%} of input images registered")
    else:
        verdict = "FAIL"
        report["warnings"].append(f"only {ratio:.0%} of input images registered")

    groups, redundant = split_groups(distorted)
    report["split_groups"] = 1 + len(groups)
    report["redundant_models"] = redundant
    if groups:
        report["warnings"].append(
            f"reconstruction split into {1 + len(groups)} groups of images; only the largest is used")
        verdict = "WARN" if verdict == "PASS" else verdict
    error = model["mean_reprojection_error_px"]
    if error is not None and error > REPROJECTION_WARN_PX:
        report["warnings"].append(f"mean reprojection error {error:.2f} px > {REPROJECTION_WARN_PX} px")
        verdict = "WARN" if verdict == "PASS" else verdict
    if not model["points3d"]:  # 3DGS khởi tạo Gaussian từ điểm SfM: không có điểm thì không train được
        report["warnings"].append("the COLMAP model has no 3D points")
        verdict = "FAIL"
    report["verdict"] = verdict

    # Giải thích cho người dùng: ảnh nào không ghép được, có bị tách nhóm không
    source_type = frames.get("source_type")
    video = source_type == "video"
    what = "ảnh" if source_type == "images" else "frame"
    unregistered = report["unregistered_images"]
    if unregistered and verdict != "PASS":
        cause = ("frame bị mờ, xoay máy quá nhanh, nhìn vào tường trơn, hoặc ít chồng lấp với phần còn lại "
                 "(vd. đoạn quay ở phòng khác)" if video else
                 f"{what} chụp ở phòng khác, ít chồng lấp với phần còn lại, nhìn vào tường trơn, hoặc bị mờ")
        report["hints"].append(
            f"{len(unregistered)}/{len(input_names)} {what} không ghép được vào mô hình (không có vị trí camera): "
            f"{_examples(unregistered, labels)}. Thường do {cause}.")
    if groups:
        sizes = ", ".join(f"nhóm {k + 1}: {count} {what}"
                          for k, count in enumerate([distorted[0]["registered"], *map(len, groups)]))
        missing = "đoạn quay vùng nối giữa chúng" if video else f"{what} chụp vùng nối giữa chúng"
        report["hints"].append(
            f"COLMAP tách {what} thành {1 + len(groups)} nhóm không nối được với nhau ({sizes}); chỉ nhóm lớn nhất "
            f"được dùng. Các nhóm có thể là các phòng hoặc khu vực khác nhau, hoặc thiếu {missing}. "
            f"Ví dụ nhóm 2: {_examples(groups[0], labels, 5)}.")
    if not model["points3d"]:
        report["hints"].append(
            f"Mô hình COLMAP không có điểm 3D nào, nên không khởi tạo được 3DGS. Thường do camera đứng một chỗ chỉ "
            f"xoay máy (không có thị sai để tính độ sâu): cần {'đi lại trong phòng khi quay' if video else 'di chuyển vài bước giữa các lần chụp'}.")
    if verdict == "FAIL":
        report["hints"].append(
            f"Không train vì vị trí camera chưa đủ tin cậy. Cách xử lý: {capture_advice(source_type)}.")

    try:
        report["coverage"] = analyze_coverage(model_dir, labels, source_type)
    except (OSError, ValueError, KeyError, IndexError) as exc:  # thiếu hoặc hỏng file model: không chặn báo cáo
        report["coverage"] = {"status": "UNKNOWN", "reason": f"cannot analyse coverage: {exc}", "warnings": []}
    return report


def format_report(report: dict) -> str:
    lines = [f"scene: {report['scene']}", f"input images: {report['input_images']}"]
    model = report.get("model")
    if model:
        ratio = report.get("registration_ratio")
        lines += [
            f"registered images: {model['registered_images']}"
            + (f" ({ratio:.1%})" if ratio is not None else ""),
            f"3D points: {model['points3d']}",
            f"mean reprojection error: {model['mean_reprojection_error_px']:.3f} px"
            if model["mean_reprojection_error_px"] is not None else "mean reprojection error: -",
            f"mean track length: {model['mean_track_length']:.2f}"
            if model["mean_track_length"] is not None else "mean track length: -",
            f"models before undistortion: {len(report['distorted_models'])}"
            + (f" ({report['redundant_models']} only repeat images of the largest model, ignored)"
               if report.get("redundant_models") else ""),
        ]
    coverage = report.get("coverage")
    if coverage and coverage.get("status") != "UNKNOWN":
        spread = coverage.get("camera_extent_over_depth")
        lines.append(f"view coverage: {coverage['azimuth_coverage_deg']}/360 deg horizontal, largest gap "
                     f"{coverage['largest_gap_deg']} deg | camera spread "
                     + ("-" if spread is None else f"{spread:.2f}") + " x scene depth")
    for warning in report["warnings"]:
        lines.append(f"WARN: {warning}")
    lines.append(f"SFM VERDICT: {report['verdict']}")
    if coverage:
        lines.append(f"VIEW COVERAGE: {coverage['status']}"
                     + (f" ({coverage['reason']})" if coverage.get("reason") else ""))
        lines += [f"  - {warning}" for warning in coverage.get("warnings") or []]
    for hint in report.get("hints") or []:
        lines.append(f"HINT: {hint}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Báo cáo chất lượng COLMAP cho một cảnh")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="mặc định <scene>/sfm_report.json")
    args = parser.parse_args(argv)

    report = analyze_scene(args.scene)
    out = args.out or args.scene / "sfm_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(report))
    print(f"report: {out}")
    return 0 if report["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
