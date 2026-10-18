"""Chuẩn bị ảnh đầu vào cho ước lượng camera: trích và chọn frame từ video, hoặc dùng thẳng thư mục ảnh chụp.

Thư mục ảnh (`--images`, ảnh JPG/JPEG/PNG chụp cùng một căn phòng từ nhiều vị trí): KHÔNG trích hay chọn frame.
Mọi ảnh dùng được đều vào `<scene-dir>/input/`, sau khi chuẩn hóa giống frame của video (xoay theo EXIF, cạnh dài
≤ `--max-side`, ghi JPEG). Khác frame của video: tiêu cự quy đổi 35 mm trong EXIF của ảnh gốc được giữ lại, vì COLMAP
dùng nó làm tiêu cự ban đầu. Chỉ bỏ ảnh không đọc được và ảnh trùng y hệt (cùng nội dung file): ảnh trùng mà một
bản rơi vào tập test sẽ làm điểm đánh giá bị thổi phồng. Ảnh mờ chỉ được cảnh báo, không bị bỏ. Dừng khi không đủ
ảnh (`src.input.sources`: một ảnh thì từ chối, dưới `--min-images` thì báo thiếu) hoặc khi ảnh khác kích thước
(COLMAP dùng một camera chung cho cả cảnh, như với video).

Video (`--video`):
  1. Trích frame ứng viên bằng ffmpeg (ffmpeg tự xoay theo metadata video). Mặc định tốc độ trích
     được tính từ độ dài video để mỗi đoạn có khoảng `--candidates-per-window` ứng viên
     (giới hạn trong [1, `--max-fps`]); `--fps` đặt cố định nếu cần.
  2. Bỏ frame trùng y hệt frame trước (bộ lọc fps của ffmpeg lặp frame khi video có tốc độ khung
     thay đổi và bị hụt khung).
  3. Chấm độ nét mỗi frame bằng phương sai Laplacian (ảnh xám, thu nhỏ cạnh dài về 640 px).
  4. Chia dãy frame thành `--target` đoạn đều nhau theo thời gian, lấy frame nét nhất mỗi đoạn.
  5. Bỏ frame được chọn nếu độ nét < `--blur-ratio` x trung vị độ nét của các frame lân cận
     (±`--blur-window-s` giây). Dùng trung vị cục bộ vì phương sai Laplacian phụ thuộc nhiều vào
     họa tiết: tường trơn nét vẫn có điểm thấp hơn nhiều so với kệ sách mờ.
  6. Ghi frame đã chọn vào `<scene-dir>/input/` (đúng cấu trúc COLMAP/3DGS mong đợi)
     và báo cáo vào `<scene-dir>/frames_report.json`.

Mặc định giới hạn cạnh dài 1600 px: 3DGS gốc (Inria) tự thu ảnh rộng hơn 1600 px, gsplat thì
không, nên hai bên chỉ nhận cùng một ảnh khi ảnh đầu vào đã ≤ 1600 px.

Cách dùng:
  python -m src.input.frames --video phong.mp4 --scene-dir /workspace/outputs/custom_scenes/phong/colmap
  python -m src.input.frames --images thu_muc_anh --scene-dir /workspace/outputs/custom_scenes/phong/colmap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.input.sources import (
    MIN_PHOTOS,
    PHOTO_EXTS,
    RECOMMENDED_PHOTOS,
    InputError,
    photo_count_problem,
    scan_photo_folder,
)

IMAGE_EXTS = PHOTO_EXTS
DEFAULT_MAX_SIDE = 1600
DEFAULT_FPS = 4.0  # khi không đọc được độ dài video
# Mọi thứ ghi vào thư mục cảnh từ bước này trở đi; --overwrite xóa hết để không còn kết quả cũ trông như hợp lệ.
SCENE_OUTPUTS = ("input", "images", "sparse", "distorted", "stereo", "colmap_logs", "frames_candidates",
                 "frames_report.json", "sfm_report.json")


EXIF_FOCAL_35MM_TAG = 0xA405  # FocalLengthIn35mmFilm, nằm trong Exif IFD (con trỏ 0x8769 ở IFD0)


def exif_focal_35mm(data: bytes) -> int | None:
    """Tiêu cự quy đổi 35 mm (thẻ EXIF FocalLengthIn35mmFilm) của một file JPEG; None nếu không có hoặc không đọc được.

    COLMAP lấy tiêu cự ban đầu của camera từ thẻ này (khi không có thì đoán 1,2 x cạnh dài). Ảnh ghi lại bằng OpenCV
    mất EXIF, nên giá trị được đọc từ ảnh gốc và ghi lại vào ảnh trong input/ (`with_exif_focal_35mm`).
    """
    try:
        if data[:2] != b"\xff\xd8":
            return None
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 0xFF:
            marker = data[pos + 1]
            if marker in (0xD9, 0xDA):  # hết ảnh / bắt đầu dữ liệu nén: không còn đoạn header nào
                return None
            length = int.from_bytes(data[pos + 2:pos + 4], "big")
            if length < 2:
                return None
            if marker == 0xE1 and data[pos + 4:pos + 10] == b"Exif\x00\x00":
                return _tiff_focal_35mm(data[pos + 10:pos + 2 + length])
            pos += 2 + length
    except (IndexError, ValueError, struct.error):
        return None
    return None


def _tiff_focal_35mm(tiff: bytes) -> int | None:
    order = {b"II": "<", b"MM": ">"}.get(tiff[:2])
    if order is None or struct.unpack(order + "H", tiff[2:4])[0] != 42:
        return None

    def entries(offset: int):
        count = struct.unpack(order + "H", tiff[offset:offset + 2])[0]
        for index in range(count):
            start = offset + 2 + 12 * index
            tag, kind, n = struct.unpack(order + "HHI", tiff[start:start + 8])
            yield tag, kind, n, tiff[start + 8:start + 12]

    exif_ifd = next((struct.unpack(order + "I", raw)[0] for tag, kind, n, raw in entries(struct.unpack(
        order + "I", tiff[4:8])[0]) if tag == 0x8769 and kind == 4), None)
    if exif_ifd is None:
        return None
    for tag, kind, n, raw in entries(exif_ifd):
        if tag == EXIF_FOCAL_35MM_TAG and n == 1 and kind in (3, 4):  # SHORT hoặc LONG
            value = struct.unpack(order + ("H" if kind == 3 else "I"), raw[:2] if kind == 3 else raw)[0]
            return value if 0 < value < 2000 else None  # 0 = không rõ
    return None


def with_exif_focal_35mm(jpeg: bytes, focal_35mm: int) -> bytes:
    """Chèn một đoạn APP1 Exif tối thiểu, chỉ có FocalLengthIn35mmFilm (không có Orientation: điểm ảnh đã được xoay)."""
    ifd0 = struct.pack("<H", 1) + struct.pack("<HHII", 0x8769, 4, 1, 26) + struct.pack("<I", 0)  # Exif IFD ở offset 26
    exif_ifd = struct.pack("<H", 1) + struct.pack("<HHIHH", EXIF_FOCAL_35MM_TAG, 3, 1, focal_35mm, 0) \
        + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + ifd0 + exif_ifd
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":  # JFIF APP0 phải đứng ngay sau SOI
        pos = 4 + int.from_bytes(jpeg[4:6], "big")
    return jpeg[:pos] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + jpeg[pos:]


def sharpness(path: Path, long_side: int = 640) -> float:
    """Độ nét: phương sai của Laplacian trên ảnh xám đã thu nhỏ (càng lớn càng nét)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image {path}")
    return sharpness_of(gray, long_side)


def sharpness_of(gray: np.ndarray, long_side: int = 640) -> float:
    """Như `sharpness`, cho ảnh xám đã đọc sẵn."""
    height, width = gray.shape
    scale = long_side / max(height, width)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def limit_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Thu nhỏ để cạnh dài ≤ max_side (giữ tỉ lệ); không phóng to."""
    height, width = image.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def ffprobe_for(ffmpeg: str) -> str:
    """ffprobe đi kèm ffmpeg (cùng thư mục) nếu có, nếu không thì tìm trong PATH."""
    sibling = Path(ffmpeg).with_name("ffprobe")
    return str(sibling) if Path(ffmpeg).parent != Path(".") and sibling.is_file() else "ffprobe"


def probe_duration(video: Path, ffprobe: str = "ffprobe") -> float | None:
    try:
        proc = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                              capture_output=True, text=True, check=True)
        value = float(proc.stdout.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None
    return value if value > 0 else None


def choose_fps(duration: float | None, target: int, per_window: int, max_fps: float) -> float:
    """Tốc độ trích để có khoảng `per_window` ứng viên cho mỗi đoạn."""
    if not duration:
        return DEFAULT_FPS
    return round(min(max(target * per_window / duration, 1.0), max_fps), 3)


def extract_frames(video: Path, out_dir: Path, fps: float, max_side: int = 0,
                   ffmpeg: str = "ffmpeg") -> list[Path]:
    """Trích frame JPEG từ video vào out_dir; trả về danh sách file theo thứ tự thời gian."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = [f"fps={fps}"]
    if max_side > 0:
        # Giới hạn cạnh dài, giữ tỉ lệ, kích thước chẵn.
        filters.append(
            f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
        )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", ",".join(filters), "-q:v", "2", str(out_dir / "frame_%06d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def drop_duplicates(paths: list[Path]) -> tuple[list[int], int]:
    """Chỉ số các file khác file liền trước (so nội dung byte); trả về (giữ lại, số bị bỏ)."""
    kept: list[int] = []
    previous = None
    for index, path in enumerate(paths):
        digest = hashlib.blake2b(path.read_bytes(), digest_size=16).digest()
        if digest != previous:
            kept.append(index)
        previous = digest
    return kept, len(paths) - len(kept)


def local_median(values: np.ndarray, neighborhood: int | None) -> np.ndarray:
    """Trung vị của ±neighborhood phần tử quanh mỗi vị trí; None hoặc ≤ 0: trung vị toàn bộ."""
    if not neighborhood or neighborhood <= 0:
        return np.full(len(values), float(np.median(values)))
    return np.array([np.median(values[max(0, i - neighborhood): i + neighborhood + 1])
                     for i in range(len(values))], dtype=float)


def select_frames(scores: list[float], target: int, blur_ratio: float,
                  neighborhood: int | None = None) -> tuple[list[int], dict]:
    """Chọn frame nét nhất trong mỗi đoạn; bỏ frame quá mờ so với trung vị (cục bộ nếu có neighborhood).

    Trả về (chỉ số các frame được chọn theo thứ tự, thống kê).
    """
    count = len(scores)
    if count == 0:
        return [], {"candidates": 0, "windows": 0, "selected": 0, "rejected_blurry": 0}
    values = np.asarray(scores, dtype=float)
    thresholds = blur_ratio * local_median(values, neighborhood)
    windows = max(1, min(target, count))
    edges = np.linspace(0, count, windows + 1).round().astype(int)

    selected: list[int] = []
    rejected = 0
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        best = int(start + np.argmax(values[start:end]))
        if values[best] >= thresholds[best]:
            selected.append(best)
        else:
            rejected += 1
    stats = {
        "candidates": count,
        "windows": windows,
        "candidates_per_window": round(count / windows, 2),
        "selected": len(selected),
        "rejected_blurry": rejected,
        "blur_ratio": blur_ratio,
        "blur_reference": f"local ±{neighborhood} candidates" if neighborhood and neighborhood > 0 else "global",
        "blur_threshold_median": float(np.median(thresholds)),
        "sharpness_min": float(values.min()),
        "sharpness_median": float(np.median(values)),
        "sharpness_max": float(values.max()),
    }
    return selected, stats


def prepare_from_video(video: Path, scene_dir: Path, fps: float | None, target: int, blur_ratio: float,
                       max_side: int = DEFAULT_MAX_SIDE, keep_candidates: bool = False,
                       ffmpeg: str = "ffmpeg", candidates_per_window: int = 4, max_fps: float = 15.0,
                       blur_window_s: float = 2.0) -> dict:
    """fps=None: tự chọn theo độ dài video (cần ffprobe)."""
    duration = probe_duration(video, ffprobe_for(ffmpeg))
    fps_auto = fps is None
    if fps is None:
        fps = choose_fps(duration, target, candidates_per_window, max_fps)
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if keep_candidates:
        work = scene_dir / "frames_candidates"
        shutil.rmtree(work, ignore_errors=True)  # không trộn ứng viên của lần chạy trước
    else:
        work = Path(tempfile.mkdtemp(prefix="frames_"))
    try:
        extracted = extract_frames(video, work, fps, max_side=max_side, ffmpeg=ffmpeg)
        if not extracted:
            raise RuntimeError(f"ffmpeg extracted no frames from {video}")
        kept, duplicates = drop_duplicates(extracted)
        candidates = [extracted[i] for i in kept]
        scores = [sharpness(path) for path in candidates]
        neighborhood = round(blur_window_s * fps) if blur_window_s > 0 else None
        selected, stats = select_frames(scores, target, blur_ratio, neighborhood)
        records = []
        for order, index in enumerate(selected, start=1):
            name = f"{order:05d}.jpg"
            shutil.copy2(candidates[index], input_dir / name)
            records.append({"name": name, "candidate_index": kept[index],
                            "time_s": round(kept[index] / fps, 3), "sharpness": scores[index]})
    finally:
        if not keep_candidates:
            shutil.rmtree(work, ignore_errors=True)
    return {"source": str(video), "source_type": "video", "duration_s": duration, "fps": fps,
            "fps_auto": fps_auto, "max_side": max_side, "extracted": len(extracted),
            "duplicates_dropped": duplicates, "blur_window_s": blur_window_s, **stats, "frames": records}


def _names(items: list, limit: int = 8) -> str:
    names = [item if isinstance(item, str) else item["name"] for item in items]
    return ", ".join(names[:limit]) + (f" ... (+{len(names) - limit})" if len(names) > limit else "")


def prepare_from_images(images_dir: Path, scene_dir: Path, blur_ratio: float = 0.4,
                        max_side: int = DEFAULT_MAX_SIDE, min_photos: int = MIN_PHOTOS,
                        before_commit: Callable[[], None] | None = None) -> dict:
    """Ảnh chụp sẵn: mọi ảnh JPG/JPEG/PNG dùng được vào input/ (không chọn frame), chuẩn hóa như frame của video.

    Thứ tự là thứ tự tên tự nhiên của file (thường là thứ tự chụp); tập test lấy theo thứ tự này (cứ 8 ảnh lấy 1).
    Ảnh được đọc bằng OpenCV (xoay theo EXIF), thu cạnh dài về `max_side`, ghi JPEG chất lượng 95. Tiêu cự quy đổi
    35 mm trong EXIF của ảnh gốc được ghi lại vào ảnh mới để COLMAP có tiêu cự ban đầu (`exif_focal_35mm`).
    Không đủ ảnh hoặc ảnh khác kích thước -> InputError, và input/ không được tạo.
    `before_commit` được gọi ngay trước khi ảnh mới vào input/ (sau khi mọi kiểm tra đã đạt).
    """
    scan = scan_photo_folder(images_dir)
    input_dir = scene_dir / "input"
    staging = scene_dir / ".input_staging"  # ghi trước vào đây, chỉ đổi tên thành input/ khi mọi kiểm tra đạt
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    staged, unreadable, duplicates, seen = [], [], [], {}
    try:
        for path in scan["photos"]:
            raw = path.read_bytes()
            digest = hashlib.blake2b(raw, digest_size=16).digest()
            if digest in seen:
                duplicates.append({"name": path.name, "same_as": seen[digest]})
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable.append({"name": path.name, "reason": "OpenCV không đọc được (file hỏng hoặc sai định dạng)"})
                continue
            seen[digest] = path.name
            source_size = [image.shape[1], image.shape[0]]
            image = limit_side(image, max_side)
            tmp = staging / f"{len(staged) + 1:05d}.jpg"
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError(f"cannot encode {path}")
            focal = exif_focal_35mm(raw)
            tmp.write_bytes(with_exif_focal_35mm(encoded.tobytes(), focal) if focal else encoded.tobytes())
            staged.append({"name": tmp.name, "source_name": path.name, "source_size": source_size,
                           "size": [image.shape[1], image.shape[0]], "focal_length_35mm": focal,
                           "sharpness": sharpness_of(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))})

        dropped = []
        if duplicates:
            dropped.append(f"{len(duplicates)} ảnh trùng y hệt ảnh khác")
        if unreadable:
            dropped.append(f"{len(unreadable)} file không đọc được ({_names(unreadable, 3)})")
        problem = photo_count_problem(len(staged), min_photos)
        if problem:
            found = f"{images_dir}: có {len(scan['photos'])} file ảnh JPG/JPEG/PNG"
            raise InputError(f"{found}" + (f", bỏ {'; '.join(dropped)}" if dropped else "") + f". {problem}")

        groups: dict[tuple, list] = {}
        for record in staged:
            groups.setdefault(tuple(record["size"]), []).append(record["source_name"])
        if len(groups) > 1:
            ordered = sorted(groups.items(), key=lambda item: -len(item[1]))
            listing = "; ".join(f"{w}x{h}: {len(names)} ảnh" + ("" if index == 0 else f" ({_names(names, 5)})")
                                for index, ((w, h), names) in enumerate(ordered))
            raise InputError(
                f"{images_dir}: ảnh khác kích thước sau khi chuẩn hóa ({listing}). COLMAP dùng một camera chung cho cả "
                "cảnh (như với video), nên mọi ảnh phải cùng kích thước và cùng chiều máy: bỏ hoặc chụp lại các ảnh ở "
                "nhóm ít hơn (thường là ảnh chụp dọc), chụp bằng cùng một ống kính và không zoom.")

        scores = np.array([record["sharpness"] for record in staged])
        median = float(np.median(scores))
        blurry = [record["source_name"] for record in staged if record["sharpness"] < blur_ratio * median]
        warnings = []
        if len(staged) < RECOMMENDED_PHOTOS:
            warnings.append(f"chỉ có {len(staged)} ảnh (nên từ {RECOMMENDED_PHOTOS} ảnh trở lên): có thể thiếu góc nhìn")
        if blurry:
            warnings.append(f"{len(blurry)} ảnh có độ nét thấp hơn hẳn các ảnh khác (< {blur_ratio} x trung vị: bị mờ, "
                            f"hoặc chỉ nhìn vào vùng ít chi tiết), vẫn được giữ; ảnh nào bị mờ thì nên chụp lại: "
                            f"{_names(blurry)}")
        if duplicates:
            warnings.append(f"bỏ {len(duplicates)} ảnh trùng y hệt ảnh khác (tránh một bản ở tập train, một bản ở tập "
                            f"test): {_names(duplicates)}")
        if unreadable:
            warnings.append(f"bỏ {len(unreadable)} file không đọc được: {_names(unreadable)}")
        if scan["skipped"]:
            warnings.append(f"bỏ qua {len(scan['skipped'])} mục không phải ảnh JPG/JPEG/PNG: {_names(scan['skipped'])}")
        focals: dict[int, list] = {}
        for record in staged:
            if record["focal_length_35mm"]:
                focals.setdefault(record["focal_length_35mm"], []).append(record["source_name"])
        no_focal = [record["source_name"] for record in staged if not record["focal_length_35mm"]]
        if not focals:
            warnings.append("không ảnh nào có tiêu cự trong EXIF (FocalLengthIn35mmFilm): COLMAP phải tự đoán tiêu cự "
                            "ban đầu, dễ ghép thiếu ảnh hơn. Nên dùng file ảnh gốc của máy (ảnh gửi qua ứng dụng nhắn tin, "
                            "ảnh chụp màn hình hay ảnh đã chỉnh sửa thường mất EXIF)")
        elif no_focal:
            warnings.append(f"{len(no_focal)} ảnh không có tiêu cự trong EXIF: {_names(no_focal)}")
        if len(focals) > 1:
            ordered = sorted(focals.items(), key=lambda item: -len(item[1]))
            listing = "; ".join(f"{value} mm: {len(names)} ảnh" + ("" if index == 0 else f" ({_names(names, 5)})")
                                for index, (value, names) in enumerate(ordered))
            warnings.append(f"ảnh có nhiều tiêu cự khác nhau theo EXIF ({listing}): có thể lẫn ống kính hoặc có zoom. "
                            "COLMAP dùng một camera chung cho cả cảnh, nên bỏ hoặc chụp lại các ảnh ở nhóm ít hơn")
        if before_commit is not None:
            before_commit()  # vd. xóa kết quả cũ của cảnh (--overwrite): chỉ làm khi ảnh mới đã đạt mọi kiểm tra
        staging.rename(input_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    width, height = staged[0]["size"]
    return {"source": str(images_dir), "source_type": "images", "max_side": max_side, "image_size": [width, height],
            "photos_found": len(scan["photos"]), "candidates": len(scan["photos"]), "selected": len(staged),
            "duplicates_dropped": len(duplicates), "unreadable": unreadable, "duplicates": duplicates,
            "skipped": scan["skipped"], "blur_ratio": blur_ratio, "blurry_kept": blurry,
            "sharpness_median": median, "min_photos": min_photos,
            "focal_length_35mm": {str(value): len(names) for value, names in sorted(focals.items())},
            "warnings": warnings,
            "frames": [{key: record[key] for key in ("name", "source_name", "source_size", "focal_length_35mm",
                                                      "sharpness")}
                       for record in staged]}


def existing_outputs(scene_dir: Path) -> list[str]:
    found = []
    for name in SCENE_OUTPUTS:
        path = scene_dir / name
        if path.is_dir() and any(path.iterdir()) or path.is_file():
            found.append(name)
    return found


def clear_outputs(scene_dir: Path) -> None:
    for name in SCENE_OUTPUTS:
        path = scene_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chuẩn bị ảnh cho COLMAP/3DGS: frame nét từ video, hoặc thư mục ảnh")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="file video đầu vào")
    source.add_argument("--images", type=Path,
                        help="thư mục ảnh JPG/JPEG/PNG chụp cùng một căn phòng (dùng hết, không chọn frame)")
    parser.add_argument("--scene-dir", type=Path, required=True, help="thư mục cảnh; frame ghi vào <scene-dir>/input")
    parser.add_argument("--fps", type=float, default=None,
                        help="tốc độ trích frame ứng viên; mặc định tự tính theo độ dài video")
    parser.add_argument("--candidates-per-window", type=int, default=4,
                        help="số ứng viên mong muốn cho mỗi đoạn khi tự tính fps (mặc định 4)")
    parser.add_argument("--max-fps", type=float, default=15.0, help="trần của fps tự tính (mặc định 15)")
    parser.add_argument("--target", type=int, default=None,
                        help="video: số đoạn chia = số frame tối đa được chọn (mặc định 250)")
    parser.add_argument("--blur-ratio", type=float, default=0.4,
                        help="video: bỏ frame có độ nét < tỉ lệ này x trung vị lân cận; ảnh chụp: chỉ cảnh báo "
                             "(mặc định 0.4)")
    parser.add_argument("--min-images", type=int, default=MIN_PHOTOS,
                        help=f"ảnh chụp: số ảnh tối thiểu (mặc định {MIN_PHOTOS}; một ảnh luôn bị từ chối)")
    parser.add_argument("--blur-window-s", type=float, default=2.0,
                        help="trung vị độ nét lấy trong ±N giây quanh frame (0 = cả video; mặc định 2)")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"giới hạn cạnh dài (0 = giữ nguyên; mặc định {DEFAULT_MAX_SIDE})")
    parser.add_argument("--keep-candidates", action="store_true", help="giữ lại toàn bộ frame ứng viên")
    parser.add_argument("--overwrite", action="store_true",
                        help="xóa mọi kết quả cũ của cảnh (frame, COLMAP, báo cáo) rồi làm lại")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="đường dẫn ffmpeg (ffprobe lấy cùng thư mục)")
    args = parser.parse_args(argv)

    source_path = (args.video or args.images).resolve()
    if source_path == args.scene_dir.resolve() or args.scene_dir.resolve() in source_path.parents:
        parser.error("the source must be outside the scene folder (it would be deleted with --overwrite)")
    old = existing_outputs(args.scene_dir)
    if old and not args.overwrite:
        parser.error(f"{args.scene_dir} already has {', '.join(old)} (use --overwrite to start the scene over)")

    if args.video:
        if old:
            clear_outputs(args.scene_dir)
        report = prepare_from_video(args.video, args.scene_dir, args.fps,
                                    args.target if args.target is not None else 250, args.blur_ratio,
                                    max_side=args.max_side, keep_candidates=args.keep_candidates,
                                    ffmpeg=args.ffmpeg, candidates_per_window=args.candidates_per_window,
                                    max_fps=args.max_fps, blur_window_s=args.blur_window_s)
    else:
        if args.target is not None:
            print("NOTE: --target is for videos; every usable photo is kept")
        try:
            # kết quả cũ (--overwrite) chỉ bị xóa khi bộ ảnh mới đã đạt mọi kiểm tra
            report = prepare_from_images(args.images, args.scene_dir, args.blur_ratio, max_side=args.max_side,
                                         min_photos=args.min_images,
                                         before_commit=(lambda: clear_outputs(args.scene_dir)) if old else None)
        except InputError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

    report_path = args.scene_dir / "frames_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["source_type"] == "video":
        print(f"candidates: {report['candidates']} | selected: {report['selected']} "
              f"| rejected (blurry): {report['rejected_blurry']} "
              f"| sharpness median: {report['sharpness_median']:.1f}"
              f" | fps {report['fps']}{' (auto)' if report['fps_auto'] else ''}"
              f" | duplicates dropped: {report['duplicates_dropped']}")
    else:
        width, height = report["image_size"]
        focals = ", ".join(f"{value} mm" for value in report["focal_length_35mm"]) or "none"
        print(f"photos: {report['selected']} used / {report['photos_found']} JPG/JPEG/PNG files "
              f"| size {width}x{height} | EXIF focal (35 mm): {focals} "
              f"| sharpness median: {report['sharpness_median']:.1f}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    print(f"frames: {args.scene_dir / 'input'}\nreport: {report_path}")
    return 0 if report["selected"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
