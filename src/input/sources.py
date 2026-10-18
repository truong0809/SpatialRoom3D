"""Nhận diện và kiểm tra nhanh nguồn đầu vào của một cảnh: file video hay thư mục ảnh JPG/JPEG/PNG.

Chỉ dùng thư viện chuẩn và không giải mã ảnh, để `scripts/pod/run_scene.sh` kiểm tra được ngay bằng `python3` của máy
(trước khi xếp hàng chờ GPU) và báo lỗi rõ khi đầu vào không dùng được:
  * thư mục không có ảnh JPG/JPEG/PNG nào;
  * chỉ có một ảnh (một file ảnh, hoặc thư mục có một ảnh): pipeline 3DGS nhiều góc nhìn không dựng được cả căn
    phòng từ một ảnh, nên từ chối và yêu cầu chụp thêm (không có chế độ dựng từ một ảnh);
  * ít hơn `MIN_PHOTOS` ảnh.
`src.input.frames` dùng lại các hàm này khi chuẩn bị ảnh (sau khi bỏ ảnh hỏng và ảnh trùng).

Cách dùng:
  python3 -m src.input.sources check /workspace/data/custom_scenes/phong1  # in "photos" hoặc "video"
  python3 -m src.input.sources check /workspace/data/custom_scenes/phong2/phong2.mp4 --min-images 20
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
# Ảnh không đọc được trực tiếp: báo rõ thay vì coi là video
OTHER_IMAGE_EXTS = {".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".dng", ".raw"}
# Ngưỡng kinh nghiệm ban đầu (hiệu chỉnh sau các lần chụp thử). Dưới MIN_PHOTOS: không chạy; COLMAP mặc định cũng
# coi mô hình dưới 10 ảnh là quá nhỏ (Mapper.min_model_size = 10). Dưới RECOMMENDED_PHOTOS: chỉ cảnh báo.
MIN_PHOTOS = 10
RECOMMENDED_PHOTOS = 30

SINGLE_IMAGE_MESSAGE = (
    "Chỉ có 1 ảnh. Pipeline 3DGS ước lượng vị trí camera và cấu trúc 3D từ nhiều ảnh chụp cùng một căn phòng "
    "ở các vị trí khác nhau (các ảnh chồng lấp nhau), nên một ảnh không đủ để dựng lại căn phòng: phần không nhìn "
    "thấy trong ảnh không thể khôi phục đúng, và hệ thống không có chế độ dựng từ một ảnh. Hãy chụp thêm ảnh của "
    f"cùng căn phòng (tối thiểu {MIN_PHOTOS}, nên từ {RECOMMENDED_PHOTOS} ảnh trở lên, xem "
    "docs/data/capture_protocol.md) rồi chạy lại.")


class InputError(RuntimeError):
    """Đầu vào không dùng được; thông điệp viết cho người dùng."""


def natural_key(name: str) -> list:
    """'IMG_2.jpg' xếp trước 'IMG_10.jpg' (thứ tự chụp của điện thoại / máy ảnh)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def scan_photo_folder(folder: Path) -> dict:
    """Ảnh JPG/JPEG/PNG trong thư mục (không xét thư mục con), theo thứ tự tên tự nhiên, và các mục bị bỏ qua."""
    photos, skipped = [], []
    for path in sorted(Path(folder).iterdir(), key=lambda p: natural_key(p.name)):
        if path.name.startswith("."):
            continue  # file ẩn, .ipynb_checkpoints của Jupyter, ._* của macOS
        if path.is_dir():
            skipped.append({"name": path.name, "reason": "thư mục con (chỉ đọc ảnh nằm trực tiếp trong thư mục)"})
        elif path.suffix.lower() in PHOTO_EXTS:
            photos.append(path)
        elif path.suffix.lower() in OTHER_IMAGE_EXTS:
            skipped.append({"name": path.name, "reason": f"định dạng {path.suffix} chưa hỗ trợ: xuất sang JPG"})
        else:
            skipped.append({"name": path.name, "reason": "không phải ảnh JPG/JPEG/PNG"})
    return {"photos": photos, "skipped": skipped}


def photo_count_problem(count: int, min_photos: int = MIN_PHOTOS) -> str | None:
    """Thông báo khi số ảnh dùng được không đủ cho pipeline nhiều góc nhìn; None nếu đủ."""
    if count == 0:
        return "Không có ảnh JPG/JPEG/PNG nào dùng được."
    if count == 1:
        return SINGLE_IMAGE_MESSAGE
    if count < min_photos:
        return (f"Chỉ có {count} ảnh dùng được, cần ít nhất {min_photos} ảnh chụp cùng một căn phòng từ nhiều vị trí "
                f"(nên từ {RECOMMENDED_PHOTOS} ảnh trở lên). Hãy chụp thêm, hoặc đặt --min-images nếu cố ý thử với ít "
                "ảnh.")
    return None


def check_source(source: Path, min_photos: int = MIN_PHOTOS) -> dict:
    """Loại nguồn ('photos' | 'video') và thông tin kiểm tra nhanh; nguồn không dùng được -> InputError."""
    source = Path(source)
    if not source.exists():
        raise InputError(f"Không thấy nguồn: {source}")
    if source.is_dir():
        scan = scan_photo_folder(source)
        count = len(scan["photos"])
        problem = photo_count_problem(count, min_photos)
        if problem:
            videos = [item["name"] for item in scan["skipped"]
                      if Path(item["name"]).suffix.lower() in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}]
            hint = f" Thư mục có video ({', '.join(videos[:3])}): với video, hãy đưa đường dẫn tới file video." \
                if videos and count == 0 else ""
            raise InputError(f"{source}: {problem}{hint}")
        return {"kind": "photos", "photos": count, "skipped": scan["skipped"]}
    suffix = source.suffix.lower()
    if suffix in PHOTO_EXTS:
        raise InputError(f"{source} là một ảnh. {SINGLE_IMAGE_MESSAGE} Nhiều ảnh thì đặt chung một thư mục và đưa "
                         "đường dẫn thư mục.")
    if suffix in OTHER_IMAGE_EXTS:
        raise InputError(f"{source} là một ảnh {suffix}: định dạng này chưa hỗ trợ, và một ảnh không đủ để dựng căn "
                         "phòng. Hãy chụp nhiều ảnh, xuất sang JPG và đặt chung một thư mục.")
    return {"kind": "video"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh nguồn đầu vào của một cảnh (video / thư mục ảnh)")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="in 'photos' hoặc 'video'; nguồn không dùng được thì báo lỗi, mã thoát 1")
    check.add_argument("source", type=Path)
    check.add_argument("--min-images", type=int, default=MIN_PHOTOS)
    # các tham số khác của frames.py được truyền kèm (run_scene.sh / prepare_scene.sh) và bỏ qua ở đây
    args, _unknown = parser.parse_known_args(argv)
    try:
        info = check_source(args.source, args.min_images)
    except InputError as error:
        print(f"ERROR: {error}")
        return 1
    print(info["kind"])
    if info["kind"] == "photos":
        print(f"{info['photos']} ảnh JPG/JPEG/PNG", file=sys.stderr)
        for item in info["skipped"][:10]:
            print(f"  bỏ qua {item['name']}: {item['reason']}", file=sys.stderr)
        if len(info["skipped"]) > 10:
            print(f"  ... và {len(info['skipped']) - 10} mục khác", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
