"""Test kiểm tra nhanh nguồn đầu vào (src.input.sources): chỉ dùng thư viện chuẩn, không giải mã ảnh."""

import subprocess
import sys

import pytest

from shell_env import REPO
from src.input.sources import (
    MIN_PHOTOS,
    InputError,
    check_source,
    main,
    natural_key,
    photo_count_problem,
    scan_photo_folder,
)


def touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")  # chỉ cần tên: module này không đọc nội dung ảnh
    return folder


def test_natural_order():
    names = ["IMG_10.jpg", "IMG_2.jpg", "img_1.JPG", "IMG_2a.jpg"]
    assert sorted(names, key=natural_key) == ["img_1.JPG", "IMG_2.jpg", "IMG_2a.jpg", "IMG_10.jpg"]


def test_scan_lists_photos_and_reasons_for_the_rest(tmp_path):
    folder = touch(tmp_path / "up", "IMG_10.JPG", "IMG_2.jpeg", "a.PNG", "b.heic", "clip.mp4", ".DS_Store",
                   "._IMG_10.JPG")
    (folder / ".ipynb_checkpoints").mkdir()
    (folder / "sub").mkdir()
    scan = scan_photo_folder(folder)
    assert [p.name for p in scan["photos"]] == ["a.PNG", "IMG_2.jpeg", "IMG_10.JPG"]
    reasons = {item["name"]: item["reason"] for item in scan["skipped"]}
    assert set(reasons) == {"b.heic", "clip.mp4", "sub"}  # file / thư mục ẩn bị bỏ qua lặng lẽ
    assert "xuất sang JPG" in reasons["b.heic"] and "thư mục con" in reasons["sub"]


def test_count_problems():
    assert "Không có ảnh" in photo_count_problem(0)
    single = photo_count_problem(1)
    assert single.startswith("Chỉ có 1 ảnh") and "không có chế độ dựng từ một ảnh" in single
    assert f"cần ít nhất {MIN_PHOTOS}" in photo_count_problem(MIN_PHOTOS - 1)
    assert photo_count_problem(MIN_PHOTOS) is None
    assert photo_count_problem(1, min_photos=1) == single  # một ảnh luôn bị từ chối
    assert photo_count_problem(3, min_photos=3) is None


def test_check_source_kinds(tmp_path):
    photos = touch(tmp_path / "photos", *[f"IMG_{i}.jpg" for i in range(12)])
    assert check_source(photos)["kind"] == "photos" and check_source(photos)["photos"] == 12
    video = touch(tmp_path / "v", "room.mp4") / "room.mp4"
    assert check_source(video) == {"kind": "video"}
    with pytest.raises(InputError, match="là một ảnh. Chỉ có 1 ảnh"):
        check_source(photos / "IMG_1.jpg")
    with pytest.raises(InputError, match=r"\.heic: định dạng này chưa hỗ trợ"):
        check_source(touch(tmp_path / "h", "x.heic") / "x.heic")
    with pytest.raises(InputError, match="Chỉ có 1 ảnh"):
        check_source(touch(tmp_path / "one", "a.jpg"))
    with pytest.raises(InputError, match="Chỉ có 5 ảnh"):
        check_source(touch(tmp_path / "five", *[f"{i}.png" for i in range(5)]))
    assert check_source(tmp_path / "five", min_photos=5)["kind"] == "photos"
    with pytest.raises(InputError, match="hãy đưa đường dẫn tới file video"):
        check_source(tmp_path / "v")  # thư mục chỉ có video
    with pytest.raises(InputError, match="Không thấy nguồn"):
        check_source(tmp_path / "missing")


def test_cli_prints_kind_and_ignores_frames_options(tmp_path, capsys):
    photos = touch(tmp_path / "photos", *[f"IMG_{i}.jpg" for i in range(10)], "notes.txt")
    assert main(["check", str(photos), "--target", "300", "--blur-ratio", "0.5"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "photos" and "10 ảnh" in out.err and "notes.txt" in out.err
    assert main(["check", str(photos / "IMG_1.jpg")]) == 1
    assert capsys.readouterr().out.startswith("ERROR: ")
    assert main(["check", str(photos), "--min-images", "11"]) == 1


def test_cli_runs_without_numpy_or_opencv(tmp_path):
    """run_scene.sh gọi module này bằng python3 của máy, có thể không có numpy / OpenCV."""
    photos = touch(tmp_path / "photos", *[f"IMG_{i}.jpg" for i in range(10)])
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['cv2'] = None; "
            "from src.input.sources import main; sys.exit(main(['check', sys.argv[1]]))")
    out = subprocess.run([sys.executable, "-c", code, str(photos)], capture_output=True, text=True,
                         env={"PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "photos"
