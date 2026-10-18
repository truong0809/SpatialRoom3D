"""Test chuẩn bị ảnh đầu vào: trích và chọn frame từ video; dùng thẳng thư mục ảnh chụp.

Video và ảnh trong test là dữ liệu tổng hợp. Luồng video và luồng ảnh được kiểm tra riêng.
"""

import json
import shutil
import struct
import subprocess

import cv2
import numpy as np
import pytest

from src.input.frames import (
    choose_fps,
    drop_duplicates,
    exif_focal_35mm,
    main,
    prepare_from_images,
    prepare_from_video,
    select_frames,
    sharpness,
    with_exif_focal_35mm,
)
from src.input.sources import InputError

FFMPEG = shutil.which("ffmpeg")
COLMAP = shutil.which("colmap")


def texture(height=360, width=1200, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 1.0)
    tiles = ((np.indices((height, width)).sum(axis=0) // 24) % 2 * 120).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(base, 0.6, tiles, 0.4, 0), cv2.COLOR_GRAY2BGR)


def heavy_blur(image):
    return cv2.GaussianBlur(image, (0, 0), 6.0)


def photo_set(folder, count, blurred=(), size=(640, 360), names=None):
    """`count` ảnh khác nhau (các đoạn cắt của một texture lớn), tên IMG_<i>.JPG theo thứ tự."""
    folder.mkdir(parents=True, exist_ok=True)
    base = texture(height=size[1], width=size[0] + 20 * count, seed=1)
    paths = []
    for i in range(1, count + 1):
        image = np.ascontiguousarray(base[:, i * 20: i * 20 + size[0]])
        path = folder / (names[i - 1] if names else f"IMG_{i}.JPG")
        cv2.imwrite(str(path), heavy_blur(image) if i in blurred else image)
        paths.append(path)
    return paths


def test_select_frames_picks_sharpest_per_window():
    scores = [1, 9, 2, 3, 3, 8, 7, 1]  # two windows of four
    selected, stats = select_frames(scores, target=2, blur_ratio=0.0)
    assert selected == [1, 5]
    assert stats["windows"] == 2 and stats["selected"] == 2


def test_select_frames_rejects_blurry_windows():
    scores = [100, 90, 95, 2, 3, 1, 98, 97, 99]  # middle window is blurry
    selected, stats = select_frames(scores, target=3, blur_ratio=0.4)
    assert selected == [0, 8]
    assert stats["rejected_blurry"] == 1


def test_select_frames_handles_more_windows_than_frames():
    selected, stats = select_frames([5.0, 6.0], target=10, blur_ratio=0.4)
    assert selected == [0, 1]
    assert stats["windows"] == 2


def test_select_frames_empty():
    selected, stats = select_frames([], target=10, blur_ratio=0.4)
    assert selected == [] and stats["candidates"] == 0


def test_sharpness_orders_sharp_above_blurred(tmp_path):
    image = texture()
    cv2.imwrite(str(tmp_path / "sharp.png"), image)
    cv2.imwrite(str(tmp_path / "blur.png"), heavy_blur(image))
    assert sharpness(tmp_path / "sharp.png") > 5 * sharpness(tmp_path / "blur.png")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_prepare_from_video_skips_blurred_frames(tmp_path):
    frames_dir = tmp_path / "src"
    frames_dir.mkdir()
    base = texture()
    blurred = set(range(1, 48, 4)) | set(range(32, 40))  # periodic blur + one fully blurred stretch
    for i in range(48):
        frame = np.ascontiguousarray(base[:, i * 8: i * 8 + 640])
        if i in blurred:
            frame = heavy_blur(frame)
        cv2.imwrite(str(frames_dir / f"f_{i:03d}.png"), frame)
    video = tmp_path / "pan.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-framerate", "12",
                    "-i", str(frames_dir / "f_%03d.png"), "-c:v", "mpeg4", "-q:v", "2", str(video)],
                   check=True)

    scene = tmp_path / "scene"
    report = prepare_from_video(video, scene, fps=12, target=12, blur_ratio=0.4)

    assert report["candidates"] == 48
    assert report["rejected_blurry"] == 2  # windows 32-35 and 36-39
    assert report["selected"] == 10
    chosen = {record["candidate_index"] for record in report["frames"]}
    assert not chosen & blurred
    assert sorted(p.name for p in (scene / "input").iterdir()) == [f"{k:05d}.jpg" for k in range(1, 11)]
    assert not (scene / "frames_candidates").exists()


def test_photos_are_used_directly_without_frame_selection(tmp_path):
    # 12 ảnh, ảnh 5 bị mờ: luồng video sẽ loại nó, luồng ảnh chỉ cảnh báo và giữ lại
    photo_set(tmp_path / "photos", 12, blurred={5})
    report = prepare_from_images(tmp_path / "photos", tmp_path / "scene", blur_ratio=0.4)
    assert report["source_type"] == "images" and report["selected"] == 12 == report["photos_found"]
    assert [r["source_name"] for r in report["frames"]] == [f"IMG_{i}.JPG" for i in range(1, 13)]
    assert [r["name"] for r in report["frames"]] == [f"{i:05d}.jpg" for i in range(1, 13)]
    assert report["blurry_kept"] == ["IMG_5.JPG"]
    assert any("mờ" in warning and "IMG_5.JPG" in warning for warning in report["warnings"])
    assert any("nên từ 30 ảnh" in warning for warning in report["warnings"])  # 12 < số ảnh nên có
    written = sorted(p.name for p in (tmp_path / "scene" / "input").iterdir())
    assert written == [f"{i:05d}.jpg" for i in range(1, 13)]
    assert cv2.imread(str(tmp_path / "scene" / "input" / "00005.jpg")).shape[:2] == (360, 640)
    assert not (tmp_path / "scene" / ".input_staging").exists()


def test_photo_order_is_natural_and_keeps_png(tmp_path):
    names = [f"IMG_{i}.{'png' if i % 2 else 'jpeg'}" for i in (10, 9, 11, 2, 1, 12, 3, 4, 5, 6)]
    photo_set(tmp_path / "photos", len(names), names=names)
    report = prepare_from_images(tmp_path / "photos", tmp_path / "scene")
    order = [r["source_name"] for r in report["frames"]]
    assert order == sorted(names, key=lambda n: int(n.split("_")[1].split(".")[0]))  # IMG_2 trước IMG_10
    assert all(r["name"].endswith(".jpg") for r in report["frames"])


def test_photo_folder_skips_hidden_unsupported_broken_and_duplicates(tmp_path):
    src = tmp_path / "photos"
    paths = photo_set(src, 11)
    shutil.copy2(paths[2], src / "IMG_3_copy.JPG")          # trùng y hệt IMG_3
    (src / "broken.png").write_bytes(b"not an image")        # hỏng
    (src / ".hidden.jpg").write_bytes(b"x")                  # file ẩn: bỏ qua lặng lẽ
    (src / ".ipynb_checkpoints").mkdir()                     # Jupyter tạo khi mở thư mục
    (src / "phone.HEIC").write_bytes(b"x")                   # định dạng chưa hỗ trợ
    (src / "notes.txt").write_text("x")
    (src / "more").mkdir()                                   # thư mục con: không đọc
    report = prepare_from_images(src, tmp_path / "scene")
    assert report["selected"] == 11 and report["photos_found"] == 13
    assert report["duplicates"] == [{"name": "IMG_3_copy.JPG", "same_as": "IMG_3.JPG"}]
    assert [item["name"] for item in report["unreadable"]] == ["broken.png"]
    assert sorted(item["name"] for item in report["skipped"]) == ["more", "notes.txt", "phone.HEIC"]
    heic = next(item for item in report["skipped"] if item["name"] == "phone.HEIC")
    assert "xuất sang JPG" in heic["reason"]
    assert ".hidden.jpg" not in json.dumps(report)


def test_single_photo_is_refused_with_explanation(tmp_path):
    photo_set(tmp_path / "photos", 1)
    with pytest.raises(InputError, match="Chỉ có 1 ảnh") as error:
        prepare_from_images(tmp_path / "photos", tmp_path / "scene")
    assert "không có chế độ dựng từ một ảnh" in str(error.value)
    assert not (tmp_path / "scene" / "input").exists()
    # cùng thông báo qua dòng lệnh, mã thoát 1, không ghi báo cáo
    assert main(["--images", str(tmp_path / "photos"), "--scene-dir", str(tmp_path / "scene")]) == 1
    assert not (tmp_path / "scene" / "frames_report.json").exists()


def test_too_few_photos_and_min_images_override(tmp_path):
    src = tmp_path / "photos"
    paths = photo_set(src, 4)
    shutil.copy2(paths[0], src / "dup.JPG")
    with pytest.raises(InputError, match=r"bỏ 1 ảnh trùng.*Chỉ có 4 ảnh dùng được, cần ít nhất 10"):
        prepare_from_images(src, tmp_path / "scene")
    report = prepare_from_images(src, tmp_path / "scene", min_photos=3)  # cố ý thử với ít ảnh
    assert report["selected"] == 4
    assert main(["--images", str(src), "--scene-dir", str(tmp_path / "s2"), "--min-images", "3"]) == 0
    # một ảnh luôn bị từ chối, kể cả khi hạ --min-images
    photo_set(tmp_path / "one", 1)
    with pytest.raises(InputError, match="Chỉ có 1 ảnh"):
        prepare_from_images(tmp_path / "one", tmp_path / "s3", min_photos=1)


def test_main_refuses_to_overwrite_existing_input(tmp_path):
    src = tmp_path / "photos"
    photo_set(src, 10)
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    (scene / "input" / "old.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["--images", str(src), "--scene-dir", str(scene)])
    assert main(["--images", str(src), "--scene-dir", str(scene), "--overwrite"]) == 0
    report = json.loads((scene / "frames_report.json").read_text(encoding="utf-8"))
    assert report["selected"] == 10
    assert not (scene / "input" / "old.jpg").exists()


def test_local_threshold_keeps_sharp_plain_walls():
    # 60 khung kệ sách nét (1000), 20 khung tường trơn nét (80), trong đoạn kệ có 4 khung mờ (5)
    scores = [1000.0] * 60 + [80.0] * 20
    for i in (16, 17, 18, 19):
        scores[i] = 5.0
    global_sel, _ = select_frames(scores, target=20, blur_ratio=0.4)
    local_sel, local_stats = select_frames(scores, target=20, blur_ratio=0.4, neighborhood=8)
    assert all(i < 60 for i in global_sel)  # trung vị toàn bộ (1000) loại hết tường trơn
    assert sum(i >= 60 for i in local_sel) == 5  # trung vị cục bộ giữ cả 5 đoạn tường trơn
    assert local_stats["rejected_blurry"] == 1  # đoạn 16-19 toàn khung mờ vẫn bị loại
    assert 16 not in local_sel and 17 not in local_sel


def test_choose_fps():
    assert choose_fps(100.0, target=250, per_window=4, max_fps=15) == 10.0
    assert choose_fps(20.0, target=250, per_window=4, max_fps=15) == 15  # trần
    assert choose_fps(None, target=250, per_window=4, max_fps=15) == 4.0  # không đọc được độ dài


def test_drop_duplicates(tmp_path):
    paths = []
    for i, content in enumerate([b"a", b"b", b"b", b"b", b"c", b"a"]):
        path = tmp_path / f"f{i}.jpg"
        path.write_bytes(content)
        paths.append(path)
    kept, dropped = drop_duplicates(paths)
    assert kept == [0, 1, 4, 5] and dropped == 2


def test_prepare_from_images_limits_size_and_rejects_mixed_orientation(tmp_path):
    src = tmp_path / "photos"
    photo_set(src, 10, size=(2400, 720))
    report = prepare_from_images(src, tmp_path / "scene", blur_ratio=0.0, max_side=1200)
    assert report["image_size"] == [1200, 360]
    assert report["frames"][0]["source_size"] == [2400, 720]
    assert cv2.imread(str(tmp_path / "scene" / "input" / "00001.jpg")).shape[:2] == (360, 1200)

    portrait = cv2.rotate(cv2.imread(str(src / "IMG_1.JPG")), cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(src / "IMG_99.JPG"), portrait)
    with pytest.raises(InputError, match="khác kích thước") as error:
        prepare_from_images(src, tmp_path / "scene2", blur_ratio=0.0, max_side=1200)
    message = str(error.value)
    assert "1200x360: 10 ảnh" in message and "360x1200: 1 ảnh (IMG_99.JPG)" in message
    assert "cùng chiều máy" in message
    assert not (tmp_path / "scene2" / "input").exists()


def with_exif_orientation(data, value):
    """Chèn đoạn APP1 Exif chỉ có thẻ Orientation (0x0112) ngay sau SOI của một file JPEG."""
    ifd = struct.pack("<H", 1) + struct.pack("<HHIHH", 0x0112, 3, 1, value, 0) + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + ifd
    return data[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + data[2:]


def test_photos_are_turned_upright_by_exif_orientation(tmp_path):
    """Điện thoại thường lưu ảnh chụp dọc bằng điểm ảnh ngang kèm thẻ EXIF Orientation; ảnh vào input/ phải đứng thẳng."""
    src = tmp_path / "photos"
    originals = [cv2.imread(str(path)) for path in photo_set(src, 10, size=(640, 360))]
    for path in sorted(src.iterdir()):
        path.write_bytes(with_exif_orientation(path.read_bytes(), 6))  # 6: xoay 90 độ theo chiều kim đồng hồ
    report = prepare_from_images(src, tmp_path / "scene", blur_ratio=0.0)
    assert report["image_size"] == [360, 640]
    out = cv2.imread(str(tmp_path / "scene" / "input" / "00001.jpg")).astype(np.float32)
    assert out.shape[:2] == (640, 360)
    clockwise = cv2.rotate(originals[0], cv2.ROTATE_90_CLOCKWISE).astype(np.float32)
    counter = cv2.rotate(originals[0], cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.float32)
    assert np.abs(out - clockwise).mean() < 4.0 < np.abs(out - counter).mean()


def phone_exif(focal_35mm, orientation=1, order=">"):
    """APP1 Exif giống ảnh điện thoại: IFD0 có Make, Orientation, con trỏ Exif IFD; Exif IFD có vài thẻ trước
    FocalLengthIn35mmFilm (0xA405). Mặc định big-endian ("MM") như nhiều máy Android / iPhone."""
    def entry(tag, kind, count, value):
        return struct.pack(order + "HHI", tag, kind, count) + value

    make = b"Phone\x00"
    ifd0_count, exif_count = 3, 3
    ifd0_offset = 8
    exif_offset = ifd0_offset + 2 + 12 * ifd0_count + 4
    make_offset = exif_offset + 2 + 12 * exif_count + 4
    short = (lambda v: struct.pack(order + "HH", v, 0))
    ifd0 = struct.pack(order + "H", ifd0_count) + entry(0x010F, 2, len(make), struct.pack(order + "I", make_offset)) \
        + entry(0x0112, 3, 1, short(orientation)) + entry(0x8769, 4, 1, struct.pack(order + "I", exif_offset)) \
        + struct.pack(order + "I", 0)
    exif = struct.pack(order + "H", exif_count) + entry(0x829A, 5, 1, struct.pack(order + "I", 0)) \
        + entry(0x9003, 2, 0, b"\x00" * 4) + entry(0xA405, 3, 1, short(focal_35mm)) + struct.pack(order + "I", 0)
    tiff = (b"MM\x00*" if order == ">" else b"II*\x00") + struct.pack(order + "I", ifd0_offset) + ifd0 + exif + make
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def with_phone_exif(data, focal_35mm, orientation=1, order=">"):
    return data[:2] + phone_exif(focal_35mm, orientation, order) + data[2:]


def test_exif_focal_is_read_from_phone_photos_and_written_back():
    jpeg = cv2.imencode(".jpg", texture(64, 80))[1].tobytes()
    assert exif_focal_35mm(jpeg) is None  # OpenCV không ghi EXIF
    for order in (">", "<"):
        assert exif_focal_35mm(with_phone_exif(jpeg, 26, orientation=6, order=order)) == 26
    tagged = with_exif_focal_35mm(jpeg, 28)
    assert exif_focal_35mm(tagged) == 28
    assert tagged[2:4] == b"\xff\xe0"  # JFIF APP0 vẫn đứng ngay sau SOI
    assert cv2.imdecode(np.frombuffer(tagged, np.uint8), cv2.IMREAD_COLOR).shape == (64, 80, 3)
    assert exif_focal_35mm(b"not a jpeg") is None and exif_focal_35mm(jpeg[:2] + b"\xff\xe1\x00") is None


def test_photos_keep_the_exif_focal_and_warn_when_missing_or_mixed(tmp_path):
    src = tmp_path / "photos"
    paths = photo_set(src, 10)
    for path in paths:
        path.write_bytes(with_phone_exif(path.read_bytes(), 26))
    report = prepare_from_images(src, tmp_path / "scene", blur_ratio=0.0)
    assert report["focal_length_35mm"] == {"26": 10}
    assert exif_focal_35mm((tmp_path / "scene" / "input" / "00001.jpg").read_bytes()) == 26
    assert report["frames"][0]["focal_length_35mm"] == 26
    assert not any("tiêu cự" in warning for warning in report["warnings"])

    for path in paths[7:]:
        path.write_bytes(with_phone_exif(cv2.imencode(".jpg", cv2.imread(str(path)))[1].tobytes(), 52))
    report = prepare_from_images(src, tmp_path / "scene2", blur_ratio=0.0)
    warning = next(w for w in report["warnings"] if "nhiều tiêu cự" in w)
    assert "26 mm: 7 ảnh" in warning and "52 mm: 3 ảnh (IMG_8.JPG, IMG_9.JPG, IMG_10.JPG)" in warning

    plain = photo_set(tmp_path / "plain", 10)
    report = prepare_from_images(plain[0].parent, tmp_path / "scene3", blur_ratio=0.0)
    assert report["focal_length_35mm"] == {}
    assert any("không ảnh nào có tiêu cự trong EXIF" in w for w in report["warnings"])


@pytest.mark.skipif(COLMAP is None, reason="colmap not installed")
def test_colmap_reads_the_focal_written_back(tmp_path):
    """COLMAP thật (3.x đọc EXIF qua FreeImage, 4.x qua OpenImageIO) lấy tiêu cự ban đầu từ ảnh trong input/."""
    from src.reconstruction.colmap_runner import colmap_env, detect_gpu_option

    src = tmp_path / "photos"
    for path in photo_set(src, 10, size=(640, 480)):
        path.write_bytes(with_phone_exif(path.read_bytes(), 26, orientation=1))
    prepare_from_images(src, tmp_path / "scene", blur_ratio=0.0)
    option = detect_gpu_option(COLMAP, "feature_extractor", ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"))
    database = tmp_path / "db.db"
    subprocess.run([COLMAP, "feature_extractor", "--database_path", str(database), "--image_path",
                    str(tmp_path / "scene" / "input"), "--ImageReader.single_camera", "1",
                    "--ImageReader.camera_model", "OPENCV", f"--{option}", "0"],
                   check=True, capture_output=True, env=colmap_env(), timeout=300)
    import sqlite3
    with sqlite3.connect(database) as db:
        params, prior = db.execute("SELECT params, prior_focal_length FROM cameras").fetchone()
    focal = struct.unpack("8d", params)[0]
    assert prior == 1
    # 3.x: f35 / 35 * cạnh dài = 475 px; 4.x: f35 / 43,27 * đường chéo = 481 px (ảnh 4:3)
    assert 460 < focal < 495, focal


def test_main_overwrite_clears_old_colmap_outputs(tmp_path):
    src = tmp_path / "photos"
    photo_set(src, 10)
    scene = tmp_path / "scene"
    for sub in ("sparse/0", "images", "colmap_logs"):
        (scene / sub).mkdir(parents=True)
    (scene / "sparse" / "0" / "images.bin").write_bytes(b"old")
    (scene / "sfm_report.json").write_text('{"verdict": "PASS"}', encoding="utf-8")
    with pytest.raises(SystemExit):  # kết quả cũ mà không có --overwrite
        main(["--images", str(src), "--scene-dir", str(scene)])
    assert main(["--images", str(src), "--scene-dir", str(scene), "--overwrite"]) == 0
    assert not (scene / "sparse").exists() and not (scene / "sfm_report.json").exists()
    assert (scene / "input" / "00001.jpg").is_file()


def test_overwrite_with_unusable_photos_keeps_the_old_scene(tmp_path):
    scene = tmp_path / "scene"
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "sparse" / "0" / "images.bin").write_bytes(b"old")
    (scene / "sfm_report.json").write_text('{"verdict": "PASS"}', encoding="utf-8")
    photo_set(tmp_path / "one", 1)
    assert main(["--images", str(tmp_path / "one"), "--scene-dir", str(scene), "--overwrite"]) == 1
    assert (scene / "sparse" / "0" / "images.bin").read_bytes() == b"old"  # ảnh mới không dùng được: không xóa gì
    assert (scene / "sfm_report.json").is_file()


def test_main_refuses_source_inside_scene(tmp_path):
    scene = tmp_path / "scene"
    photo_set(scene / "photos", 10)
    with pytest.raises(SystemExit):
        main(["--images", str(scene / "photos"), "--scene-dir", str(scene)])


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_keep_candidates_starts_from_an_empty_folder(tmp_path):
    frames_dir = tmp_path / "src"
    frames_dir.mkdir()
    base = texture()
    for i in range(24):
        cv2.imwrite(str(frames_dir / f"f_{i:03d}.png"), np.ascontiguousarray(base[:, i * 8: i * 8 + 640]))
    video = tmp_path / "pan.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-framerate", "12",
                    "-i", str(frames_dir / "f_%03d.png"), "-c:v", "mpeg4", "-q:v", "2", str(video)],
                   check=True)
    scene = tmp_path / "scene"
    first = prepare_from_video(video, scene, fps=12, target=6, blur_ratio=0.0, keep_candidates=True)
    shutil.rmtree(scene / "input")
    second = prepare_from_video(video, scene, fps=6, target=6, blur_ratio=0.0, keep_candidates=True)
    assert first["candidates"] == 24 and second["candidates"] == 12
    assert len(list((scene / "frames_candidates").glob("frame_*.jpg"))) == 12
    assert max(record["time_s"] for record in second["frames"]) < 2.0
