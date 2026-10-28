"""Test src/training/mono_depth.py (depth prior của biến thể mono) trên cảnh COLMAP tổng hợp, với bộ ước lượng độ sâu giả.

Cảnh: 4 ảnh 64×48, camera PINHOLE ở gốc tọa độ nhìn theo +z, các điểm SfM nằm trên mặt z = 2 + 0,05·u. Bộ ước lượng
giả nhận ra ảnh theo màu: ảnh a trả nghịch đảo độ sâu đúng, b trả độ sâu (phải được đảo), c trả bản đồ phẳng (không dùng
được), d chỉ có 10 điểm SfM (không kiểm được, vẫn dùng). Depth Anything V2 thật không chạy ở đây (cần trọng số và GPU):
scripts/setup/smoke_test_gsplat.py kiểm nó trên pod.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from colmap_writer import write_cameras, write_images, write_points3d
from src.reconstruction.colmap_model import (
    INVALID_POINT3D_ID,
    qvec_to_rotation,
    read_images_binary,
    read_points3d_positions,
)
from src.reconstruction.report import _rotation
from src.training import mono_depth

cv2 = pytest.importorskip("cv2")

REPO = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT, FOCAL = 64, 48, 50.0
COLORS = {"a.png": 10, "b.png": 20, "c.png": 30, "d.png": 40}


def plane_depth(u):
    return 2.0 + 0.05 * np.asarray(u, dtype=float)


def make_scene(root: Path, camera_model: int = 1, points: dict | None = None) -> Path:
    """Cảnh có images/ và sparse/0/; `points`: số điểm SfM mỗi ảnh (mặc định 60, ảnh d: 10)."""
    points = points or {"a.png": 60, "b.png": 60, "c.png": 60, "d.png": 10}
    scene = root / "scene"
    (scene / "images").mkdir(parents=True)
    model = scene / "sparse" / "0"
    model.mkdir(parents=True)
    params = [FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2] + ([0.01, 0.0, 0.0, 0.0] if camera_model == 4 else [])
    write_cameras(model / "cameras.bin", [(1, camera_model, WIDTH, HEIGHT, params)])
    images, points3d, point_id = [], [], 1
    rng = np.random.default_rng(0)
    for image_id, (name, count) in enumerate(points.items(), start=1):
        cv2.imwrite(str(scene / "images" / name), np.full((HEIGHT, WIDTH, 3), COLORS[name], dtype=np.uint8))
        points2d = []
        for k in range(count):
            u, v = int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT))
            z = float(plane_depth(u))
            xyz = ((u + 0.5 - WIDTH / 2) * z / FOCAL, (v + 0.5 - HEIGHT / 2) * z / FOCAL, z)
            points2d.append((u + 0.5, v + 0.5, point_id))
            points3d.append({"id": point_id, "xyz": xyz, "rgb": (1, 2, 3), "error": 0.5, "track": [(image_id, k)]})
            point_id += 1
        points2d.append((1.0, 1.0, INVALID_POINT3D_ID))
        images.append({"image_id": image_id, "name": name, "camera_id": 1, "qvec": (1.0, 0.0, 0.0, 0.0),
                       "tvec": (0.0, 0.0, 0.0), "points2d": points2d})
    write_images(model / "images.bin", images)
    write_points3d(model / "points3D.bin", points3d[::-1])  # thứ tự id không tăng dần: reader phải sắp xếp
    return scene


def fake_estimator():
    calls = []

    def estimate(bgr):
        value = int(bgr[0, 0, 0])
        calls.append(value)
        columns = np.broadcast_to(np.arange(bgr.shape[1], dtype=float), bgr.shape[:2])
        if value == COLORS["a.png"] or value == COLORS["d.png"]:
            return 1.0 / plane_depth(columns)  # như Depth Anything V2: lớn = gần
        if value == COLORS["b.png"]:
            return plane_depth(columns)  # "độ sâu": phải đảo
        return np.ones(bgr.shape[:2])  # phẳng: không dùng được

    return estimate, calls


def prepare(scene, cache, fingerprint="f" * 64, estimator=None):
    estimate = estimator or fake_estimator()[0]
    return mono_depth.prepare(scene, cache, fingerprint, lambda: estimate, log=lambda message: None)


def test_colmap_reader_keeps_2d_points_and_sorts_point_ids(tmp_path):
    scene = make_scene(tmp_path)
    images = read_images_binary(scene / "sparse" / "0" / "images.bin", with_points2d=True)
    first = images[1]
    assert first.points2d is not None and len(first.points2d) == 61 and first.num_observations == 60
    assert read_images_binary(scene / "sparse" / "0" / "images.bin")[1].points2d is None
    ids, xyz = read_points3d_positions(scene / "sparse" / "0" / "points3D.bin")
    assert list(ids) == sorted(ids) and len(ids) == 190 and xyz.shape == (190, 3)
    assert np.allclose(qvec_to_rotation((1, 0, 0, 0)), np.eye(3))
    qvec = (0.3, -0.5, 0.7, 0.1)
    assert np.allclose(qvec_to_rotation(qvec), _rotation(qvec))


def test_prior_per_image_with_sfm_check_and_flip(tmp_path):
    scene = make_scene(tmp_path)
    manifest = prepare(scene, tmp_path / "cache")
    entries = manifest["images"]
    assert set(entries) == set(COLORS) and manifest["complete"] and manifest["fingerprint"] == "f" * 64
    assert entries["a.png"]["use"] and entries["a.png"]["corr"] == pytest.approx(1.0, abs=1e-6)
    assert not entries["a.png"]["flipped"]
    assert entries["b.png"]["use"] and entries["b.png"]["flipped"] and entries["b.png"]["corr"] > 0.999
    assert not entries["c.png"]["use"] and entries["c.png"]["corr"] == pytest.approx(0.0, abs=1e-6)
    assert entries["d.png"]["use"] and entries["d.png"]["corr"] is None and entries["d.png"]["points"] == 10
    stored = np.load(tmp_path / "cache" / entries["b.png"]["file"])
    assert stored.dtype == np.float16 and stored.shape == (HEIGHT, WIDTH) and float(stored.max()) == 1.0
    # sau khi đảo: lớn = gần (cột trái gần hơn)
    assert stored[0, 0] > stored[0, -1]


def test_cache_is_reused_only_for_the_same_model_and_parameters(tmp_path):
    scene, cache = make_scene(tmp_path), tmp_path / "cache"
    estimate, calls = fake_estimator()
    prepare(scene, cache, estimator=estimate)
    assert len(calls) == 4

    def must_not_load():
        raise AssertionError("the cache should have been reused")

    assert mono_depth.prepare(scene, cache, "f" * 64, must_not_load, log=lambda m: None)["complete"]
    prepare(scene, cache, fingerprint="0" * 64, estimator=estimate)  # mô hình COLMAP khác: tính lại
    assert len(calls) == 8
    (cache / mono_depth.MANIFEST).unlink()  # cache dở dang
    prepare(scene, cache, fingerprint="0" * 64, estimator=estimate)
    assert len(calls) == 12
    (cache / "a.npy").unlink()
    assert not mono_depth.cache_is_complete(cache, "0" * 64)


def test_a_cache_folder_with_foreign_files_is_never_cleared(tmp_path):
    scene, cache = make_scene(tmp_path), tmp_path / "not_a_cache"
    cache.mkdir()
    (cache / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(mono_depth.MonoDepthError, match="not a depth prior"):
        prepare(scene, cache)
    assert (cache / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_scene_check_needs_agreement_with_sfm(tmp_path):
    manifest = prepare(make_scene(tmp_path), tmp_path / "cache")
    ok = mono_depth.check_scene(manifest, ["a.png", "b.png", "c.png"])
    assert ok["status"] == "OK" and ok["used"] == 2 and ok["low_corr"] == ["c.png"] and ok["flipped"] == 1
    assert ok["median_corr"] == pytest.approx(1.0, abs=1e-3)
    bad = mono_depth.check_scene(manifest, ["c.png"])
    assert bad["status"] == "FAILED" and bad["used"] == 0
    unchecked = mono_depth.check_scene(manifest, ["d.png"])
    assert unchecked["status"] == "OK" and unchecked["median_corr"] is None and unchecked["unchecked"] == 1
    with pytest.raises(mono_depth.MonoDepthError, match="not registered"):
        mono_depth.check_scene(manifest, ["zz.png"])


def write_jpeg_with_orientation(path: Path, image: np.ndarray, orientation: int) -> None:
    """JPEG có thẻ EXIF Orientation (0x0112): chèn một đoạn APP1 tối thiểu ngay sau SOI."""
    import struct

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    tiff = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) \
        + struct.pack("<I", 0)
    body = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
    data = encoded.tobytes()
    path.write_bytes(data[:2] + app1 + data[2:])


def test_exif_orientation_is_ignored_like_imageio(tmp_path):
    """gsplat đọc ảnh train bằng imageio (không xoay theo EXIF); prior phải tính trên đúng pixel đó, kể cả khi ảnh
    có thẻ xoay 180° (cùng kích thước nên kiểm tra kích thước không bắt được)."""
    scene = make_scene(tmp_path)
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :] = COLORS["a.png"]
    image[:, WIDTH // 2:] = 200  # nửa phải sáng
    write_jpeg_with_orientation(scene / "images" / "a.png", image, orientation=3)
    rotated = cv2.imread(str(scene / "images" / "a.png"), cv2.IMREAD_COLOR)
    seen = []
    estimate, _calls = fake_estimator()

    def recording(bgr):
        seen.append(bgr.copy())
        return estimate(bgr)

    prepare(scene, tmp_path / "cache", estimator=recording)
    first = seen[0]  # ảnh a.png là ảnh đầu tiên theo tên
    assert int(first[0, 0, 0]) < 60 and int(first[0, -1, 0]) > 150  # trái tối, phải sáng: không bị xoay
    if int(rotated[0, 0, 0]) > 150:  # OpenCV này có xoay theo EXIF khi không có cờ: test thật sự phân biệt hai cách đọc
        assert not np.array_equal(rotated, first)


def test_distorted_cameras_and_wrong_image_sizes_are_refused(tmp_path):
    with pytest.raises(mono_depth.MonoDepthError, match="not undistorted"):
        prepare(make_scene(tmp_path / "distorted", camera_model=4), tmp_path / "c1")
    scene = make_scene(tmp_path / "size")
    cv2.imwrite(str(scene / "images" / "a.png"), np.full((32, 32, 3), 10, dtype=np.uint8))
    with pytest.raises(mono_depth.MonoDepthError, match="COLMAP camera is 64x48"):
        prepare(scene, tmp_path / "c2")


def test_command_line_writes_the_record_and_fails_on_a_bad_prior(tmp_path, monkeypatch, capsys):
    scene = make_scene(tmp_path)
    estimate, _calls = fake_estimator()
    monkeypatch.setattr(mono_depth, "load_depth_anything", lambda code_dir: estimate)
    split = tmp_path / "split.json"
    record = tmp_path / "run" / "mono_depth.json"
    args = ["--scene", str(scene), "--fingerprint", "f" * 64, "--cache", str(tmp_path / "cache"), "--split", str(split),
            "--record", str(record), "--depth-anything-dir", str(tmp_path / "da2")]
    split.write_text(json.dumps({"train": ["a.png", "b.png", "c.png"], "test": ["d.png"]}), encoding="utf-8")
    assert mono_depth.main(args) == 0
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["status"] == "OK" and saved["used"] == 2 and len(saved["manifest_sha256"]) == 64
    assert "2/3 train images use the depth prior" in capsys.readouterr().out
    split.write_text(json.dumps({"train": ["c.png"], "test": ["d.png"]}), encoding="utf-8")
    assert mono_depth.main(args) == 1
    assert json.loads(record.read_text(encoding="utf-8"))["status"] == "FAILED"
    assert "does not agree with the SfM points" in capsys.readouterr().err


def test_depth_anything_checkout_and_weights_are_checked_before_loading(tmp_path, monkeypatch):
    with pytest.raises(mono_depth.MonoDepthError, match="setup_depth_anything.sh"):
        mono_depth.load_depth_anything(tmp_path)  # không phải checkout git
    monkeypatch.setattr(mono_depth, "checkout_state", lambda code_dir: (mono_depth.DA2_COMMIT, False))
    weights = tmp_path / mono_depth.WEIGHTS_FILE
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not the weights")
    with pytest.raises(mono_depth.MonoDepthError, match="not the pinned Depth Anything V2 Small weights"):
        mono_depth.load_depth_anything(tmp_path)
    monkeypatch.setattr(mono_depth, "checkout_state", lambda code_dir: (mono_depth.DA2_COMMIT, True))
    with pytest.raises(mono_depth.MonoDepthError, match="modified"):
        mono_depth.load_depth_anything(tmp_path)


def test_pins_match_the_environment_config():
    text = (REPO / "configs" / "environment" / "gs-gsplat.conf").read_text(encoding="utf-8")
    conf = dict(re.findall(r'^(DA2_[A-Z0-9_]+)="([^"]*)"', text, flags=re.MULTILINE))
    assert conf["DA2_COMMIT"] == mono_depth.DA2_COMMIT == mono_depth.MODEL["commit"]
    assert conf["DA2_WEIGHTS_FILE"] == mono_depth.WEIGHTS_FILE
    assert conf["DA2_WEIGHTS_SHA256"] == mono_depth.WEIGHTS_SHA256 == mono_depth.MODEL["weights_sha256"]
    assert conf["DA2_WEIGHTS_URL"].endswith("/" + Path(mono_depth.WEIGHTS_FILE).name)
