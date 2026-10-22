"""Test kiểm tra điểm SfM cho loss độ sâu (src/training/depth_check.py).

Parser giả có đúng các thuộc tính mà Parser của gsplat @6e8c837 (examples/datasets/colmap.py) cung cấp: image_names,
camera_ids, Ks_dict, imsize_dict, camtoworlds, points, point_indices. Camera ở gốc tọa độ, nhìn theo +z; ảnh 100 × 80.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.training.depth_check import main, valid_point_counts

K = [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
POINTS = [[0.0, 0.0, 5.0],    # 0: chiếu vào (50, 40), trong ảnh
          [0.0, 0.0, -5.0],   # 1: sau camera
          [10.0, 0.0, 5.0],   # 2: u = 250, ngoài ảnh
          [0.4, 0.3, 2.0]]    # 3: (70, 55), trong ảnh
NAMES = [f"img{i}.jpg" for i in range(9)]  # test_every 8: img0 và img8 là ảnh test
TRACKS = {"img0.jpg": [], "img1.jpg": [0, 1, 2, 3], "img2.jpg": [1, 2], "img4.jpg": [3], "img5.jpg": [0],
          "img6.jpg": [0, 3], "img7.jpg": [3], "img8.jpg": [1]}  # img3 không có trong track của điểm nào

FAKE_COLMAP = '''import json
from pathlib import Path

import numpy as np


class Parser:  # như Parser của gsplat, đọc cảnh từ spec.json
    def __init__(self, data_dir, factor=1, normalize=False, test_every=8):
        assert factor == 1 and normalize, (factor, normalize)
        spec = json.loads((Path(data_dir) / "spec.json").read_text())
        self.image_names = spec["names"]
        self.camera_ids = [1] * len(self.image_names)
        self.Ks_dict = {1: np.array(spec["K"])}
        self.imsize_dict = {1: (100, 80)}
        self.camtoworlds = np.stack([np.eye(4)] * len(self.image_names))
        self.points = np.array(spec["points"], dtype=np.float32)
        self.point_indices = {k: np.array(v, dtype=np.int32) for k, v in spec["tracks"].items() if v}
'''


def fake_parser(tracks):
    return SimpleNamespace(image_names=NAMES, camera_ids=[1] * len(NAMES), Ks_dict={1: np.array(K)},
                           imsize_dict={1: (100, 80)}, camtoworlds=np.stack([np.eye(4)] * len(NAMES)),
                           points=np.array(POINTS, dtype=np.float32),
                           point_indices={k: np.array(v, dtype=np.int32) for k, v in tracks.items() if v})


def test_counts_only_points_in_front_of_the_camera_and_inside_the_image():
    counts = valid_point_counts(fake_parser(TRACKS), test_every=8)
    assert "img0.jpg" not in counts and "img8.jpg" not in counts  # ảnh test không dùng loss độ sâu
    assert counts == {"img1.jpg": 2, "img2.jpg": 0, "img3.jpg": -1, "img4.jpg": 1, "img5.jpg": 1, "img6.jpg": 2,
                      "img7.jpg": 1}
    # điểm sát mép: u = 99.9 còn trong ảnh, u = 100 thì không (giống selector của gsplat: 0 <= u < rộng)
    edge = fake_parser({name: [0] for name in NAMES})
    edge.points = np.array([[0.4999, 0.0, 1.0]], dtype=np.float32)
    assert set(valid_point_counts(edge, 8).values()) == {1}
    edge.points = np.array([[0.5, 0.0, 1.0]], dtype=np.float32)
    assert set(valid_point_counts(edge, 8).values()) == {0}


@pytest.fixture
def examples(tmp_path, monkeypatch):
    """examples/ giả của gsplat (datasets/colmap.py), nạp như simple_trainer.py; dọn module sau test."""
    root = tmp_path / "examples"
    (root / "datasets").mkdir(parents=True)
    (root / "datasets" / "__init__.py").write_text("", encoding="utf-8")
    (root / "datasets" / "colmap.py").write_text(FAKE_COLMAP, encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("datasets", "datasets.colmap"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield root
    for name in ("datasets", "datasets.colmap"):
        sys.modules.pop(name, None)


def scene(tmp_path, name, tracks):
    path = tmp_path / name
    path.mkdir()
    (path / "spec.json").write_text(json.dumps({"names": NAMES, "K": K, "points": POINTS, "tracks": tracks}),
                                    encoding="utf-8")
    return path


def test_main_passes_good_scenes_and_names_the_images_that_would_break_training(tmp_path, examples, capsys):
    good = scene(tmp_path, "good", {name: [0, 3] for name in NAMES})
    bad = scene(tmp_path, "bad", TRACKS)
    assert main(["--examples-dir", str(examples), "--test-every", "8", str(good)]) == 0
    assert "good: OK, 7 training images, SfM points per image: min 2, median 2" in capsys.readouterr().out
    assert main(["--examples-dir", str(examples), str(good), str(bad), str(tmp_path / "missing")]) == 1
    out, err = capsys.readouterr()
    assert "good: OK" in out
    assert ("bad: FAIL: 7 training images, 1 with no SfM point in their tracks (gsplat: KeyError when loading them) "
            "['img3.jpg']") in out
    assert "missing: FAIL: gsplat cannot read the scene (FileNotFoundError" in out
    assert "2 scene(s) cannot be trained with the SfM depth loss" in err


def test_images_whose_points_all_fall_outside_only_warn(tmp_path, examples, capsys):
    # gsplat @6e8c837: loss độ sâu trên tập điểm rỗng là NaN ở giá trị ghi ra, gradient vẫn hữu hạn: lượt train không hỏng
    tracks = {**TRACKS, "img3.jpg": [0]}  # mọi ảnh train có track; img2 chỉ có điểm sau camera / ngoài ảnh
    assert main(["--examples-dir", str(examples), str(scene(tmp_path, "edge", tracks))]) == 0
    assert ("edge: OK, 7 training images, SfM points per image: min 0, median 1; WARN: 1 whose SfM points all fall "
            "outside the image (no depth supervision there; the logged loss is NaN on those steps, the gradients are "
            "not) ['img2.jpg']") in capsys.readouterr().out


def test_main_stops_when_gsplat_cannot_be_imported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.delitem(sys.modules, "datasets.colmap", raising=False)
    (tmp_path / "empty").mkdir()
    assert main(["--examples-dir", str(tmp_path / "empty"), str(tmp_path)]) == 2
    assert "cannot import gsplat's COLMAP parser" in capsys.readouterr().err
    assert main(["--examples-dir", str(tmp_path), "--test-every", "0", str(tmp_path)]) == 2


def test_the_parser_must_come_from_the_given_examples_folder(tmp_path, examples, monkeypatch):
    from src.training.depth_check import load_parser

    assert load_parser(examples).__module__ == "datasets.colmap"
    other = tmp_path / "other_examples"
    (other / "datasets").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="not from"):
        load_parser(other)  # datasets.colmap đã nạp từ chỗ khác (vd. gói datasets cùng tên)
