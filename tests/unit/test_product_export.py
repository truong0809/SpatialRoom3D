"""Test phần xuất cho web của luồng sản phẩm: đọc PLY 3DGS, đổi sang .splat, cameras.json, thư viện (manifest.json).

Không cần PyTorch hay gsplat: Parser của gsplat được thay bằng một module giả trong examples/datasets/colmap.py của một
thư mục gsplat giả. Đối chiếu với exporter và Parser thật của gsplat @6e8c837 (cần PyTorch): docs/design/product.md
mục 7.
"""

import json
import math
import struct
import sys
from pathlib import Path

import pytest
import yaml

from product_fakes import FAKE_PARSER
from shell_env import REPO
from src.product import config as product_config
from src.product import export_web
from src.product import ply as ply_io
from src.product import train_main

np = pytest.importorskip("numpy")

GSPLAT_ORDER = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + [
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]



def write_ply(path: Path, rows: dict, order: list[str] | None = None, extra_header: str = "") -> Path:
    """PLY nhị phân float32 như gsplat ghi; `rows`: tên thuộc tính -> mảng giá trị (thiếu thì 0)."""
    order = order or GSPLAT_ORDER
    count = len(next(iter(rows.values())))
    data = np.zeros((count, len(order)), dtype="<f4")
    for index, name in enumerate(order):
        if name in rows:
            data[:, index] = rows[name]
    header = (f"ply\nformat binary_little_endian 1.0\n{extra_header}element vertex {count}\n"
              + "".join(f"property float {name}\n" for name in order) + "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode() + data.tobytes())
    return path


def gaussians(count: int, **overrides) -> dict:
    rows = {"x": np.arange(count, dtype=float), "y": np.zeros(count), "z": np.zeros(count),
            "scale_0": np.full(count, math.log(0.5)), "scale_1": np.full(count, math.log(0.5)),
            "scale_2": np.full(count, math.log(0.5)), "opacity": np.zeros(count), "rot_0": np.ones(count)}
    rows.update({key: np.asarray(value, dtype=float) for key, value in overrides.items()})
    return rows


def unpack(records) -> list[tuple]:
    data = records.tobytes()
    return [struct.unpack("<3f3f4B4B", data[i:i + 32]) for i in range(0, len(data), 32)]


# ------------------------------------------------------------------------------------------------- PLY
def test_the_gsplat_ply_layout_is_read_by_name(tmp_path):
    path = write_ply(tmp_path / "a.ply", gaussians(3, opacity=[1, 2, 3]), extra_header="comment gsplat\n")
    assert ply_io.vertex_count(path) == 3
    vertices = ply_io.read_vertices(path)
    assert list(vertices.dtype.names) == GSPLAT_ORDER and vertices["opacity"].tolist() == [1, 2, 3]
    other = ["nx", "ny", "nz"] + list(reversed(GSPLAT_ORDER))  # thứ tự khác, thêm thuộc tính (bản gốc Inria)
    path = write_ply(tmp_path / "b.ply", gaussians(3, opacity=[1, 2, 3]), order=other)
    assert ply_io.read_vertices(path)["opacity"].tolist() == [1, 2, 3]


def test_broken_ply_files_are_refused(tmp_path):
    path = write_ply(tmp_path / "a.ply", gaussians(4))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ply_io.PlyError, match="truncated"):
        ply_io.read_vertices(path)
    (tmp_path / "ascii.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n")
    with pytest.raises(ply_io.PlyError, match="only binary_little_endian"):
        ply_io.read_vertices(tmp_path / "ascii.ply")
    (tmp_path / "no.ply").write_bytes(b"not a ply\n")
    with pytest.raises(ply_io.PlyError, match="not a PLY file"):
        ply_io.read_header(tmp_path / "no.ply")
    (tmp_path / "open.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n")
    with pytest.raises(ply_io.PlyError, match="no end_header"):
        ply_io.read_header(tmp_path / "open.ply")


# ---------------------------------------------------------------------------------------------- .splat
def test_a_splat_record_follows_the_gsplat_exporter_byte_layout():
    """32 byte: vị trí, exp(scale) (float32), RGBA = (0,5 + C0 * f_dc, sigmoid(opacity)) × 255, quaternion × 128 + 128
    (uint8, cắt phần lẻ), như splat2splat_bytes của gsplat @6e8c837."""
    rows = gaussians(1, x=[1.5], y=[-2.0], z=[3.25], f_dc_0=[1.0], f_dc_1=[-1.0], f_dc_2=[10.0], opacity=[0.0],
                     scale_0=[math.log(0.5)], scale_1=[0.0], scale_2=[math.log(2.0)], rot_0=[2.0], rot_1=[0.0],
                     rot_2=[-2.0], rot_3=[0.0])
    records, stats = export_web.splat_records(write_ply_vertices(rows))
    assert records.itemsize == 32 and stats == {"input": 1, "kept": 1, "dropped_invalid": 0, "dropped_transparent": 0}
    (x, y, z, sx, sy, sz, r, g, b, a, q0, q1, q2, q3), = unpack(records)
    assert (x, y, z) == (1.5, -2.0, 3.25)
    assert sx == pytest.approx(0.5) and sy == pytest.approx(1.0) and sz == pytest.approx(2.0)
    c0 = export_web.SH_C0
    assert (r, g, b, a) == (int((0.5 + c0) * 255), int((0.5 - c0) * 255), 255, 127)  # 0,5 × 255 = 127,5 -> 127
    assert (q0, q1, q2, q3) == (int(2 / math.sqrt(8) * 128 + 128), 128, int(-2 / math.sqrt(8) * 128 + 128), 128)


def write_ply_vertices(rows: dict):
    """Mảng có cấu trúc như ply_io.read_vertices trả về (không qua file)."""
    count = len(next(iter(rows.values())))
    vertices = np.zeros(count, dtype=[(name, "<f4") for name in GSPLAT_ORDER])
    for name, values in rows.items():
        vertices[name] = values
    return vertices


def test_invalid_and_fully_transparent_gaussians_are_left_out_and_the_rest_sorted_by_importance():
    rows = gaussians(6, opacity=[0.0, 5.0, -20.0, 0.0, 0.0, 2.0],
                     scale_0=[0.0, 0.0, 0.0, float("nan"), 0.0, 1.0], rot_0=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                     x=[0, 1, 2, 3, 4, 5])
    records, stats = export_web.splat_records(write_ply_vertices(rows))
    assert stats == {"input": 6, "kept": 3, "dropped_invalid": 2, "dropped_transparent": 1}
    # thể tích × độ đục: x=5 (scale lớn, opacity 2), rồi x=1 (opacity 5), rồi x=0 (opacity 0)
    assert records["position"][:, 0].tolist() == [5.0, 1.0, 0.0]
    with pytest.raises(export_web.ExportError, match="missing rot_0"):
        export_web.splat_records(np.zeros(2, dtype=[(name, "<f4") for name in GSPLAT_ORDER if name != "rot_0"]))


# ---------------------------------------------------------------------------------------------- camera
def camera_spec(rotation=None) -> dict:
    """3 camera ở x = -1, 0, 1 nhìn theo +z, đứng thẳng (OpenCV: y xuống), tiêu cự 100, ảnh 100 × 80."""
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    c2w = []
    for x in (-1.0, 0.0, 1.0):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = (x, 0.0, 0.0)
        c2w.append(matrix.tolist())
    return {"names": ["00001.jpg", "00002.jpg", "00003.jpg"], "c2w": c2w,
            "K": [[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1]], "size": [100, 80]}


def test_camera_layout_gives_each_view_and_the_room_frame():
    spec = camera_spec()
    positions = np.array([[-2.0, -1.0, 0.0], [2.0, 1.0, 4.0]] * 50)
    layout = export_web.camera_layout(spec["c2w"], spec["names"], spec["K"], spec["size"], positions)
    assert layout["up"] == [0.0, -1.0, 0.0] and layout["center"] == [0.0, 0.0, 0.0] and layout["scene_radius"] == 1.0
    assert layout["right_handed"] is True and layout["default_view"] == 1
    assert layout["frames"][0] == {"name": "00001.jpg", "position": [-1.0, 0.0, 0.0], "forward": [0.0, 0.0, 1.0],
                                   "up": [0.0, -1.0, 0.0]}
    assert layout["intrinsics"] == {"width": 100, "height": 80, "fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0,
                                    "fov_y_deg": round(math.degrees(2 * math.atan(80 / 200)), 3)}
    assert layout["bounds"]["min"][0] == -2.0 and layout["bounds"]["max"][2] == 4.0
    mirrored = camera_spec(np.diag([-1.0, 1.0, 1.0]))  # nhánh phản xạ trong chuẩn hóa của gsplat
    assert export_web.camera_layout(mirrored["c2w"], mirrored["names"], mirrored["K"], mirrored["size"])[
        "right_handed"] is False
    with pytest.raises(export_web.ExportError, match="do not match"):
        export_web.camera_layout(spec["c2w"], spec["names"][:2], spec["K"], spec["size"])


# ------------------------------------------------------------------------------------------- xuất phòng
@pytest.fixture
def room(tmp_path, monkeypatch):
    """Một phòng đã train (train_meta.json OK, PLY), gsplat giả có Parser giả, thư mục web trống."""
    examples = tmp_path / "gsplat" / "examples"
    (examples / "datasets").mkdir(parents=True)
    (examples / "datasets" / "colmap.py").write_text(FAKE_PARSER, encoding="utf-8")
    data = yaml.safe_load((REPO / "configs" / "product.yaml").read_text(encoding="utf-8"))
    data["trainer"]["dir"] = str(tmp_path / "gsplat")
    config = tmp_path / "product.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = product_config.load(config, ws=str(tmp_path / "ws"))
    scene = tmp_path / "phong1" / "colmap"
    (scene / "images").mkdir(parents=True)
    for name in camera_spec()["names"]:
        write_image(scene / "images" / name)
    (scene / "fake_parser.json").write_text(json.dumps(camera_spec()), encoding="utf-8")
    (scene / "frames_report.json").write_text(json.dumps({"source_type": "video", "selected": 3, "duration_s": 61.5}),
                                              encoding="utf-8")
    (scene / "sfm_report.json").write_text(json.dumps({"input_images": 3, "verdict": "PASS",
                                                       "model": {"registered_images": 3}}), encoding="utf-8")
    model = tmp_path / "phong1" / "model"
    write_ply(model / "ply" / "point_cloud_29999.ply", gaussians(5))
    meta = {"status": "OK", "config": train_main.training_key(cfg), "num_gaussians": 5, "duration_s": 650.5,
            "finished_at": "2026-11-04T01:00:00+00:00", "files": {"ply": "ply/point_cloud_29999.ply"}}
    (model / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # mỗi lần xuất một thời điểm mới (thứ tự trong manifest), và datasets.colmap nạp lại từ gsplat giả của test này
    clock = iter(f"2026-11-04T02:00:{second:02d}+00:00" for second in range(60))
    monkeypatch.setattr(export_web, "now", lambda: next(clock))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("datasets", "datasets.colmap"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    class Room:
        pass

    env = Room()
    env.cfg, env.config, env.scene, env.model, env.web = cfg, config, scene, model, tmp_path / "web"
    yield env
    for name in ("datasets", "datasets.colmap"):  # module giả của test này không ở lại cho các test sau
        sys.modules.pop(name, None)


def write_image(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:  # không có Pillow: thumbnail là bản chép nguyên ảnh JPEG
        path.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg \xff\xd9")
        return
    Image.new("RGB", (1600, 1200), (200, 120, 40)).save(path, "JPEG")


def test_a_room_is_exported_with_its_cameras_and_added_to_the_library(room):
    entry = export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg, name="Phòng khách")
    folder = room.web / "rooms" / "phong1"
    assert sorted(path.name for path in folder.iterdir()) == ["cameras.json", "model.splat", "room.json",
                                                              "thumbnail.jpg"]
    assert (folder / "model.splat").stat().st_size == 5 * 32 == entry["size_bytes"]
    assert entry["files"] == {"model": "rooms/phong1/model.splat", "cameras": "rooms/phong1/cameras.json",
                              "thumbnail": "rooms/phong1/thumbnail.jpg"}
    assert entry["name"] == "Phòng khách" and entry["num_gaussians"] == 5 and entry["format"] == "splat"
    assert entry["viewer_modes"] == ["orbit", "walk", "tour"] and entry["metrics"] is None and entry["warnings"] == []
    assert entry["source"] == {"kind": "video", "input_images": 3, "registered_images": 3, "sfm_verdict": "PASS",
                               "video_duration_s": 61.5}
    assert entry["model"] == {"name": "mcmc", "cap_max": 1000000, "max_steps": 30000,
                              "trainer_commit": "6e8c837f35c9b025c0e92fb07bfda4e7c9361c49", "num_gaussians_trained": 5,
                              "train_seconds": 650.5, "trained_at": "2026-11-04T01:00:00+00:00"}
    assert len(entry["version"]) == 12
    # Parser của gsplat được gọi đúng như simple_trainer.py (chuẩn hóa hệ tọa độ, cùng data_factor, test_every)
    assert json.loads((room.scene / "parser_args.json").read_text()) == {"factor": 1, "normalize": True,
                                                                         "test_every": 1000000000}
    cameras = json.loads((folder / "cameras.json").read_text(encoding="utf-8"))
    assert [frame["name"] for frame in cameras["frames"]] == camera_spec()["names"]
    thumbnail = (folder / "thumbnail.jpg").read_bytes()
    assert thumbnail[:2] == b"\xff\xd8"
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        with Image.open(folder / "thumbnail.jpg") as image:
            assert max(image.size) == export_web.THUMBNAIL_SIDE
    manifest = json.loads((room.web / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1 and manifest["rooms"] == [entry]


def test_the_library_lists_the_latest_trained_room_first_and_keeps_names(room):
    import shutil

    later = room.model.parent.parent / "phong2" / "model"  # phòng thứ hai, train sau phòng thứ nhất
    shutil.copytree(room.model, later)
    meta = json.loads((later / "train_meta.json").read_text(encoding="utf-8"))
    (later / "train_meta.json").write_text(json.dumps({**meta, "finished_at": "2026-11-04T01:30:00+00:00"}),
                                           encoding="utf-8")
    export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg, name="Phòng khách")
    export_web.export_room("phong2", later, room.scene, room.web, room.cfg)
    manifest = json.loads((room.web / "manifest.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["rooms"]] == ["phong2", "phong1"]
    entry = export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg)  # xuất lại, không --name
    manifest = json.loads((room.web / "manifest.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["rooms"]] == ["phong2", "phong1"]  # xuất lại không đổi thứ tự
    assert entry["name"] == "Phòng khách" and manifest["rooms"][0]["name"] == "phong2"
    (room.web / "rooms" / "broken").mkdir()
    (room.web / "rooms" / "broken" / "room.json").write_text("{not json", encoding="utf-8")
    import shutil

    shutil.rmtree(room.web / "rooms" / "phong2")  # xóa một phòng rồi dựng lại thư viện
    assert export_web.main(["--rebuild-manifest", "--web-dir", str(room.web), "--config", str(room.config)]) == 0
    manifest = json.loads((room.web / "manifest.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["rooms"]] == ["phong1"]


def test_without_cameras_the_room_can_only_be_orbited(room):
    export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg)
    entry = export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg, cameras=False)
    assert entry["files"]["cameras"] is None and entry["viewer_modes"] == ["orbit"]
    assert not (room.web / "rooms" / "phong1" / "cameras.json").exists()  # của lần xuất trước: đã bỏ
    spec = camera_spec()
    spec["raise"] = "pycolmap could not read the model"
    (room.scene / "fake_parser.json").write_text(json.dumps(spec), encoding="utf-8")
    entry = export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg)
    assert entry["viewer_modes"] == ["orbit"] and entry["files"]["thumbnail"] is not None
    assert entry["warnings"] == ["Không đọc được vị trí camera (RuntimeError: pycolmap could not read the model): "
                                 "chỉ xem xoay quanh, tắt đi bộ và tham quan"]


def test_a_mirrored_world_is_reported(room):
    (room.scene / "fake_parser.json").write_text(json.dumps(camera_spec(np.diag([-1.0, 1.0, 1.0]))), encoding="utf-8")
    entry = export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg)
    assert entry["warnings"] == [export_web.WARN_MIRRORED] and "ảnh gương" in export_web.WARN_MIRRORED
    assert entry["viewer_modes"] == ["orbit", "walk", "tour"]  # vẫn có camera, chỉ cảnh báo


def test_only_a_finished_model_of_the_current_config_is_exported_cleanly(room, capsys):
    meta_path = room.model / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "status": "RUNNING"}), encoding="utf-8")
    with pytest.raises(export_web.ExportError, match="status RUNNING"):
        export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg)
    old = json.loads(json.dumps(meta))
    old["config"]["model"]["cap_max"] = 2_000_000
    meta_path.write_text(json.dumps(old), encoding="utf-8")
    entry = export_web.export_room("phong1", room.model, room.scene, room.web, room.cfg)
    assert entry["model"]["cap_max"] == 2_000_000  # ghi cấu hình đã thực sự train
    assert entry["warnings"] == ["Mô hình được train với cấu hình khác configs/product.yaml hiện tại; chạy lại "
                                 "scripts/build_room.sh để train lại"]
    with pytest.raises(export_web.ExportError, match="invalid room id"):
        export_web.export_room("../x", room.model, room.scene, room.web, room.cfg)
    args = ["--model", str(room.model), "--scene", str(room.scene), "--web-dir", str(room.web), "--config",
            str(room.config)]
    assert export_web.main(["--room", "phong1", *args]) == 0
    assert "WARN: Mô hình được train với cấu hình khác" in capsys.readouterr().out
    meta_path.unlink()
    assert export_web.main(["--room", "phong1", *args]) == 1
    assert "ERROR: export failed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        export_web.main(["--room", "phong1", "--config", str(room.config)])
