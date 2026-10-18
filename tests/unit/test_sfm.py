"""Test đọc model COLMAP, báo cáo chất lượng và điều phối COLMAP (dùng chương trình colmap giả).

Mọi model và số liệu trong file này là dữ liệu tổng hợp để kiểm tra logic.
"""

import json
import os
import stat
import sys
import textwrap

import pytest

import numpy as np

from colmap_writer import make_model, write_cameras, write_images, write_points3d
from src.reconstruction.colmap_model import read_cameras_binary, read_images_binary, read_points3d_summary
from src.reconstruction.colmap_runner import main as colmap_main
from src.reconstruction.colmap_runner import run_colmap
from src.reconstruction.report import analyze_coverage, analyze_scene, failure_hints, format_report


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


def make_scene(tmp_path, n_input, models, error=0.8, final_from=0, redundant=False):
    """models: list số ảnh đăng ký của từng model trong distorted/sparse/<k>.

    Model k lấy các ảnh tiếp theo sau model k-1 (các nhóm không chung ảnh, như khi COLMAP tách cảnh); `redundant=True`:
    mọi model đều lấy từ ảnh đầu tiên (model phụ chỉ lặp lại ảnh của model lớn nhất).
    """
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(n_input):
        (scene / "input" / name).write_bytes(b"jpg")
    start, model_names = 0, []
    for registered in models:
        model_names.append(names(start + registered)[start:])
        start = 0 if redundant else start + registered
    for index, image_names in enumerate(model_names):
        make_model(scene / "distorted" / "sparse" / str(index), image_names, camera_model_id=4, error=error)
    if models:
        make_model(scene / "sparse" / "0", model_names[final_from], camera_model_id=1, error=error)
    return scene


def test_readers_roundtrip(tmp_path):
    model = make_model(tmp_path / "m", names(4), camera_model_id=4, error=0.5, points_per_image=3)
    cameras = read_cameras_binary(model / "cameras.bin")
    assert cameras[1].model == "OPENCV" and len(cameras[1].params) == 8
    assert (cameras[1].width, cameras[1].height) == (640, 480)
    images = read_images_binary(model / "images.bin")
    assert [images[i].name for i in sorted(images)] == names(4)
    assert images[2].num_points2d == 4 and images[2].num_observations == 3
    assert images[3].tvec == (0.0, 0.0, 3.0)
    points = read_points3d_summary(model / "points3D.bin")
    assert points.count == 12
    assert points.mean_error == pytest.approx(0.5)
    assert points.mean_track_length == pytest.approx(2.0)


def test_report_pass(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 20, [20]))
    assert report["verdict"] == "PASS"
    assert report["registration_ratio"] == 1.0
    assert report["model"]["cameras"][0]["model"] == "PINHOLE"
    assert report["unregistered_images"] == []


def test_report_warn_when_split_into_models(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 21, [20, 1]))
    assert report["verdict"] == "WARN"  # 20/21 = 95 % đăng ký, nhưng có ảnh nằm ở nhóm riêng
    assert report["split_groups"] == 2 and report["redundant_models"] == 0
    assert any("split into 2 groups" in w for w in report["warnings"])


def test_model_repeating_registered_images_is_not_a_split(tmp_path):
    """COLMAP 3.9.1 trên ảnh tổng hợp: model 36/36 ảnh, cộng một model phụ 2 ảnh lấy lại từ chính các ảnh đó."""
    report = analyze_scene(make_scene(tmp_path, 20, [20, 2], redundant=True))
    assert report["verdict"] == "PASS" and report["hints"] == []
    assert report["split_groups"] == 1 and report["redundant_models"] == 1
    assert not any("split" in w for w in report["warnings"])
    assert "(1 only repeat images of the largest model, ignored)" in format_report(report)


def test_report_warn_and_fail_on_low_registration(tmp_path):
    assert analyze_scene(make_scene(tmp_path / "a", 20, [17]))["verdict"] == "WARN"
    report = analyze_scene(make_scene(tmp_path / "b", 20, [10]))
    assert report["verdict"] == "FAIL"
    assert report["unregistered_images"] == names(20)[10:]


def test_report_warn_on_high_reprojection_error(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 10, [10], error=2.0))
    assert report["verdict"] == "WARN"
    assert any("reprojection error" in w for w in report["warnings"])


def test_report_fail_without_model(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 5, []))
    assert report["verdict"] == "FAIL"


FAKE_COLMAP = textwrap.dedent("""\
    #!{python}
    import json, os, shutil, sys
    from pathlib import Path
    args = sys.argv[1:]
    with open(os.environ["FAKE_COLMAP_LOG"], "a") as log:
        log.write(json.dumps(args) + "\\n")
    new = os.environ.get("FAKE_COLMAP_STYLE", "new") == "new"
    cmd = args[0]
    if args == ["-h"]:
        # dòng đầu của `colmap -h` ở COLMAP 4.2.1 (src/colmap/exe/colmap.cc, util/version.cc.in)
        print(os.environ.get("FAKE_COLMAP_HEADER", "COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 with CUDA)"))
        print("Usage:")
        sys.exit(0)
    if "-h" in args:
        if cmd == "feature_extractor":
            print("--FeatureExtraction.use_gpu" if new else "--SiftExtraction.use_gpu")
        else:
            print("--FeatureMatching.use_gpu" if new else "--SiftMatching.use_gpu")
        sys.exit(0)
    def opt(name):
        return args[args.index("--" + name) + 1]
    if cmd == "feature_extractor":
        Path(opt("database_path")).write_bytes(b"db")
    elif cmd in ("exhaustive_matcher", "sequential_matcher"):
        assert Path(opt("database_path")).exists()
    elif cmd in ("mapper", "global_mapper"):
        if os.environ.get("FAKE_COLMAP_FAIL_MAPPER"):
            print("mapper: no good initial image pair found")
            sys.exit(1)
        for model in Path(os.environ["FAKE_COLMAP_TEMPLATE"]).iterdir():
            shutil.copytree(model, Path(opt("output_path")) / model.name)
    elif cmd == "image_undistorter":
        out, src = Path(opt("output_path")), Path(opt("input_path"))
        shutil.copytree(opt("image_path"), out / "images")
        (out / "sparse").mkdir(exist_ok=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy(src / name, out / "sparse" / name)
    else:
        sys.exit(2)
    """)


@pytest.fixture
def fake_colmap(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "colmap"
    exe.parent.mkdir()
    exe.write_text(FAKE_COLMAP.format(python=sys.executable), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    template = tmp_path / "template"
    make_model(template / "0", names(2), camera_model_id=4)
    make_model(template / "1", names(5), camera_model_id=4)  # largest model, not index 0
    log = tmp_path / "colmap_calls.jsonl"
    monkeypatch.setenv("FAKE_COLMAP_TEMPLATE", str(template))
    monkeypatch.setenv("FAKE_COLMAP_LOG", str(log))
    return exe, log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("style,extract_opt,match_opt", [
    ("new", "FeatureExtraction.use_gpu", "FeatureMatching.use_gpu"),
    ("old", "SiftExtraction.use_gpu", "SiftMatching.use_gpu"),
])
def test_runner_orchestrates_colmap(tmp_path, fake_colmap, monkeypatch, style, extract_opt, match_opt):
    exe, log = fake_colmap
    monkeypatch.setenv("FAKE_COLMAP_STYLE", style)
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(6):
        (scene / "input" / name).write_bytes(b"jpg")

    report = run_colmap(scene, colmap=str(exe))

    settings = report["colmap_settings"]
    assert settings["extract_gpu_option"] == extract_opt
    assert settings["match_gpu_option"] == match_opt
    assert settings["undistorted_model"].endswith(os.sep + "1")
    run_calls = [c for c in calls(log) if "-h" not in c]
    assert [c[0] for c in run_calls] == ["feature_extractor", "exhaustive_matcher", "mapper", "image_undistorter"]
    assert run_calls[0][run_calls[0].index(f"--{extract_opt}") + 1] == "1"
    assert "--Mapper.ba_global_function_tolerance=0.000001" in run_calls[2]
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (scene / "sparse" / "0" / name).is_file()
    assert report["registration_ratio"] == pytest.approx(5 / 6)
    assert report["unregistered_images"] == ["00006.jpg"]
    assert report["verdict"] == "WARN"  # 83 % registered (model phụ chỉ lặp lại ảnh, không tính là tách)
    assert report["split_groups"] == 1 and report["redundant_models"] == 1
    assert json.loads((scene / "sfm_report.json").read_text(encoding="utf-8"))["verdict"] == "WARN"


def test_runner_options_and_overwrite_guard(tmp_path, fake_colmap):
    exe, log = fake_colmap
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(5):
        (scene / "input" / name).write_bytes(b"jpg")

    report = run_colmap(scene, colmap=str(exe), gpu="off", matcher="sequential",
                        sequential_overlap=15, loop_detection=True, mapper="global")
    assert report["verdict"] == "PASS"  # 5/5 ảnh; model phụ (2 ảnh) chỉ lặp lại ảnh của model lớn nhất
    assert report["redundant_models"] == 1
    run_calls = [c for c in calls(log) if "-h" not in c]
    assert run_calls[0][run_calls[0].index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert run_calls[1][0] == "sequential_matcher" and "15" in run_calls[1]
    assert run_calls[1][run_calls[1].index("--SequentialMatching.loop_detection") + 1] == "1"
    assert run_calls[2][0] == "global_mapper"

    with pytest.raises(RuntimeError, match="--overwrite"):
        run_colmap(scene, colmap=str(exe))
    assert run_colmap(scene, colmap=str(exe), overwrite=True)["verdict"] == "PASS"


@pytest.mark.parametrize("header,expect_gpu,expect_cuda", [
    ("COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 without GPU support)", "0", False),  # bản CPU conda-forge
    ("COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 with CUDA)", "1", True),
    ("COLMAP 3.9.1 -- Structure-from-Motion and Multi-View Stereo\n(Commit 1 on 2023-01-01 without CUDA)", "0", False),
    ("COLMAP (header without build info)", "0", None),  # không nhận ra: dùng CPU cho an toàn
])
def test_runner_picks_cpu_unless_colmap_has_cuda(tmp_path, fake_colmap, monkeypatch, header, expect_gpu, expect_cuda):
    exe, log = fake_colmap
    monkeypatch.setenv("FAKE_COLMAP_HEADER", header)
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(5):
        (scene / "input" / name).write_bytes(b"jpg")

    report = run_colmap(scene, colmap=str(exe))  # gpu="auto" mặc định

    run_calls = [c for c in calls(log) if "-h" not in c]
    assert run_calls[0][run_calls[0].index("--FeatureExtraction.use_gpu") + 1] == expect_gpu
    assert run_calls[1][run_calls[1].index("--FeatureMatching.use_gpu") + 1] == expect_gpu
    settings = report["colmap_settings"]
    assert settings["colmap_cuda"] is expect_cuda
    assert settings["use_gpu"] is (expect_gpu == "1") and settings["gpu_requested"] == "auto"


def test_runner_refuses_gpu_on_cpu_build(tmp_path, fake_colmap, monkeypatch):
    exe, _ = fake_colmap
    monkeypatch.setenv("FAKE_COLMAP_HEADER", "COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 without GPU support)")
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    (scene / "input" / "00001.jpg").write_bytes(b"jpg")
    with pytest.raises(RuntimeError, match="no GPU support"):
        run_colmap(scene, colmap=str(exe), gpu="on")


def test_runner_failure_replaces_old_report(tmp_path, fake_colmap, monkeypatch):
    exe, _ = fake_colmap
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(5):
        (scene / "input" / name).write_bytes(b"jpg")
    assert run_colmap(scene, colmap=str(exe))["verdict"] == "PASS"  # 5/5 ảnh (model phụ chỉ lặp lại ảnh)
    assert (scene / "sparse" / "0" / "images.bin").is_file()

    monkeypatch.setenv("FAKE_COLMAP_FAIL_MAPPER", "1")
    with pytest.raises(RuntimeError, match="3_mapper"):
        run_colmap(scene, colmap=str(exe), overwrite=True)
    report = json.loads((scene / "sfm_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL" and "3_mapper" in report["error"]
    assert not (scene / "sparse").exists() and not (scene / "images").exists()


# ------------------------------------------------------------------------- kiểm tra trước khi train: gợi ý, độ phủ

def _quat(rotation):
    """Ma trận quay -> quaternion (w, x, y, z) theo quy ước của COLMAP."""
    m = np.asarray(rotation, dtype=float)
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    return (w, x, y, z)


def camera_looking(azimuth_deg, pitch_deg=0.0):
    """cam_from_world của camera đứng thẳng (z thế giới hướng lên), nhìn theo phương vị / độ ngẩng cho trước."""
    a, p = np.radians(azimuth_deg), np.radians(pitch_deg)
    forward = np.array([np.cos(a) * np.cos(p), np.sin(a) * np.cos(p), np.sin(p)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])  # hàng = trục camera (x phải, y xuống, z nhìn) trong hệ thế giới


def make_posed_model(model_dir, cameras, points):
    """cameras: list (tên, R cam_from_world, tâm camera). Ảnh 640x480, fx = 500 (góc nhìn ngang ~65 độ)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    write_cameras(model_dir / "cameras.bin", [(1, 1, 640, 480, [500.0, 500.0, 320.0, 240.0])])
    images = [{"image_id": i, "name": name, "camera_id": 1, "qvec": _quat(rotation),
               "tvec": tuple(-rotation @ np.asarray(center, dtype=float)), "points2d": []}
              for i, (name, rotation, center) in enumerate(cameras, start=1)]
    write_images(model_dir / "images.bin", images)
    write_points3d(model_dir / "points3D.bin", [
        {"id": j, "xyz": tuple(map(float, xyz)), "rgb": (1, 2, 3), "error": 0.5, "track": [(1, 0), (2, 0)]}
        for j, xyz in enumerate(points, start=1)])
    return model_dir


def room_points(half=3.0, height=2.6, per_side=12):
    """Điểm trên 4 bức tường của một phòng vuông cạnh 2*half quanh gốc tọa độ."""
    t = np.linspace(-half, half, per_side)
    z = np.linspace(0.2, height - 0.2, 4)
    pts = []
    for u in t:
        for h in z:
            pts += [(half, u, h), (-half, u, h), (u, half, h), (u, -half, h)]
    return pts


def test_camera_helper_matches_colmap_convention(tmp_path):
    rotation = camera_looking(30, 10)
    model = make_posed_model(tmp_path / "m", [("a.jpg", rotation, (1.0, 2.0, 1.5))], room_points())
    image = read_images_binary(model / "images.bin")[1]
    from src.reconstruction.report import _rotation
    back = _rotation(image.qvec)
    assert np.allclose(back, rotation, atol=1e-9)
    assert np.allclose(-back.T @ np.array(image.tvec), [1.0, 2.0, 1.5])


def test_coverage_ok_for_walk_around_capture(tmp_path):
    cams = [(f"{k + 1:05d}.jpg", camera_looking(30 * k), (np.cos(np.radians(30 * k)), np.sin(np.radians(30 * k)), 1.5))
            for k in range(12)]
    coverage = analyze_coverage(make_posed_model(tmp_path / "m", cams, room_points()))
    assert coverage["status"] == "OK", coverage
    assert coverage["azimuth_coverage_deg"] == 360 and coverage["largest_gap_deg"] == 0
    assert coverage["horizontal_fov_deg"] == pytest.approx(65.2, abs=0.1)
    assert coverage["camera_extent_over_depth"] > 0.2


def test_coverage_warns_about_unseen_directions_with_photo_names(tmp_path):
    cams = [(f"{k + 1:05d}.jpg", camera_looking(10 * k), (0.3 * k, -0.2 * k, 1.5)) for k in range(7)]  # 0..60 độ
    labels = {name: f"{name} (IMG_{i}.JPG)" for i, (name, _, _) in enumerate(cams, start=1)}
    coverage = analyze_coverage(make_posed_model(tmp_path / "m", cams, room_points()), labels)
    assert coverage["status"] == "LOW"
    assert 120 <= coverage["azimuth_coverage_deg"] <= 130 and coverage["largest_gap_deg"] >= 230
    warning = coverage["warnings"][0]
    assert "không ảnh nào nhìn tới" in warning
    assert "00007.jpg (IMG_7.JPG)" in warning and "00001.jpg (IMG_1.JPG)" in warning  # hai ảnh ở hai mép khoảng trống
    assert "một phần căn phòng" in warning


def test_coverage_warns_when_camera_only_rotates_in_place(tmp_path):
    rng = np.random.default_rng(0)
    cams = [(f"{k + 1:05d}.jpg", camera_looking(30 * k), tuple(rng.normal(0, 0.01, 3) + [0, 0, 1.5]))
            for k in range(12)]
    coverage = analyze_coverage(make_posed_model(tmp_path / "m", cams, room_points()))
    assert coverage["azimuth_coverage_deg"] == 360
    assert coverage["camera_extent_over_depth"] < 0.05
    assert coverage["status"] == "LOW" and any("đứng một chỗ" in w for w in coverage["warnings"])


def test_coverage_ignores_cameras_looking_at_floor_or_ceiling(tmp_path):
    cams = [(f"{k + 1:05d}.jpg", camera_looking(30 * k, 80 if k % 2 else -80),
             (np.cos(np.radians(30 * k)), np.sin(np.radians(30 * k)), 1.5)) for k in range(12)]
    cams += [(f"{k + 13:05d}.jpg", camera_looking(90 * k), (0.5 * k, 0.0, 1.5)) for k in range(4)]
    coverage = analyze_coverage(make_posed_model(tmp_path / "m", cams, room_points()))
    assert coverage["steep_cameras"] == 12 and coverage["level_cameras"] == 4
    assert coverage["azimuth_coverage_deg"] < 300  # 4 camera ngang, mỗi camera ~65 độ


def test_report_explains_unregistered_photos_and_split_groups(tmp_path):
    scene = make_scene(tmp_path, 20, [12, 6])
    frames = {"source_type": "images", "frames": [{"name": name, "source_name": f"IMG_{i}.JPG"}
                                                  for i, name in enumerate(names(20), start=1)]}
    (scene / "frames_report.json").write_text(json.dumps(frames), encoding="utf-8")
    report = analyze_scene(scene)
    assert report["verdict"] == "FAIL" and report["source_type"] == "images"
    hints = "\n".join(report["hints"])
    assert "8/20 ảnh không ghép được" in hints and "00013.jpg (IMG_13.JPG)" in hints
    assert "tách ảnh thành 2 nhóm" in hints and "nhóm 1: 12 ảnh, nhóm 2: 6 ảnh" in hints
    assert "phòng" in hints and "Không train" in hints
    text = format_report(report)
    assert "SFM VERDICT: FAIL" in text and "HINT: " in text and "VIEW COVERAGE: " in text


def test_report_for_video_talks_about_frames_and_filming(tmp_path):
    scene = make_scene(tmp_path, 20, [12, 6])
    (scene / "frames_report.json").write_text('{"source_type": "video"}', encoding="utf-8")
    report = analyze_scene(scene)
    hints = "\n".join(report["hints"])
    assert "8/20 frame không ghép được" in hints and "đoạn quay vùng nối" in hints
    assert "quay lại theo docs/data/capture_protocol.md" in hints and "chụp thêm ảnh" not in hints
    assert report["coverage"]["status"] == "LOW"  # make_model: mọi camera cùng hướng
    warning = report["coverage"]["warnings"][0]
    assert "không frame nào nhìn tới" in warning and "quay thêm về phía đó" in warning
    assert any("quay từ cùng một chỗ" in h for h in failure_hints("no good initial image pair", "video"))
    assert all("chụp thêm ảnh" in h.split("Cách xử lý:")[-1] for h in failure_hints(None, "images"))


def test_model_without_3d_points_fails_with_a_parallax_hint(tmp_path):
    """Đứng một chỗ xoay máy (COLMAP 3.9.1 trên ảnh tổng hợp): ảnh có pose nhưng 0 điểm 3D."""
    scene = make_scene(tmp_path, 12, [12])
    for sub in (scene / "distorted" / "sparse" / "0", scene / "sparse" / "0"):
        make_model(sub, names(12), camera_model_id=4 if "distorted" in str(sub) else 1, points_per_image=0)
    (scene / "frames_report.json").write_text('{"source_type": "images"}', encoding="utf-8")
    report = analyze_scene(scene)
    assert report["registration_ratio"] == 1.0 and report["verdict"] == "FAIL"
    assert "the COLMAP model has no 3D points" in report["warnings"]
    assert any("không có điểm 3D" in hint and "di chuyển vài bước" in hint for hint in report["hints"])


def test_report_pass_has_no_hints_and_coverage_does_not_change_verdict(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 20, [20]))
    assert report["verdict"] == "PASS" and report["hints"] == []
    # make_model đặt mọi camera cùng hướng: độ phủ thấp chỉ là cảnh báo riêng, không đổi kết luận
    assert report["coverage"]["status"] == "LOW"


def test_report_without_model_gives_failure_hints(tmp_path):
    report = analyze_scene(make_scene(tmp_path, 5, []))
    assert report["verdict"] == "FAIL" and any("không ước lượng được vị trí camera" in h for h in report["hints"])


def test_runner_failure_report_and_cli_print_hints(tmp_path, fake_colmap, monkeypatch, capsys):
    exe, _ = fake_colmap
    scene = tmp_path / "scene"
    (scene / "input").mkdir(parents=True)
    for name in names(5):
        (scene / "input" / name).write_bytes(b"jpg")
    (scene / "frames_report.json").write_text('{"source_type": "images"}', encoding="utf-8")
    monkeypatch.setenv("FAKE_COLMAP_FAIL_MAPPER", "1")
    assert colmap_main(["--scene", str(scene), "--colmap", str(exe)]) == 1
    report = json.loads((scene / "sfm_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "FAIL" and report["source_type"] == "images"
    assert any("cặp ảnh" in hint for hint in report["hints"])  # log của mapper: "no good initial image pair"
    err = capsys.readouterr().err
    assert "HINT: COLMAP không tìm được cặp ảnh" in err
    # lỗi không phải của COLMAP (vd. còn kết quả cũ): không in gợi ý về chụp ảnh
    monkeypatch.delenv("FAKE_COLMAP_FAIL_MAPPER")
    (scene / "sparse").mkdir()
    assert colmap_main(["--scene", str(scene), "--colmap", str(exe)]) == 1
    assert "HINT" not in capsys.readouterr().err
