"""Test thống kê PLY cuối của một lượt (src/evaluation/splat_stats.py). PLY giả theo đúng định dạng export_splats của
gsplat @6e8c837: float32 little-endian x, y, z, f_dc_*, f_rest_*, opacity (logit), scale_* (log), rot_*."""

import json
import math
import os

import numpy as np
import pytest

from src.evaluation.gsplat_results import load_splat_stats, load_summary
from src.evaluation.splat_stats import main, ply_stats, read_header, sh_degree, update_run


def logit(p):
    return math.log(p / (1 - p))


def write_ply(path, opacities, largest_axes, *, rest=45, extra_header="", broken_type=False):
    """PLY của gsplat: opacity theo xác suất, trục lớn nhất theo đơn vị cảnh (hai trục còn lại nhỏ hơn)."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(rest)], "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = np.zeros((len(opacities), len(names)), dtype="<f4")
    rows[:, names.index("opacity")] = [logit(p) for p in opacities]
    rows[:, names.index("scale_0")] = np.log(np.asarray(largest_axes, dtype=np.float64) / 2)
    rows[:, names.index("scale_1")] = np.log(np.asarray(largest_axes, dtype=np.float64))
    rows[:, names.index("scale_2")] = np.log(np.asarray(largest_axes, dtype=np.float64) / 3)
    rows[:, names.index("rot_0")] = 1.0
    kind = "double" if broken_type else "float"
    header = (["ply", "format binary_little_endian 1.0", f"element vertex {len(opacities)}"]
              + [f"property {kind} {name}" for name in names] + ([extra_header] if extra_header else []) + ["end_header"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())
    return path


def make_run(root, name="mcmc_s42", *, scene_scale=2.0, opacities=None, largest=None, rest=45):
    run = root / "room" / name
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"spec": {"protocol": {"max_steps": 30000}}, "seed": 42}),
                                       encoding="utf-8")
    if scene_scale is not None:
        (run / "train.log").write_text(f"Loading data\nScene scale: {scene_scale}\nStep 0\n", encoding="utf-8")
    opacities = opacities or [0.001, 0.01, 0.015, 0.05, 0.5, 0.9, 0.9, 0.95, 0.99, 0.3]
    largest = largest or [0.01, 0.018, 0.05, 0.1, 0.3, 0.01, 0.01, 0.018, 0.03, 0.04]
    write_ply(run / "ply" / "point_cloud_29999.ply", opacities, largest, rest=rest)
    return run


def test_statistics_of_a_gsplat_ply(tmp_path):
    run = make_run(tmp_path)
    assert update_run(run)[0] == "OK"
    stats = json.loads((run / "splat_stats.json").read_text())
    assert stats["status"] == "OK" and stats["num_gaussians"] == 10 and stats["non_finite"] == 0
    assert stats["bytes_per_gaussian"] == 236 and stats["sh_degree"] == 3  # 59 float32: đúng PLY của gsplat, SH bậc 3
    assert stats["opacity"]["below"] == {"0.005": pytest.approx(0.1), "0.02": pytest.approx(0.3),
                                         "0.1": pytest.approx(0.4)}
    assert stats["scene_scale"] == 2.0 and stats["scene_scale_source"] == "train_log"
    # trục lớn nhất 0,3 > 0,1 × 2,0: một Gaussian trong 10; trục > 0,01 × 2,0 = 0,02: 5 Gaussian
    assert stats["largest_axis_relative"]["above"] == {"0.01": pytest.approx(0.5), "0.1": pytest.approx(0.1)}
    assert stats["largest_axis"]["p50"] == pytest.approx(0.024, rel=1e-5)  # trung vị của 0,018 và 0,03
    assert load_splat_stats(run)["num_gaussians"] == 10
    assert load_summary(run).splat_stats["bytes_per_gaussian"] == 236  # gsplat_results đọc kèm vào kết quả của lượt


def test_bytes_per_gaussian_and_sh_degree_follow_the_ply(tmp_path):
    run = make_run(tmp_path, rest=9)  # SH bậc 1
    update_run(run)
    stats = load_splat_stats(run)
    assert stats["bytes_per_gaussian"] == 92 and stats["sh_degree"] == 1
    assert sh_degree(["f_rest_0"]) is None and sh_degree([]) == 0


def test_cached_until_the_ply_changes(tmp_path, capsys):
    run = make_run(tmp_path)
    assert main(["--root", str(tmp_path)]) == 0 and "OK" in capsys.readouterr().out
    assert update_run(run)[0] == "CACHED"
    ply = run / "ply" / "point_cloud_29999.ply"
    write_ply(ply, [0.5] * 4, [0.01] * 4)  # train lại: PLY khác
    assert load_splat_stats(run) is None  # thống kê cũ không còn khớp PLY
    assert update_run(run)[0] == "OK" and load_splat_stats(run)["num_gaussians"] == 4
    os.utime(ply, ns=(ply.stat().st_atime_ns, ply.stat().st_mtime_ns + 10**9))
    assert load_splat_stats(run) is None and update_run(run)[0] == "OK"
    assert update_run(run, force=True)[0] == "OK"


def test_scene_scale_comes_from_the_filter_first(tmp_path):
    run = make_run(tmp_path, scene_scale=2.0)
    (run / "filtered").mkdir()
    (run / "filtered" / "filter_meta.json").write_text(json.dumps({"status": "OK", "scene_scale": 4.0}))
    update_run(run)
    stats = load_splat_stats(run)
    assert stats["scene_scale"] == 4.0 and stats["scene_scale_source"] == "filter_meta"
    assert stats["largest_axis_relative"]["above"]["0.1"] == 0.0  # 0,3 < 0,1 × 4,0


def test_without_scene_scale_only_absolute_sizes(tmp_path):
    run = make_run(tmp_path, scene_scale=None)
    update_run(run)
    stats = load_splat_stats(run)
    assert stats["scene_scale"] is None and stats["scene_scale_source"] is None
    assert "largest_axis_relative" not in stats and stats["largest_axis"]["p99"] > 0


def test_broken_files_are_reported_not_used(tmp_path, capsys):
    run = make_run(tmp_path)
    ply = run / "ply" / "point_cloud_29999.ply"
    ply.write_bytes(ply.read_bytes()[:-10])  # thiếu dữ liệu
    assert main(["--root", str(tmp_path)]) == 1
    assert "truncated" in capsys.readouterr().err
    assert json.loads((run / "splat_stats.json").read_text())["status"] == "FAILED" and load_splat_stats(run) is None
    write_ply(ply, [0.5], [0.1], broken_type=True)
    status, text = update_run(run)
    assert status == "FAILED" and "only float properties" in text
    write_ply(ply, [0.5], [0.1], extra_header="element face 0")
    assert update_run(run)[0] == "FAILED"
    with pytest.raises(ValueError, match="unsupported format"):
        ply.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n")
        read_header(ply)


def test_runs_without_a_final_ply_are_skipped(tmp_path, capsys):
    run = make_run(tmp_path)
    (run / "ply" / "point_cloud_29999.ply").unlink()
    old = tmp_path / "room" / "mcmc_s42.prev_20261020"  # thư mục cũ do chạy lại: không đọc
    old.mkdir()
    (old / "run_meta.json").write_text("{}")
    assert main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "SKIP" in out and "no final PLY" in out and ".prev_" not in out
    assert not (run / "splat_stats.json").exists()


def test_ply_stats_rejects_an_empty_model(tmp_path):
    ply = write_ply(tmp_path / "empty.ply", [], [])
    with pytest.raises(ValueError, match="no Gaussians"):
        ply_stats(ply)


def test_recomputed_when_the_scene_scale_appears(tmp_path):
    run = make_run(tmp_path, scene_scale=None)
    assert update_run(run)[0] == "OK" and load_splat_stats(run)["scene_scale"] is None
    assert update_run(run)[0] == "CACHED"
    (run / "filtered").mkdir()  # lượt vừa được lọc: filter_meta.json có scene_scale
    (run / "filtered" / "filter_meta.json").write_text(json.dumps({"status": "OK", "scene_scale": 2.0}))
    assert update_run(run)[0] == "OK"
    stats = load_splat_stats(run)
    assert stats["scene_scale"] == 2.0 and stats["largest_axis_relative"]["above"]["0.1"] == pytest.approx(0.1)


def test_statistics_stay_usable_without_the_ply(tmp_path):
    run = make_run(tmp_path)
    update_run(run)
    assert load_splat_stats(run)["ply_checked"] is True
    (run / "ply" / "point_cloud_29999.ply").unlink()  # gói kết quả nhẹ: có JSON, không có PLY
    stats = load_splat_stats(run)
    assert stats["ply_checked"] is False and stats["num_gaussians"] == 10
    assert update_run(run)[0] == "SKIP"  # không tính lại khi không có PLY, giữ thống kê cũ
    assert load_splat_stats(run) is not None
