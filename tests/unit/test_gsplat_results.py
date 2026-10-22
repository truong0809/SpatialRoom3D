"""Test đọc kết quả một lượt chạy gsplat. Mọi con số là dữ liệu giả để kiểm tra logic."""

import json

import pytest

from src.evaluation.gsplat_results import latest_stats, load_filtered, load_run, load_summary

PROTOCOL = {"data_factor": 1, "test_every": 8, "max_steps": 30000, "lpips_net": "vgg",
            "antialiased": False, "save_ply": True, "disable_video": False}


def make_run(root, *, val_steps=(6999, 29999), status="DONE", applied_ok=True, vertices=2_000_000):
    run = root / "run"
    (run / "stats").mkdir(parents=True)
    (run / "ply").mkdir()
    (run / "ckpts").mkdir()
    meta = {"scene": "playroom", "mode": "default", "seed": 42, "status": status, "train_seconds": 1500.0,
            "eval_seconds": 60.0, "spec": {"protocol": PROTOCOL, "deviations": {}},
            "split": {"sha256": "f" * 64}, "gsplat_commit": "6e8c837", "gpu": "NVIDIA RTX A6000",
            "cap_max": None}
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for step in val_steps:
        stats = {"psnr": 25.0 + step / 10000, "ssim": 0.8, "lpips": 0.3, "ellipse_time": 0.004, "num_GS": vertices}
        (run / "stats" / f"val_step{step:04d}.json").write_text(json.dumps(stats), encoding="utf-8")
    (run / "stats" / "train_step29999_rank0.json").write_text(
        json.dumps({"mem": 7.25, "ellipse_time": 1400.0, "num_GS": vertices}), encoding="utf-8")
    (run / "gpu_log.csv").write_text("2026/10/09 10:00:00.000, 9000, 99\n2026/10/09 10:00:02.000, 11000, 99\n",
                                     encoding="utf-8")
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {vertices}\nend_header\n".encode("ascii")
    (run / "ply" / "point_cloud_29999.ply").write_bytes(header + b"\x00" * (2**20))
    (run / "ckpts" / "ckpt_29999_rank0.pt").write_bytes(b"\x00" * (2**20))
    names = [f"{i:05d}.jpg" for i in range(1, 10)]
    split = {"train": names[1:8], "test": [names[0], names[8]]}
    (run / "split.json").write_text(json.dumps(split), encoding="utf-8")
    applied_val = split["test"] if applied_ok else names[:2]
    (run / "split_applied.json").write_text(
        json.dumps({"splits": {"train": split["train"], "val": split["test"]}}), encoding="utf-8")
    (run / "split_applied_eval.json").write_text(
        json.dumps({"splits": {"train": split["train"], "val": applied_val}}), encoding="utf-8")
    return run


def test_latest_stats_picks_highest_step(tmp_path):
    run = make_run(tmp_path)
    step, stats = latest_stats(run / "stats", "val")
    assert step == 29999 and stats["num_GS"] == 2_000_000
    assert latest_stats(run / "stats", "train")[0] == 29999
    assert latest_stats(tmp_path / "missing", "val") is None


def test_load_run_collects_everything(tmp_path):
    result = load_run(make_run(tmp_path))
    assert result.status == "OK" and result.eval_step == 29999
    assert result.metrics == pytest.approx({"PSNR": 27.9999, "SSIM": 0.8, "LPIPS": 0.3})
    assert result.num_gaussians == 2_000_000
    assert result.render_ms_per_image == 4.0
    assert result.torch_peak_gib == 7.25 and result.peak_vram_mib == 11000
    assert result.ply_size_mb == 1.0 and result.ckpt_size_mb == 1.0
    assert result.split_applied_ok is True and result.lpips_net == "vgg"
    scene = result.to_scene_result()
    assert scene.status == "OK" and scene.metrics["PSNR"] == pytest.approx(27.9999)
    assert scene.scene == "playroom"


def test_status_rules(tmp_path):
    assert load_run(make_run(tmp_path / "a", val_steps=(6999,))).status == "MISSING"  # chưa có bước cuối
    assert load_run(make_run(tmp_path / "b", status="FAILED")).status == "FAILED"
    assert load_run(make_run(tmp_path / "c", applied_ok=False)).split_applied_ok is False


def test_load_summary_prefers_summary_file(tmp_path):
    run = make_run(tmp_path)
    (run / "summary.json").write_text(json.dumps({"run_dir": str(run), "status": "OK", "num_gaussians": 5,
                                                  "unknown_key": 1}), encoding="utf-8")
    assert load_summary(run).num_gaussians == 5
    (run / "summary.json").unlink()
    assert load_summary(run).num_gaussians == 2_000_000


def test_load_summary_reports_where_the_run_is_now(tmp_path):
    """Sau khi dời thư mục kết quả, báo cáo in chỗ mới, không phải đường dẫn cũ ghi trong summary.json."""
    run = make_run(tmp_path)
    (run / "summary.json").write_text(json.dumps({"run_dir": "/workspace/outputs/modes_db/playroom/default_s42",
                                                  "status": "OK"}), encoding="utf-8")
    assert load_summary(run).run_dir == str(run.resolve())


def test_non_finite_metrics_are_not_ok(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "stats").mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"status": "DONE", "spec": {"protocol": {"max_steps": 10}}}),
                                           encoding="utf-8")
    (run_dir / "stats" / "val_step0009.json").write_text('{"psnr": NaN, "ssim": 0.5, "lpips": 0.3}', encoding="utf-8")
    result = load_run(run_dir)
    assert result.status == "FAILED" and "non-finite" in result.problem


def write_filtered(run, *, status="OK", num_gs=1_200_000, psnr=24.9, kept=1_200_000):
    folder = run / "filtered"
    (folder / "stats").mkdir(parents=True)
    meta = {"status": status, "thresholds": {"min_opacity": 0.02, "max_scale": 0.1}, "scene_scale": 3.3,
            "counts": {"input": 2_000_000, "removed": {"non_finite": 0, "opacity": 800_000, "max_scale": 0},
                       "kept": kept, "kept_fraction": 0.6}}
    (folder / "filter_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    stats = {"psnr": psnr, "ssim": 0.79, "lpips": 0.31, "ellipse_time": 0.003, "num_GS": num_gs}
    (folder / "stats" / "val_step29999.json").write_text(json.dumps(stats), encoding="utf-8")
    return folder


def test_filtered_copies_are_read_from_their_own_folder(tmp_path):
    run = make_run(tmp_path)
    assert load_run(run).filtered is None  # chưa lọc
    write_filtered(run)
    result = load_run(run)
    assert result.status == "OK" and result.metrics["PSNR"] == pytest.approx(27.9999)  # bảng chính: chưa lọc
    assert result.filtered["status"] == "OK" and result.filtered["metrics"]["PSNR"] == 24.9
    assert result.filtered["num_gaussians"] == 1_200_000 and result.filtered["kept_fraction"] == 0.6
    assert result.filtered["eval_step"] == 29999 and result.filtered["ply_size_mb"] is None  # chưa có PLY
    (run / "filtered" / "stats" / "val_step29999.json").write_text(
        json.dumps({"psnr": 24.9, "ssim": 0.79, "lpips": 0.31, "num_GS": 7}), encoding="utf-8")
    assert load_filtered(run)["status"] == "FAILED" and "saw 7 Gaussians" in load_filtered(run)["problem"]
    (run / "filtered" / "stats" / "val_step29999.json").write_text(
        json.dumps({"psnr": float("nan"), "ssim": 0.79, "lpips": 0.31, "num_GS": 1_200_000}), encoding="utf-8")
    assert load_filtered(run)["status"] == "FAILED"
    (run / "filtered" / "stats" / "val_step29999.json").unlink()
    assert load_filtered(run)["problem"] == "no evaluation stats in filtered/stats"
    (run / "filtered" / "filter_meta.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    assert load_filtered(run)["status"] == "RUNNING" and "metrics" not in load_filtered(run)
