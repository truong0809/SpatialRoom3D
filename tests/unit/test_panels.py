"""Test ảnh so sánh nhiều lượt (src/evaluation/panels.py) và scripts/run_panels.sh. Ảnh render giả theo đúng cách
simple_trainer.py @6e8c837 ghi khi chấm: renders/val_step<N>_<i>.png là [ảnh gốc | ảnh render]."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.evaluation import panels  # noqa: E402
from src.evaluation.panels import (auto_crop, error_map, main, make_panels, parse_crop, psnr, run_label,  # noqa: E402
                                   scene_runs)

REPO = Path(__file__).resolve().parents[2]
H, W = 48, 64


def truth_image(index):
    rng = np.random.default_rng(index)
    return rng.integers(0, 240, size=(H, W, 3), dtype=np.uint8)  # không chạm 255: sai số cộng thêm không bị cắt


def write_run(scene_dir, name, *, mode, seed=42, tag=None, noise=0, region=None, images=range(3), step=29999,
              truth=truth_image, status="OK", deviations=None):
    """Một lượt có ảnh render: sai số đều `noise`, cộng một vùng sai lớn `region` = (x, y, w, h) nếu có."""
    run = scene_dir / name
    (run / "renders").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"mode": mode, "seed": seed, "tag": tag}), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"run_dir": str(run), "mode": mode, "seed": seed, "status": status,
                                                  "deviations": deviations or {}}), encoding="utf-8")
    for index in images:
        gt = truth(index)
        render = np.clip(gt.astype(int) + noise, 0, 255).astype(np.uint8)
        if region:
            x, y, w, h = region
            render[y:y + h, x:x + w] = 255 - gt[y:y + h, x:x + w]
        cv2.imwrite(str(run / "renders" / f"val_step{step}_{index:04d}.png"), np.hstack([gt, render]))
    return run


def write_video(path, frames=5, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * W, H))
    if not writer.isOpened():
        pytest.skip("OpenCV in this environment cannot write mp4")
    for i in range(frames):
        writer.write(np.full((H, 2 * W, 3), (value + 10 * i) % 256, dtype=np.uint8))
    writer.release()


def test_panels_pick_the_hardest_region_and_record_psnr(tmp_path):
    scene = tmp_path / "room"
    good = write_run(scene, "mcmc_s42", mode="mcmc", noise=2)
    bad = write_run(scene, "default_s42", mode="default", noise=2, region=(40, 30, 16, 12))
    record = make_panels([bad, good], tmp_path / "out", labels=["default", "mcmc"])
    assert record["test_images"] == [0, 1, 2] and [p["image"] for p in record["panels"]] == [0, 1, 2]
    assert record["images_rule"].startswith("default: first, middle and last") and record["warnings"] == []
    crop = record["panels"][0]["crop"]
    assert (crop["w"], crop["h"]) == (16, 12) and 32 <= crop["x"] <= 48 and 24 <= crop["y"] <= 36  # gần vùng sai
    assert record["panels"][0]["crop_rule"].startswith("auto")
    assert record["mean_psnr_png"]["mcmc"] > record["mean_psnr_png"]["default"]
    assert record["mean_psnr_png"]["mcmc"] == pytest.approx(10 * np.log10(255.0 ** 2 / 4), abs=0.01)  # sai số 2 mức
    panel = cv2.imread(str(tmp_path / "out" / "panel_0000.png"))
    assert panel is not None and panel.shape[0] > 3 * 26  # hàng ảnh gốc + 2 lượt, mỗi hàng có dải nhãn
    text = (tmp_path / "out" / "panels.md").read_text(encoding="utf-8")
    assert "| default |" in text and "panel_0002.png" in text and "PNG 8 bit" in text
    assert json.loads((tmp_path / "out" / "panels.json").read_text())["labels"] == ["default", "mcmc"]


def test_given_crop_and_images(tmp_path):
    scene = tmp_path / "room"
    runs = [write_run(scene, "default_s42", mode="default", noise=3), write_run(scene, "mcmc_s42", mode="mcmc")]
    record = make_panels(runs, tmp_path / "out", images=[1], crop="0.5,0.5,0.25,0.25")
    assert [p["image"] for p in record["panels"]] == [1] and record["images_rule"] == "given (--images)"
    assert record["panels"][0]["crop"] == {"x": 32, "y": 24, "w": 16, "h": 12}
    assert record["panels"][0]["crop_rule"] == "given (--crop)"
    assert record["labels"] == ["default_s42", "mcmc_s42"] and record["mean_psnr_png"]["mcmc_s42"] == 100.0
    with pytest.raises(ValueError, match="outside the image"):
        parse_crop("0.9,0.1,0.5,0.5", W, H)
    with pytest.raises(ValueError, match="fractions"):
        parse_crop("a,b", W, H)
    with pytest.raises(ValueError, match="not in the runs"):
        make_panels(runs, tmp_path / "out2", images=[7])


def test_runs_of_different_test_images_are_refused(tmp_path):
    scene = tmp_path / "room"
    a = write_run(scene, "default_s42", mode="default")
    b = write_run(scene, "mcmc_s42", mode="mcmc", truth=lambda index: truth_image(index + 100))
    with pytest.raises(ValueError, match="different ground truth"):
        make_panels([a, b], tmp_path / "out")
    c = write_run(scene, "mcmc_s43", mode="mcmc", images=[0, 1])  # thiếu một ảnh test: không lặng lẽ co bộ ảnh lại
    with pytest.raises(ValueError, match="different test images"):
        make_panels([a, c], tmp_path / "out")
    d = write_run(scene, "mcmc_40k_s42", mode="mcmc_40k", step=39999)  # chế độ 40.000 vòng, chỉ rõ bằng --runs
    record = make_panels([a, d], tmp_path / "out")  # khác bước: chỉ cảnh báo, panels.md ghi bước từng lượt
    assert record["steps"] == {"default_s42": 29999, "mcmc_40k_s42": 39999}
    assert "different steps" in record["warnings"][0]
    assert "- cảnh báo: the runs were evaluated at different steps" in (tmp_path / "out" / "panels.md").read_text()
    with pytest.raises(ValueError, match="labels must be distinct"):
        make_panels([a, a], tmp_path / "out")
    empty = scene / "empty_s42"
    empty.mkdir()
    with pytest.raises(ValueError, match="no renders"):
        make_panels([a, empty], tmp_path / "out")


def test_latest_step_is_used(tmp_path):
    scene = tmp_path / "room"
    run = write_run(scene, "default_s42", mode="default", noise=50, step=6999)
    write_run_step = write_run(scene, "tmp", mode="default", noise=1, step=29999)  # ảnh của bước sau, chép sang
    for path in (write_run_step / "renders").iterdir():
        path.rename(run / "renders" / path.name)
    other = write_run(scene, "mcmc_s42", mode="mcmc", noise=1)
    record = make_panels([run, other], tmp_path / "out")
    assert record["steps"]["default_s42"] == 29999
    assert record["mean_psnr_png"]["default_s42"] == pytest.approx(record["mean_psnr_png"]["mcmc_s42"])


def test_scene_runs_selects_untagged_runs_of_one_seed(tmp_path):
    scene = tmp_path / "room"
    for name, mode, seed, tag in (("mcmc_s42", "mcmc", 42, None), ("default_s42", "default", 42, None),
                                  ("mcmc_indoor_s42", "mcmc_indoor", 42, None), ("mcmc_rbg_s42", "mcmc", 42, "rbg"),
                                  ("default_s43", "default", 43, None)):
        write_run(scene, name, mode=mode, seed=seed, tag=tag)
    (scene / "norender_s42").mkdir()
    (scene / "norender_s42" / "run_meta.json").write_text(json.dumps({"mode": "x", "seed": 42}))
    write_run(scene, "mcmc_s42.prev_20261020_101010", mode="mcmc")  # thư mục cũ do chạy lại
    write_run(scene, "mcmc_x_s42", mode="mcmc_x", status="RUNNING")  # chưa train xong
    write_run(scene, "mcmc_40k_s42", mode="mcmc_40k", step=39999,
              deviations={"max_steps": {"base": 30000, "used": 40000}})  # khác protocol chung: không vào mặc định
    assert [run.name for run in scene_runs(scene, 42)] == ["default_s42", "mcmc_s42", "mcmc_indoor_s42"]
    assert [run.name for run in scene_runs(scene, 43)] == ["default_s43"]


def test_video_rows_stop_at_the_shortest(tmp_path):
    scene = tmp_path / "room"
    a = write_run(scene, "default_s42", mode="default")
    b = write_run(scene, "mcmc_s42", mode="mcmc")
    write_video(a / "videos" / "traj_29999.mp4", frames=6)
    write_video(b / "videos" / "traj_29999.mp4", frames=4, value=100)
    write_video(b / "videos" / "traj_6999.mp4", frames=9)  # bước trước: không dùng
    record = make_panels([a, b], tmp_path / "out", video=True)
    assert record["video"]["frames"] == 4 and record["video"]["sources"][1].endswith("traj_29999.mp4")
    assert record["video"]["codec"] == ("h264" if importlib.util.find_spec("imageio_ffmpeg") else "mp4v")
    reader = cv2.VideoCapture(str(tmp_path / "out" / "traj_compare.mp4"))
    ok, frame = reader.read()
    reader.release()
    assert ok and frame.shape[0] >= 2 * (H // 2)  # hai hàng (thu nhỏ một nửa), mỗi hàng có nhãn
    os.remove(b / "videos" / "traj_29999.mp4")
    os.remove(b / "videos" / "traj_6999.mp4")
    record = make_panels([a, b], tmp_path / "out2", video=True)  # thiếu video: chỉ cảnh báo, ảnh so sánh vẫn có
    assert "no videos" in record["video"]["error"] and len(record["panels"]) == 3
    saved = json.loads((tmp_path / "out2" / "panels.json").read_text())
    assert "no videos" in saved["video"]["error"] and "không ghép được" in (tmp_path / "out2" / "panels.md").read_text()


def test_a_video_that_cannot_be_written_is_only_a_warning(tmp_path, capsys):
    scene = tmp_path / "room"
    runs = [write_run(scene, f"m{i}_s42", mode=f"m{i}") for i in range(2)]
    for run in runs:
        write_video(run / "videos" / "traj_29999.mp4", frames=3)
    out = tmp_path / "out"
    (out / "traj_compare.mp4").mkdir(parents=True)  # không ghi được file ra (cả hai bộ mã hóa)
    assert main(["--runs", *map(str, runs), "--out", str(out), "--video"]) == 0
    assert "WARN: video skipped: " in capsys.readouterr().err
    record = json.loads((out / "panels.json").read_text())
    assert "cannot write" in record["video"]["error"] and len(record["panels"]) == 3


def test_stacked_video_is_kept_under_the_height_limit(tmp_path, monkeypatch):
    scene = tmp_path / "room"
    runs = [write_run(scene, f"m{i}_s42", mode=f"m{i}") for i in range(3)]
    for run in runs:
        write_video(run / "videos" / "traj_29999.mp4", frames=2)
    monkeypatch.setattr(panels, "MAX_VIDEO_HEIGHT", 120)
    record = make_panels(runs, tmp_path / "out", video=True)
    reader = cv2.VideoCapture(str(tmp_path / "out" / record["video"]["file"]))
    ok, frame = reader.read()
    reader.release()
    assert ok and frame.shape[0] <= 120 and record["video"]["frames"] == 2


def test_filtered_models_have_their_own_label(tmp_path):
    assert run_label(tmp_path / "room" / "mcmc_s42" / "filtered") == "mcmc_s42+filtered"
    assert run_label(tmp_path / "room" / "mcmc_s42") == "mcmc_s42"


def test_helpers():
    gt = np.zeros((4, 4, 3), np.uint8)
    assert psnr(gt, gt) == 100.0 and error_map(gt + 51, gt).max() == pytest.approx(0.2)
    errors = [np.zeros((40, 40)), np.zeros((40, 40))]
    errors[1][30:40, 0:10] = 1.0
    assert auto_crop(errors) == (0, 30, 10, 10)


def test_cli(tmp_path, capsys):
    scene = tmp_path / "room"
    write_run(scene, "default_s42", mode="default", noise=4)
    write_run(scene, "mcmc_s42", mode="mcmc", noise=2)
    assert main(["--scene-dir", str(scene), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "3 panel(s)" in out and "default_s42" in out
    assert main(["--scene-dir", str(scene), "--seed", "7", "--out", str(tmp_path / "o2")]) == 1
    assert "no finished run of seed 7" in capsys.readouterr().err
    assert main(["--runs", str(scene / "default_s42"), str(scene / "default_s42"), "--out", str(tmp_path / "o3")]) == 1


def test_run_panels_script(tmp_path):
    """scripts/run_panels.sh trên /workspace giả: thư mục kết quả theo configs/, python của env gs-gsplat."""
    ws = tmp_path / "ws"
    scene = ws / "outputs" / "benchmarks" / "mipnerf360" / "gsplat" / "counter"
    write_run(scene, "default_s42", mode="default", noise=3)
    write_run(scene, "mcmc_s42", mode="mcmc", noise=1)
    env_python = ws / "miniforge3" / "envs" / "gs-gsplat" / "bin" / "python"  # python của env: python đang chạy test
    env_python.parent.mkdir(parents=True)
    env_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    env_python.chmod(0o755)
    env = {**os.environ, "WS": str(ws), "SR3D_SYSTEM_PYTHON": sys.executable, "IMAGES": "0 2"}
    script = str(REPO / "scripts" / "run_panels.sh")
    out = subprocess.run(["bash", script, "mipnerf360", "counter"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    folders = sorted((ws / "outputs" / "panels").glob("mipnerf360_counter_*"))  # ngoài reports/: không vào gói nhẹ
    assert len(folders) == 1 and sorted(p.name for p in folders[0].glob("panel_*.png")) == ["panel_0000.png",
                                                                                            "panel_0002.png"]
    assert not (ws / "outputs" / "reports").exists()
    filtered = write_run(scene / "mcmc_s42", "filtered", mode="mcmc", noise=2)  # bản đã lọc của một lượt
    (filtered / "filter_meta.json").write_text("{}")
    out = subprocess.run(["bash", script, "mipnerf360", "counter", "mcmc_s42", "mcmc_s42/filtered"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "mcmc_s42+filtered" in out.stdout
    out = subprocess.run(["bash", script, "mipnerf360", "counter", "default_s42/filtered"], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 1 and "no filtered model" in out.stdout
    quick = ws / "outputs" / "benchmarks" / "mipnerf360" / "gsplat_quick" / "counter"
    write_run(quick, "default_s42", mode="default")
    out = subprocess.run(["bash", script, "mipnerf360", "counter"], capture_output=True, text=True,
                         env={**env, "QUICK": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert sorted((ws / "outputs" / "panels").glob("mipnerf360_quick_counter_*"))
    out = subprocess.run(["bash", script, "mipnerf360", "counter", "mcmc_s42", "nope_s42"], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 1 and "no run nope_s42" in out.stdout
    out = subprocess.run(["bash", script, "mipnerf360", "kitchen"], capture_output=True, text=True, env=env)
    assert out.returncode == 1 and "no results for kitchen" in out.stdout
    out = subprocess.run(["bash", script, "mipnerf360", "counter"], capture_output=True, text=True,
                         env={**env, "IMAGES": "a"})
    assert out.returncode == 1 and "IMAGES must be" in out.stdout
    assert subprocess.run(["bash", script, "mipnerf360"], capture_output=True, text=True, env=env).returncode == 2
