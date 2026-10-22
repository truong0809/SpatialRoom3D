"""Test lọc Gaussian sau khi train (src/training/prune.py, scripts/run_filter.sh).

gsplat, GPU và PyTorch được thay bằng đồ giả: checkpoint là pickle của các mảng numpy (FakeIO), Parser giả cho scene_scale,
runner giả ghi đúng các file mà lệnh đánh giá `--ckpt` của simple_trainer.py ghi. Test riêng với PyTorch thật (TorchIO)
tự bỏ qua khi môi trường không có torch.
"""

import json
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from shell_env import ShellEnv, make_fake_repo
from src.evaluation.gsplat_results import load_filtered, load_summary
from src.training.prune import (TorchIO, filtered_eval_command, keep_mask, logged_scene_scale, main,
                                same_test_images)
from src.training.run import scene_fingerprint

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

FINAL = 29999
TEST = ["00008.jpg", "00016.jpg"]


def splats(count=10, seed=0):
    """Checkpoint gsplat giả: scales là log của trục, opacities là logit."""
    rng = np.random.default_rng(seed)
    return {"means": rng.normal(size=(count, 3)).astype(np.float32),
            "scales": np.full((count, 3), np.log(0.01), dtype=np.float32),
            "quats": rng.normal(size=(count, 4)).astype(np.float32),
            "opacities": np.full(count, 2.0, dtype=np.float32),  # sigmoid(2) = 0,88
            "sh0": rng.normal(size=(count, 1, 3)).astype(np.float32),
            "shN": np.zeros((count, 15, 3), dtype=np.float32)}


class FakeIO:
    """Như TorchIO nhưng checkpoint là pickle của mảng numpy."""

    def __init__(self):
        self.exported = []

    def load(self, path):
        return pickle.loads(Path(path).read_bytes())

    def arrays(self, ckpt):
        return ckpt["splats"]

    def subset(self, ckpt, keep):
        return {**ckpt, "splats": {key: value[keep] for key, value in ckpt["splats"].items()}}

    def save(self, ckpt, path):
        Path(path).write_bytes(pickle.dumps(ckpt))

    def export_ply(self, ckpt, path):
        count = len(ckpt["splats"]["means"])
        Path(path).write_bytes(f"ply\nelement vertex {count}\nend_header\n".encode() + b"\0" * count * 236)
        self.exported.append((path, count))


class FakeParser:
    seen = []

    def __init__(self, data_dir, factor=1, normalize=False, test_every=8):
        FakeParser.seen.append({"data_dir": data_dir, "factor": factor, "normalize": normalize, "test_every": test_every})
        self.scene_scale = 2.0


class FakeRunner:
    """Lệnh đánh giá `--ckpt` của gsplat: đọc checkpoint đã lọc, ghi stats / renders / video / split vào --result-dir."""

    def __init__(self, code=0, num_gs=None, val=None):
        self.calls, self.code, self.num_gs, self.val = [], code, num_gs, val

    def __call__(self, cmd, cwd, env, log_path):
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        Path(log_path).write_text("fake eval\n", encoding="utf-8")
        if self.code:
            return self.code
        out = Path(cmd[cmd.index("--result-dir") + 1])
        ckpt = pickle.loads(Path(cmd[cmd.index("--ckpt") + 1]).read_bytes())
        count = len(ckpt["splats"]["means"]) if self.num_gs is None else self.num_gs
        for sub in ("stats", "renders", "videos"):
            (out / sub).mkdir(exist_ok=True)
        (out / "stats" / f"val_step{ckpt['step']:04d}.json").write_text(
            json.dumps({"psnr": 29.5, "ssim": 0.9, "lpips": 0.21, "ellipse_time": 0.004, "num_GS": count}))
        (out / "renders" / f"val_step{ckpt['step']}_0000.png").write_bytes(b"png")
        (out / "videos" / f"traj_{ckpt['step']}.mp4").write_bytes(b"mp4")
        record = Path(cmd[cmd.index("--record-split") + 1])
        record.write_text(json.dumps({"splits": {"train": ["00001.jpg"], "val": self.val or TEST}}), encoding="utf-8")
        return 0


def make_run(root, *, status="DONE", data=None, scene_scale_log=2.2, name="mcmc_s42"):
    """Một lượt gsplat đã train xong như src.training.run ghi ra (rút gọn), checkpoint dạng FakeIO."""
    scene = root / "colmap"
    if not (scene / "sparse").exists():
        (scene / "sparse" / "0").mkdir(parents=True)
        (scene / "images").mkdir()
        for model in ("cameras", "images", "points3D"):
            (scene / "sparse" / "0" / f"{model}.bin").write_bytes(model.encode())
    examples = root / "gsplat" / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    run = root / "out" / "room" / name
    (run / "ckpts").mkdir(parents=True)
    (run / "ply").mkdir()
    (run / "ckpts" / f"ckpt_{FINAL}_rank0.pt").write_bytes(pickle.dumps({"step": FINAL, "splats": data or splats()}))
    (run / "ply" / f"point_cloud_{FINAL}.ply").write_bytes(b"ply-original")
    launcher = ["/env/bin/python", "-m", "src.training.gsplat_launcher", "--examples-dir", str(examples), "--seed", "42",
                "--record-split", str(run / "split_applied_eval.json"),
                "--resolved-config", str(run / "resolved_config_eval.json")]
    trainer = ["mcmc", "--data-dir", str(scene), "--result-dir", str(run), "--data-factor", "1", "--test-every", "8",
               "--max-steps", "30000", "--no-disable-video", "--disable-viewer", "--strategy.cap-max", "10",
               "--ckpt", str(run / "ckpts" / f"ckpt_{FINAL}_rank0.pt")]
    meta = {"scene": "room", "mode": "mcmc", "seed": 42, "status": status, "scene_dir": str(scene),
            "data_dir": str(scene), "scene_info": {"fingerprint": scene_fingerprint(scene)},
            "spec": {"protocol": {"max_steps": 30000, "test_every": 8}},
            "commands": {"train": launcher + ["--", *trainer[:-2]], "eval": launcher + ["--", *trainer]}}
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "split.json").write_text(json.dumps({"train": ["00001.jpg"], "test": TEST}), encoding="utf-8")
    (run / "resolved_config.json").write_text(json.dumps({"normalize_world_space": True, "global_scale": 1.0}))
    if scene_scale_log is not None:
        (run / "train.log").write_text(f"Model initialized.\nScene scale: {scene_scale_log}\n", encoding="utf-8")
    summary = {"run_dir": str(run), "scene": "room", "mode": "mcmc", "seed": 42, "status": "OK",
               "metrics": {"PSNR": 30.0, "SSIM": 0.91, "LPIPS": 0.2}, "num_gaussians": 10, "ply_size_mb": 1.5}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def prune(*argv, runner=None, io=None):
    return main([*map(str, argv)], runner=runner or FakeRunner(), parser_class=FakeParser, io=io or FakeIO())


# ---------------------------------------------------------------------------------------------- lọc
def test_keep_mask_counts_each_gaussian_under_its_first_criterion():
    data = splats(8)
    data["means"][0, 1] = np.nan                      # 0: không hữu hạn
    data["opacities"][1] = -5.0                       # 1: opacity 0,0067
    data["scales"][2, 0] = np.log(0.5)                # 2: trục 0,5 > 0,1 × 2
    data["opacities"][3] = -5.0                       # 3: cả hai, tính cho opacity
    data["scales"][3, 2] = np.log(0.5)
    data["shN"][4, 3, 1] = np.inf                     # 4: không hữu hạn
    keep, removed = keep_mask(data, scene_scale=2.0, min_opacity=0.02, max_scale=0.1)
    assert removed == {"non_finite": 2, "opacity": 2, "max_scale": 1}
    assert keep.tolist() == [False, False, False, False, False, True, True, True]
    keep, removed = keep_mask(data, scene_scale=2.0, min_opacity=0.0, max_scale=0.1)  # 0 = tắt tiêu chí
    assert removed == {"non_finite": 2, "opacity": 0, "max_scale": 2} and keep.sum() == 4
    keep, removed = keep_mask(data, scene_scale=10.0, min_opacity=0.02, max_scale=0.1)  # 0,5 < 0,1 × 10
    assert removed["max_scale"] == 0 and keep.sum() == 4


def test_the_filtered_model_is_saved_beside_the_run_and_evaluated_with_the_recorded_command(tmp_path, capsys):
    data = splats(10)
    data["opacities"][:3] = -6.0
    data["scales"][9] = np.log(1.0)
    run = make_run(tmp_path, data=data)
    original = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    io, runner = FakeIO(), FakeRunner()
    FakeParser.seen.clear()
    assert prune(run, runner=runner, io=io) == 0
    filtered = run / "filtered"
    meta = json.loads((filtered / "filter_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "OK" and meta["thresholds"] == {"min_opacity": 0.02, "max_scale": 0.1}
    assert meta["scene_scale"] == pytest.approx(2.2) and meta["scene_scale_logged"] == 2.2  # 2,0 × 1,1 × 1,0
    assert meta["counts"] == {"input": 10, "removed": {"non_finite": 0, "opacity": 3, "max_scale": 1}, "kept": 6,
                              "kept_fraction": 0.6}
    assert meta["before"]["metrics"]["PSNR"] == 30.0 and meta["after"]["metrics"]["PSNR"] == 29.5
    assert meta["same_test_images"] is True and meta["exit_code"] == 0
    assert FakeParser.seen == [{"data_dir": str(tmp_path / "colmap"), "factor": 1, "normalize": True, "test_every": 8}]
    saved = pickle.loads((filtered / "ckpts" / f"ckpt_{FINAL}_rank0.pt").read_bytes())
    assert saved["step"] == FINAL and len(saved["splats"]["means"]) == 6
    assert np.array_equal(saved["splats"]["means"], data["means"][3:9])  # tham số giữ nguyên
    assert io.exported == [(filtered / "ply" / f"point_cloud_{FINAL}.ply", 6)]
    # lệnh đánh giá: đúng lệnh đã ghi, chỉ đổi thư mục ghi, checkpoint và file ghi lại của launcher
    cmd = runner.calls[0]["cmd"]
    cut = cmd.index("--")
    assert cmd[cmd.index("--result-dir") + 1] == str(filtered)
    assert cmd[cmd.index("--ckpt") + 1] == str(filtered / "ckpts" / f"ckpt_{FINAL}_rank0.pt")
    assert cmd[cmd.index("--record-split") + 1] == str(filtered / "split_applied_eval.json") and cmd.index("--seed") < cut
    assert cmd[cut + 1:cut + 4] == ["mcmc", "--data-dir", str(tmp_path / "colmap")]
    assert cmd.count("--result-dir") == 1 and cmd.count("--ckpt") == 1 and "--strategy.cap-max" in cmd
    assert runner.calls[0]["cwd"] == tmp_path / "gsplat" / "examples"
    assert runner.calls[0]["env"]["MPLBACKEND"] == "Agg"
    assert {path: path.read_bytes() for path in original} == original  # lượt gốc không đổi
    assert not [p for p in run.rglob("*") if p.is_file() and p not in original and filtered not in p.parents]
    # báo cáo đọc lại bản đã lọc từ stats của gsplat
    info = load_filtered(run)
    assert info["status"] == "OK" and info["metrics"]["LPIPS"] == 0.21 and info["num_gaussians"] == 6
    assert info["render_ms_per_image"] == 4.0 and info["ply_size_mb"] == pytest.approx(6 * 236 / 2**20, abs=0.01)
    assert load_summary(run).filtered["status"] == "OK"  # summary.json cũ, bản đã lọc đọc mới
    assert "Gaussians 10 -> 6" in capsys.readouterr().out

    # chạy lại cùng ngưỡng: bỏ qua; ngưỡng khác: thư mục cũ đổi tên, lọc lại
    runner = FakeRunner()
    assert prune(run, runner=runner) == 0 and not runner.calls
    assert prune(run, "--min-opacity", "0.5", runner=runner) == 0 and len(runner.calls) == 1
    assert len(list(run.glob("filtered.prev_*"))) == 1
    assert json.loads((filtered / "filter_meta.json").read_text())["thresholds"]["min_opacity"] == 0.5
    assert prune(run, "--min-opacity", "0.5", "--force", runner=runner) == 0 and len(runner.calls) == 2


def test_unfinished_runs_are_skipped_only_on_request(tmp_path, capsys):
    run = make_run(tmp_path, status="RUNNING")
    assert prune(run) == 1 and "not a finished run" in capsys.readouterr().err
    assert prune(run, "--skip-unfinished") == 0 and not (run / "filtered").exists()
    assert prune(tmp_path / "nope", "--skip-unfinished") == 0  # thư mục không có lượt nào


def test_a_failed_evaluation_is_recorded_and_reported(tmp_path, capsys):
    run = make_run(tmp_path)
    assert prune(run, runner=FakeRunner(code=3)) == 1
    meta = json.loads((run / "filtered" / "filter_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED" and meta["failed_step"] == "eval" and meta["exit_code"] == 3
    assert load_filtered(run)["status"] == "FAILED" and "exit code 3" in load_filtered(run)["problem"]
    # lần sau làm lại từ đầu: bản dở dang được đổi tên, không bị coi là đã lọc
    assert prune(run) == 0 and len(list(run.glob("filtered.prev_*"))) == 1
    capsys.readouterr()


@pytest.mark.parametrize("runner, message", [
    (FakeRunner(num_gs=99), "the evaluation saw 99 Gaussians"),
    (FakeRunner(val=["00001.jpg"]), "other images than the run's test set"),
])
def test_the_evaluation_must_see_the_filtered_model_on_the_test_images(tmp_path, capsys, runner, message):
    run = make_run(tmp_path)
    assert prune(run, runner=runner) == 1 and message in capsys.readouterr().err
    assert json.loads((run / "filtered" / "filter_meta.json").read_text())["failed_step"] == "collect"


def test_checks_before_touching_anything(tmp_path, capsys):
    run = make_run(tmp_path, scene_scale_log=5.0)  # scene_scale tính lại khác số train.log ghi
    assert prune(run) == 1 and "differs from the one training printed" in capsys.readouterr().err
    assert not (run / "filtered").exists()
    (run / "train.log").unlink()  # không có log: tin Parser
    assert prune(run, "--dry-run") == 0 and not (run / "filtered").exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["filtered"] == str(run / "filtered") and "--ckpt" in plan["eval"]
    (tmp_path / "colmap" / "sparse" / "0" / "images.bin").write_bytes(b"rebuilt")  # COLMAP làm lại sau khi train
    assert prune(run) == 1 and "changed after this run was trained" in capsys.readouterr().err
    for bad in (["--min-opacity", "1"], ["--max-scale", "-1"], ["--min-opacity", "0", "--max-scale", "0"]):
        assert prune(run, *bad) == 2


def test_eval_command_needs_the_recorded_flags(tmp_path):
    run = make_run(tmp_path)
    meta = json.loads((run / "run_meta.json").read_text())
    cmd = filtered_eval_command(meta, run / "filtered", run / "filtered" / "x.pt", "/py")
    assert cmd[:3] == ["/py", "-m", "src.training.gsplat_launcher"]
    meta["commands"]["eval"] = [token for token in meta["commands"]["eval"] if token != "--ckpt"]
    with pytest.raises(ValueError, match="lacks"):
        filtered_eval_command(meta, run / "filtered", run / "filtered" / "x.pt", "/py")
    assert logged_scene_scale(run) == 2.2 and logged_scene_scale(tmp_path) is None
    assert same_test_images(run, tmp_path) is None


class BrokenParser:
    def __init__(self, data_dir, **_):
        if "broken" in str(data_dir):
            raise ValueError("cannot read images.bin")  # Parser của gsplat không đọc được cảnh
        self.scene_scale = 2.0


def test_an_error_in_one_run_does_not_stop_the_next_ones(tmp_path, capsys):
    broken = make_run(tmp_path / "broken")
    good = make_run(tmp_path / "good")
    code = main([str(broken), str(good)], runner=FakeRunner(), parser_class=BrokenParser, io=FakeIO())
    assert code == 1
    assert "cannot compute gsplat's scene_scale for this run: ValueError: cannot read images.bin" in capsys.readouterr().err
    assert not (broken / "filtered").exists()
    assert json.loads((good / "filtered" / "filter_meta.json").read_text())["status"] == "OK"


def test_the_test_images_of_the_evaluation_must_be_checked(tmp_path, capsys):
    run = make_run(tmp_path)
    (run / "split.json").unlink()  # không còn tập test đã ghi của lượt: không kiểm được
    assert prune(run) == 1 and "cannot check that the filtered model was evaluated" in capsys.readouterr().err
    assert load_filtered(run)["status"] == "FAILED"


def test_a_filtered_copy_with_lost_files_is_filtered_again(tmp_path, capsys):
    run = make_run(tmp_path)
    assert prune(run) == 0
    for stats in (run / "filtered" / "stats").glob("*.json"):
        stats.unlink()  # chỉ số của lượt chấm lại mất: báo cáo sẽ ghi FAILED, không được coi là đã lọc
    runner = FakeRunner()
    assert prune(run, runner=runner) == 0 and len(runner.calls) == 1
    assert "is incomplete (no evaluation stats in filtered/stats); filtering again" in capsys.readouterr().out
    assert load_filtered(run)["status"] == "OK" and len(list(run.glob("filtered.prev_*"))) == 1
    (run / "filtered" / "ply" / f"point_cloud_{FINAL}.ply").unlink()
    assert prune(run, runner=runner) == 0 and len(runner.calls) == 2  # PLY đã lọc mất: lọc lại


class NanRunner(FakeRunner):
    """Lượt chấm lại ra chỉ số NaN (gsplat ghi NaN vào stats)."""

    def __call__(self, cmd, cwd, env, log_path):
        code = super().__call__(cmd, cwd, env, log_path)
        for stats in (Path(cmd[cmd.index("--result-dir") + 1]) / "stats").glob("val_step*.json"):
            data = json.loads(stats.read_text())
            stats.write_text(json.dumps({**data, "psnr": float("nan")}))
        return code


def test_non_finite_metrics_after_filtering_fail_the_run(tmp_path, capsys):
    run = make_run(tmp_path)
    assert prune(run, runner=NanRunner()) == 1 and "non-finite metrics after filtering" in capsys.readouterr().err
    meta = json.loads((run / "filtered" / "filter_meta.json").read_text())
    assert meta["status"] == "FAILED" and meta["failed_step"] == "collect"
    assert load_filtered(run)["status"] == "FAILED"


def test_keeps_every_gaussian_or_refuses_to_keep_none(tmp_path, capsys):
    data = splats(4)
    data["opacities"][:] = -9.0
    run = make_run(tmp_path, data=data)
    assert prune(run) == 1 and "remove every Gaussian" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------ PyTorch thật
def test_torch_io_keeps_the_gsplat_checkpoint_layout(tmp_path):
    torch = pytest.importorskip("torch")
    data = splats(5)
    ckpt = {"step": FINAL, "splats": OrderedDict((key, torch.from_numpy(value)) for key, value in data.items())}
    torch.save(ckpt, tmp_path / "ckpt.pt")
    exported = {}

    def exporter(**kwargs):
        exported.update(kwargs)

    io = TorchIO(exporter=exporter)
    loaded = io.load(tmp_path / "ckpt.pt")
    keep = np.array([True, False, True, True, False])
    subset = io.subset(loaded, keep)
    io.save(subset, tmp_path / "filtered.pt")
    again = torch.load(tmp_path / "filtered.pt", map_location="cpu", weights_only=True)  # như lệnh --ckpt của gsplat
    assert again["step"] == FINAL and list(again["splats"]) == list(data)
    assert again["splats"]["sh0"].shape == (3, 1, 3) and again["splats"]["shN"].shape == (3, 15, 3)
    assert torch.equal(again["splats"]["means"], torch.from_numpy(data["means"][keep]))
    assert set(io.arrays(again)) == set(data) and io.arrays(again)["opacities"].shape == (3,)
    io.export_ply(again, tmp_path / "out.ply")
    assert exported["format"] == "ply" and exported["save_to"] == str(tmp_path / "out.ply")
    assert exported["opacities"].shape == (3,) and exported["quats"].shape == (3, 4)


# ------------------------------------------------------------------------------------------ run_filter.sh
FAKE_PYTHON = r'''#!/usr/bin/env bash
echo "python $*" >> "$CALLS"
exit 0
'''


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["run_filter.sh"], with_src=True, with_configs=True)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    env.env["PATH"] = f"{bin_dir}:{env.env['PATH']}"
    conda = env.ws / "miniforge3" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
    yield env
    env.cleanup()


def test_run_filter_script_filters_the_finished_runs_of_a_scene(sh, tmp_path):
    scene = sh.ws / "outputs" / "custom_scenes" / "phong1"
    for name, done in (("default_s42", True), ("mcmc_s42", True), ("mcmc_s42.prev_20261020_101010", True),
                       ("mcmc_or0001_s42", False)):
        (scene / name).mkdir(parents=True)
        (scene / name / "run_meta.json").write_text("{}")
        if done:
            (scene / name / "summary.json").write_text("{}")
    (scene / "colmap").mkdir()
    script = str(sh.repo / "scripts" / "run_filter.sh")
    out = sh.run(script, "custom", "phong1")
    assert out.returncode == 0, out.stdout + out.stderr
    call = (tmp_path / "calls.txt").read_text().strip()
    assert call == (f"python -m src.training.prune --min-opacity 0.02 --max-scale 0.1 --skip-unfinished "
                    f"{scene / 'default_s42'} {scene / 'mcmc_s42'}")
    assert sorted((sh.ws / "outputs" / "logs").glob("filter_phong1_*.log"))
    (tmp_path / "calls.txt").unlink()
    out = sh.run(script, "custom", "phong1", "mcmc_or0001_s42", env={"MIN_OPACITY": "0.05", "MAX_SCALE": "0", "FORCE": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "calls.txt").read_text().strip() == (
        f"python -m src.training.prune --min-opacity 0.05 --max-scale 0 --force {scene / 'mcmc_or0001_s42'}")
    missing = sh.run(script, "custom", "phong1", "nope_s42")
    assert missing.returncode == 1 and "no run at" in missing.stdout
    assert sh.run(script, "custom").returncode == 2
    assert sh.run(script, "custom", "../x").returncode == 1
    assert sh.run(script, "mipnerf360", "room").returncode == 1  # chưa có kết quả nào của cảnh
