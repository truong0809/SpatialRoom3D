"""Test src.rendering.render và scripts/run_render.sh: render lại từ checkpoint của một lượt đã train (không train,
không ghi đè lượt gốc) và xuất PLY. gsplat và GPU được thay bằng runner giả / stub; mọi file là dữ liệu giả.
"""

import json
import os
from pathlib import Path

import pytest

from src.rendering.render import TRAJECTORIES, main, render_command
from src.training.run import scene_fingerprint
from shell_env import ShellEnv, make_fake_repo

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")


def make_run(root, status="DONE", with_ply=True, split_file=False, scene_name="phong1"):
    """Một lượt gsplat đã train xong như src.training.run ghi ra (rút gọn)."""
    scene = root / "colmap"
    (scene / "sparse" / "0").mkdir(parents=True)
    for name in ("cameras", "images", "points3D"):
        (scene / "sparse" / "0" / f"{name}.bin").write_bytes(name.encode())
    examples = root / "gsplat" / "examples"
    examples.mkdir(parents=True)
    run = root / "out" / scene_name / "default_s42"
    (run / "ckpts").mkdir(parents=True)
    (run / "ckpts" / "ckpt_29999_rank0.pt").write_bytes(b"ckpt")
    if with_ply:
        (run / "ply").mkdir()
        (run / "ply" / "point_cloud_6999.ply").write_bytes(b"early")
        (run / "ply" / "point_cloud_29999.ply").write_bytes(b"ply-final")
    launcher = ["/env/bin/python", "-m", "src.training.gsplat_launcher", "--examples-dir", str(examples),
                "--seed", "7", "--record-split", str(run / "split_applied_eval.json"),
                "--resolved-config", str(run / "resolved_config_eval.json")]
    if split_file:
        launcher += ["--split-file", str(run / "split.json")]
    trainer = ["default", "--data-dir", str(scene), "--result-dir", str(run), "--data-factor", "1",
               "--test-every", "8", "--max-steps", "30000", "--no-disable-video", "--disable-viewer",
               "--ckpt", str(run / "ckpts" / "ckpt_29999_rank0.pt")]
    meta = {"scene": scene_name, "mode": "default", "seed": 7, "status": status, "finished": "2026-10-19T08:30:05+00:00",
            "scene_dir": str(scene), "scene_info": {"fingerprint": scene_fingerprint(scene)},
            "commands": {"train": launcher + ["--", *trainer[:-2]], "eval": launcher + ["--", *trainer]}}
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run, meta


def test_render_command_reuses_the_recorded_evaluation_and_writes_elsewhere(tmp_path):
    run, meta = make_run(tmp_path, split_file=True)
    target = run / "render_ellipse_x"
    cmd = render_command(meta, target, "ellipse", "/py")
    cut = cmd.index("--")
    launcher, trainer = cmd[:cut], cmd[cut + 1:]
    assert launcher[:3] == ["/py", "-m", "src.training.gsplat_launcher"]
    assert launcher[launcher.index("--seed") + 1] == "7"
    assert launcher[launcher.index("--split-file") + 1] == str(run / "split.json")  # cùng tập test
    assert launcher[launcher.index("--record-split") + 1].startswith(str(target))  # chỉ ghi vào thư mục render mới
    assert trainer.count("--result-dir") == 1 and trainer[trainer.index("--result-dir") + 1] == str(target)
    assert trainer[trainer.index("--render-traj-path") + 1] == "ellipse" and "--no-disable-video" in trainer
    assert trainer[trainer.index("--ckpt") + 1] == str(run / "ckpts" / "ckpt_29999_rank0.pt")
    assert trainer[:3] == ["default", "--data-dir", str(tmp_path / "colmap")]
    with pytest.raises(ValueError, match="unknown trajectory"):
        render_command(meta, target, "orbit", "/py")
    assert set(TRAJECTORIES) == {"interp", "ellipse", "spiral"}


def test_dry_run_writes_nothing(tmp_path, capsys):
    run, _ = make_run(tmp_path)
    before = sorted(run.rglob("*"))
    assert main(["--run", str(run), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert "-m src.training.gsplat_launcher" in plan["render"] and plan["render_dir"].startswith(str(run.resolve()))
    assert sorted(run.rglob("*")) == before


def test_render_runs_into_a_new_folder_and_exports_the_final_ply(tmp_path, capsys):
    run, _ = make_run(tmp_path)
    seen = {}

    def fake_runner(cmd, cwd, env, log_path):  # thay gsplat: ghi đúng các file mà chế độ --ckpt ghi
        out = Path(cmd[cmd.index("--result-dir") + 1])
        (out / "renders").mkdir(parents=True)
        (out / "renders" / "val_step29999_0000.png").write_bytes(b"png")
        (out / "stats").mkdir()
        (out / "stats" / "val_step29999.json").write_text('{"psnr": 30.1, "ssim": 0.9, "lpips": 0.2}')
        (out / "videos").mkdir()
        (out / "videos" / "traj_29999.mp4").write_bytes(b"mp4")
        log_path.write_text("ok")
        seen.update(cwd=cwd, env=env)
        return 0

    export = tmp_path / "checkpoints" / "custom_scene"
    assert main(["--run", str(run), "--traj", "spiral", "--export", str(export)], runner=fake_runner) == 0
    renders = sorted(run.glob("render_spiral_*"))
    assert len(renders) == 1
    meta = json.loads((renders[0] / "render_meta.json").read_text())
    assert meta["exit_code"] == 0 and meta["images"] == 1 and meta["metrics"]["psnr"] == 30.1
    assert meta["videos"][0].endswith("traj_29999.mp4") and meta["source_run"] == str(run)
    assert seen["cwd"] == tmp_path / "gsplat" / "examples" and seen["env"]["MPLBACKEND"] == "Agg"
    target = export / "phong1" / "gsplat_default_phong1_s42.ply"  # tên bản (gsplat_default), cảnh, seed
    assert target.read_bytes() == b"ply-final"  # PLY của bước cuối
    assert target.stat().st_ino == (run / "ply" / "point_cloud_29999.ply").stat().st_ino  # liên kết cứng
    assert "(hardlink)" in capsys.readouterr().out
    assert (run / "run_meta.json").is_file() and not (run / "renders").exists()  # lượt gốc không đổi
    # xuất lại: không ghi đè, báo đã có
    assert main(["--run", str(run), "--no-render", "--export", str(export)]) == 0
    assert "(exists)" in capsys.readouterr().out


def test_refuses_unfinished_runs_and_missing_files(tmp_path, capsys):
    run, _ = make_run(tmp_path / "a", status="RUNNING")
    assert main(["--run", str(run), "--dry-run"]) == 2
    assert "not a finished run" in capsys.readouterr().err
    run, _ = make_run(tmp_path / "b")
    (run / "ckpts" / "ckpt_29999_rank0.pt").unlink()
    assert main(["--run", str(run), "--dry-run"]) == 2
    assert "checkpoint not found" in capsys.readouterr().err
    run, _ = make_run(tmp_path / "c", with_ply=False)
    assert main(["--run", str(run), "--no-render", "--export", str(tmp_path / "exp")]) == 2
    assert "no PLY" in capsys.readouterr().err
    run, _ = make_run(tmp_path / "d")
    assert main(["--run", str(run), "--no-render"]) == 2  # không có gì để làm
    meta = json.loads((run / "run_meta.json").read_text())
    del meta["scene_info"]  # không biết mô hình COLMAP lúc train: không kiểm tra được camera, không render
    (run / "run_meta.json").write_text(json.dumps(meta))
    assert main(["--run", str(run), "--dry-run"]) == 2
    assert "does not record the COLMAP model" in capsys.readouterr().err


def test_export_never_overwrites_another_ply(tmp_path, capsys):
    """Tên PLY xuất ra đã là PLY của lượt khác (vd. lượt cũ trước khi train lại): xuất thêm tên có thời điểm train xong."""
    run, _ = make_run(tmp_path)
    export = tmp_path / "exp"
    other = export / "phong1" / "gsplat_default_phong1_s42.ply"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"older run")
    assert main(["--run", str(run), "--no-render", "--export", str(export)]) == 0
    assert other.read_bytes() == b"older run"  # không ghi đè
    second = export / "phong1" / "gsplat_default_phong1_s42_20261019_083005.ply"
    assert second.read_bytes() == b"ply-final" and f"[export] {second}" in capsys.readouterr().out
    assert main(["--run", str(run), "--no-render", "--export", str(export)]) == 0  # xuất lại: đã có, không thêm file
    assert "(exists)" in capsys.readouterr().out and len(list((export / "phong1").iterdir())) == 2


def test_export_names_follow_the_version_scene_and_seed():
    from src.rendering.render import export_name
    assert export_name(Path("/x/drjohnson/mcmc_s42"), "drjohnson") == "gsplat_mcmc_drjohnson_s42"  # notebook gsplat_mcmc
    assert export_name(Path("/x/room/mcmc_or0001_s43"), "room") == "gsplat_mcmc_or0001_room_s43"  # biến thể có tag
    assert export_name(Path("/x/room/custom"), "room") == "gsplat_custom_room"  # thư mục không theo mẫu <lượt>_s<seed>
    quick = Path("/ws/outputs/benchmarks/deep_blending/gsplat_quick/playroom/default_s42")
    assert export_name(quick, "playroom") == "gsplat_default_playroom_s42_quick"  # lượt chạy thử 3.000 vòng
    previous = Path("/ws/outputs/custom_scenes/phong1/default_s42.prev_20261028_101500")
    assert export_name(previous, "phong1") == "gsplat_default_phong1_s42.prev_20261028_101500"  # lượt cũ đã đổi tên


def test_render_refuses_when_the_colmap_model_changed(tmp_path, capsys):
    """Làm lại COLMAP sau khi train: checkpoint không còn khớp camera, không render (ảnh / chỉ số sẽ sai)."""
    run, _ = make_run(tmp_path)
    (tmp_path / "colmap" / "sparse" / "0" / "images.bin").write_bytes(b"new model")
    called = []
    assert main(["--run", str(run), "--export", str(tmp_path / "exp")], runner=lambda *a: called.append(a)) == 2
    assert "changed after this run was trained" in capsys.readouterr().err
    assert not called and not list(run.glob("render_*")) and not (tmp_path / "exp").exists()
    # xuất PLY không phụ thuộc camera: vẫn được
    assert main(["--run", str(run), "--no-render", "--export", str(tmp_path / "exp")]) == 0


def test_render_without_ply_stops_before_rendering(tmp_path, capsys):
    run, _ = make_run(tmp_path, with_ply=False)
    called = []
    assert main(["--run", str(run), "--export", str(tmp_path / "exp")], runner=lambda *a: called.append(a)) == 2
    assert "no PLY" in capsys.readouterr().err and not called and not list(run.glob("render_*"))


def test_failed_render_does_not_export(tmp_path, capsys):
    run, _ = make_run(tmp_path)
    export = tmp_path / "exp"
    assert main(["--run", str(run), "--export", str(export)], runner=lambda *a: 3) == 1
    assert "rendering failed with exit code 3" in capsys.readouterr().err and not export.exists()


FAKE_PYTHON = r'''#!/usr/bin/env bash
echo "python $*" >> "$CALLS"
exit 0
'''


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["run_render.sh"], with_src=True, with_configs=True)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    yield env
    env.cleanup()


def test_run_render_script_finds_the_run_and_exports_without_gpu(sh, tmp_path):
    out_root = sh.ws / "outputs" / "custom_scenes"
    run, _ = make_run(tmp_path / "src")
    (out_root / "phong1").mkdir(parents=True)
    run.rename(out_root / "phong1" / "default_s42")
    script = str(sh.repo / "scripts" / "run_render.sh")
    out = sh.run(script, "custom", "phong1", env={"RENDER": "0"})
    assert out.returncode == 0, out.stdout + out.stderr
    exported = sh.ws / "outputs" / "checkpoints" / "custom_scene" / "phong1" / "gsplat_default_phong1_s42.ply"
    assert exported.read_bytes() == b"ply-final"
    missing = sh.run(script, "custom", "phong1", "mcmc_s42", env={"RENDER": "0"})
    assert missing.returncode == 1 and "no run at" in missing.stdout
    assert sh.run(script, "custom").returncode == 2
    assert sh.run(script, "custom", "../x").returncode == 1  # tên không hợp lệ


def test_quick_runs_export_to_their_own_folder(sh, tmp_path):
    """Lượt chạy thử (gsplat_quick) và lượt chuẩn cùng tên: PLY xuất vào hai thư mục khác nhau, không lẫn."""
    quick = sh.ws / "outputs" / "benchmarks" / "deep_blending" / "gsplat_quick" / "playroom"
    run, _ = make_run(tmp_path / "src", scene_name="playroom")
    quick.mkdir(parents=True)
    run.rename(quick / "default_s42")
    out = sh.run(str(sh.repo / "scripts" / "run_render.sh"), "deep_blending", "playroom",
                 env={"RENDER": "0", "QUICK": "1"})
    assert out.returncode == 0, out.stdout + out.stderr
    assert (sh.ws / "outputs" / "checkpoints" / "deep_blending_quick" / "playroom" /
            "gsplat_default_playroom_s42_quick.ply").is_file()  # tải về vẫn phân biệt được với lượt chuẩn
    assert not (sh.ws / "outputs" / "checkpoints" / "deep_blending").exists()


def test_run_render_script_renders_in_the_gsplat_env(sh, tmp_path):
    out_root = sh.ws / "outputs" / "benchmarks" / "deep_blending" / "gsplat"
    run, _ = make_run(tmp_path / "src")
    (out_root / "playroom").mkdir(parents=True)
    run.rename(out_root / "playroom" / "mcmc_s42")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    conda = sh.ws / "miniforge3" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
    out = sh.run(str(sh.repo / "scripts" / "run_render.sh"), "deep_blending", "playroom", "mcmc_s42",
                 env={"PATH": f"{bin_dir}:{sh.env['PATH']}", "TRAJ": "ellipse"})
    assert out.returncode == 0, out.stdout + out.stderr
    call = (tmp_path / "calls.txt").read_text().strip()
    assert call == (f"python -m src.rendering.render --run {out_root / 'playroom' / 'mcmc_s42'} --traj ellipse "
                    f"--export {sh.ws / 'outputs' / 'checkpoints' / 'deep_blending'}")
    assert sorted((sh.ws / "outputs" / "logs").glob("render_playroom_mcmc_s42_*.log"))
