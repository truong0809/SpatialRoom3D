"""Test điều phối một lượt huấn luyện (src.training.run) với "runner" giả thay cho gsplat.

Runner giả ghi đúng các file mà simple_trainer.py và gsplat_launcher ghi (stats, ckpt, ply, split_applied).
Mọi con số là dữ liệu giả để kiểm tra logic.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from colmap_writer import make_model, write_cameras, write_points3d
from src.reconstruction.colmap_model import read_points3d_binary
from src.training.run import main, run_command

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"  # mục train_modes


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


@pytest.fixture()
def env(tmp_path):
    scene = tmp_path / "scenes" / "phong01"
    (scene / "images").mkdir(parents=True)
    for name in names(17):
        (scene / "images" / name).write_bytes(b"jpg")
    make_model(scene / "sparse" / "0", names(17))
    gsplat = tmp_path / "gsplat"
    (gsplat / "examples").mkdir(parents=True)
    (gsplat / "examples" / "simple_trainer.py").write_text("# placeholder\n", encoding="utf-8")
    return {"scene": scene, "gsplat": gsplat, "out": tmp_path / "out"}


def value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _rel_paths(folder):
    """Như `_get_rel_paths` của gsplat: mọi file trong thư mục, đệ quy."""
    return [os.path.relpath(os.path.join(root, name), folder) for root, _, files in os.walk(folder) for name in files]


def gsplat_image_pairs(cmd):
    """Như Parser của gsplat (examples/datasets/colmap.py @6e8c837) với --data-factor N > 1: ảnh JPG thu nhỏ thì
    gsplat tự thu nhỏ images/ thành images_N_png/<tên>.png (bỏ qua file đã có), rồi ghép hai danh sách theo thứ tự."""
    data_dir = Path(value_after(cmd, "--data-dir"))
    factor = int(value_after(cmd, "--data-factor")) if "--data-factor" in cmd else 1
    colmap_files = sorted(_rel_paths(data_dir / "images"))
    if factor == 1:
        return [(name, name) for name in colmap_files]
    image_dir = data_dir / f"images_{factor}"
    image_files = sorted(_rel_paths(image_dir))
    if os.path.splitext(image_files[0])[1].lower() == ".jpg":
        resized = data_dir / f"images_{factor}_png"
        os.makedirs(resized, exist_ok=True)
        for name in colmap_files:
            target = resized / (os.path.splitext(name)[0] + ".png")
            if not target.is_file():
                target.write_bytes(b"png")
        image_files = sorted(_rel_paths(resized))
    return list(zip(colmap_files, image_files))


class FakeRunner:
    """Giả lập launcher + simple_trainer: ghi kết quả vào --result-dir.

    Ghép ảnh như gsplat; cặp ảnh lệch tên (file lạ trong thư mục ảnh thu nhỏ, cache của cảnh khác) thì "lỗi" như
    gsplat thật (imageio đọc file lạ, hoặc huấn luyện trên ảnh sai).
    """

    def __init__(self, num_gs=1_500_000, fail_on=None):
        self.calls = []
        self.num_gs = num_gs
        self.fail_on = fail_on

    def __call__(self, cmd, cwd, env, log_path):
        step = "eval" if "--ckpt" in cmd else "train"
        self.calls.append({"step": step, "cmd": cmd, "cwd": cwd, "env": env})
        Path(log_path).write_text("fake log\n", encoding="utf-8")
        if step == self.fail_on:
            return 3
        pairs = gsplat_image_pairs(cmd)
        if any(os.path.splitext(a)[0] != os.path.splitext(b)[0] for a, b in pairs):
            Path(log_path).write_text(f"image pairing broken: {pairs[:3]}\n", encoding="utf-8")
            return 4
        result = Path(value_after(cmd, "--result-dir"))
        final = int(value_after(cmd, "--max-steps")) - 1
        split = json.loads((result / "split.json").read_text(encoding="utf-8"))
        record = Path(value_after(cmd, "--record-split"))
        record.write_text(json.dumps({"splits": {"train": split["train"], "val": split["test"]}}), encoding="utf-8")
        Path(value_after(cmd, "--resolved-config")).write_text("{}", encoding="utf-8")
        (result / "stats").mkdir(exist_ok=True)
        if step == "train":
            for sub in ("ckpts", "ply"):
                (result / sub).mkdir(exist_ok=True)
            (result / "ckpts" / f"ckpt_{final}_rank0.pt").write_bytes(b"0" * 1024)
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {self.num_gs}\nend_header\n"
            (result / "ply" / f"point_cloud_{final}.ply").write_bytes(header.encode("ascii"))
            stats = {"mem": 6.5, "ellipse_time": 1000.0, "num_GS": self.num_gs}
            (result / "stats" / f"train_step{final:04d}_rank0.json").write_text(json.dumps(stats))
        else:
            stats = {"psnr": 28.0, "ssim": 0.88, "lpips": 0.2, "ellipse_time": 0.005, "num_GS": self.num_gs}
            (result / "stats" / f"val_step{final:04d}.json").write_text(json.dumps(stats))
        return 0


def run(env, *extra, runner=None, mode="default", out=None, allow_mismatch=True):
    # thư mục gsplat giả không phải git checkout, nên phải cho phép lệch commit (trừ test dành riêng)
    argv = ["--config", str(CONFIG), "--mode", mode, "--scene", str(env["scene"]), "--out", str(out or env["out"]),
            "--gsplat-dir", str(env["gsplat"]), *(["--allow-gsplat-mismatch"] if allow_mismatch else []), *extra]
    return main(argv, runner=runner or FakeRunner())


def test_default_then_mcmc_with_equal_budget(env, tmp_path):
    runner = FakeRunner(num_gs=1_234_567)
    assert run(env, runner=runner, out=tmp_path / "default") == 0
    summary = json.loads((tmp_path / "default" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "OK" and summary["num_gaussians"] == 1_234_567
    assert summary["metrics"] == {"PSNR": 28.0, "SSIM": 0.88, "LPIPS": 0.2}
    assert summary["split_applied_ok"] is True and summary["seed"] == 42
    meta = json.loads((tmp_path / "default" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "DONE" and meta["split"]["num_test"] == 3 and meta["cap_max"] is None
    train = runner.calls[0]
    assert train["cmd"][1:3] == ["-m", "src.training.gsplat_launcher"]
    assert value_after(train["cmd"], "--seed") == "42" and train["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert train["cwd"] == (env["gsplat"] / "examples").resolve()
    assert "--split-file" not in train["cmd"]  # quy tắc mỗi ảnh thứ 8: dùng nguyên logic của gsplat

    runner = FakeRunner(num_gs=1_234_567)
    assert run(env, "--cap-max-from", str(tmp_path / "default"), runner=runner, mode="mcmc",
               out=tmp_path / "mcmc") == 0
    assert value_after(runner.calls[0]["cmd"], "--strategy.cap-max") == "1234567"
    meta = json.loads((tmp_path / "mcmc" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["cap_max"]["policy"] == "match_baseline" and meta["cap_max"]["value"] == 1_234_567


def test_mcmc_without_budget_source_is_refused(env):
    assert run(env, mode="mcmc") == 2
    assert not env["out"].exists()


def test_dry_run_writes_nothing(env, capsys):
    assert run(env, "--dry-run", "--protocol", "max_steps=3000") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["spec"]["deviations"]["max_steps"]["used"] == 3000
    assert "--max-steps 3000" in plan["train"] and "--ckpt" in plan["eval"]
    assert not env["out"].exists()


def test_test_list_uses_split_file_and_filtered_points(env, tmp_path):
    # điểm 1: hai ảnh train thấy; điểm 2: chỉ ảnh test thấy; điểm 3: một train + một test -> chỉ giữ điểm 1
    write_points3d(env["scene"] / "sparse" / "0" / "points3D.bin", [
        {"id": 1, "xyz": (0.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(1, 0), (2, 0)]},
        {"id": 2, "xyz": (1.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(16, 0), (17, 0)]},
        {"id": 3, "xyz": (2.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(1, 1), (17, 1)]},
    ])
    test_list = tmp_path / "test.txt"
    test_list.write_text("00016.jpg\n00017.jpg\n", encoding="utf-8")
    runner = FakeRunner()
    assert run(env, "--test-list", str(test_list), runner=runner) == 0
    out = env["out"].resolve()
    assert value_after(runner.calls[0]["cmd"], "--split-file") == str(out / "split.json")
    assert value_after(runner.calls[0]["cmd"], "--data-dir") == str(out / "data")
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["rule"] == "test_list" and split["test"] == ["00016.jpg", "00017.jpg"]
    assert [p.point_id for p in read_points3d_binary(out / "data" / "sparse" / "0" / "points3D.bin")] == [1]
    assert (out / "data" / "images" / "00016.jpg").is_file()  # ảnh test vẫn có để render khi đánh giá
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["train_view"]["points_kept"] == 1 and meta["train_view"]["points_total"] == 3


def test_failures_and_reruns(env):
    assert run(env, runner=FakeRunner(fail_on="train")) == 1
    meta = json.loads((env["out"] / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED" and meta["failed_step"] == "train" and meta["exit_code"] == 3
    assert run(env) == 2  # thư mục dở dang: phải --rerun-incomplete hoặc --force
    assert run(env, "--rerun-incomplete") == 0
    backups = [p for p in env["out"].parent.iterdir() if ".prev_" in p.name]
    assert len(backups) == 1  # bản cũ được đổi tên, không bị xóa
    runner = FakeRunner()
    assert run(env, "--rerun-incomplete", runner=runner) == 0 and runner.calls == []  # cùng cấu hình, đã OK: bỏ qua
    # lượt OK nhưng cấu hình khác (vd. chạy thử ít vòng) thì không được bỏ qua lặng lẽ
    assert run(env, "--protocol", "max_steps=3000", "--rerun-incomplete", runner=runner) == 2
    assert run(env, "--seed", "43", runner=runner) == 2 and runner.calls == []
    assert run(env, "--seed", "43", "--force", runner=runner) == 0 and len(runner.calls) == 2


def test_gsplat_commit_must_match_configuration(env):
    assert run(env, allow_mismatch=False) == 2
    assert not env["out"].exists()


def test_mcmc_budget_from_other_seed_is_refused(env, tmp_path):
    assert run(env, runner=FakeRunner(), out=tmp_path / "default") == 0
    assert run(env, "--seed", "43", "--cap-max-from", str(tmp_path / "default"), mode="mcmc",
               out=tmp_path / "mcmc") == 2


def test_run_command_throttles_progress_lines(tmp_path):
    script = tmp_path / "progress.py"
    script.write_text("import sys\n"
                      "for i in range(200):\n"
                      "    sys.stdout.write(f'\\rstep {i}')\n"
                      "sys.stdout.write('\\nStep 600: done\\r\\nend')\n", encoding="utf-8")
    log = tmp_path / "log.txt"
    assert run_command([sys.executable, str(script)], tmp_path, {}, log, progress_interval=3600) == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("$ ")
    # tiến độ đầu tiên được ghi, các cập nhật sau bị bỏ trong khoảng progress_interval;
    # dòng kết thúc bằng xuống dòng luôn được ghi (kể cả "\r\n")
    assert lines[1:] == ["step 0", "step 199", "Step 600: done", "end"]


def test_scene_checks(env, tmp_path):
    (env["scene"] / "images" / "00003.jpg").unlink()
    assert run(env) == 2
    (env["scene"] / "images" / "00003.jpg").write_bytes(b"jpg")
    write_cameras(env["scene"] / "sparse" / "0" / "cameras.bin", [(1, 1, 1920, 1080, [1500.0, 1500.0, 960.0, 540.0])])
    assert run(env) == 0
    meta = json.loads((env["out"] / "run_meta.json").read_text(encoding="utf-8"))
    assert any("wider than 1600" in warning for warning in meta["warnings"])
    assert main(["--config", str(CONFIG), "--mode", "default", "--scene", str(tmp_path / "none"),
                 "--out", str(tmp_path / "x"), "--gsplat-dir", str(env["gsplat"])]) == 2


def test_rerun_after_scene_is_rebuilt_is_not_skipped(env, tmp_path):
    runner = FakeRunner()
    assert run(env, runner=runner) == 0
    assert runner.calls[0]["env"]["MPLBACKEND"] == "Agg"
    # COLMAP chạy lại: cùng tên ảnh nhưng model khác -> kết quả cũ không còn đúng cho cảnh này
    make_model(env["scene"] / "sparse" / "0", names(17), error=0.3, points_per_image=4)
    runner = FakeRunner()
    assert run(env, "--rerun-incomplete", runner=runner) == 2 and runner.calls == []
    assert run(env, "--force", runner=runner) == 0 and len(runner.calls) == 2


def test_mcmc_budget_from_other_scene_is_refused(env, tmp_path):
    other = tmp_path / "scenes" / "phong02"
    (other / "images").mkdir(parents=True)
    for name in names(17):
        (other / "images" / name).write_bytes(b"jpg")
    make_model(other / "sparse" / "0", names(17))
    assert run(env, runner=FakeRunner(), out=tmp_path / "default") == 0
    argv = ["--config", str(CONFIG), "--mode", "mcmc", "--scene", str(other), "--out", str(tmp_path / "mcmc2"),
            "--gsplat-dir", str(env["gsplat"]), "--allow-gsplat-mismatch", "--cap-max-from", str(tmp_path / "default")]
    assert main(argv, runner=FakeRunner()) == 2
    assert not (tmp_path / "mcmc2").exists()


def test_live_run_is_not_replaced(env):
    import socket
    import subprocess

    assert run(env, runner=FakeRunner(fail_on="eval")) == 1
    meta_path = env["out"] / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # một tiến trình khác (vẫn sống) đang ghi vào thư mục này
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "src.training.run"])
    try:
        meta.update(status="RUNNING", owner={"host": socket.gethostname(), "pid": other.pid})
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert run(env, "--rerun-incomplete") == 2
        assert not [p for p in env["out"].parent.iterdir() if ".prev_" in p.name]
    finally:
        other.kill()
        other.wait()
    assert run(env, "--rerun-incomplete") == 0  # tiến trình đó đã chết: lượt dở dang được chạy lại


def test_failed_checks_keep_the_old_run_in_place(env):
    assert run(env) == 0
    before = sorted(p.name for p in env["out"].iterdir())
    assert run(env, "--seed", "43", "--force", allow_mismatch=False) == 2  # commit gsplat sai: dừng trước khi dời
    assert sorted(p.name for p in env["out"].iterdir()) == before
    assert not [p for p in env["out"].parent.iterdir() if ".prev_" in p.name]


def test_unreadable_json_counts_as_incomplete(env):
    assert run(env) == 0
    (env["out"] / "summary.json").write_text("{\"status\": \"OK\", \"metr", encoding="utf-8")
    (env["out"] / "run_meta.json").write_text("{\"scene\": ", encoding="utf-8")
    assert run(env) == 2  # không còn là lượt OK, phải chạy lại có chủ đích
    assert run(env, "--rerun-incomplete") == 0
    assert json.loads((env["out"] / "summary.json").read_text(encoding="utf-8"))["status"] == "OK"


def add_downscaled(env, factor=2, count=17):
    folder = env["scene"] / f"images_{factor}"
    folder.mkdir()
    for name in names(count):
        (folder / name).write_bytes(b"small")
    return folder


def test_data_factor_needs_the_downscaled_folder(env):
    assert run(env, "--dataset-protocol", "data_factor=2") == 2  # không có images_2/
    assert run(env, "--protocol", "data_factor=2") == 2
    add_downscaled(env, count=16)
    assert run(env, "--dataset-protocol", "data_factor=2") == 2  # images_2 thiếu một ảnh: gsplat ghép sai tên
    assert run(env, "--dataset-protocol", "max_steps=3000") == 2  # chỉ data_factor là quy ước của benchmark
    assert not env["out"].exists()


def test_dataset_protocol_uses_a_data_view_and_a_shared_png_cache(env, tmp_path):
    add_downscaled(env)
    scene_before = sorted(str(path.relative_to(env["scene"])) for path in env["scene"].rglob("*"))
    out_root = tmp_path / "modes_mipnerf360" / "phong01"
    runner = FakeRunner(num_gs=777)
    assert run(env, "--dataset-protocol", "data_factor=2", runner=runner, out=out_root / "default_s42") == 0
    default = (out_root / "default_s42").resolve()
    cmd = runner.calls[0]["cmd"]
    assert value_after(cmd, "--data-factor") == "2" and value_after(cmd, "--data-dir") == str(default / "data")
    view = default / "data"
    assert (view / "images").resolve() == (env["scene"] / "images").resolve()
    assert (view / "sparse").resolve() == (env["scene"] / "sparse").resolve()
    assert (view / "images_2").resolve() == (env["scene"] / "images_2").resolve()
    meta = json.loads((default / "run_meta.json").read_text(encoding="utf-8"))
    cache = out_root.resolve() / f".images_2_png_{meta['scene_info']['fingerprint'][:12]}"
    assert (view / "images_2_png").resolve() == cache
    assert sorted(os.listdir(cache)) == [name.replace(".jpg", ".png") for name in names(17)]  # chỉ ảnh, không dấu
    assert json.loads(cache.with_name(cache.name + ".complete").read_text())["files"] == 17
    assert [call["step"] for call in runner.calls] == ["train", "eval"]  # eval ghép ảnh đúng sau khi đánh dấu
    assert sorted(str(path.relative_to(env["scene"])) for path in env["scene"].rglob("*")) == scene_before
    assert meta["spec"]["deviations"] == {} and meta["spec"]["dataset_protocol"] == {"data_factor": 2}
    assert meta["spec"]["base_protocol"]["data_factor"] == 2 and meta["scene_info"]["data_factor"] == 2
    assert meta["factor_view"]["png_cache_reused"] is False

    runner = FakeRunner(num_gs=777)
    assert run(env, "--dataset-protocol", "data_factor=2", "--cap-max-from", str(default), mode="mcmc",
               runner=runner, out=out_root / "mcmc_s42") == 0  # cùng protocol với default: ngân sách dùng được
    meta = json.loads((out_root / "mcmc_s42" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["factor_view"]["png_cache_reused"] is True and meta["cap_max"]["value"] == 777

    runner = FakeRunner()
    assert run(env, "--dataset-protocol", "data_factor=2", runner=runner, out=out_root / "default_s42") == 0
    assert runner.calls == []  # cùng cấu hình: bỏ qua
    assert run(env, runner=FakeRunner(), out=out_root / "default_s42") == 2  # data_factor khác: không ghi đè


def png_cache(env, out):
    from src.training.run import png_cache_dir, scene_fingerprint
    return png_cache_dir(out, 2, scene_fingerprint(env["scene"]))


def test_unfinished_png_cache_is_cleared(env, tmp_path):
    add_downscaled(env)
    out = tmp_path / "scene_runs" / "default_s42"
    cache = png_cache(env, out)
    cache.mkdir(parents=True)
    (cache / "00001.png").write_bytes(b"truncated")  # lượt trước dừng giữa lúc gsplat ghi PNG (chưa có dấu)
    assert run(env, "--dataset-protocol", "data_factor=2", out=out) == 0
    assert (cache / "00001.png").read_bytes() == b"png"  # ghi lại từ đầu
    assert cache.with_name(cache.name + ".complete").is_file()


def test_png_cache_with_a_stray_file_is_rebuilt(env, tmp_path):
    add_downscaled(env)
    first = FakeRunner()
    assert run(env, "--dataset-protocol", "data_factor=2", runner=first, out=tmp_path / "r" / "default_s42") == 0
    cache = png_cache(env, tmp_path / "r" / "default_s42")
    (cache / ".DS_Store").write_bytes(b"x")  # file lạ: gsplat sẽ ghép lệch mọi ảnh
    runner = FakeRunner()
    assert run(env, "--dataset-protocol", "data_factor=2", "--seed", "43", runner=runner,
               out=tmp_path / "r" / "default_s43") == 0
    assert not (cache / ".DS_Store").exists() and len(runner.calls) == 2
    meta = json.loads((tmp_path / "r" / "default_s43" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["factor_view"]["png_cache_reused"] is False


def test_png_downscaled_images_are_read_directly_without_cache(env, tmp_path, capsys):
    folder = env["scene"] / "images_2"
    folder.mkdir()
    for name in names(17):
        (folder / name.replace(".jpg", ".png")).write_bytes(b"small")  # gsplat đọc thẳng, không thu nhỏ lại
    assert run(env, "--dataset-protocol", "data_factor=2", out=tmp_path / "r" / "default_s42") == 0
    meta = json.loads((tmp_path / "r" / "default_s42" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["factor_view"]["png_cache_used"] is False
    assert not [warning for warning in meta["warnings"] if "PNG cache" in warning]
    assert "not marked complete" not in capsys.readouterr().out


def test_two_scenes_under_one_folder_do_not_share_a_png_cache(env, tmp_path):
    add_downscaled(env)
    other = tmp_path / "scenes" / "phong02"
    (other / "images").mkdir(parents=True)
    (other / "images_2").mkdir()
    for name in [f"b{i:04d}.jpg" for i in range(1, 15)]:
        (other / "images" / name).write_bytes(b"jpg")
        (other / "images_2" / name).write_bytes(b"small")
    make_model(other / "sparse" / "0", [f"b{i:04d}.jpg" for i in range(1, 15)])
    shared = tmp_path / "m360"  # chạy tay: hai cảnh chung thư mục cha
    assert run(env, "--dataset-protocol", "data_factor=2", out=shared / "phong01_default") == 0
    runner = FakeRunner()
    assert main(["--config", str(CONFIG), "--mode", "default", "--scene", str(other), "--out",
                 str(shared / "phong02_default"), "--gsplat-dir", str(env["gsplat"]), "--allow-gsplat-mismatch",
                 "--dataset-protocol", "data_factor=2"], runner=runner) == 0
    caches = sorted(path.name for path in shared.iterdir() if path.is_dir() and path.name.startswith(".images_2_png"))
    assert len(caches) == 2 and len(runner.calls) == 2  # mỗi cảnh một cache, ghép ảnh đúng ở cả train và eval


def test_protocol_data_factor_is_a_deviation(env, tmp_path):
    add_downscaled(env)
    assert run(env, "--protocol", "data_factor=2", out=tmp_path / "dev") == 0
    meta = json.loads((tmp_path / "dev" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["spec"]["deviations"]["data_factor"]["used"] == 2 and "dataset_protocol" not in meta["spec"]
    # cùng protocol cuối nhưng một bên là sai khác, một bên là quy ước của benchmark: không coi là cùng một lượt
    runner = FakeRunner()
    assert run(env, "--dataset-protocol", "data_factor=2", runner=runner, out=tmp_path / "dev") == 2
    assert runner.calls == []
    assert run(env, "--dataset-protocol", "data_factor=2", out=tmp_path / "bench") == 0
    assert run(env, "--protocol", "data_factor=2", runner=runner, out=tmp_path / "bench") == 2
    assert runner.calls == []


def test_replace_stale_replaces_runs_made_from_older_inputs(env):
    assert run(env) == 0
    make_model(env["scene"] / "sparse" / "0", names(17), error=0.3, points_per_image=4)  # COLMAP chạy lại
    runner = FakeRunner()
    assert run(env, "--rerun-incomplete", "--replace-stale", runner=runner) == 0 and len(runner.calls) == 2
    assert len([p for p in env["out"].parent.iterdir() if ".prev_" in p.name]) == 1  # bản cũ đổi tên, không xóa
    # cấu hình do người chạy chọn (seed, protocol) khác thì vẫn phải --force
    runner = FakeRunner()
    assert run(env, "--seed", "43", "--replace-stale", runner=runner) == 2 and runner.calls == []
    assert run(env, "--protocol", "max_steps=3000", "--replace-stale", runner=runner) == 2 and runner.calls == []
    # mô hình COLMAP đổi thì cả lượt đã lỗi thời, kể cả khi cấu hình cũng khác (vd. train_modes trong configs/base.yaml đổi)
    make_model(env["scene"] / "sparse" / "0", names(17), error=0.5, points_per_image=5)
    runner = FakeRunner()
    assert run(env, "--protocol", "max_steps=3000", "--replace-stale", runner=runner) == 0 and runner.calls


def test_replace_stale_keeps_runs_the_user_configured(env, tmp_path):
    write_points3d(env["scene"] / "sparse" / "0" / "points3D.bin", [  # điểm 1: hai ảnh train cùng thấy
        {"id": 1, "xyz": (0.0, 0.0, 1.0), "rgb": (1, 2, 3), "error": 0.5, "track": [(1, 0), (2, 0)]}])
    test_list = tmp_path / "heldout.txt"
    test_list.write_text("00016.jpg\n00017.jpg\n", encoding="utf-8")
    assert run(env, "--test-list", str(test_list)) == 0
    runner = FakeRunner()
    # cùng mô hình, chỉ khác tập test: do --test-list (người chạy chọn), không phải dữ liệu cũ
    assert run(env, "--replace-stale", runner=runner) == 2 and runner.calls == []
    default, mcmc = tmp_path / "default", tmp_path / "mcmc"
    assert run(env, runner=FakeRunner(num_gs=1000), out=default) == 0
    assert run(env, "--cap-max", "3000", runner=FakeRunner(), mode="mcmc", out=mcmc) == 0  # ngân sách tự chọn
    assert run(env, "--cap-max-from", str(default), "--replace-stale", runner=runner, mode="mcmc", out=mcmc) == 2
    assert runner.calls == []


def test_replace_stale_follows_a_retrained_baseline(env, tmp_path):
    default, mcmc = tmp_path / "default", tmp_path / "mcmc"
    assert run(env, runner=FakeRunner(num_gs=1000), out=default) == 0
    assert run(env, "--cap-max-from", str(default), runner=FakeRunner(), mode="mcmc", out=mcmc) == 0
    assert run(env, "--force", runner=FakeRunner(num_gs=2000), out=default) == 0  # default train lại
    runner = FakeRunner()
    # không có --replace-stale: lượt mcmc OK cũ (ngân sách 1000) không bị thay lặng lẽ
    assert run(env, "--cap-max-from", str(default), "--rerun-incomplete", runner=runner, mode="mcmc",
               out=mcmc) == 2 and runner.calls == []
    assert run(env, "--cap-max-from", str(default), "--rerun-incomplete", "--replace-stale", runner=runner,
               mode="mcmc", out=mcmc) == 0
    assert value_after(runner.calls[0]["cmd"], "--strategy.cap-max") == "2000"
    # đã khớp: lần sau bỏ qua
    runner = FakeRunner()
    assert run(env, "--cap-max-from", str(default), "--replace-stale", runner=runner, mode="mcmc", out=mcmc) == 0
    assert runner.calls == []


def test_run_command_passes_the_gpu_lock_fd(tmp_path, monkeypatch):
    lock = open(tmp_path / "gpu.lock", "a")  # Python mở fd không kế thừa: chỉ pass_fds mới đưa được sang con
    try:
        script = tmp_path / "check_fd.py"
        script.write_text(f"import os, sys\ntry:\n    os.fstat({lock.fileno()})\nexcept OSError:\n    sys.exit(1)\n",
                          encoding="utf-8")
        log = tmp_path / "log.txt"
        monkeypatch.delenv("SR3D_GPU_LOCK_FD", raising=False)
        assert run_command([sys.executable, str(script)], tmp_path, {}, log) == 1
        monkeypatch.setenv("SR3D_GPU_LOCK_FD", str(lock.fileno()))
        assert run_command([sys.executable, str(script)], tmp_path, {}, log) == 0
        monkeypatch.setenv("SR3D_GPU_LOCK_FILE", str(tmp_path / "gpu.lock"))  # đúng file khóa: vẫn truyền
        assert run_command([sys.executable, str(script)], tmp_path, {}, log) == 0
        (tmp_path / "other.lock").write_text("")
        monkeypatch.setenv("SR3D_GPU_LOCK_FILE", str(tmp_path / "other.lock"))  # fd là file khác: không truyền
        assert run_command([sys.executable, str(script)], tmp_path, {}, log) == 1
        monkeypatch.delenv("SR3D_GPU_LOCK_FILE")
        monkeypatch.setenv("SR3D_GPU_LOCK_FD", "987654")  # fd không mở: bỏ qua, không lỗi
        assert run_command([sys.executable, str(script)], tmp_path, {}, log) == 1
    finally:
        lock.close()
