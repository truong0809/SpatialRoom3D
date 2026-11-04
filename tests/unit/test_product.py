"""Test luồng sản phẩm (src/product/): cấu hình mô hình chính, lệnh train, train với trainer giả, tiến độ (progress.json).

Không cần GPU, PyTorch hay gsplat: simple_trainer.py được thay bằng một script giả ghi đúng các file gsplat ghi (PLY,
checkpoint, stats) và in thanh tiến độ như tqdm. Phần xuất cho web: test_product_export.py; script build_room.sh:
test_build_room.py. Đối chiếu với code thật của gsplat @6e8c837: docs/design/product.md mục 7.
"""

import io
import json
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from product_fakes import FAKE_TRAINER, make_gsplat_checkout, write_fake_gpu_modules
from shell_env import REPO
from src.product import config as product_config
from src.product import progress
from src.product import train_main

PRODUCT = REPO / "configs" / "product.yaml"
BASE = REPO / "configs" / "base.yaml"



# ---------------------------------------------------------------------------------------------- tiện ích
def product_yaml() -> dict:
    return yaml.safe_load(PRODUCT.read_text(encoding="utf-8"))


def write_config(path: Path, changes: dict | None = None, data: dict | None = None) -> Path:
    data = data if data is not None else product_yaml()
    for dotted, value in (changes or {}).items():
        holder = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            holder = holder[part]
        if value is DELETE:
            del holder[leaf]
        else:
            holder[leaf] = value
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


DELETE = object()


def make_scene(root: Path, registered: int = 12) -> Path:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(b"cameras")
    (sparse / "images.bin").write_bytes(struct.pack("<Q", registered) + b"images")
    (sparse / "points3D.bin").write_bytes(b"points")
    (root / "images").mkdir()
    (root / "images" / "00001.jpg").write_bytes(b"jpg")
    return root


@pytest.fixture
def product(tmp_path, monkeypatch):
    """Cấu hình sản phẩm trỏ tới một gsplat giả (simple_trainer.py giả, 40 vòng), một phòng đã chuẩn bị."""
    examples = tmp_path / "gsplat" / "examples"
    (examples / "datasets").mkdir(parents=True)  # namespace package như examples/datasets thật
    (examples / "simple_trainer.py").write_text(FAKE_TRAINER, encoding="utf-8")
    config = write_config(tmp_path / "product.yaml", {"trainer.dir": str(tmp_path / "gsplat"),
                                                      "training.max_steps": 40})
    monkeypatch.setattr(train_main, "check_trainer",
                        lambda cfg: {"dir": cfg["trainer"]["dir"], "commit": cfg["trainer"]["commit"], "installed": "x"})
    gpu_checks = []
    monkeypatch.setattr(train_main, "check_gpu", lambda cfg, env: gpu_checks.append(env["CUDA_VISIBLE_DEVICES"]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("FAKE_TRAINER_RECORD", str(tmp_path / "trainer_call.json"))
    room = tmp_path / "ws" / "outputs" / "product" / "phong1"
    scene = make_scene(room / "colmap")

    class Env:
        pass

    env = Env()
    env.root, env.config, env.examples, env.room, env.scene = tmp_path, config, examples, room, scene
    env.out, env.progress, env.record = room / "model", room / "progress.json", tmp_path / "trainer_call.json"
    env.gpu_checks = gpu_checks
    env.args = ["--scene", str(scene), "--out", str(env.out), "--config", str(config), "--ws", str(tmp_path / "ws")]
    return env


def flag_map(tokens: list[str]) -> dict:
    """['--a', '1', '--b', '--no-c'] -> {'a': '1', 'b': True, 'c': False}."""
    result, index = {}, 0
    while index < len(tokens):
        flag = tokens[index].lstrip("-")
        values = []
        index += 1
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        if values:
            result[flag] = " ".join(values)
        else:
            prefix, _, leaf = flag.rpartition(".")
            if leaf.startswith("no-"):
                result[(prefix + "." if prefix else "") + leaf[3:]] = False
            else:
                result[flag] = True
    return result


# ------------------------------------------------------------------------------------------------ cấu hình
def test_the_main_model_is_the_mcmc_mode_of_the_main_experiment():
    """Mô hình chính = chế độ mcmc đã đo trong thực nghiệm chính: cùng preset, tham số, điều kiện train và commit của
    gsplat. Chỉ khác có chủ ý: số Gaussian tối đa cố định và mọi ảnh dùng để train (docs/design/product.md mục 2)."""
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))["train_modes"]
    cfg = product_config.load(PRODUCT, ws="/ws")
    mcmc, protocol = base["modes"]["mcmc"], base["protocol"]
    assert cfg["model"]["name"] == "mcmc" and cfg["model"]["preset"] == mcmc["preset"] == "mcmc"
    assert cfg["model"]["args"] == mcmc["args"] == {}
    assert cfg["trainer"]["commit"] == base["trainer"]["commit"]
    assert cfg["trainer"]["seed"] == base["default_seed"] == 42
    for key in ("data_factor", "max_steps", "antialiased", "init_type", "lpips_net"):
        assert cfg["training"][key] == protocol[key], key
    assert set(protocol) - set(cfg["training"]) == {"save_ply", "disable_video"}  # công tắc đầu ra, không đổi mô hình
    assert protocol["test_every"] == 8 and cfg["training"]["test_every"] >= 10**9
    assert mcmc["cap_max"]["policy"] == "match_baseline" and cfg["model"]["cap_max"] == 1_000_000


def test_the_train_command_matches_the_experiment_mcmc_command_except_the_two_choices():
    """So từng tham số với lệnh của thực nghiệm (src.training.profiles) cùng cap_max: chỉ test_every khác."""
    from src.training.profiles import load_config, resolve_mode, trainer_args

    cfg = product_config.load(PRODUCT, ws="/ws")
    ours = train_main.trainer_command(cfg, Path("/r/colmap"), Path("/r/model"), python="py")
    assert ours[:3] == ["py", "/ws/code/gsplat/examples/simple_trainer.py", "mcmc"]
    spec = resolve_mode(load_config(BASE), "mcmc")
    theirs = trainer_args(spec, Path("/r/colmap"), Path("/r/model"), cfg["model"]["cap_max"])
    assert theirs[0] == "mcmc"
    ours_flags, theirs_flags = flag_map(ours[3:]), flag_map(theirs[1:])
    theirs_flags.setdefault("disable-video", False)
    ours_flags.setdefault("disable-video", False)  # mặc định của simple_trainer.py: không ảnh hưởng mô hình
    assert ours_flags.pop("test-every") == "1000000000" and theirs_flags.pop("test-every") == "8"
    assert ours_flags == theirs_flags


def test_the_train_command_lists_every_setting():
    cfg = product_config.load(PRODUCT, ws="/ws")
    assert train_main.trainer_command(cfg, Path("/r/colmap"), Path("/r/model"), python="py") == [
        "py", "/ws/code/gsplat/examples/simple_trainer.py", "mcmc", "--data-dir", "/r/colmap", "--result-dir",
        "/r/model", "--data-factor", "1", "--test-every", "1000000000", "--max-steps", "30000", "--no-antialiased",
        "--init-type", "sfm", "--lpips-net", "vgg", "--strategy.cap-max", "1000000", "--eval-steps", "-1",
        "--save-steps", "30000", "--ply-steps", "30000", "--save-ply", "--disable-viewer"]
    cfg["model"]["args"] = {"opacity_reg": 0.001, "random_bkgd": True, "strategy.noise_lr": 100000, "packed": False}
    assert train_main.trainer_command(cfg, Path("/s"), Path("/o"), python="py")[-7:] == [
        "--disable-viewer", "--opacity-reg", "0.001", "--random-bkgd", "--strategy.noise-lr", "100000", "--no-packed"]


def test_paths_use_the_ws_folder(monkeypatch):
    monkeypatch.setenv("WS", "/data/ws")
    cfg = product_config.load(PRODUCT)
    assert cfg["output_root"] == "/data/ws/outputs/product" and cfg["web"]["dir"] == "/data/ws/outputs/product/web"
    assert cfg["trainer"]["dir"] == "/data/ws/code/gsplat"
    assert product_config.load(PRODUCT, ws="/x")["output_root"] == "/x/outputs/product"
    out = subprocess.run([sys.executable, "-m", "src.product.config", "--ws", "/p", "--get", "web.dir"], cwd=REPO,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "/p/outputs/product/web"
    out = subprocess.run([sys.executable, "-m", "src.product.config", "--get", "web.nope"], cwd=REPO,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 1 and "no key web.nope" in out.stderr


@pytest.mark.parametrize("changes, message", [
    ({"model.cap_max": 0}, "cap_max must be a positive"),
    ({"model.cap_max": True}, "model.cap_max must be int"),
    ({"model.preset": "fancy"}, "model.preset must be one of"),
    ({"training.antialiased": "no"}, "training.antialiased must be bool"),
    ({"training.test_every": 1}, "test_every must be >= 2"),
    ({"training.max_steps": 0}, "max_steps must be >= 1"),
    ({"trainer.commit": "6e8c837"}, "40-character"),
    ({"web.format": "ply"}, "web.format must be one of"),
    ({"output_root": DELETE}, "missing key output_root"),
    ({"model.args": {"max_steps": 100}}, "--max-steps is set by src/product/train_main.py"),
    ({"model.args": {"strategy.cap_max": 5}}, "--strategy.cap-max is set by"),
    ({"model.args": {"bad key!": 1}}, "invalid trainer argument name"),
    ({"model.args": {"opacity_reg": None}}, "must be a number"),
])
def test_a_wrong_config_is_rejected_with_the_key(tmp_path, changes, message):
    path = write_config(tmp_path / "product.yaml", changes)
    with pytest.raises(product_config.ConfigError, match=re.escape(message)):
        product_config.load(path)


# ---------------------------------------------------------------------------------------------- train
def test_dry_run_prints_the_command_and_writes_nothing(product, capsys):
    assert train_main.main([*product.args, "--dry-run"]) == 0
    printed = capsys.readouterr().out.splitlines()
    cmd = shlex.split(printed[0])
    assert cmd[1:3] == [str(product.examples / "simple_trainer.py"), "mcmc"]
    assert cmd[cmd.index("--max-steps") + 1] == "40" and cmd[cmd.index("--result-dir") + 1] == str(product.out)
    assert printed[1] == f"(cwd: {product.examples})"
    assert not product.out.exists() and not product.record.exists()


def test_training_runs_the_trainer_and_records_the_model(product, capsys):
    progress.start(product.progress, "phong1", "video")
    progress.set_stage(product.progress, "train")
    assert train_main.main([*product.args, "--progress", str(product.progress)]) == 0
    meta = json.loads((product.out / "train_meta.json").read_text(encoding="utf-8"))
    cfg = product_config.load(product.config, ws=str(product.root / "ws"))
    assert meta["status"] == "OK" and meta["num_gaussians"] == 7 and meta["error"] is None
    assert meta["registered_images"] == 12 and meta["train_images"] == 11
    assert meta["gsplat_stats"] == {"mem": 1.5, "ellipse_time": 12.0, "num_GS": 7}
    assert meta["config"] == train_main.training_key(cfg) and meta["scene_fingerprint"] == \
        train_main.scene_fingerprint(product.scene)
    assert meta["files"] == {"ply": "ply/point_cloud_39.ply", "ckpt": "ckpts/ckpt_39_rank0.pt"}
    assert meta["command"] == train_main.trainer_command(cfg, product.scene.resolve(), product.out.resolve())
    call = json.loads(product.record.read_text(encoding="utf-8"))
    assert call["argv"][1:] == meta["command"][2:] and Path(call["cwd"]) == product.examples
    assert call["cuda"] == "0" and call["unbuffered"] == "1" and call["mpl"] == "Agg"
    train = progress.read(product.progress)["stages"][2]
    assert train["id"] == "train" and train["percent"] == 100.0 and train["message"] == "Train xong: 0,00 triệu Gaussian"
    log = capsys.readouterr().out
    bars = [line for line in log.splitlines() if "/40 [" in line]
    assert 2 <= len(bars) <= 21 and bars[-1].endswith("40/40 [00:00<00:00, 999it/s]")  # mỗi 5 % một dòng
    assert "Now having 1234 GSs" in log and "528M/528M" in log and "[train_main] OK: 7 Gaussians" in log


def test_a_matching_model_is_not_trained_again(product, capsys):
    assert train_main.main(product.args) == 0
    assert product.gpu_checks == ["0"]
    product.record.unlink()
    assert train_main.main(product.args) == 0
    assert not product.record.exists() and "not training again" in capsys.readouterr().out
    assert product.gpu_checks == ["0"]  # không train thì không cần GPU (vd. chỉ xuất lại file web)
    assert train_main.main([*product.args, "--force"]) == 0  # --force: train lại, kết quả cũ giữ lại
    assert product.record.exists()
    kept = sorted(product.room.glob("model.prev_*"))
    assert len(kept) == 1 and (kept[0] / "train_meta.json").is_file()


def test_a_new_colmap_model_or_a_new_config_trains_again(product):
    assert train_main.main(product.args) == 0
    (product.scene / "sparse" / "0" / "points3D.bin").write_bytes(b"points of a new COLMAP run")
    product.record.unlink()
    assert train_main.main(product.args) == 0 and product.record.exists()
    write_config(product.config, {"trainer.dir": str(product.root / "gsplat"), "training.max_steps": 20})
    product.record.unlink()
    assert train_main.main(product.args) == 0 and product.record.exists()
    meta = json.loads((product.out / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["training"]["max_steps"] == 20 and meta["files"]["ply"] == "ply/point_cloud_19.ply"
    assert len(list(product.room.glob("model.prev_*"))) == 2


def test_a_failed_training_is_recorded(product, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_TRAINER_EXIT", "3")
    assert train_main.main(product.args) == 1
    meta = json.loads((product.out / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED" and meta["error"] == "the trainer exited with code 3"
    assert "RuntimeError: CUDA out of memory (fake)" in meta["output_tail"]
    assert "ERROR: training failed" in capsys.readouterr().out
    monkeypatch.setenv("FAKE_TRAINER_EXIT", "0")
    monkeypatch.setenv("FAKE_TRAINER_NO_PLY", "1")
    assert train_main.main(product.args) == 1  # thoát 0 nhưng không có PLY
    meta = json.loads((product.out / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED" and "without writing" in meta["error"]


def test_problems_found_before_training_change_nothing(product, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert train_main.main(product.args) == 1
    assert "several GPUs" in capsys.readouterr().out and not product.record.exists() and not product.out.exists()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    (product.scene / "sparse" / "0" / "images.bin").unlink()
    assert train_main.main(product.args) == 1
    assert "no COLMAP model (images.bin missing" in capsys.readouterr().out and not product.out.exists()


def test_the_trainer_check_wants_the_pinned_commit_and_the_same_installed_package(tmp_path, monkeypatch):
    import product_fakes

    assert product_fakes.GUARDED_FILES == train_main.GUARDED_FILES
    checkout, installed, head = make_gsplat_checkout(tmp_path)
    cfg = product_config.load(write_config(tmp_path / "p.yaml", {"trainer.dir": str(checkout), "trainer.commit": head}))
    monkeypatch.delitem(sys.modules, "gsplat", raising=False)
    monkeypatch.syspath_prepend(str(installed.parent))
    info = train_main.check_trainer(cfg)
    assert info["commit"] == head and Path(info["installed"]) == installed.resolve()
    with pytest.raises(train_main.TrainError, match="expects " + "0" * 40):
        train_main.check_trainer({**cfg, "trainer": {**cfg["trainer"], "commit": "0" * 40}})
    (installed / "strategy" / "mcmc.py").write_text("# another version\n", encoding="utf-8")
    with pytest.raises(train_main.TrainError, match=r"differs from .* \['strategy/mcmc.py'\]"):
        train_main.check_trainer(cfg)
    (checkout / "examples" / "simple_trainer.py").unlink()
    with pytest.raises(train_main.TrainError, match="simple_trainer.py not found"):
        train_main.check_trainer(cfg)


def test_a_datasets_package_that_hides_the_gsplat_examples_is_refused(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    (examples / "datasets").mkdir(parents=True)
    train_main.check_examples_imports(examples)
    other = tmp_path / "site"
    (other / "datasets").mkdir(parents=True)
    (other / "datasets" / "__init__.py").write_text("", encoding="utf-8")  # gói thường, như datasets của Hugging Face
    monkeypatch.syspath_prepend(str(other))
    with pytest.raises(train_main.TrainError, match="pip uninstall datasets"):
        train_main.check_examples_imports(examples)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name != "posix", reason="needs POSIX signals")
def test_a_stop_signal_stops_the_trainer_and_is_recorded(tmp_path):
    """Chạy thật như build_room.sh gọi (tiến trình riêng, kiểm tra trainer thật với kho git giả), rồi SIGTERM."""
    checkout, installed, head = make_gsplat_checkout(tmp_path)
    config = write_config(tmp_path / "p.yaml", {"trainer.dir": str(checkout), "trainer.commit": head})
    scene, out = make_scene(tmp_path / "room" / "colmap"), tmp_path / "room" / "model"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(installed.parent)]),
           "FAKE_TRAINER_RECORD": str(tmp_path / "call.json"), "FAKE_TRAINER_SLEEP": "30"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.Popen([sys.executable, "-m", "src.product.train_main", "--scene", str(scene), "--out", str(out),
                             "--config", str(config)], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 30
    while not (tmp_path / "call.json").exists():
        assert proc.poll() is None, proc.stdout.read()
        assert time.monotonic() < deadline, "the fake trainer did not start"
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    output, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143, output
    meta = json.loads((out / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "STOPPED" and meta["trainer"]["commit"] == head
    assert "[train_main] stopped" in output


# ------------------------------------------------------------------------------------------- tiến độ
def test_progress_goes_through_the_stages(tmp_path):
    path = tmp_path / "room" / "progress.json"
    assert progress.queue(path, "phong1", "photos", "room_phong1")
    data = progress.read(path)
    assert data["status"] == "queued" and data["message"] == "Đang chờ GPU" and data["started_at"] is None
    assert [stage["label"] for stage in data["stages"]] == [
        "Chuẩn hóa ảnh", "COLMAP: ước lượng camera", "Train mô hình chính (mcmc)", "Xuất mô hình web và thêm vào thư viện"]
    assert not progress.queue(path, "phong1", "video")  # lần dựng đang chờ / đang chạy: không ghi đè
    created = data["created_at"]
    data = progress.start(path, "phong1", log="/ws/log.txt")
    assert data["status"] == "running" and data["created_at"] == created and data["job"] == "room_phong1"
    assert data["source_kind"] == "photos" and data["started_at"] is not None
    progress.set_stage(path, "images")
    progress.set_stage(path, "colmap", "COLMAP: ghép cặp ảnh")
    data = progress.set_stage(path, "train")
    assert [stage["status"] for stage in data["stages"]] == ["done", "done", "running", "waiting"]
    assert data["stage"] == "train" and data["message"] == "Train mô hình chính (mcmc)"
    data = progress.update(path, "train", 150, "Vòng 30.000/30.000")
    assert data["stages"][2]["percent"] == 100.0 and data["message"] == "Vòng 30.000/30.000"
    progress.set_stage(path, "export")
    data = progress.done(path)
    assert data["status"] == "done" and data["stage"] is None and data["finished_at"]
    assert [stage["status"] for stage in data["stages"]] == ["done"] * 4 and data["stages"][2]["percent"] == 100.0
    assert progress.queue(path, "phong1", "video")  # lần dựng trước đã xong: lần mới xếp hàng được
    assert progress.read(path)["stages"][0]["label"] == "Tách khung ảnh từ video"


def test_progress_records_failures_stops_and_skipped_stages(tmp_path):
    path = tmp_path / "progress.json"
    progress.start(path, "phong1")
    assert progress.read(path)["stages"][0]["label"] == "Chuẩn bị ảnh"  # chưa biết loại nguồn
    progress.skip(path, ["images", "colmap"], "Dùng lại ảnh và COLMAP đã có")
    progress.set_stage(path, "train")
    data = progress.fail(path, "Train mô hình chính lỗi")
    assert data["status"] == "failed" and data["stage"] == "train" and data["error"] == "Train mô hình chính lỗi"
    assert [stage["status"] for stage in data["stages"]] == ["skipped", "skipped", "failed", "waiting"]
    progress.start(path, "phong1")
    progress.set_stage(path, "colmap")
    data = progress.fail(path, "Đã dừng", stopped=True)
    assert data["status"] == "stopped" and data["error"] is None and data["stages"][1]["status"] == "stopped"
    assert data["stages"][0]["status"] == "done"
    assert progress.update(tmp_path / "none.json", "train", 5) is None and progress.fail(tmp_path / "none.json", "x") is None
    with pytest.raises(ValueError, match="no progress file"):
        progress.set_stage(tmp_path / "none.json", "train")


def test_follow_copies_the_output_and_switches_stages_on_the_markers(tmp_path):
    path = tmp_path / "progress.json"
    progress.start(path, "phong1")
    progress.set_stage(path, "images")
    lines = [b"===== 1. Frames -> /ws/colmap/input =====\n", b"ffmpeg \xff\xfe not utf-8\n",
             b"===== 2. COLMAP =====\n", b"feature extraction\n"]
    out = io.BytesIO()
    assert progress.follow(path, io.BytesIO(b"".join(lines)), out) == 0
    assert out.getvalue() == b"".join(lines)
    data = progress.read(path)
    assert data["source_kind"] == "video" and data["stages"][0]["label"] == "Tách khung ảnh từ video"
    assert [stage["status"] for stage in data["stages"]] == ["done", "running", "waiting", "waiting"]


def test_progress_command_line(tmp_path):
    path = tmp_path / "progress.json"

    def run(*args):
        return subprocess.run([sys.executable, "-m", "src.product.progress", str(path), *args], cwd=REPO,
                              capture_output=True, text=True, timeout=60)

    assert run("stage", "train").returncode == 1  # chưa có file
    assert run("start", "--room", "phong1", "--source-kind", "photos", "--log", "/l").returncode == 0
    assert run("queue", "--room", "phong1").stdout.startswith("progress: ")  # đang chạy: không ghi đè
    assert run("skip", "images", "colmap", "--message", "Dùng lại").returncode == 0
    assert run("stage", "train").returncode == 0
    assert run("fail", "--message", "lỗi").returncode == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "failed" and data["log"] == "/l" and data["stages"][2]["status"] == "failed"
    assert run("stage", "nope").returncode == 2


def test_the_gpu_check_runs_the_gpu_libraries_of_the_env(tmp_path):
    cfg = product_config.load(PRODUCT, ws="/ws")
    good, bad = tmp_path / "good", tmp_path / "bad"
    write_fake_gpu_modules(good)
    write_fake_gpu_modules(bad, error="CUDA error: no kernel image is available for execution on the device")
    train_main.check_gpu(cfg, {**os.environ, "PYTHONPATH": str(good)})
    with pytest.raises(train_main.TrainError, match="gs-gsplat env is not ready for this GPU .*no kernel image"):
        train_main.check_gpu(cfg, {**os.environ, "PYTHONPATH": str(bad)})
