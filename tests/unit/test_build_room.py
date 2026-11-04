"""Test scripts/build_room.sh thật (luồng sản phẩm: video / thư mục ảnh -> ảnh + COLMAP -> train mô hình chính -> web).

prepare_scene.sh, conda, GPU, src.product.train_main và src.product.export_web được thay bằng stub; kiểm tra nguồn
(src.input.sources), đọc cấu hình (src.product.config), tiến độ (src.product.progress), khóa GPU và việc nền
(scripts/pod/job.sh) là thật. Kiểm tra: thứ tự và tham số các bước, cả hai loại đầu vào (video, thư mục ảnh),
dùng lại COLMAP, dừng khi COLMAP FAIL hoặc train lỗi (progress.json ghi đúng bước), chạy nền, DRY_RUN.
"""

import json
import os
import shlex

import pytest
import yaml

from app.cli.main import COMMANDS, command_for
from product_fakes import REAL_PREPARE, make_build_env, make_gsplat_checkout
from shell_env import REPO, HoldLock, calls, write_stub

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

@pytest.fixture
def sh(tmp_path):
    env = make_build_env(tmp_path)
    yield env
    env.cleanup()


def build(sh, *args, env=None, timeout=60):
    return sh.run(str(sh.repo / "scripts" / "build_room.sh"), *args, env=env, timeout=timeout)


def progress_of(sh, name="phong1"):
    return json.loads((sh.room / name / "progress.json").read_text(encoding="utf-8"))


def step_calls(sh):
    return [line for line in calls(sh.root) if not line.startswith(("python -c", "conda"))]


def test_photos_go_through_every_step_with_the_main_model(sh):
    out = build(sh, str(sh.photos), "phong1", "--name", "Phòng khách")
    assert out.returncode == 0, out.stdout + out.stderr
    prepare, train, export = step_calls(sh)
    scene, model = sh.room / "phong1" / "colmap", sh.room / "phong1" / "model"
    config = sh.repo / "configs" / "product.yaml"
    progress_file = sh.room / "phong1" / "progress.json"
    assert prepare == f"prepare_scene {sh.photos} phong1 OVERWRITE=0 OUT_ROOT={sh.room} HELD=1"
    assert shlex.split(train) == ["train_main", "--scene", str(scene), "--out", str(model), "--config", str(config),
                                  "--ws", str(sh.ws), "--progress", str(progress_file)]
    assert shlex.split(export) == ["export_web", "--room", "phong1", "--model", str(model), "--scene", str(scene),
                                   "--web-dir", str(sh.room / "web"), "--config", str(config), "--ws", str(sh.ws),
                                   "--progress", str(progress_file), "--name=Phòng khách"]
    assert "conda activate gs-gsplat" in calls(sh.root)
    data = progress_of(sh)
    assert data["status"] == "done" and data["source_kind"] == "photos" and data["error"] is None
    assert [stage["status"] for stage in data["stages"]] == ["done"] * 4
    assert data["stages"][0]["label"] == "Chuẩn hóa ảnh" and data["job"] is None
    assert os.path.isfile(data["log"]) and "DONE. Room phong1" in out.stdout
    assert sh.lock_free()


def test_a_video_is_split_into_frames_and_a_prepared_room_is_reused(sh):
    assert build(sh, str(sh.video), "phong2").returncode == 0
    data = progress_of(sh, "phong2")
    assert data["source_kind"] == "video" and data["stages"][0]["label"] == "Tách khung ảnh từ video"
    (sh.root / "calls.txt").unlink()
    out = build(sh, "-", "phong2", "--retrain")  # đã có COLMAP: không chuẩn bị lại, train lại khi có --retrain
    assert out.returncode == 0, out.stdout
    train, export = step_calls(sh)
    assert shlex.split(train)[-1] == "--force" and shlex.split(export)[0] == "export_web"
    data = progress_of(sh, "phong2")
    assert data["source_kind"] == "video"  # đọc từ frames_report.json của lần chuẩn bị trước
    assert [stage["status"] for stage in data["stages"]] == ["skipped", "skipped", "done", "done"]
    (sh.root / "calls.txt").unlink()
    assert build(sh, str(sh.video), "phong2", "--redo-colmap").returncode == 0
    assert step_calls(sh)[0].startswith(f"prepare_scene {sh.video} phong2 OVERWRITE=1 ")


def test_a_failed_colmap_stops_before_training(sh):
    out = build(sh, str(sh.photos), "phong1", env={"FAKE_VERDICT": "FAIL", "FAKE_PREP_EXIT": "1"})
    assert out.returncode == 1 and "Quay chậm hơn" in out.stdout
    assert [line.split()[0] for line in step_calls(sh)] == ["prepare_scene"]
    data = progress_of(sh)
    assert data["status"] == "failed" and data["stage"] == "colmap" and "COLMAP FAIL" in data["error"]
    assert [stage["status"] for stage in data["stages"]] == ["done", "failed", "waiting", "waiting"]
    out = build(sh, "-", "phong1")  # COLMAP có mô hình nhưng đánh giá FAIL: vẫn không train
    assert out.returncode == 1 and "COLMAP đánh giá FAIL" in out.stdout
    assert progress_of(sh)["error"].startswith("COLMAP đánh giá FAIL")


def test_a_failed_training_is_shown_at_the_train_step(sh):
    out = build(sh, str(sh.photos), "phong1", env={"FAKE_TRAIN_EXIT": "1"})
    assert out.returncode == 1 and "Train mô hình chính lỗi" in out.stdout
    assert [line.split()[0] for line in step_calls(sh)] == ["prepare_scene", "train_main"]
    data = progress_of(sh)
    assert data["status"] == "failed" and data["stage"] == "train"
    assert [stage["status"] for stage in data["stages"]] == ["done", "done", "failed", "waiting"]
    out = build(sh, "-", "phong1", env={"FAKE_EXPORT_EXIT": "1"})
    assert out.returncode == 1 and progress_of(sh)["stage"] == "export"


def test_bad_inputs_are_refused_before_anything_is_created(sh):
    single = sh.root / "one.jpg"
    single.write_bytes(b"jpg")
    few = sh.root / "few"
    few.mkdir()
    for index in range(3):
        (few / f"{index}.jpg").write_bytes(b"jpg")
    for args, message in [((str(single), "phong1"), "Chỉ có 1 ảnh"),
                          ((str(few), "phong1"), "Chỉ có 3 ảnh dùng được"),
                          ((str(sh.root / "missing.mp4"), "phong1"), "source not found"),
                          ((str(sh.photos), "../phong"), "room name must start with a letter or digit"),
                          ((str(sh.photos), "web"), "trùng thư mục của hệ thống"),  # thư mục web/ của sản phẩm
                          ((str(sh.photos), "Uploads"), "trùng thư mục của hệ thống"),
                          (("-", "phong1"), "chưa có mô hình COLMAP"),
                          ((str(sh.photos), "phong1", "--name"), "--name needs a value"),
                          ((str(sh.photos), "phong1", "--fast"), "unknown option '--fast'")]:
        out = build(sh, *args)
        assert out.returncode != 0 and message in out.stdout + out.stderr, (args, out.stdout, out.stderr)
    assert not calls(sh.root) and not sh.room.exists()  # không xếp hàng, không tạo thư mục, không ghi tiến độ


def test_a_room_name_belongs_to_one_source(sh):
    assert build(sh, str(sh.photos), "phong1").returncode == 0
    out = build(sh, str(sh.video), "phong1")
    assert out.returncode == 1 and f"đã được dựng từ {sh.photos}" in out.stdout


def test_dry_run_prints_the_steps_and_creates_nothing(sh):
    out = build(sh, str(sh.photos), "phong1", "--name", "Phòng khách", env={"DRY_RUN": "1"})
    assert out.returncode == 0, out.stdout
    lines = [line for line in out.stdout.splitlines() if line.startswith("+ ")]
    assert lines[0].startswith("+ env OVERWRITE=0 ") and "prepare_scene.sh" in lines[0] and lines[0].endswith(" follow")
    assert lines[1] == "+ conda activate gs-gsplat"
    assert "src.product.train_main" in lines[2] and "src.product.export_web" in lines[3]
    assert shlex.split(lines[3][2:])[-1] == "--name=Phòng khách"
    assert not calls(sh.root) and not sh.room.exists()
    out = build(sh, str(sh.photos), "phong1", "--background", env={"DRY_RUN": "1"})
    assert out.returncode == 0 and "job.sh start room_phong1 -- bash" in out.stdout and not sh.room.exists()


def test_background_runs_as_a_job_and_waits_for_the_gpu(sh):
    with HoldLock(sh.lock):  # một việc khác đang dùng GPU: việc mới xếp hàng
        out = build(sh, str(sh.video), "phong1", "--background", "--name", "Phòng khách")
        assert out.returncode == 0 and "room_phong1" in out.stdout, out.stdout + out.stderr
        sh.wait_state("room_phong1", "QUEUED")
        data = progress_of(sh)
        assert data["status"] == "queued" and data["job"] == "room_phong1" and data["source_kind"] == "video"
    state = sh.wait_state("room_phong1", "DONE", "FAILED", timeout=60)
    assert state["state"] == "DONE", sh.log("room_phong1")
    data = progress_of(sh)
    assert data["status"] == "done" and data["job"] == "room_phong1" and data["started_at"]
    assert shlex.split(step_calls(sh)[-1])[-1] == "--name=Phòng khách"


def test_the_cli_command_runs_the_script():
    assert COMMANDS["build-room"][0] == "scripts/build_room.sh"
    assert command_for("build-room", ["/x/video.mp4", "phong1", "--background"]) == [
        "bash", str(REPO / "scripts" / "build_room.sh"), "/x/video.mp4", "phong1", "--background"]


def test_the_real_train_and_export_steps_produce_the_web_files(sh, tmp_path):
    """build_room.sh với train_main.py và export_web.py thật: kiểm tra trainer thật (kho git giả đúng commit trong cấu
    hình, gói gsplat giả trùng checkout), simple_trainer.py và Parser giả. Chỉ GPU, conda và COLMAP là giả."""
    write_stub(sh.repo, "scripts/pipeline/prepare_scene.sh", REAL_PREPARE)
    checkout, installed, head = make_gsplat_checkout(tmp_path / "gs")
    data = yaml.safe_load((REPO / "configs" / "product.yaml").read_text(encoding="utf-8"))
    data["trainer"].update(dir=str(checkout), commit=head)
    data["training"]["max_steps"] = 40
    config = tmp_path / "product.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    env = {"REAL_STEPS": "1", "PRODUCT_CONFIG": str(config), "PYTHONPATH": str(installed.parent)}
    out = build(sh, str(sh.video), "phong1", "--name", "Phòng khách", env=env, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
    web = sh.room / "web"
    entry = json.loads((web / "manifest.json").read_text(encoding="utf-8"))["rooms"][0]
    assert entry["id"] == "phong1" and entry["name"] == "Phòng khách" and entry["num_gaussians"] == 7
    assert (web / entry["files"]["model"]).stat().st_size == 7 * 32 and (web / entry["files"]["cameras"]).is_file()
    assert entry["viewer_modes"] == ["orbit", "walk", "tour"] and entry["source"]["kind"] == "video"
    meta = json.loads((sh.room / "phong1" / "model" / "train_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "OK" and meta["trainer"]["commit"] == head and meta["cuda_visible_devices"] == "0"
    data = progress_of(sh)
    assert data["status"] == "done" and data["stages"][2]["percent"] == 100.0
    assert data["stages"][2]["message"] == "Train xong: 0,00 triệu Gaussian"
    out = build(sh, "-", "phong1", env=env, timeout=180)  # chạy lại: không train lại, xuất lại
    assert out.returncode == 0 and "not training again" in out.stdout, out.stdout


def test_existing_data_on_the_volume_is_not_touched(sh):
    """Dữ liệu đã có trên volume (nguồn, kết quả thực nghiệm, báo cáo, việc nền, gói kết quả) giữ nguyên từng byte;
    luồng sản phẩm chỉ ghi vào outputs/product/ và log của chính nó, kể cả khi tên phòng trùng tên một cảnh thực nghiệm."""
    import hashlib

    existing = {
        "data/custom_scenes/room/video2.mp4": b"video of the group room",
        "outputs/custom_scenes/phong_nhom/colmap/sparse/0/images.bin": b"colmap of the experiment",
        "outputs/custom_scenes/phong_nhom/mcmc_s42/ply/point_cloud_29999.ply": b"ply of the experiment",
        "outputs/custom_scenes/phong_nhom/mcmc_s42/summary.json": b'{"status": "OK"}',
        "outputs/benchmarks/mipnerf360/gsplat/room/mcmc_s42/summary.json": b'{"status": "OK"}',
        "outputs/reports/main_20261031_114145.md": b"# main",
        "outputs/logs/jobs/main.log": b"log of main",
        "outputs/jobs/main.state": b"state=DONE\n",
        "ket_qua_main_20261031_114145.tar.gz": b"package",
    }
    for relative, content in existing.items():
        path = sh.ws / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def snapshot():
        return {str(path.relative_to(sh.ws)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sh.ws.rglob("*") if path.is_file() and "miniforge3" not in path.parts}

    before = snapshot()
    out = build(sh, str(sh.ws / "data" / "custom_scenes" / "room" / "video2.mp4"), "phong_nhom", "--background")
    assert out.returncode == 0, out.stdout + out.stderr
    assert sh.wait_state("room_phong_nhom", "DONE", "FAILED", timeout=60)["state"] == "DONE", sh.log("room_phong_nhom")
    after = snapshot()
    assert {path: after.get(path) for path in before} == before  # không file cũ nào bị sửa hay xóa
    new = sorted(set(after) - set(before))
    allowed = ("outputs/product/", "outputs/logs/build_room_phong_nhom_", "outputs/jobs/room_phong_nhom.",
               "outputs/logs/jobs/room_phong_nhom.")
    assert new and all(path.startswith(allowed) for path in new), new
    assert progress_of(sh, "phong_nhom")["status"] == "done"
