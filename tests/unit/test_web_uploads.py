"""Test app/web/uploads.py (phiên tải lên của trang web) và app/web/rooms.py (trạng thái phòng cho trang web).

Không cần server: gọi thẳng các hàm. Kiểm tra quy tắc đầu vào (giống scripts/build_room.sh), nhận từng phần đúng
thứ tự, chuyển nguồn sang data/uploads/<phòng>/ chỉ khi đủ, không đụng dữ liệu đã có; trạng thái phòng (đang chờ,
đang chạy, mất khi pod tắt, đã dừng khi còn xếp hàng), log, file PLY tải về.
"""

import json
import os
import subprocess
import time
import uuid

import pytest

from app.web import rooms as room_info
from app.web import uploads
from app.web.uploads import UploadError, UploadStore
from src.product import progress


def photos(count, size=10):
    return [{"name": f"IMG_{i:04d}.jpg", "size": size} for i in range(count)]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "CHUNK_BYTES", 4)  # phần nhỏ để thử nhiều phần
    ws = tmp_path / "ws"
    return UploadStore(ws, ws / "outputs" / "product", max_total=10_000)


# ------------------------------------------------------------------------------------------ đầu vào
@pytest.mark.parametrize("kind, files, message", [
    ("video", [{"name": "a.mp4", "size": 5}, {"name": "b.mp4", "size": 5}], "Chỉ chọn một video"),
    ("video", [{"name": "IMG_1.JPG", "size": 5}], "File đã chọn là ảnh"),
    ("video", [{"name": "phong.txt", "size": 5}], "Video phải là"),
    ("photos", photos(9), "Chỉ có 9 ảnh dùng được, cần ít nhất 10"),
    ("photos", photos(1), "Chỉ có 1 ảnh"),
    ("photos", photos(11) + [{"name": "x.heic", "size": 3}], "HEIC của iPhone"),
    ("photos", photos(12) + [{"name": "notes.txt", "size": 3}], "Chỉ nhận ảnh JPG/JPEG/PNG"),
    ("photos", photos(12, size=2000), "vượt giới hạn"),
    ("photos", photos(11) + [{"name": "a.jpg", "size": 0}], "file rỗng"),
    ("model", photos(12), "Loại nguồn"),
    ("photos", [], "Chưa chọn file"),
])
def test_bad_selections_are_refused_with_a_message_for_the_user(kind, files, message):
    with pytest.raises(UploadError, match=message):
        uploads.check_files(kind, files, max_total=10_000)


def test_the_photo_message_does_not_mention_command_line_options():
    with pytest.raises(UploadError) as error:
        uploads.check_files("photos", photos(5), max_total=10_000)
    assert "--min-images" not in str(error.value)


def test_file_names_are_made_safe_and_unique():
    used = set()
    assert uploads.safe_file_name("C:\\fakepath\\IMG 0001.JPG", used) == "IMG_0001.JPG"
    assert uploads.safe_file_name("../../etc/passwd", used) == "passwd"
    assert uploads.safe_file_name("img_0001.jpg", used) == "img_0001_2.jpg"  # trùng (không phân biệt hoa thường)
    assert uploads.safe_file_name(".hidden.png", used) == "hidden.png"
    assert uploads.safe_file_name("ảnh phòng.mp4", used) == "nh_ph_ng.mp4"


@pytest.mark.parametrize("room, message", [
    ("web", "dành cho thư mục của hệ thống"), ("Uploads", "dành cho thư mục của hệ thống"),
    ("../x", "Tên phòng chỉ gồm"), ("_x", "Tên phòng chỉ gồm"), ("a" * 65, "tối đa 64"), ("", "Tên phòng chỉ gồm"),
])
def test_room_names_follow_the_rules_of_build_room(store, room, message):
    assert message in store.room_problem(room)


def test_a_room_name_already_used_on_the_volume_is_refused(store):
    (store.product_root / "phong1").mkdir(parents=True)
    (store.sources_dir / "phong2").mkdir(parents=True)
    assert "Đã có phòng 'phong1'" in store.room_problem("phong1")
    assert "Đã có dữ liệu tải lên" in store.room_problem("phong2")
    with pytest.raises(UploadError) as error:
        store.create("phong1", "Phòng 1", "video", [{"name": "a.mp4", "size": 8}])
    assert error.value.status == 409
    assert store.room_problem("phong3") is None


# ----------------------------------------------------------------------------------- nhận từng phần
def test_a_video_is_received_in_order_then_becomes_the_source_of_the_room(store):
    data = b"0123456789"
    session = store.create("phong_khach", "  Phòng khách\x07 ", "video", [{"name": "C:\\fakepath\\Phòng 1.MP4", "size": 10}])
    assert session["name"] == "Phòng khách" and session["chunk_bytes"] == 4 and session["files"][0]["name"] == "Ph_ng_1.MP4"
    assert "source" not in session and "original" not in session["files"][0]  # không gửi đường dẫn / tên gốc ra ngoài
    assert store.write_chunk(session["id"], 0, 0, data[:4])["received"] == 4
    with pytest.raises(UploadError) as error:  # gửi lại phần đầu (vd. trả lời bị mất): server báo đã nhận tới đâu
        store.write_chunk(session["id"], 0, 0, data[:4])
    assert error.value.status == 409 and error.value.extra == {"received": 4}
    with pytest.raises(UploadError, match="Chưa nhận đủ"):
        store.finish(session["id"])
    for bad in (b"", b"12345", b"456789x"):  # rỗng, quá chunk_bytes, vượt kích thước đã khai
        with pytest.raises(UploadError, match="sai kích thước"):
            store.write_chunk(session["id"], 0, 4, bad)
    store.write_chunk(session["id"], 0, 4, data[4:8])
    store.write_chunk(session["id"], 0, 8, data[8:])
    assert store.get(session["id"])["files"][0]["received"] == 10
    finished, source = store.finish(session["id"])
    assert finished["status"] == "finished" and source == store.sources_dir / "phong_khach" / "Ph_ng_1.MP4"
    assert source.read_bytes() == data
    assert store.source_of("phong_khach") == source
    with pytest.raises(UploadError, match="đã kết thúc"):
        store.write_chunk(session["id"], 0, 0, data[:4])
    store.forget(session["id"])
    with pytest.raises(UploadError) as error:
        store.get(session["id"])
    assert error.value.status == 404


def test_photos_become_a_folder_and_a_failed_start_is_rolled_back(store):
    session = store.create("phong2", None, "photos", photos(12, size=3))
    for index in range(12):
        store.write_chunk(session["id"], index, 0, f"{index:03d}".encode())
    _, source = store.finish(session["id"])
    assert source == store.sources_dir / "phong2" and sorted(p.name for p in source.iterdir())[:2] == [
        uploads.UPLOAD_INFO, "IMG_0000.jpg"]
    assert (source / "IMG_0011.jpg").read_bytes() == b"011" and store.source_of("phong2") == source
    assert not list(store.sources_dir.glob(".*incoming*"))
    store.rollback(session["id"], source)  # build_room.sh từ chối: xóa nguồn vừa tạo và phiên
    assert not source.exists() and not (store.sessions_dir / session["id"]).exists()
    assert store.room_problem("phong2") is None  # đặt lại được tên đó


def test_the_upload_keeps_its_display_name_next_to_the_source(store):
    from src.input import sources

    session = store.create("phong3", "Phòng ngủ", "photos", photos(12, size=3))
    for index in range(12):
        store.write_chunk(session["id"], index, 0, b"abc")
    _, source = store.finish(session["id"])
    info = uploads.read_upload_info(store.sources_dir, "phong3")
    assert info["name"] == "Phòng ngủ" and info["kind"] == "photos" and len(info["files"]) == 12
    assert info["files"][0] == {"name": "IMG_0000.jpg", "original": "IMG_0000.jpg", "size": 3}
    scan = sources.scan_photo_folder(source)  # file ẩn: build_room.sh / prepare_scene.sh không thấy, không cảnh báo
    assert len(scan["photos"]) == 12 and scan["skipped"] == [] and store.source_of("phong3") == source
    # thư viện: phòng chưa có trong manifest.json hiện tên đặt lúc tải lên
    root = store.product_root
    progress.queue(root / "phong3" / "progress.json", "phong3", "photos")
    assert room_info.list_rooms(root / "web", root, store.ws)[0]["name"] == "Phòng ngủ"
    assert room_info.room_detail(root / "web", root, store.ws, "phong3")["name"] == "Phòng ngủ"
    assert room_info.display_name(store.ws, "phong3", {"name": "Tên trong thư viện"}) == "Tên trong thư viện"
    (source / uploads.UPLOAD_INFO).write_text('{"name": ["x"]}', encoding="utf-8")  # file hỏng: dùng mã phòng
    assert room_info.display_name(store.ws, "phong3") == "phong3" and room_info.display_name(store.ws, "khac") == "khac"


def test_only_a_few_unfinished_uploads_are_kept_at_once(store):
    """Mỗi phiên dở giữ tới max_total byte trên volume: tối đa MAX_OPEN phiên; phiên ngừng nhận quá lâu nhường chỗ."""
    opened = [store.create(f"p{i}", None, "video", [{"name": "a.mp4", "size": 8}]) for i in range(uploads.MAX_OPEN)]
    with pytest.raises(UploadError) as error:
        store.create("p_more", None, "video", [{"name": "a.mp4", "size": 8}])
    assert error.value.status == 429 and "lượt tải lên dở" in str(error.value)
    idle = store.sessions_dir / opened[0]["id"] / "session.json"
    old = time.time() - uploads.IDLE_SECONDS - 60
    os.utime(idle, (old, old))
    assert store.create("p_more", None, "video", [{"name": "a.mp4", "size": 8}])["room"] == "p_more"
    assert not idle.parent.exists() and (store.sessions_dir / opened[1]["id"]).exists()


def test_a_new_upload_for_the_same_room_replaces_an_unfinished_one(store):
    first = store.create("phong1", None, "video", [{"name": "a.mp4", "size": 8}])
    store.write_chunk(first["id"], 0, 0, b"abcd")
    second = store.create("phong1", None, "video", [{"name": "b.mp4", "size": 8}])
    assert second["id"] != first["id"] and not (store.sessions_dir / first["id"]).exists()


def test_session_ids_cannot_reach_other_folders(store):
    for bad in ("../../x", "abc", "0" * 15, "Z" * 16):
        with pytest.raises(UploadError) as error:
            store.get(bad)
        assert error.value.status == 404


def test_stale_sessions_are_cleaned_up(store):
    session = store.create("phong1", None, "video", [{"name": "a.mp4", "size": 8}])
    path = store.sessions_dir / session["id"] / "session.json"
    old = time.time() - uploads.STALE_SECONDS - 60
    os.utime(path, (old, old))
    store.cleanup_stale()
    assert not path.parent.exists()


def test_uploads_do_not_touch_existing_data(store):
    keep = store.ws / "data" / "custom_scenes" / "room" / "video2.mp4"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"original")
    run = store.ws / "outputs" / "benchmarks" / "mipnerf360" / "room" / "mcmc_s42" / "results.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")
    session = store.create("room", None, "video", [{"name": "v.mp4", "size": 4}])  # trùng tên cảnh thực nghiệm
    store.write_chunk(session["id"], 0, 0, b"vvvv")
    _, source = store.finish(session["id"])
    assert source.parent == store.sources_dir / "room" and keep.read_bytes() == b"original" and run.read_text() == "{}"


# ------------------------------------------------------------------------------------- trạng thái phòng
def write_state(ws, job, **values):
    path = ws / "outputs" / "jobs" / f"{job}.state"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")


def test_room_status_follows_the_background_job(tmp_path):
    ws = tmp_path / "ws"
    root = ws / "outputs" / "product"
    web = root / "web"
    # phòng đang chờ mà việc nền đã bị dừng (không ai ghi progress.json): "stopped"
    progress.queue(root / "p_stopped" / "progress.json", "p_stopped", "video", "room_p_stopped")
    write_state(ws, "room_p_stopped", state="STOPPED", token="t-gone")
    # đang chạy nhưng không còn tiến trình (pod khởi động lại): "lost"
    progress.start(root / "p_lost" / "progress.json", "p_lost", "photos", "room_p_lost")
    write_state(ws, "room_p_lost", state="RUNNING", token="t-gone-2")
    # đang chạy, còn một tiến trình mang token của việc (như job.sh nhận tiến trình của việc)
    token = f"t-alive-{uuid.uuid4().hex}"
    progress.start(root / "p_run" / "progress.json", "p_run", "photos", "room_p_run")
    write_state(ws, "room_p_run", state="RUNNING", token=token)
    proc = subprocess.Popen(["sleep", "30"], env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "SR3D_JOB_TOKEN": token})
    try:
        statuses = {r["id"]: r["status"] for r in room_info.list_rooms(web, root, ws)}
    finally:
        proc.kill()
        proc.wait()
    assert statuses == {"p_stopped": "stopped", "p_lost": "lost", "p_run": "running"}
    assert {r["id"]: r["status"] for r in room_info.list_rooms(web, root, ws)}["p_run"] == "lost"  # tiến trình đã tắt
    # thư mục hệ thống không phải phòng
    (root / "uploads").mkdir()
    (root / "web").mkdir()
    assert {"uploads", "web"}.isdisjoint(r["id"] for r in room_info.list_rooms(web, root, ws))


def test_a_live_job_is_found_by_its_token(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    (proc / "123" / "environ").write_bytes(b"PATH=/bin\0SR3D_JOB_TOKEN=abc-1\0")
    assert room_info.token_alive("abc-1", proc) and not room_info.token_alive("abc", proc)
    assert not room_info.token_alive("", proc)


def test_the_library_lists_rooms_being_built_before_finished_ones(tmp_path):
    ws = tmp_path / "ws"
    root = ws / "outputs" / "product"
    web = root / "web"
    (web / "rooms").mkdir(parents=True)
    manifest = {"format_version": 1, "rooms": [{"id": "done1", "name": "Phòng xong", "files": {}}]}
    (web / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    progress.start(root / "done1" / "progress.json", "done1")
    progress.done(root / "done1" / "progress.json")
    progress.start(root / "new1" / "progress.json", "new1", "video")  # không có job: không đối chiếu được
    progress.set_stage(root / "new1" / "progress.json", "train")
    progress.update(root / "new1" / "progress.json", "train", 41.3, "Vòng 12.390/30.000")
    rooms = room_info.list_rooms(web, root, ws)
    assert [r["id"] for r in rooms] == ["new1", "done1"]
    assert rooms[0]["status"] == "running" and rooms[0]["build"]["percent"] == 41.3 and rooms[0]["entry"] is None
    assert rooms[1]["status"] == "done" and rooms[1]["build"] is None and rooms[1]["name"] == "Phòng xong"
    detail = room_info.room_detail(web, root, ws, "done1")
    assert detail["status"] == "done" and detail["entry"]["id"] == "done1"
    assert room_info.room_detail(web, root, ws, "none") is None and room_info.room_detail(web, root, ws, "../x") is None


def test_the_log_tail_keeps_the_last_state_of_progress_bars_and_stays_in_the_log_folder(tmp_path):
    logs = tmp_path / "ws" / "outputs" / "logs"
    logs.mkdir(parents=True)
    log = logs / "build_room_p.log"
    log.write_bytes(b"===== 1. Frames\n 10%|#\r 50%|##\r100%|###\nDONE \xff\n\n")
    assert room_info.log_tail(str(log), logs, 10) == ["===== 1. Frames", "100%|###", "DONE \ufffd"]
    assert room_info.log_tail(str(log), logs, 1) == ["DONE \ufffd"]
    outside = tmp_path / "secret.txt"
    outside.write_text("token")
    assert room_info.log_tail(str(outside), logs) == [] and room_info.log_tail(None, logs) == []
    assert room_info.log_tail(str(logs / ".." / ".." / ".." / "secret.txt"), logs) == []


def test_the_ply_download_needs_a_finished_training(tmp_path):
    root = tmp_path / "product"
    model = root / "p1" / "model"
    (model / "ply").mkdir(parents=True)
    (model / "ply" / "point_cloud_29999.ply").write_bytes(b"ply")
    assert room_info.ply_file(root, "p1") is None
    (model / "train_meta.json").write_text(json.dumps({"status": "RUNNING", "files": {"ply": "ply/point_cloud_29999.ply"}}))
    assert room_info.ply_file(root, "p1") is None
    (model / "train_meta.json").write_text(json.dumps({"status": "OK", "files": {"ply": "ply/point_cloud_29999.ply"}}))
    assert room_info.ply_file(root, "p1") == (model / "ply" / "point_cloud_29999.ply").resolve()
    (model / "train_meta.json").write_text(json.dumps({"status": "OK", "files": {"ply": "../../../etc/passwd"}}))
    assert room_info.ply_file(root, "p1") is None and room_info.ply_file(root, "../p1") is None
