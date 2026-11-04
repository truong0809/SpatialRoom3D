"""Test app/web/server.py qua HTTP thật (server chạy trong một thread, cổng ngẫu nhiên trên 127.0.0.1).

build_room.sh được thay bằng một script giả ghi lại tham số (trừ test cuối: build_room.sh thật với train_main /
export_web thật, chỉ GPU, conda, COLMAP là giả). Kiểm tra: đăng nhập, phiên và cookie, header chống CSRF, request lạ
(mật khẩu ngoài ASCII, Content-Length sai, đường dẫn có NUL) không làm lỗi server, file tĩnh không ra ngoài thư mục,
ETag / Range, tải lên từng phần rồi dựng phòng, lỗi / chậm của build_room.sh, tiến độ, log, hủy, dựng lại, tải PLY,
chế độ chỉ xem, script chạy server.
"""

import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest
import yaml

from app.cli.main import COMMANDS
from app.web import server as web_server
from app.web import uploads
from product_fakes import REAL_PREPARE, make_build_env, make_gsplat_checkout
from shell_env import REPO, ShellEnv, make_fake_repo, write_stub
from src.product import progress

TOKEN = "secret-token"
FAKE_BUILD = r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_BUILD_CALLS"
if [ -n "${FAKE_BUILD_JOB_TOKEN:-}" ]; then  # như job.sh: việc nền đã có (tiến trình mang token) trước khi lệnh trả lời
  mkdir -p "$WS/outputs/jobs"
  printf 'name=room_%s\ntoken=%s\nstate=QUEUED\n' "$2" "$FAKE_BUILD_JOB_TOKEN" > "$WS/outputs/jobs/room_$2.state"
  SR3D_JOB_TOKEN="$FAKE_BUILD_JOB_TOKEN" sleep 30 < /dev/null > /dev/null 2>&1 &
  echo "$!" > "$FAKE_BUILD_HOLDER"
fi
[ -z "${FAKE_BUILD_SLEEP:-}" ] || sleep "$FAKE_BUILD_SLEEP"
if [ -n "${FAKE_BUILD_OUTPUT:-}" ]; then echo "$FAKE_BUILD_OUTPUT"; exit 0; fi
if [ "${FAKE_BUILD_EXIT:-0}" != 0 ]; then echo "ERROR: ${FAKE_BUILD_MESSAGE:-lỗi giả}"; exit "$FAKE_BUILD_EXIT"; fi
if [ -n "${FAKE_BUILD_QUEUED:-}" ]; then echo "Đã xếp hàng 'room_$2': đang có việc khác dùng GPU"; else echo "Đã chạy nền 'room_$2'."; fi
'''


class Client:
    def __init__(self, port: int):
        self.port = port
        self.cookie = None

    def request(self, method, path, body=None, headers=None, json_body=None, write=True):
        headers = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        if write and method in ("POST", "PUT", "DELETE"):
            headers.setdefault("X-SR3D", "1")
        if self.cookie:
            headers.setdefault("Cookie", self.cookie)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        response.data = data
        try:
            response.json = json.loads(data) if data else None
        except ValueError:
            response.json = None
        return response

    def login(self, token=TOKEN):
        response = self.request("POST", "/api/login", json_body={"token": token})
        if response.status == 200:
            self.cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        return response


def raw_request(port, data: bytes) -> bytes:
    """Gửi nguyên các byte của một request (http.client không gửi được header sai) và đọc tới khi server đóng."""
    with socket.create_connection(("127.0.0.1", port), timeout=30) as conn:
        conn.sendall(data)
        chunks = []
        while True:
            block = conn.recv(65536)
            if not block:
                return b"".join(chunks)
            chunks.append(block)


def start_server(settings):
    httpd = web_server.make_server(settings, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    return httpd, thread


@pytest.fixture
def web(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    root = ws / "outputs" / "product"
    web_dir = root / "web"
    (web_dir / "rooms" / "p1").mkdir(parents=True)
    (web_dir / "rooms" / "p1" / "model.splat").write_bytes(bytes(range(64)))
    (web_dir / "rooms" / "p1" / ".secret").write_text("x")
    entry = {"id": "p1", "name": "Phòng 1", "version": "abc", "files": {"model": "rooms/p1/model.splat"},
             "num_gaussians": 2, "size_bytes": 64, "viewer_modes": ["orbit"]}
    (web_dir / "manifest.json").write_text(json.dumps({"format_version": 1, "rooms": [entry]}), encoding="utf-8")
    (root / "server").mkdir()
    (root / "server" / "token").write_text(TOKEN)
    build = tmp_path / "fake_build.sh"
    build.write_text(FAKE_BUILD, encoding="utf-8")
    build.chmod(0o755)
    monkeypatch.setenv("FAKE_BUILD_CALLS", str(tmp_path / "build_calls.txt"))
    monkeypatch.setenv("JOB_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("STOP_TIMEOUT", "2")
    monkeypatch.delenv("SR3D_WEB_TOKEN", raising=False)
    settings = web_server.Settings(ws=ws, web_dir=web_dir, product_root=root, token=TOKEN, build_script=build,
                                   max_upload=1_000_000)
    httpd, thread = start_server(settings)
    client = Client(httpd.server_address[1])
    client.ws, client.root, client.web_dir, client.tmp, client.settings = ws, root, web_dir, tmp_path, settings
    client.calls = lambda: (tmp_path / "build_calls.txt").read_text(encoding="utf-8").splitlines() \
        if (tmp_path / "build_calls.txt").exists() else []
    yield client
    httpd.shutdown()
    httpd.server_close()
    thread.join(5)


def upload(client, room, kind, files, name=None, chunk=None):
    """Tải lên như trang web: tạo phiên, gửi từng phần theo chunk_bytes của phiên, rồi finish."""
    created = client.request("POST", "/api/uploads", json_body={
        "room": room, "name": name, "kind": kind, "files": [{"name": n, "size": len(d)} for n, d in files]})
    assert created.status == 201, created.json
    session = created.json
    step = chunk or session["chunk_bytes"]
    for index, (_name, data) in enumerate(files):
        for offset in range(0, len(data), step):
            r = client.request("PUT", f"/api/uploads/{session['id']}/files/{index}?offset={offset}",
                               body=data[offset:offset + step], headers={"Content-Type": "application/octet-stream"})
            assert r.status == 200, r.json
    return session, client.request("POST", f"/api/uploads/{session['id']}/finish", json_body={})


# ------------------------------------------------------------------------------------------ đăng nhập
def test_the_page_shell_is_public_but_data_needs_the_password(web):
    assert web.request("GET", "/").status == 200
    js = web.request("GET", "/app/js/main.js")
    assert js.status == 200 and js.getheader("Content-Type").startswith("text/javascript")
    status = web.request("GET", "/api/status")
    assert status.status == 200 and status.json["auth"] == {"required": True, "ok": False} and "gpu" not in status.json
    for path in ("/api/rooms", "/api/rooms/p1", "/web/manifest.json", "/web/rooms/p1/model.splat", "/download/p1/model.ply"):
        assert web.request("GET", path).status == 401, path
    assert web.request("GET", "/api/rooms", headers={"Authorization": f"Bearer {TOKEN}"}).status == 200  # curl


def test_login_sets_a_strict_http_only_cookie(web):
    assert web.login("wrong").status == 401 and web.cookie is None
    response = web.login()
    cookie = response.getheader("Set-Cookie")
    assert response.status == 200 and "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Secure" not in cookie
    assert TOKEN not in cookie  # cookie là một phiên ngẫu nhiên, không phải mật khẩu
    status = web.request("GET", "/api/status").json
    assert status["auth"]["ok"] and status["gpu"]["busy"] is False and status["limits"]["min_photos"] == 10
    proxied = web.request("POST", "/api/login", json_body={"token": TOKEN}, headers={"X-Forwarded-Proto": "https"})
    assert "Secure" in proxied.getheader("Set-Cookie")
    assert "Max-Age=0" in web.request("POST", "/api/logout", json_body={}).getheader("Set-Cookie")
    assert web.request("GET", "/api/rooms").status == 401  # cookie cũ sau khi đăng xuất không dùng lại được


def test_sessions_are_random_limited_and_end_at_logout(web, monkeypatch):
    monkeypatch.setattr(web_server, "MAX_SESSIONS", 2)
    cookies = []
    for _ in range(3):
        web.login()
        cookies.append(web.cookie)
    assert len(set(cookies)) == 3

    def check(cookie):
        return web.request("GET", "/api/rooms", headers={"Cookie": cookie}).status

    assert [check(cookie) for cookie in cookies] == [401, 200, 200]  # quá số phiên: phiên cũ nhất hết hiệu lực
    assert check("sr3d_session=guess") == 401 and check("sr3d_session=") == 401
    web.request("POST", "/api/logout", json_body={}, headers={"Cookie": cookies[2]})
    assert check(cookies[2]) == 401 and check(cookies[1]) == 200


def test_odd_passwords_are_refused_without_a_server_error(web, monkeypatch):
    monkeypatch.setattr(web_server, "LOGIN_DELAY", 0)
    for token in ("mật khẩu", "\ud800", 123, None, TOKEN + "é"):
        assert web.request("POST", "/api/login", json_body={"token": token}).status == 401, token
    for header in ("Bearer café", "Bearer", "Basic " + TOKEN, "Bearer " + TOKEN + "\xe9"):
        assert web.request("GET", "/api/rooms", headers={"Authorization": header}).status == 401, header


def test_a_bad_content_length_is_refused_without_a_server_error(web):
    head = b"POST /api/login HTTP/1.1\r\nHost: 127.0.0.1\r\nX-SR3D: 1\r\n"
    for value in (b"\xb2", b"-1", b"1e3", b"9" * 20, b"0x10"):  # "\xb2" = "²": str.isdigit() đúng nhưng int() lỗi
        reply = raw_request(web.port, head + b"Content-Length: " + value + b"\r\n\r\n")
        assert reply.startswith(b"HTTP/1.1 400 ") and b"Connection: close" in reply, (value, reply[:300])
        assert "Content-Length không hợp lệ".encode() in reply
    reply = raw_request(web.port, head + b"Connection: close\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 411 ") and "Thiếu Content-Length".encode() in reply
    reply = raw_request(web.port, head + b"Content-Length: 2000000\r\n\r\n")  # quá giới hạn JSON: không đọc body
    assert reply.startswith(b"HTTP/1.1 413 ") and b"Connection: close" in reply
    reply = raw_request(web.port, b"GET /api/status HTTP/1.1\r\nHost: x\r\nContent-Length: \xb2\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 200 ") and b"Connection: close" in reply  # body không đọc: đóng kết nối


def test_writes_need_the_custom_header(web):
    web.login()
    response = web.request("POST", "/api/uploads", json_body={"room": "x"}, write=False)
    assert response.status == 403 and "X-SR3D" in response.json["error"]
    assert web.request("DELETE", "/api/uploads/0123456789abcdef", write=False).status == 403


# ------------------------------------------------------------------------------------------- file tĩnh
@pytest.mark.parametrize("path", ["/app/../server.py", "/app/%2e%2e/server.py", "/app/.%2e/uploads.py",
                                  "/web/rooms/p1/.secret", "/web/%2e%2e/server/token", "/web/rooms//p1/model.splat",
                                  "/app/vendor/../../server.py", "/nothing", "/web/rooms/p1/%00x", "/app/%00",
                                  "/app/" + "a" * 300, "/web/" + "b/" * 600 + "c"])
def test_files_never_leave_their_folder(web, path):
    web.login()
    assert web.request("GET", path).status == 404, path


def test_files_have_etags_and_ranges(web):
    web.login()
    full = web.request("GET", "/web/rooms/p1/model.splat?v=abc")
    assert full.status == 200 and full.data == bytes(range(64)) and "max-age" in full.getheader("Cache-Control")
    etag = full.getheader("ETag")
    assert web.request("GET", "/web/rooms/p1/model.splat", headers={"If-None-Match": etag}).status == 304
    part = web.request("GET", "/web/rooms/p1/model.splat", headers={"Range": "bytes=32-63"})
    assert part.status == 206 and part.data == bytes(range(32, 64)) and part.getheader("Content-Range") == "bytes 32-63/64"
    assert web.request("GET", "/web/rooms/p1/model.splat", headers={"Range": "bytes=-8"}).data == bytes(range(56, 64))
    assert web.request("GET", "/web/rooms/p1/model.splat", headers={"Range": "bytes=99-"}).status == 416
    assert web.request("HEAD", "/web/rooms/p1/model.splat").data == b""
    manifest = web.request("GET", "/web/manifest.json")
    assert manifest.getheader("Cache-Control") == "no-cache" and manifest.json["rooms"][0]["id"] == "p1"


# ------------------------------------------------------------------------------------- tải lên, dựng phòng
def test_photos_are_uploaded_in_parts_then_build_room_is_started(web, monkeypatch):
    monkeypatch.setattr(uploads, "CHUNK_BYTES", 7)
    web.login()
    files = [(f"IMG_{i:04d}.JPG", f"photo-{i:02d}-data".encode()) for i in range(12)]
    session, finished = upload(web, "phong_khach", "photos", files, name="Phòng khách")
    assert finished.status == 202 and finished.json == {"ok": True, "room": "phong_khach", "queued": False,
                                                        "message": "Đã bắt đầu dựng phòng"}
    source = web.ws / "data" / "uploads" / "phong_khach"
    assert web.calls() == [f"{source} phong_khach --name Phòng khách --background"]
    assert sorted(p.name for p in source.iterdir() if not p.name.startswith(".")) == [name for name, _ in files]
    assert (source / "IMG_0011.JPG").read_bytes() == files[11][1]
    info = json.loads((source / uploads.UPLOAD_INFO).read_text(encoding="utf-8"))
    assert info["name"] == "Phòng khách" and info["kind"] == "photos" and len(info["files"]) == 12
    assert web.request("GET", f"/api/uploads/{session['id']}").status == 404  # phiên đã xong, phần đã nhận đã chuyển
    again = web.request("POST", "/api/uploads", json_body={"room": "phong_khach", "kind": "video",
                                                           "files": [{"name": "a.mp4", "size": 3}]})
    assert again.status == 409 and "Đã có dữ liệu tải lên" in again.json["error"]
    # build_room.sh ghi progress.json: thư viện hiện tên đã đặt lúc tải lên (chưa có trong manifest.json)
    progress.queue(web.root / "phong_khach" / "progress.json", "phong_khach", "photos")
    listed = web.request("GET", "/api/rooms").json["rooms"]
    assert listed[0]["id"] == "phong_khach" and listed[0]["name"] == "Phòng khách"


def test_a_queued_video_and_resuming_after_a_lost_answer(web, monkeypatch):
    monkeypatch.setenv("FAKE_BUILD_QUEUED", "1")
    web.login()
    created = web.request("POST", "/api/uploads", json_body={"room": "v1", "kind": "video",
                                                             "files": [{"name": "quay.mov", "size": 10}]}).json
    path = f"/api/uploads/{created['id']}/files/0"
    assert web.request("PUT", f"{path}?offset=0", body=b"01234").json["received"] == 5
    stale = web.request("PUT", f"{path}?offset=0", body=b"01234")  # trả lời trước bị mất, trình duyệt gửi lại
    assert stale.status == 409 and stale.json["received"] == 5
    assert web.request("PUT", f"{path}?offset=5", body=b"56789").status == 200
    finished = web.request("POST", f"/api/uploads/{created['id']}/finish", json_body={})
    assert finished.status == 202 and finished.json["queued"] and "Đang chờ GPU" in finished.json["message"]
    assert web.calls()[0].startswith(f"{web.ws / 'data' / 'uploads' / 'v1' / 'quay.mov'} v1 --name v1")


def test_a_refused_build_returns_its_message_and_removes_the_upload(web, monkeypatch):
    monkeypatch.setenv("FAKE_BUILD_EXIT", "1")
    monkeypatch.setenv("FAKE_BUILD_MESSAGE", "Chỉ có 3 ảnh dùng được, cần ít nhất 10 ảnh")
    web.login()
    _, finished = upload(web, "p2", "photos", [(f"{i}.jpg", b"x") for i in range(10)])
    assert finished.status == 422 and finished.json["error"] == "Chỉ có 3 ảnh dùng được, cần ít nhất 10 ảnh"
    assert not (web.ws / "data" / "uploads" / "p2").exists()  # đặt lại tên đó và tải lên lại được
    assert web.request("POST", "/api/uploads", json_body={"room": "p2", "kind": "video",
                                                          "files": [{"name": "a.mp4", "size": 3}]}).status == 201


def test_answers_of_build_room_that_started_no_job_are_failures(web, monkeypatch):
    web.login()
    # job.sh: việc cũ cùng tên còn sống -> exit 0 nhưng không chạy gì
    monkeypatch.setenv("FAKE_BUILD_OUTPUT", "Việc 'room_dup1' (đang dừng): không chạy thêm bản thứ hai.")
    _, finished = upload(web, "dup1", "video", [("a.mp4", b"abc")])
    assert finished.status == 422 and "vẫn đang chạy hoặc đang dừng" in finished.json["error"]
    assert not (web.ws / "data" / "uploads" / "dup1").exists()
    monkeypatch.delenv("FAKE_BUILD_OUTPUT")
    # build_room.sh không trả lời kịp (trước giới hạn 100 giây của proxy), chưa có việc nền: lỗi, xóa nguồn
    web.settings.build_timeout = 1
    monkeypatch.setenv("FAKE_BUILD_SLEEP", "3")
    _, finished = upload(web, "slow1", "video", [("a.mp4", b"abc")])
    assert finished.status == 422 and "không trả lời kịp" in finished.json["error"]
    assert not (web.ws / "data" / "uploads" / "slow1").exists()
    # không trả lời kịp nhưng việc nền đã có: coi như đã xếp, giữ nguồn cho việc đó
    holder = web.tmp / "holder.pid"
    monkeypatch.setenv("FAKE_BUILD_JOB_TOKEN", f"tok-{uuid.uuid4().hex}")
    monkeypatch.setenv("FAKE_BUILD_HOLDER", str(holder))
    try:
        _, finished = upload(web, "slow2", "video", [("a.mp4", b"abc")])
        assert finished.status == 202 and finished.json["queued"] and "chậm" in finished.json["message"]
        assert (web.ws / "data" / "uploads" / "slow2" / "a.mp4").is_file()
    finally:
        if holder.exists():
            os.kill(int(holder.read_text()), signal.SIGKILL)


def test_bad_uploads_are_refused_with_messages(web, monkeypatch):
    monkeypatch.setattr(web_server, "CHUNK_BYTES", 4)  # giới hạn một request của server
    web.login()
    few = web.request("POST", "/api/uploads", json_body={"room": "p3", "kind": "photos",
                                                         "files": [{"name": f"{i}.jpg", "size": 1} for i in range(4)]})
    assert few.status == 400 and "cần ít nhất 10" in few.json["error"]
    big = web.request("POST", "/api/uploads", json_body={"room": "p3", "kind": "video",
                                                         "files": [{"name": "a.mp4", "size": 2_000_000}]})
    assert big.status == 400 and "vượt giới hạn" in big.json["error"]
    assert web.request("POST", "/api/uploads", json_body={"room": "web", "kind": "video",
                                                          "files": [{"name": "a.mp4", "size": 3}]}).status == 400
    assert web.request("POST", "/api/uploads", body=b"{not json", headers={"Content-Type": "application/json"}).status == 400
    created = web.request("POST", "/api/uploads", json_body={"room": "p4", "kind": "video",
                                                             "files": [{"name": "a.mp4", "size": 3}]}).json
    huge = web.request("PUT", f"/api/uploads/{created['id']}/files/0?offset=0", body=b"x" * 5)
    assert huge.status == 413 and huge.getheader("Connection") == "close"  # body chưa đọc: không dùng lại kết nối
    assert web.request("PUT", f"/api/uploads/{created['id']}/files/0", body=b"x").status == 400  # thiếu offset
    assert web.request("POST", f"/api/uploads/{created['id']}/finish", json_body={}).status == 409  # chưa đủ
    assert web.request("DELETE", f"/api/uploads/{created['id']}").status == 200
    assert web.calls() == []


# ------------------------------------------------------------------------------- tiến độ, log, hủy, dựng lại
def test_progress_log_cancel_rebuild_and_ply_download(web):
    web.login()
    log = web.ws / "outputs" / "logs" / "build_room_p5.log"
    log.parent.mkdir(parents=True)
    log.write_text("line 1\nVòng 10/100\r Vòng 50/100\n", encoding="utf-8")
    file = web.root / "p5" / "progress.json"
    progress.queue(file, "p5", "photos", "room_p5")
    progress.start(file, "p5", job="room_p5", log=str(log))
    progress.set_stage(file, "colmap")
    progress.fail(file, "COLMAP đánh giá FAIL: không train")
    state = web.ws / "outputs" / "jobs" / "room_p5.state"
    state.parent.mkdir(parents=True)
    state.write_text("name=room_p5\ntoken=gone\nstate=FAILED\nexit=1\n")
    rooms = web.request("GET", "/api/rooms").json["rooms"]
    assert [r["id"] for r in rooms] == ["p5", "p1"] and rooms[0]["status"] == "failed"
    detail = web.request("GET", "/api/rooms/p5").json
    assert detail["status"] == "failed" and "log" not in detail["progress"] and detail["has_ply"] is False
    assert detail["progress"]["stages"][1]["status"] == "failed"
    assert web.request("GET", "/api/rooms/p5/log?lines=5").json == {"lines": ["line 1", " Vòng 50/100"],
                                                                    "file": "build_room_p5.log"}
    assert web.request("GET", "/api/rooms/none").status == 404
    assert web.request("POST", "/api/rooms/none/cancel", json_body={}).status == 404
    assert web.request("POST", "/api/rooms/p5/cancel", json_body={}).status == 202  # job.sh stop: việc không còn chạy
    # dựng lại: chỉ phòng tải lên từ web (có nguồn trong data/uploads/)
    assert web.request("POST", "/api/rooms/p5/rebuild", json_body={}).status == 409
    source = web.ws / "data" / "uploads" / "p5"
    source.mkdir(parents=True)
    for i in range(12):
        (source / f"{i}.jpg").write_bytes(b"x")
    rebuilt = web.request("POST", "/api/rooms/p5/rebuild", json_body={})
    assert rebuilt.status == 202 and web.calls() == [f"{source} p5 --name p5 --background"]
    # PLY đầy đủ: chỉ khi train xong
    assert web.request("GET", "/download/p5/model.ply").status == 404
    model = web.root / "p5" / "model"
    (model / "ply").mkdir(parents=True)
    (model / "ply" / "point_cloud_29999.ply").write_bytes(b"ply data")
    (model / "train_meta.json").write_text(json.dumps({"status": "OK", "files": {"ply": "ply/point_cloud_29999.ply"}}))
    ply = web.request("GET", "/download/p5/model.ply")
    assert ply.status == 200 and ply.data == b"ply data" and 'filename="p5.ply"' in ply.getheader("Content-Disposition")


def test_rebuild_waits_for_the_old_job_and_clears_a_stale_progress(web):
    web.login()
    source = web.ws / "data" / "uploads" / "p6"
    source.mkdir(parents=True)
    for i in range(12):
        (source / f"{i}.jpg").write_bytes(b"x")
    (source / uploads.UPLOAD_INFO).write_text(json.dumps({"name": "Phòng sáu"}), encoding="utf-8")
    file = web.root / "p6" / "progress.json"
    progress.queue(file, "p6", "photos", "room_p6")
    progress.start(file, "p6", job="room_p6")
    progress.set_stage(file, "train")
    token = f"tok-{uuid.uuid4().hex}"
    state = web.ws / "outputs" / "jobs" / "room_p6.state"
    state.parent.mkdir(parents=True)
    state.write_text(f"name=room_p6\ntoken={token}\nstate=RUNNING\n")
    holder = subprocess.Popen(["sleep", "60"], env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                                    "SR3D_JOB_TOKEN": token})
    try:
        assert web.request("GET", "/api/rooms/p6").json["status"] == "running"
        assert web.request("POST", "/api/rooms/p6/rebuild", json_body={}).status == 409
        progress.fail(file, "Đã dừng", stopped=True)  # trap của build_room.sh đã ghi, tiến trình chưa thoát hẳn
        refused = web.request("POST", "/api/rooms/p6/rebuild", json_body={})
        assert refused.status == 409 and "đang dừng" in refused.json["error"] and web.calls() == []
    finally:
        holder.kill()
        holder.wait()
    # pod khởi động lại giữa lúc train: progress.json còn ghi "đang chạy" mà không còn tiến trình nào
    progress.queue(file, "p6", "photos", "room_p6")
    progress.start(file, "p6", job="room_p6")
    progress.set_stage(file, "train")
    assert web.request("GET", "/api/rooms/p6").json["status"] == "lost"
    rebuilt = web.request("POST", "/api/rooms/p6/rebuild", json_body={})
    assert rebuilt.status == 202, rebuilt.json
    assert web.calls() == [f"{source} p6 --name Phòng sáu --background"]  # tên đặt lúc tải lên, không phải id
    data = progress.read(file)  # ghi "đã dừng" trước, để build_room.sh xếp hàng lại được
    assert data["status"] == "stopped" and data["stage"] == "train" and data["stages"][2]["status"] == "stopped"


def test_a_view_only_server_shows_the_library_but_builds_nothing(tmp_path):
    web_dir = tmp_path / "copy" / "web"
    (web_dir / "rooms").mkdir(parents=True)
    (web_dir / "manifest.json").write_text('{"format_version": 1, "rooms": []}')
    args = web_server.parse_args(["--web-dir", str(web_dir), "--view-only", "--host", "127.0.0.1", "--no-auth"])
    settings = web_server.settings_from(args)
    assert settings.view_only and settings.token is None and settings.product_root == web_dir.resolve().parent
    httpd, thread = start_server(settings)
    try:
        client = Client(httpd.server_address[1])
        status = client.request("GET", "/api/status").json
        assert status["mode"] == "view" and status["auth"] == {"required": False, "ok": True}
        assert client.request("GET", "/api/rooms").json == {"rooms": []}
        refused = client.request("POST", "/api/uploads", json_body={"room": "x", "kind": "video", "files": []})
        assert refused.status == 503 and "chỉ để xem" in refused.json["error"]
        # không mật khẩu: chỉ nhận Host của máy này (trang lạ trỏ tên miền của nó về 127.0.0.1 không đọc được gì)
        for host, code in (("evil.example", 403), ("evil.example:8000", 403), ("localhost:8000", 200),
                           ("127.0.0.1", 200), ("[::1]:8000", 200)):
            assert client.request("GET", "/api/rooms", headers={"Host": host}).status == code, host
        assert client.request("GET", "/", headers={"Host": "evil.example"}).status == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def test_no_password_is_only_allowed_to_view_on_localhost():
    with pytest.raises(SystemExit, match="127.0.0.1"):
        web_server.settings_from(web_server.parse_args(["--no-auth", "--host", "0.0.0.0", "--view-only",
                                                        "--web-dir", "/tmp"]))
    with pytest.raises(SystemExit, match="--view-only"):  # dựng phòng luôn cần mật khẩu
        web_server.settings_from(web_server.parse_args(["--no-auth", "--host", "127.0.0.1", "--web-dir", "/tmp"]))


def test_a_password_file_is_created_once_with_private_permissions(tmp_path, monkeypatch):
    monkeypatch.delenv("SR3D_WEB_TOKEN", raising=False)
    token = web_server.read_token(tmp_path, None)
    path = tmp_path / "outputs" / "product" / "server" / "token"
    assert len(token) >= 16 and path.read_text().strip() == token and (path.stat().st_mode & 0o777) == 0o600
    assert web_server.read_token(tmp_path, None) == token
    monkeypatch.setenv("SR3D_WEB_TOKEN", "from-env")
    assert web_server.read_token(tmp_path, None) == "from-env"


# ------------------------------------------------------- tải lên -> build_room.sh thật -> phòng trong thư viện
def test_an_upload_from_the_web_goes_through_the_real_build_room(tmp_path, monkeypatch):
    """Trình duyệt tải video lên -> server -> scripts/build_room.sh thật (việc nền qua job.sh) -> train_main.py và
    export_web.py thật (gsplat giả) -> phòng có trong thư viện, mô hình tải được. Chỉ GPU, conda, COLMAP là giả."""
    sh = make_build_env(tmp_path)
    try:
        write_stub(sh.repo, "scripts/pipeline/prepare_scene.sh", REAL_PREPARE)
        checkout, installed, head = make_gsplat_checkout(tmp_path / "gs")
        data = yaml.safe_load((REPO / "configs" / "product.yaml").read_text(encoding="utf-8"))
        data["trainer"].update(dir=str(checkout), commit=head)
        data["training"]["max_steps"] = 40
        config = tmp_path / "product.yaml"
        config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        for key, value in {**sh.env, "REAL_STEPS": "1", "PYTHONPATH": str(installed.parent)}.items():
            monkeypatch.setenv(key, value)
        root = sh.ws / "outputs" / "product"
        settings = web_server.Settings(ws=sh.ws, web_dir=root / "web", product_root=root, token=TOKEN,
                                       build_script=sh.repo / "scripts" / "build_room.sh", config_file=config)
        httpd, thread = start_server(settings)
        try:
            client = Client(httpd.server_address[1])
            client.login()
            _, finished = upload(client, "phong_web", "video", [("quay phong.mp4", b"video-bytes" * 100)],
                                 name="Phòng tải lên")
            assert finished.status == 202, finished.json
            state = sh.wait_state("room_phong_web", "DONE", "FAILED", timeout=180)
            assert state["state"] == "DONE", sh.log("room_phong_web")
            detail = client.request("GET", "/api/rooms/phong_web").json
            assert detail["status"] == "done" and detail["name"] == "Phòng tải lên" and detail["has_ply"]
            assert [s["status"] for s in detail["progress"]["stages"]] == ["done"] * 4
            assert all(s["started_at"] and s["finished_at"] for s in detail["progress"]["stages"])
            entry = detail["entry"]
            model = client.request("GET", f"/web/{entry['files']['model']}?v={entry['version']}")
            assert model.status == 200 and len(model.data) == entry["size_bytes"] == 7 * 32
            assert client.request("GET", "/download/phong_web/model.ply").status == 200
            prepare = [line for line in (tmp_path / "calls.txt").read_text().splitlines() if line.startswith("prepare_scene")]
            assert prepare == [f"prepare_scene {sh.ws / 'data' / 'uploads' / 'phong_web' / 'quay_phong.mp4'} phong_web"]
            assert client.request("GET", "/api/rooms/phong_web/log").json["lines"]
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(5)
    finally:
        sh.cleanup()


def test_the_launcher_script_documents_its_commands():
    text = (REPO / "scripts" / "web_server.sh").read_text(encoding="utf-8")
    for command in ("start", "status", "stop", "log", "run"):
        assert f"web_server.sh {command}" in text
    assert os.access(REPO / "scripts" / "web_server.sh", os.X_OK)
    assert Path(web_server.STATIC / "index.html").is_file()
    assert COMMANDS["web"][0] == "scripts/web_server.sh"


def test_the_launcher_starts_reports_and_stops_the_server(tmp_path):
    """scripts/web_server.sh thật trên một repo giả (thư viện xem 3D giả, đúng phiên bản của lock: không cần mạng)."""
    repo = make_fake_repo(tmp_path, ["web_server.sh", "build_room.sh"], with_src=True, with_configs=True)
    shutil.copytree(REPO / "app", repo / "app", ignore=shutil.ignore_patterns("__pycache__", "vendor"))
    lock = json.loads((REPO / "configs" / "environment" / "web_viewer.lock.json").read_text(encoding="utf-8"))
    vendor = repo / "app" / "web" / "static" / "vendor"
    for package in lock["packages"].values():
        for target in package["files"].values():
            (vendor / target).parent.mkdir(parents=True, exist_ok=True)
            (vendor / target).write_text("// stub", encoding="utf-8")
    (vendor / "VERSIONS.json").write_text(json.dumps(
        {name: {"version": p["version"], "integrity": p["integrity"]} for name, p in lock["packages"].items()}))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    sh = ShellEnv(tmp_path, repo=repo)
    env = {"PORT": str(port), "RUNPOD_POD_ID": "pod123"}
    script = str(repo / "scripts" / "web_server.sh")
    try:
        out = sh.run(script, "start", env=env, timeout=90)
        assert out.returncode == 0, out.stdout + out.stderr
        assert f"https://pod123-{port}.proxy.runpod.net/" in out.stdout and "Mật khẩu:" in out.stdout
        token = (sh.ws / "outputs" / "product" / "server" / "token").read_text().strip()
        assert token in out.stdout and (sh.ws / "outputs" / "product" / "server" / "token").stat().st_mode & 0o077 == 0
        client = Client(port)
        assert client.login(token).status == 200 and client.request("GET", "/api/rooms").json == {"rooms": []}
        assert "đang chạy" in sh.run(script, "start", env=env).stdout  # không chạy bản thứ hai
        assert sh.run(script, "status", env=env).returncode == 0
        other = sh.run(script, "status", env={"RUNPOD_POD_ID": "pod123"})  # không đặt PORT: vẫn in cổng đang dùng
        assert f"cổng {port}" in other.stdout and f"https://pod123-{port}.proxy.runpod.net/" in other.stdout
        assert "Mật khẩu" in sh.run(script, "log", env=env).stdout
        assert "Đã dừng server" in sh.run(script, "stop", env=env).stdout
        assert sh.run(script, "status", env=env).returncode == 1
        pid_file = sh.ws / "outputs" / "product" / "server" / "server.pid"
        assert not pid_file.exists()
        # run: tiền cảnh, không mang token của việc nền đang gọi nó (job.sh sẽ coi server là tiến trình của việc đó)
        proc = subprocess.Popen(["bash", script, "run"], env={**sh.env, **env, "SR3D_JOB_TOKEN": "outer-job"},
                                cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 60
            while not pid_file.exists() and time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.1)
            assert int(pid_file.read_text()) == proc.pid  # exec: chính tiến trình đã gọi
            assert b"SR3D_JOB_TOKEN=" not in Path(f"/proc/{proc.pid}/environ").read_bytes()
            again = sh.run(script, "run", env=env)
            assert again.returncode == 1 and "đang chạy" in again.stdout
        finally:
            proc.terminate()
            proc.wait(10)
    finally:
        sh.run(script, "stop", env=env)
        sh.cleanup()
