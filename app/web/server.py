"""Server của trang web sản phẩm: tải video / ảnh lên, dựng phòng trên GPU của pod, theo dõi tiến độ, xem mô hình 3D.

Chạy trên pod (từ thư mục repo; `bash scripts/web_server.sh start` làm sẵn: chọn python, tải thư viện xem 3D,
tạo mật khẩu, chạy nền):
  python3 -m app.web.server --ws /workspace --port 8000
rồi mở cổng HTTP 8000 của pod trên trang Runpod và vào https://<POD_ID>-8000.proxy.runpod.net.
Chỉ để xem (vd. trên laptop lúc bảo vệ, không dựng phòng), với bản chép của thư mục web/:
  python3 -m app.web.server --web-dir <thư mục web> --view-only --host 127.0.0.1 --no-auth

Dựng phòng vẫn là đúng lệnh của luồng sản phẩm: server gọi `scripts/build_room.sh <nguồn> <phòng> --name ...
--background` (việc nền room_<phòng> của scripts/pod/job.sh), rồi chỉ đọc lại các file luồng đó ghi
(docs/design/product.md mục 4). Server không giữ khóa GPU và không train gì.

Địa chỉ (đường dẫn trên volume không gửi cho trình duyệt):
  /, /app/...                         trang web (app/web/static/), không cần đăng nhập
  /web/...                            thư mục web của luồng sản phẩm: manifest.json, rooms/<phòng>/...
  /download/<phòng>/model.ply         mô hình đầy đủ (PLY 3DGS) để mở bằng phần mềm khác
  /api/status                         server, GPU, giới hạn tải lên
  /api/login, /api/logout             mật khẩu -> cookie
  /api/rooms, /api/rooms/<phòng>      thư viện và tiến độ; .../log: cuối log; .../cancel: hủy việc;
                                      .../rebuild: dựng lại phòng đã tải lên từ web
  /api/uploads, /api/uploads/<id>     phiên tải lên; .../files/<i>?offset=N: một phần; .../finish: dựng phòng

Bảo mật: link proxy của Runpod là công khai, nên mọi thứ trừ khung trang (/, /app/, /api/status) cần mật khẩu.
Mật khẩu ở $WS/outputs/product/server/token (tạo ngẫu nhiên lần chạy đầu, quyền 600) hoặc biến SR3D_WEB_TOKEN.
Đăng nhập xong server cấp một phiên ngẫu nhiên (cookie HttpOnly, SameSite=Strict, hết hạn sau 30 ngày; đăng xuất hoặc
khởi động lại server thì hết hiệu lực). Mọi request ghi (POST / PUT / DELETE) phải có header `X-SR3D: 1` (trang của site
khác không gửi được header này, server không bật CORS). Không mật khẩu (--no-auth) chỉ cho chế độ chỉ xem trên
127.0.0.1, và chỉ nhận Host là 127.0.0.1 / localhost (chống DNS rebinding).
"""

from __future__ import annotations

import argparse
import hmac
import http.cookies
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.input import sources
from src.product import config as product_config
from src.product import progress as progress_file

from . import rooms as room_info
from .uploads import CHUNK_BYTES, MAX_TOTAL_BYTES, VIDEO_EXTS, UploadError, UploadStore

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
API_VERSION = 1
COOKIE = "sr3d_session"
COOKIE_DAYS = 30
MAX_JSON_BYTES = 1 << 20
MAX_SESSIONS = 200
LOGIN_DELAY = 1.0  # giây chờ sau mỗi lần sai mật khẩu
WRITE_HEADER = "X-SR3D"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
DIGITS = re.compile(r"[0-9]{1,15}")  # chỉ chữ số ASCII ("²".isdigit() là True nhưng int("²") lỗi)
TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8", ".map": "application/json", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".ico": "image/x-icon",
    ".woff2": "font/woff2", ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8",
    ".splat": "application/octet-stream", ".ply": "application/octet-stream",
}


@dataclass
class Settings:
    ws: Path
    web_dir: Path
    product_root: Path
    token: str | None            # None: không cần đăng nhập (chỉ cho chế độ chỉ xem trên 127.0.0.1)
    view_only: bool = False      # chỉ xem: không tải lên, không dựng phòng
    build_script: Path = REPO / "scripts" / "build_room.sh"
    job_script: Path = REPO / "scripts" / "pod" / "job.sh"
    config_file: Path | None = None
    max_upload: int = MAX_TOTAL_BYTES
    build_timeout: float = 80  # giây chờ build_room.sh xếp việc; < 100 giây của proxy Runpod
    pipeline: dict | None = None  # cấu hình dựng phòng (configs/product.yaml) cho trang web hiển thị; None: chỉ xem

    @property
    def logs_dir(self) -> Path:
        return self.ws / "outputs" / "logs"


# Mặc định của bước chuẩn bị ảnh mà build_room.sh dùng (src/input/frames.py: DEFAULT_TARGET, DEFAULT_MAX_SIDE; test
# kiểm tra hai số này khớp). Ghi lại ở đây để server không phải import numpy.
FRAMES_DEFAULTS = {"video_max_frames": 250, "max_side": 1600}


def pipeline_summary(cfg: dict) -> dict:
    """Phần cấu hình dựng phòng mà trang web hiển thị (đọc từ configs/product.yaml, không tự đặt số)."""
    model, training, trainer = cfg["model"], cfg["training"], cfg["trainer"]
    return {"model": model["name"], "description": model.get("description"), "cap_max": model.get("cap_max"),
            "max_steps": training.get("max_steps"), "init": training.get("init_type"),
            "data_factor": training.get("data_factor"), "trainer": "gsplat",
            "trainer_commit": str(trainer.get("commit") or "")[:7], "web_format": cfg["web"].get("format"),
            "frames": dict(FRAMES_DEFAULTS)}


def say(text: str) -> None:
    print(f"[web {time.strftime('%H:%M:%S')}] {text}", flush=True)


def same_text(a: str, b: str) -> bool:
    """So sánh thời gian hằng; chấp nhận mọi chuỗi (compare_digest với str chỉ nhận ASCII; JSON có thể chứa
    surrogate lẻ như "\\ud800", nên mã hóa bằng surrogatepass)."""
    return hmac.compare_digest(a.encode("utf-8", "surrogatepass"), b.encode("utf-8", "surrogatepass"))


# ================================================================================================ ứng dụng
class App:
    """Trạng thái dùng chung của server: cấu hình, phiên tải lên, cookie, bộ nhớ đệm trạng thái GPU."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.store = UploadStore(settings.ws, settings.product_root, settings.max_upload)
        self._sessions: dict[str, float] = {}  # phiên đăng nhập -> hạn (time.time())
        self._session_lock = threading.Lock()
        self._gpu: tuple[float, dict] | None = None
        self._gpu_name: tuple[float, str | None] | None = None
        self._gpu_lock = threading.Lock()

    # ------------------------------------------------------------------------------------- đăng nhập
    def authorized(self, headers) -> bool:
        if self.s.token is None:
            return True
        bearer = headers.get("Authorization", "")
        if bearer.startswith("Bearer ") and same_text(bearer[7:].strip(), self.s.token):
            return True
        session = self.session_of(headers)
        if not session:
            return False
        with self._session_lock:
            expires = self._sessions.get(session)
            return expires is not None and expires > time.time()

    @staticmethod
    def session_of(headers) -> str | None:
        try:
            morsel = http.cookies.SimpleCookie(headers.get("Cookie", "")).get(COOKIE)
        except http.cookies.CookieError:
            return None
        return morsel.value if morsel is not None and morsel.value else None

    def new_session(self) -> str:
        session = secrets.token_urlsafe(24)
        now = time.time()
        with self._session_lock:
            self._sessions = {key: exp for key, exp in self._sessions.items() if exp > now}
            while len(self._sessions) >= MAX_SESSIONS:  # giữ số phiên có hạn: bỏ phiên sắp hết hạn nhất
                self._sessions.pop(min(self._sessions, key=self._sessions.get))
            self._sessions[session] = now + COOKIE_DAYS * 86400
        return session

    def end_session(self, headers) -> None:
        session = self.session_of(headers)
        if session:
            with self._session_lock:
                self._sessions.pop(session, None)

    def token_ok(self, token) -> bool:
        return self.s.token is None or (isinstance(token, str) and same_text(token.strip(), self.s.token))

    def host_ok(self, headers) -> bool:
        """Không mật khẩu: chỉ nhận Host là máy này (trang lạ trỏ tên miền về 127.0.0.1 không gọi được API)."""
        if self.s.token is not None:
            return True
        host = headers.get("Host", "")
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host[1:].split("]", 1)[0]
        return name in LOCAL_HOSTS

    # ------------------------------------------------------------------------------------------- GPU
    def _env(self) -> dict:
        env = {key: value for key, value in os.environ.items() if not key.startswith("SR3D_GPU_LOCK")}
        env["WS"] = str(self.s.ws)
        if self.s.config_file:
            env["PRODUCT_CONFIG"] = str(self.s.config_file)
        return env

    def gpu_status(self) -> dict:
        """{name, busy}: tên GPU (nvidia-smi, None khi không có) và có việc nền nào đang chạy / xếp hàng (job.sh busy)."""
        with self._gpu_lock:
            now = time.monotonic()
            if self._gpu and now - self._gpu[0] < 3:
                return self._gpu[1]
            if self._gpu_name is None or now - self._gpu_name[0] > 60:
                name = None
                if shutil.which("nvidia-smi"):
                    try:
                        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                             capture_output=True, text=True, timeout=10)
                        name = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() \
                            else None
                    except (OSError, subprocess.SubprocessError):
                        name = None
                self._gpu_name = (now, name)
            busy = False
            if not self.s.view_only and self.s.job_script.is_file():
                try:
                    busy = subprocess.run(["bash", str(self.s.job_script), "busy"], env=self._env(),
                                          capture_output=True, timeout=20).returncode == 0
                except (OSError, subprocess.SubprocessError):
                    busy = False
            status = {"name": self._gpu_name[1], "busy": busy}
            self._gpu = (now, status)
            return status

    # ------------------------------------------------------------------------------------ dựng phòng
    def start_build(self, source: Path, room: str, name: str) -> tuple[bool, str, bool]:
        """Gọi build_room.sh --background. (thành công, thông điệp cho người dùng, đang xếp hàng chờ GPU)."""
        cmd = ["bash", str(self.s.build_script), str(source), room, "--name", name, "--background"]
        try:
            out = subprocess.run(cmd, cwd=REPO, env=self._env(), capture_output=True, text=True,
                                 timeout=self.s.build_timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            # job.sh có thể đã tạo việc nền trước khi hết giờ: còn việc thì coi như đã xếp, không xóa nguồn của nó
            job = room_info.job_state(self.s.ws, f"room_{room}")
            if job and job["alive"]:
                return True, "Đã xếp việc dựng phòng (server trả lời chậm): xem tiến độ", True
            return False, "Không bắt đầu được việc dựng phòng: build_room.sh không trả lời kịp", False
        except OSError as error:
            return False, f"Không chạy được build_room.sh: {error}", False
        lines = [line.strip() for line in (out.stdout + "\n" + out.stderr).splitlines() if line.strip()]
        say(f"build_room {room}: exit {out.returncode} | {' | '.join(lines[-3:])}")
        if out.returncode != 0:
            errors = [line[len("ERROR: "):] for line in lines if line.startswith("ERROR: ")]
            return False, errors[-1] if errors else (lines[-1] if lines else f"build_room.sh lỗi (mã {out.returncode})"), False
        if any("không chạy thêm bản thứ hai" in line for line in lines):  # job.sh: việc cũ của phòng chưa dừng hẳn
            return False, "Việc dựng trước của phòng này vẫn đang chạy hoặc đang dừng: thử lại sau vài giây", False
        with self._gpu_lock:
            self._gpu = None  # trạng thái GPU đã đổi
        queued = any("xếp hàng" in line for line in lines)
        return True, ("Đang chờ GPU: có việc khác đang chạy, phòng sẽ tự dựng khi việc đó xong" if queued
                      else "Đã bắt đầu dựng phòng"), queued

    def cancel(self, room: str) -> None:
        job = f"room_{room}"
        if not room_info.job_values(self.s.ws, job):
            raise UploadError("Phòng này không có việc dựng nào để hủy", 404)

        def stop() -> None:
            try:
                out = subprocess.run(["bash", str(self.s.job_script), "stop", job], env=self._env(),
                                     capture_output=True, text=True, timeout=120)
                say(f"stop {job}: exit {out.returncode} | {out.stdout.strip()[-200:]}")
            except (OSError, subprocess.SubprocessError) as error:
                say(f"stop {job}: {error}")
            with self._gpu_lock:
                self._gpu = None

        threading.Thread(target=stop, name=f"stop-{job}", daemon=True).start()


# ================================================================================================= request
ROUTES: list[tuple[str, re.Pattern, str, bool]] = []  # (method, path, tên hàm của Handler, cần đăng nhập)


def route(method: str, pattern: str, auth: bool = True):
    def wrap(func):
        ROUTES.append((method, re.compile(pattern), func.__name__, auth))
        return func
    return wrap


ROOM = r"([A-Za-z0-9][A-Za-z0-9._-]*)"
UPLOAD = r"([0-9a-f]{16})"


class Handler(BaseHTTPRequestHandler):
    server_version = "SpatialRoom3D"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 120
    app: App  # gán trong make_server

    def log_message(self, format, *args):  # noqa: A002 - tên tham số của BaseHTTPRequestHandler
        pass  # trang web hỏi tiến độ mỗi vài giây: không ghi từng request; lỗi ghi qua say()

    # -------------------------------------------------------------------------------------- điều phối
    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        self._body_read = False
        try:
            url = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(url.path)
            self.query = urllib.parse.parse_qs(url.query)
            for want, pattern, name, auth in ROUTES:
                match = pattern.fullmatch(path)
                if match is None or want != ("GET" if method == "HEAD" else method):
                    continue
                if not self.app.host_ok(self.headers):
                    return self._error("Host không hợp lệ", 403)
                if auth and not self.app.authorized(self.headers):
                    return self._error("Cần đăng nhập bằng mật khẩu của server", 401)
                if method in ("POST", "PUT", "DELETE") and self.headers.get(WRITE_HEADER) != "1":
                    return self._error(f"Thiếu header {WRITE_HEADER}", 403)
                return getattr(self, name)(*match.groups())
            if any(pattern.fullmatch(path) for _m, pattern, _n, _a in ROUTES):
                return self._error("Phương thức không hỗ trợ", 405)
            return self._error("Không có địa chỉ này", 404)
        except UploadError as error:
            self._error(str(error), error.status, **error.extra)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - lỗi bất ngờ: ghi log, trả 500, server chạy tiếp
            say(f"ERROR {method} {self.path[:200]!r}\n{traceback.format_exc(limit=6)}")
            try:
                self._error("Lỗi server (xem log của server trên pod)", 500)
            except OSError:
                pass

    # ------------------------------------------------------------------------------------- trả lời
    def _declared_length(self) -> int | None:
        """Content-Length: None khi không có, -1 khi không hợp lệ."""
        value = self.headers.get("Content-Length")
        if value is None:
            return None
        value = value.strip()
        return int(value) if DIGITS.fullmatch(value) else -1

    def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        declared = self._declared_length()
        if not self._body_read and declared not in (None, 0):
            self.close_connection = True  # body chưa đọc: không dùng lại kết nối (phần còn lại sẽ lẫn vào request sau)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200, headers: dict | None = None) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, TYPES[".json"], {"Cache-Control": "no-store", **(headers or {})})

    def _error(self, message: str, status: int, **extra) -> None:
        self._json({"error": message, **extra}, status)

    def _read_body(self, limit: int) -> bytes:
        length = self._declared_length()
        if length is None:
            raise UploadError("Thiếu Content-Length", 411)
        if length < 0:
            raise UploadError("Content-Length không hợp lệ", 400)
        if length > limit:
            raise UploadError("Dữ liệu quá lớn cho một request", 413)
        data = self.rfile.read(length)
        self._body_read = True
        if len(data) != length:
            raise UploadError("Kết nối bị ngắt giữa chừng", 400)
        return data

    def _read_json(self) -> dict:
        body = self._read_body(MAX_JSON_BYTES)  # ngoài try: UploadError (411, 413...) cũng là ValueError
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            raise UploadError("JSON không hợp lệ", 400) from None
        if not isinstance(data, dict):
            raise UploadError("JSON không hợp lệ", 400)
        return data

    def _file(self, base: Path, relative: str, cache: str = "no-cache", download: str | None = None) -> None:
        """Gửi một file trong `base` (không ra ngoài base, không file ẩn), có ETag và Range (một đoạn)."""
        if any(part.startswith(".") or part == "" for part in relative.split("/")) or len(relative) > 1024:
            return self._error("Không có file này", 404)
        try:  # đường dẫn lạ (ký tự NUL, phần tên quá dài) cũng là "không có file", không phải lỗi server
            root = base.resolve()
            path = (root / relative).resolve()
            handle = open(path, "rb") if root in path.parents and path.is_file() else None
        except (OSError, ValueError):
            handle = None
        if handle is None:
            return self._error("Không có file này", 404)
        with handle:
            return self._send_file(handle, path, cache, download)

    def _send_file(self, handle, path: Path, cache: str, download: str | None) -> None:
        stat = os.fstat(handle.fileno())  # của chính file đã mở (export_web có thể thay file trong lúc này)
        etag = f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
        headers = {"ETag": etag, "Cache-Control": cache, "Accept-Ranges": "bytes",
                   "Last-Modified": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stat.st_mtime))}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{download}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, end = 0, stat.st_size - 1
        status = 200
        ranges = re.fullmatch(r"bytes=([0-9]{0,15})-([0-9]{0,15})", self.headers.get("Range", "").strip())
        if ranges and stat.st_size > 0 and (ranges.group(1) or ranges.group(2)):
            if ranges.group(1):
                start = int(ranges.group(1))
                end = min(int(ranges.group(2)), end) if ranges.group(2) else end
            else:
                start = max(0, stat.st_size - int(ranges.group(2)))
            if start > end or start >= stat.st_size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{stat.st_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{stat.st_size}"
        length = end - start + 1 if stat.st_size else 0
        self.send_response(status)
        self.send_header("Content-Type", TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command == "HEAD" or length == 0:
            return None
        handle.seek(start)
        remaining = length
        while remaining > 0:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                self.close_connection = True  # file ngắn đi giữa chừng: báo kết thúc bằng cách đóng kết nối
                break
            self.wfile.write(block)
            remaining -= len(block)
        return None

    def _writable(self) -> None:
        if self.app.s.view_only:
            raise UploadError("Server này chỉ để xem: dựng phòng trên server chạy ở pod", 503)

    # ===================================================================================== trang web
    @route("GET", r"/", auth=False)
    def page_index(self):
        return self._file(STATIC, "index.html")

    @route("GET", r"/app/(.+)", auth=False)
    def page_static(self, relative):
        return self._file(STATIC, relative)

    @route("GET", r"/web/(.+)")
    def web_file(self, relative):
        # model.splat?v=<version>: phiên bản nằm trong URL, nên trình duyệt giữ được bản đã tải
        cache = "private, max-age=86400" if "v" in self.query else "no-cache"
        return self._file(self.app.s.web_dir, relative, cache=cache)

    @route("GET", rf"/download/{ROOM}/model\.ply")
    def download_ply(self, room):
        path = room_info.ply_file(self.app.s.product_root, room)
        if path is None:
            return self._error("Phòng này chưa có mô hình PLY trên server", 404)
        return self._file(path.parent, path.name, download=f"{room}.ply")

    # ============================================================================================ API
    @route("GET", r"/api/status", auth=False)
    def api_status(self):
        ok = self.app.authorized(self.headers)
        data = {"api": API_VERSION, "mode": "view" if self.app.s.view_only else "pod",
                "auth": {"required": self.app.s.token is not None, "ok": ok}}
        if ok:
            data["gpu"] = self.app.gpu_status()
            if self.app.s.pipeline:
                data["pipeline"] = self.app.s.pipeline
            data["limits"] = {"chunk_bytes": CHUNK_BYTES, "max_upload_bytes": self.app.s.max_upload,
                              "min_photos": sources.MIN_PHOTOS, "recommended_photos": sources.RECOMMENDED_PHOTOS,
                              "photo_exts": sorted(sources.PHOTO_EXTS), "video_exts": list(VIDEO_EXTS)}
        return self._json(data)

    @route("POST", r"/api/login", auth=False)
    def api_login(self):
        data = self._read_json()
        if not self.app.token_ok(data.get("token")):
            time.sleep(LOGIN_DELAY)  # làm chậm đoán mật khẩu
            return self._error("Sai mật khẩu", 401)
        secure = (self.headers.get("X-Forwarded-Proto", "").lower() == "https"
                  or self.headers.get("Host", "").endswith(".proxy.runpod.net"))
        session = self.app.new_session() if self.app.s.token is not None else ""
        cookie = (f"{COOKIE}={session}; Path=/; HttpOnly; SameSite=Strict; "
                  f"Max-Age={COOKIE_DAYS * 86400}{'; Secure' if secure else ''}")
        return self._json({"ok": True}, headers={"Set-Cookie": cookie})

    @route("POST", r"/api/logout", auth=False)
    def api_logout(self):
        self._read_body(MAX_JSON_BYTES)
        self.app.end_session(self.headers)
        return self._json({"ok": True}, headers={"Set-Cookie": f"{COOKIE}=; Path=/; Max-Age=0; SameSite=Strict"})

    @route("GET", r"/api/rooms")
    def api_rooms(self):
        s = self.app.s
        return self._json({"rooms": room_info.list_rooms(s.web_dir, s.product_root, s.ws)})

    @route("GET", rf"/api/rooms/{ROOM}")
    def api_room(self, room):
        s = self.app.s
        detail = room_info.room_detail(s.web_dir, s.product_root, s.ws, room)
        if detail is None:
            return self._error(f"Không có phòng '{room}'", 404)
        if detail.get("progress"):
            detail["progress"] = {key: value for key, value in detail["progress"].items() if key != "log"}
        detail["has_ply"] = room_info.ply_file(s.product_root, room) is not None
        return self._json(detail)

    @route("GET", rf"/api/rooms/{ROOM}/log")
    def api_room_log(self, room):
        s = self.app.s
        detail = room_info.room_detail(s.web_dir, s.product_root, s.ws, room)
        log = ((detail or {}).get("progress") or {}).get("log")
        try:
            lines = int((self.query.get("lines") or ["200"])[0])
        except ValueError:
            lines = 200
        return self._json({"lines": room_info.log_tail(log, s.logs_dir, lines),
                           "file": Path(log).name if log else None})

    @route("POST", rf"/api/rooms/{ROOM}/cancel")
    def api_room_cancel(self, room):
        self._read_body(MAX_JSON_BYTES)
        self._writable()
        self.app.cancel(room)
        return self._json({"ok": True, "message": "Đang dừng việc dựng phòng (tối đa khoảng 30 giây)"}, 202)

    @route("POST", rf"/api/rooms/{ROOM}/rebuild")
    def api_room_rebuild(self, room):
        """Dựng lại phòng đã tải lên từ web (lỗi, bị dừng, pod tắt giữa chừng): ảnh và COLMAP đã xong thì dùng lại."""
        self._read_body(MAX_JSON_BYTES)
        self._writable()
        s = self.app.s
        detail = room_info.room_detail(s.web_dir, s.product_root, s.ws, room)
        if detail is None:
            raise UploadError(f"Không có phòng '{room}'", 404)
        if detail["status"] in ("queued", "running") or (detail.get("job") or {}).get("alive"):
            raise UploadError("Phòng đang được dựng, hoặc việc trước vẫn đang dừng: thử lại sau vài giây", 409)
        source = self.app.store.source_of(room)
        if source is None:
            raise UploadError("Phòng này không được tải lên từ trang web: dựng lại bằng scripts/build_room.sh trên pod", 409)
        raw = (detail.get("progress") or {}).get("status")
        if raw in progress_file.ACTIVE:  # pod tắt giữa chừng: progress.json còn ghi "đang chạy", sẽ chặn lần xếp hàng mới
            progress_file.fail(s.product_root / room / "progress.json", "Bị gián đoạn: dựng lại", stopped=True)
        # detail["name"]: tên trong thư viện, không có thì tên đặt lúc tải lên (rooms.display_name), không thì id
        ok, message, queued = self.app.start_build(source, room, detail["name"])
        if not ok:
            return self._error(message, 422)
        return self._json({"ok": True, "room": room, "queued": queued, "message": message}, 202)

    @route("POST", r"/api/uploads")
    def api_upload_create(self):
        data = self._read_json()
        self._writable()
        session = self.app.store.create(data.get("room"), data.get("name"), data.get("kind"), data.get("files"))
        say(f"upload {session['id']}: room {session['room']} ({session['kind']}, {len(session['files'])} file, "
            f"{sum(f['size'] for f in session['files']) / 1e6:.1f} MB)")
        return self._json(session, 201)

    @route("GET", rf"/api/uploads/{UPLOAD}")
    def api_upload_get(self, upload_id):
        return self._json(self.app.store.get(upload_id))

    @route("PUT", rf"/api/uploads/{UPLOAD}/files/(\d+)")
    def api_upload_chunk(self, upload_id, index):
        self._writable()
        try:
            offset = int((self.query.get("offset") or [""])[0])
        except ValueError:
            raise UploadError("Thiếu offset", 400) from None
        data = self._read_body(CHUNK_BYTES)
        return self._json(self.app.store.write_chunk(upload_id, int(index), offset, data))

    @route("POST", rf"/api/uploads/{UPLOAD}/finish")
    def api_upload_finish(self, upload_id):
        self._read_body(MAX_JSON_BYTES)
        self._writable()
        session, source = self.app.store.finish(upload_id)
        ok, message, queued = self.app.start_build(source, session["room"], session["name"])
        if not ok:
            self.app.store.rollback(upload_id, source)
            return self._error(message, 422)
        self.app.store.forget(upload_id)
        return self._json({"ok": True, "room": session["room"], "queued": queued, "message": message}, 202)

    @route("DELETE", rf"/api/uploads/{UPLOAD}")
    def api_upload_delete(self, upload_id):
        self.app.store.forget(upload_id)
        return self._json({"ok": True})


# ============================================================================================= khởi động
def read_token(ws: Path, token_file: Path | None) -> str:
    """Mật khẩu: biến SR3D_WEB_TOKEN, hoặc file (tạo ngẫu nhiên lần đầu, quyền 600)."""
    if os.environ.get("SR3D_WEB_TOKEN"):
        return os.environ["SR3D_WEB_TOKEN"].strip()
    path = Path(token_file or ws / "outputs" / "product" / "server" / "token")
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(12)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token


def stop_on_signal(signum, frame):  # noqa: ARG001 - chữ ký của signal handler
    raise KeyboardInterrupt


def make_server(settings: Settings, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("SR3DHandler", (Handler,), {"app": App(settings)})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Server của trang web sản phẩm (tải lên, dựng phòng, xem 3D)")
    parser.add_argument("--ws", default=None, help="thư mục gốc trên pod (mặc định: biến WS, rồi /workspace)")
    parser.add_argument("--config", type=Path, default=None, help="cấu hình sản phẩm (mặc định configs/product.yaml)")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 trên pod (proxy của Runpod), 127.0.0.1 trên laptop")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token-file", type=Path, default=None,
                        help="file mật khẩu (mặc định $WS/outputs/product/server/token; biến SR3D_WEB_TOKEN thay được)")
    parser.add_argument("--no-auth", action="store_true",
                        help="không cần mật khẩu: chỉ cho phép cùng --view-only --host 127.0.0.1")
    parser.add_argument("--view-only", action="store_true", help="chỉ xem thư viện, không tải lên, không dựng phòng")
    parser.add_argument("--web-dir", type=Path, default=None,
                        help="thư mục web (mặc định web.dir của cấu hình; dùng với --view-only trên máy khác)")
    parser.add_argument("--max-upload-gb", type=float, default=MAX_TOTAL_BYTES / 1024 ** 3,
                        help="dung lượng tối đa của một lần tải lên (GiB)")
    parser.add_argument("--pid-file", type=Path, default=None, help="ghi pid khi đã mở cổng (scripts/web_server.sh)")
    return parser.parse_args(argv)


def settings_from(args: argparse.Namespace) -> Settings:
    ws = Path(args.ws or os.environ.get("WS") or product_config.DEFAULT_WS)
    if args.no_auth and (args.host not in LOCAL_HOSTS or not args.view_only):
        raise SystemExit("--no-auth chỉ dùng với --view-only --host 127.0.0.1: link proxy của Runpod là công khai")
    pipeline = None
    if args.web_dir and args.view_only:
        web_dir = args.web_dir.resolve()
        product_root = web_dir.parent
    else:
        cfg = product_config.load(args.config, ws=str(ws))
        web_dir = Path(args.web_dir or cfg["web"]["dir"])
        product_root = Path(cfg["output_root"])
        pipeline = pipeline_summary(cfg)
    token = None if args.no_auth else read_token(ws, args.token_file)
    build_script = Path(os.environ.get("SR3D_BUILD_SCRIPT") or REPO / "scripts" / "build_room.sh")
    return Settings(ws=ws, web_dir=web_dir, product_root=product_root, token=token,
                    view_only=args.view_only or not build_script.is_file(), build_script=build_script,
                    config_file=args.config.resolve() if args.config else None,
                    max_upload=int(args.max_upload_gb * 1024 ** 3), pipeline=pipeline)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = settings_from(args)
    if not (STATIC / "vendor" / "spark" / "spark.module.min.js").is_file():
        say("WARN: chưa có thư viện xem 3D (app/web/static/vendor/): chạy python3 -m app.web.vendor")
    server = make_server(settings, args.host, args.port)
    server.RequestHandlerClass.app.store.cleanup_stale()
    pod = os.environ.get("RUNPOD_POD_ID")
    say(f"SpatialRoom3D web: http://{args.host}:{args.port}/ | {'chỉ xem' if settings.view_only else 'dựng phòng'} | "
        f"thư viện: {settings.web_dir}")
    if pod:
        say(f"Trên Runpod: mở cổng HTTP {args.port} của pod, rồi vào https://{pod}-{args.port}.proxy.runpod.net/")
    if settings.token:
        say(f"Mật khẩu: {settings.token}  (đổi: xóa file mật khẩu rồi chạy lại server)")
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    signal.signal(signal.SIGTERM, stop_on_signal)  # web_server.sh stop: đóng cổng, xóa file pid
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        say("server stopped")
    finally:
        server.server_close()
        if args.pid_file and args.pid_file.is_file() and args.pid_file.read_text().strip() == str(os.getpid()):
            args.pid_file.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
