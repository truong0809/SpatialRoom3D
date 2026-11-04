"""Tải video / ảnh lên từ trang web: nhận theo từng phần, đủ thì thành nguồn của một phòng mới.

Vì sao theo từng phần: proxy HTTP của Runpod (Cloudflare) cắt mọi kết nối quá 100 giây, nên một video vài trăm MB
không gửi được trong một request. Trình duyệt gửi lần lượt các phần ≤ CHUNK_BYTES, mỗi phần kèm vị trí bắt đầu
(`offset`). Server chỉ nhận đúng phần tiếp theo của file, nên phần lỗi gửi lại được, và tải tiếp được khi mạng chập
chờn (hỏi lại số byte đã nhận rồi gửi tiếp).

Trên volume:
  $WS/outputs/product/uploads/<id>/session.json   phiên: tên phòng, loại nguồn, các file, số byte đã nhận
  $WS/outputs/product/uploads/<id>/part_<i>       file thứ i đang nhận
  $WS/data/uploads/<phòng>/                       nguồn của phòng khi đã nhận đủ: một video, hoặc các ảnh
Thư mục nguồn chỉ xuất hiện khi mọi file đã nhận đủ (đổi tên một thư mục tạm), nên không có nguồn dở dang. Dữ liệu đã
có trên volume không bị đụng tới: tên phòng đã dùng (ở `outputs/product/` hay `data/uploads/`) thì từ chối.

Chỉ dùng thư viện chuẩn và src.input.sources (quy tắc ảnh / video chung với scripts/build_room.sh).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from src.input import sources
from src.product import config as product_config

CHUNK_BYTES = 8 * 1024 * 1024  # mỗi request ≤ 8 MiB: gửi xong trong 100 giây từ khoảng 0,1 MB/s
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
KINDS = ("video", "photos")
MAX_PHOTOS = 3000
MAX_TOTAL_BYTES = 8 * 1024 ** 3  # mặc định của server (--max-upload-gb)
MAX_ROOM_ID = 64
MAX_DISPLAY_NAME = 80
STALE_SECONDS = 2 * 24 * 3600  # phiên bỏ dở quá 2 ngày thì xóa
MAX_OPEN = 3                   # số phiên đang nhận cùng lúc (mỗi phiên tới max_total byte trên volume)
IDLE_SECONDS = 30 * 60         # vượt MAX_OPEN: phiên không nhận thêm gì quá 30 phút coi như bỏ dở, xóa
UPLOAD_INFO = ".sr3d_upload.json"  # trong data/uploads/<phòng>/: tên hiển thị, loại nguồn, file gốc (file ẩn:
                                   # src/input/sources.py và trang web đều bỏ qua khi đọc ảnh)
# Thư mục con của outputs/product/ không phải phòng: không được dùng làm tên phòng
RESERVED_ROOMS = ("web", "uploads", "server")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(ValueError):
    """Lỗi do đầu vào của người dùng; `message` viết cho người xem trang web, `status` là mã HTTP."""

    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_file_name(name: str, used: set[str]) -> str:
    """Tên file chỉ gồm chữ không dấu, số, '.', '_', '-' (bỏ đường dẫn, kể cả 'C:\\fakepath\\'); không trùng tên đã dùng.
    Giữ phần số trong tên (IMG_0012) để thứ tự ảnh như lúc chụp."""
    base = re.split(r"[\\/]", str(name))[-1].strip()
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = _UNSAFE.sub("_", stem).strip("._") or "file"
    ext = _UNSAFE.sub("", ext)[:10]
    candidate = f"{stem}.{ext}" if ext else stem
    index = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{index}.{ext}" if ext else f"{stem}_{index}"
        index += 1
    used.add(candidate.lower())
    return candidate


def clean_display_name(name: str | None, room: str) -> str:
    text = "".join(ch for ch in str(name or "") if ch.isprintable()).strip()
    return (text or room)[:MAX_DISPLAY_NAME]


def check_files(kind: str, files: list, max_total: int) -> list[dict]:
    """Danh sách file trình duyệt khai báo ({name, size}) -> [{name, size}] đã kiểm tra; sai thì UploadError."""
    if kind not in KINDS:
        raise UploadError("Loại nguồn phải là video hoặc ảnh")
    if not isinstance(files, list) or not files:
        raise UploadError("Chưa chọn file nào")
    checked = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise UploadError("Danh sách file không hợp lệ")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise UploadError(f"{item['name']}: file rỗng hoặc không đọc được kích thước")
        checked.append({"name": item["name"], "size": size})
    total = sum(item["size"] for item in checked)
    if total > max_total:
        raise UploadError(f"Tổng dung lượng {total / 1e9:.1f} GB vượt giới hạn {max_total / 1e9:.1f} GB của server")
    suffixes = [Path(re.split(r"[\\/]", item["name"])[-1]).suffix.lower() for item in checked]
    if kind == "video":
        if len(checked) != 1:
            raise UploadError("Chỉ chọn một video cho một phòng")
        if suffixes[0] in sources.PHOTO_EXTS or suffixes[0] in sources.OTHER_IMAGE_EXTS:
            raise UploadError("File đã chọn là ảnh: chọn mục 'Ảnh' để tải nhiều ảnh lên")
        if suffixes[0] not in VIDEO_EXTS:
            raise UploadError(f"Video phải là {', '.join(ext[1:].upper() for ext in VIDEO_EXTS)}")
        return checked
    other = sorted({ext for ext in suffixes if ext not in sources.PHOTO_EXTS})
    if other:
        hint = " (ảnh HEIC của iPhone: xuất sang JPG trước)" if {".heic", ".heif"} & set(other) else ""
        raise UploadError(f"Chỉ nhận ảnh JPG/JPEG/PNG; có file {', '.join(other)}{hint}")
    if len(checked) > MAX_PHOTOS:
        raise UploadError(f"Tối đa {MAX_PHOTOS} ảnh cho một phòng")
    problem = sources.photo_count_problem(len(checked))
    if problem:
        raise UploadError(problem.replace(", hoặc đặt --min-images nếu cố ý thử với ít ảnh", ""))
    return checked


class UploadStore:
    """Các phiên tải lên trên volume. Mọi thao tác giữ một khóa chung (một người dùng, ghi tuần tự từng phần)."""

    def __init__(self, ws: Path, product_root: Path, max_total: int = MAX_TOTAL_BYTES):
        self.ws = Path(ws)
        self.product_root = Path(product_root)
        self.sessions_dir = self.product_root / "uploads"
        self.sources_dir = self.ws / "data" / "uploads"
        self.max_total = max_total
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------ phiên
    def _session_dir(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{16}", str(upload_id)):
            raise UploadError("Không có phiên tải lên này", 404)
        return self.sessions_dir / upload_id

    def _load(self, upload_id: str) -> dict:
        path = self._session_dir(upload_id) / "session.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise UploadError("Không có phiên tải lên này (có thể đã xong hoặc đã bị xóa)", 404) from None

    def _save(self, session: dict) -> None:
        folder = self._session_dir(session["id"])
        folder.mkdir(parents=True, exist_ok=True)
        session["updated_at"] = _now()
        tmp = folder / ".session.json.tmp"
        tmp.write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, folder / "session.json")

    def _sessions(self) -> list[dict]:
        found = []
        if self.sessions_dir.is_dir():
            for path in sorted(self.sessions_dir.glob("*/session.json")):
                try:
                    found.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
        return found

    @staticmethod
    def public(session: dict) -> dict:
        """Phần trả cho trình duyệt (không có đường dẫn trên pod)."""
        return {key: session[key] for key in ("id", "room", "name", "kind", "status", "chunk_bytes", "created_at")} | {
            "files": [{"name": f["name"], "size": f["size"], "received": f["received"]} for f in session["files"]]}

    # ------------------------------------------------------------------------------------- tên phòng
    def room_problem(self, room: str) -> str | None:
        """Lý do không dùng được tên phòng (viết cho người dùng); None nếu dùng được."""
        if not isinstance(room, str) or not product_config.ROOM_ID.fullmatch(room) or len(room) > MAX_ROOM_ID:
            return (f"Tên phòng chỉ gồm chữ không dấu, số, '.', '_', '-', bắt đầu bằng chữ hoặc số, tối đa "
                    f"{MAX_ROOM_ID} ký tự (vd. phong_khach)")
        if room.lower() in RESERVED_ROOMS:
            return f"Tên '{room}' dành cho thư mục của hệ thống: đặt tên khác"
        if (self.product_root / room).exists():
            return f"Đã có phòng '{room}' (trong thư viện hoặc đang dựng): đặt tên khác"
        if (self.sources_dir / room).exists():
            return f"Đã có dữ liệu tải lên cho phòng '{room}': đặt tên khác"
        return None

    # ---------------------------------------------------------------------------------------- thao tác
    def cleanup_stale(self) -> None:
        cutoff = time.time() - STALE_SECONDS
        for session in self._sessions():
            folder = self.sessions_dir / session.get("id", "")
            try:
                if folder.is_dir() and (folder / "session.json").stat().st_mtime < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                continue

    def _make_room(self) -> None:
        """Trước khi mở phiên mới: xóa phiên bỏ dở lâu; còn quá MAX_OPEN phiên đang nhận thì xóa phiên đã ngừng nhận quá
        IDLE_SECONDS, vẫn quá thì từ chối (mỗi phiên giữ tới max_total byte trên volume)."""
        self.cleanup_stale()
        open_sessions = [s for s in self._sessions() if s.get("status") == "uploading"]
        if len(open_sessions) < MAX_OPEN:
            return
        idle = time.time() - IDLE_SECONDS
        for session in sorted(open_sessions, key=lambda s: s.get("updated_at") or ""):
            folder = self.sessions_dir / session["id"]
            try:
                if (folder / "session.json").stat().st_mtime < idle:
                    shutil.rmtree(folder, ignore_errors=True)
                    open_sessions.remove(session)
            except OSError:
                continue
            if len(open_sessions) < MAX_OPEN:
                return
        raise UploadError(f"Đang có {len(open_sessions)} lượt tải lên dở: chờ chúng xong, hoặc thử lại sau "
                          f"{IDLE_SECONDS // 60} phút (lượt ngừng nhận lâu sẽ được xóa)", 429)

    def create(self, room: str, name: str | None, kind: str, files: list) -> dict:
        with self._lock:
            problem = self.room_problem(room)
            if problem:
                raise UploadError(problem, 409 if "Đã có" in problem else 400)
            checked = check_files(kind, files, self.max_total)
            for old in self._sessions():  # phiên dở của cùng tên phòng (vd. tải lại trang): thay bằng phiên mới
                if old.get("room") == room and old.get("status") == "uploading":
                    shutil.rmtree(self.sessions_dir / old["id"], ignore_errors=True)
            self._make_room()
            used: set[str] = set()
            session = {
                "format_version": 1,
                "id": secrets.token_hex(8),
                "room": room,
                "name": clean_display_name(name, room),
                "kind": kind,
                "status": "uploading",
                "chunk_bytes": CHUNK_BYTES,
                "created_at": _now(),
                "files": [{"name": safe_file_name(item["name"], used), "original": item["name"][:255],
                           "size": item["size"], "received": 0} for item in checked],
            }
            self._save(session)
            return self.public(session)

    def get(self, upload_id: str) -> dict:
        with self._lock:
            return self.public(self._load(upload_id))

    def write_chunk(self, upload_id: str, index: int, offset: int, data: bytes) -> dict:
        """Ghi phần tiếp theo của file thứ `index`. Sai vị trí -> 409 kèm số byte đã nhận (trình duyệt gửi tiếp từ đó)."""
        with self._lock:
            session = self._load(upload_id)
            if session["status"] != "uploading":
                raise UploadError("Phiên tải lên đã kết thúc", 409)
            if not 0 <= index < len(session["files"]):
                raise UploadError("Không có file này trong phiên tải lên", 404)
            entry = session["files"][index]
            if offset != entry["received"]:
                raise UploadError(f"Server đã nhận {entry['received']} byte của {entry['name']}", 409,
                                  received=entry["received"])
            if not data or len(data) > session["chunk_bytes"] or offset + len(data) > entry["size"]:
                raise UploadError("Phần dữ liệu sai kích thước", 400)
            part = self._session_dir(upload_id) / f"part_{index}"
            with open(part, "r+b" if part.exists() else "wb") as handle:
                handle.seek(offset)
                handle.write(data)
                handle.truncate()
            entry["received"] = offset + len(data)
            self._save(session)
            return {"index": index, "received": entry["received"], "size": entry["size"]}

    def finish(self, upload_id: str) -> tuple[dict, Path]:
        """Mọi file đã đủ -> chuyển sang $WS/data/uploads/<phòng>/; trả về (phiên, nguồn cho build_room.sh)."""
        with self._lock:
            session = self._load(upload_id)
            if session["status"] != "uploading":
                raise UploadError("Phiên tải lên đã kết thúc", 409)
            missing = [f["name"] for f in session["files"] if f["received"] != f["size"]]
            if missing:
                raise UploadError(f"Chưa nhận đủ {len(missing)} file (vd. {missing[0]})", 409)
            room = session["room"]
            problem = self.room_problem(room)
            if problem:
                raise UploadError(problem, 409)
            folder = self._session_dir(upload_id)
            target = self.sources_dir / room
            incoming = self.sources_dir / f".{room}.incoming-{upload_id}"
            shutil.rmtree(incoming, ignore_errors=True)
            incoming.mkdir(parents=True)
            for index, entry in enumerate(session["files"]):
                shutil.move(str(folder / f"part_{index}"), str(incoming / entry["name"]))  # kể cả khác ổ đĩa
            info = {"format_version": 1, "room": room, "name": session["name"], "kind": session["kind"],
                    "uploaded_at": _now(), "files": [{"name": f["name"], "original": f.get("original"), "size": f["size"]}
                                                     for f in session["files"]]}
            (incoming / UPLOAD_INFO).write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(incoming, target)
            source = target / session["files"][0]["name"] if session["kind"] == "video" else target
            session.update(status="finished", source=str(source))
            self._save(session)
            return self.public(session), source

    def source_of(self, room: str) -> Path | None:
        """Nguồn đã tải lên từ web của phòng (để dựng lại): video duy nhất trong thư mục, hoặc thư mục ảnh."""
        if not isinstance(room, str) or not product_config.ROOM_ID.fullmatch(room):
            return None
        folder = self.sources_dir / room
        if not folder.is_dir():
            return None
        files = [path for path in folder.iterdir() if path.is_file() and not path.name.startswith(".")]
        if len(files) == 1 and files[0].suffix.lower() in VIDEO_EXTS:
            return files[0]
        return folder if any(path.suffix.lower() in sources.PHOTO_EXTS for path in files) else None

    def forget(self, upload_id: str) -> None:
        """Xóa phiên (phần đã nhận, nếu còn). Không đụng tới nguồn đã chuyển sang data/uploads/."""
        with self._lock:
            shutil.rmtree(self._session_dir(upload_id), ignore_errors=True)

    def rollback(self, upload_id: str, source: Path) -> None:
        """Dựng phòng không bắt đầu được: xóa nguồn vừa tạo (chỉ thư mục của phiên này) và phiên."""
        with self._lock:
            target = self.sources_dir / self._load(upload_id)["room"]
            source = Path(source)
            if target.is_dir() and (source == target or source.parent == target):
                shutil.rmtree(target, ignore_errors=True)
            shutil.rmtree(self._session_dir(upload_id), ignore_errors=True)


def read_upload_info(sources_dir: Path, room: str) -> dict | None:
    """data/uploads/<phòng>/.sr3d_upload.json (UploadStore.finish ghi); None nếu không có hoặc không đọc được."""
    if not isinstance(room, str) or not product_config.ROOM_ID.fullmatch(room):
        return None
    try:
        data = json.loads((Path(sources_dir) / room / UPLOAD_INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
