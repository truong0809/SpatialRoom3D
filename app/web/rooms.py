"""Trạng thái các phòng cho trang web: phòng đã xong, phòng đang dựng hoặc lỗi, việc nền, log, file tải về.

Nguồn (đều do luồng sản phẩm ghi, trang web chỉ đọc):
  - phòng xong: mục `rooms` của $WS/outputs/product/web/manifest.json (src/product/export_web.py);
  - phòng đang dựng / lỗi: $WS/outputs/product/<phòng>/progress.json (src/product/progress.py);
  - việc nền room_<phòng>: $WS/outputs/jobs/room_<phòng>.state (scripts/pod/job.sh, dòng `state=` cuối cùng).
progress.json còn ghi "queued" / "running" nhưng việc nền không còn tiến trình nào (pod tắt hoặc khởi động lại giữa
chừng) thì trạng thái là "lost"; việc bị dừng khi còn xếp hàng (không ai ghi progress.json) thì là "stopped".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.product import config as product_config
from src.product import progress as progress_file

from .uploads import RESERVED_ROOMS, read_upload_info

LOG_TAIL_BYTES = 256 * 1024


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ===================================================================================== việc nền (job.sh)
def job_values(ws: Path, job: str) -> dict:
    """Các khóa của $WS/outputs/jobs/<job>.state (dòng sau cùng của mỗi khóa là giá trị hiện tại)."""
    values: dict[str, str] = {}
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(job)):  # cùng quy tắc tên việc của job.sh
        return values
    path = Path(ws) / "outputs" / "jobs" / f"{job}.state"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def token_alive(token: str, proc: Path = Path("/proc")) -> bool:
    """Còn tiến trình mang SR3D_JOB_TOKEN=<token> (cách job.sh nhận tiến trình của một việc, không dùng pid)?"""
    if not token:
        return False
    needle = f"SR3D_JOB_TOKEN={token}".encode()
    for environ in proc.glob("[0-9]*/environ"):
        try:
            if needle in environ.read_bytes().split(b"\0"):
                return True
        except OSError:
            continue
    return False


def job_state(ws: Path, job: str | None) -> dict | None:
    """{state, alive, started, finished} của việc nền; None khi không có việc / chưa có file trạng thái."""
    if not job:
        return None
    values = job_values(ws, job)
    if not values:
        return None
    return {"state": values.get("state"), "alive": token_alive(values.get("token", "")),
            "submitted": values.get("submitted"), "started": values.get("started"),
            "finished": values.get("finished")}


# ============================================================================================ phòng
def room_dirs(product_root: Path) -> list[Path]:
    root = Path(product_root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir()
                  if path.is_dir() and product_config.ROOM_ID.fullmatch(path.name)
                  and path.name.lower() not in RESERVED_ROOMS and not path.name.startswith("."))


def build_status(product_root: Path, ws: Path, room: str) -> dict | None:
    """progress.json của phòng kèm trạng thái đã đối chiếu với việc nền; None khi phòng chưa có progress.json."""
    data = progress_file.read(Path(product_root) / room / "progress.json")
    if data is None:
        return None
    status = data.get("status")
    job = job_state(ws, data.get("job"))
    if status in progress_file.ACTIVE and data.get("job"):
        if job is None:
            status = "lost"
        elif not job["alive"]:
            status = {"STOPPED": "stopped", "FAILED": "failed"}.get(job["state"], "lost")
    return {"status": status, "progress": data, "job": job}


def display_name(ws: Path, room: str, entry: dict | None = None) -> str:
    """Tên hiển thị: của thư viện (manifest.json), không thì của lần tải lên từ web, không thì mã phòng."""
    for name in ((entry or {}).get("name"), (read_upload_info(Path(ws) / "data" / "uploads", room) or {}).get("name")):
        if isinstance(name, str) and name:
            return name
    return room


def manifest_rooms(web_dir: Path) -> list[dict]:
    manifest = read_json(Path(web_dir) / "manifest.json")
    rooms = manifest.get("rooms") if isinstance(manifest, dict) else None
    return [entry for entry in rooms or [] if isinstance(entry, dict) and isinstance(entry.get("id"), str)]


def summary(build: dict | None) -> dict | None:
    """Phần progress.json mà thư viện cần (bước, %, thông điệp, thời điểm)."""
    if build is None:
        return None
    data = build["progress"]
    stage = next((s for s in data.get("stages", []) if s.get("id") == data.get("stage")), None)
    return {"status": build["status"], "stage": data.get("stage"), "stage_label": (stage or {}).get("label"),
            "percent": (stage or {}).get("percent"), "message": data.get("message"), "error": data.get("error"),
            "source_kind": data.get("source_kind"), "created_at": data.get("created_at"),
            "started_at": data.get("started_at"), "updated_at": data.get("updated_at"),
            "finished_at": data.get("finished_at")}


def list_rooms(web_dir: Path, product_root: Path, ws: Path) -> list[dict]:
    """Phòng chưa có trong thư viện (đang dựng / lỗi, mới nhất trước), rồi các phòng của thư viện theo thứ tự
    manifest.json. Phòng của thư viện đang được dựng lại thì kèm `build`."""
    done = manifest_rooms(web_dir)
    in_library = {entry["id"] for entry in done}
    building = []
    for folder in room_dirs(product_root):
        if folder.name in in_library:
            continue
        build = build_status(product_root, ws, folder.name)
        if build is not None:
            building.append({"id": folder.name, "name": display_name(ws, folder.name), "status": build["status"],
                             "build": summary(build), "entry": None})
    building.sort(key=lambda item: item["build"].get("updated_at") or "", reverse=True)
    rooms = []
    for entry in done:
        build = build_status(product_root, ws, entry["id"])
        active = build is not None and build["status"] in progress_file.ACTIVE
        rooms.append({"id": entry["id"], "name": display_name(ws, entry["id"], entry), "status": "done",
                      "build": summary(build) if active else None, "entry": entry})
    return building + rooms


def room_detail(web_dir: Path, product_root: Path, ws: Path, room: str) -> dict | None:
    if not product_config.ROOM_ID.fullmatch(room):
        return None
    entry = next((item for item in manifest_rooms(web_dir) if item["id"] == room), None)
    build = build_status(product_root, ws, room)
    if entry is None and build is None:
        return None
    active = build is not None and build["status"] in progress_file.ACTIVE
    status = "done" if entry is not None and not active else build["status"]  # dựng lại: bản cũ vẫn xem được
    return {"id": room, "name": display_name(ws, room, entry), "status": status, "entry": entry,
            "progress": build["progress"] if build else None, "job": build["job"] if build else None}


# ========================================================================================= log, file
def log_tail(log_path: str | None, allowed_root: Path, lines: int = 200) -> list[str]:
    """Các dòng cuối của log dựng phòng (thanh tiến độ tqdm ghi đè bằng \\r: chỉ giữ phần sau \\r cuối mỗi dòng).
    Chỉ đọc file nằm trong `allowed_root` ($WS/outputs/logs)."""
    if not log_path:
        return []
    path = Path(log_path).resolve()
    root = Path(allowed_root).resolve()
    if root not in path.parents or not path.is_file():
        return []
    with open(path, "rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - LOG_TAIL_BYTES))
        text = handle.read().decode("utf-8", errors="replace")
    out = [line.rsplit("\r", 1)[-1].rstrip() for line in text.split("\n")]
    out = [line for line in out if line.strip()]
    return [line[:400] for line in out[-max(1, min(lines, 2000)):]]


def ply_file(product_root: Path, room: str) -> Path | None:
    """PLY đầy đủ của phòng (đã train xong), để tải về mở bằng SuperSplat / phần mềm 3DGS khác."""
    if not product_config.ROOM_ID.fullmatch(room):
        return None
    model = Path(product_root) / room / "model"
    meta = read_json(model / "train_meta.json")
    if not isinstance(meta, dict) or meta.get("status") != "OK":
        return None
    relative = (meta.get("files") or {}).get("ply")
    if not isinstance(relative, str):
        return None
    path = (model / relative).resolve()
    return path if model.resolve() in path.parents and path.is_file() else None
