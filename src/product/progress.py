"""Tiến độ dựng một phòng cho web đọc: <phòng>/progress.json. Chỉ dùng thư viện chuẩn của Python.

File được ghi lại trọn vẹn mỗi lần (file tạm rồi đổi tên), nên bên đọc không bao giờ thấy file viết dở. Nội dung:

  format_version  1
  room            tên phòng (id dùng trong manifest.json của web)
  source_kind     "video" | "photos" | null (chưa biết, vd. dựng lại từ COLMAP đã có)
  status          "queued" (chờ GPU) | "running" | "done" | "failed" | "stopped"
  stage           id của bước đang chạy, hoặc bước bị lỗi / bị dừng; null khi chưa chạy hoặc đã xong
  stages          các bước theo thứ tự: {id, label, status, percent, message, started_at, finished_at}; status
                  của bước: "waiting" | "running" | "done" | "skipped" | "failed" | "stopped"; percent chỉ có ở
                  bước train; started_at / finished_at: lúc bước bắt đầu / kết thúc (null khi chưa chạy hoặc bỏ qua)
  message, error  dòng mô tả hiện tại; thông điệp lỗi (null khi không lỗi)
  job, log        tên việc nền (scripts/pod/job.sh) và đường dẫn log đầy đủ trên pod
  created_at, started_at, updated_at, finished_at   thời điểm UTC, ISO 8601

Việc nền bị dừng khi còn xếp hàng thì không có ai ghi file này: trạng thái của việc nằm ở
$WS/outputs/jobs/<job>.state (scripts/pod/job.sh, dòng state= cuối cùng).

Cách dùng (thường do scripts/build_room.sh gọi):
  python3 -m src.product.progress <file> queue --room <phòng> [--source-kind video|photos] [--job <tên việc>]
  python3 -m src.product.progress <file> start --room <phòng> [--source-kind ...] [--job ...] [--log <file log>]
  python3 -m src.product.progress <file> stage <bước> [--message ...]
  python3 -m src.product.progress <file> skip <bước> [<bước> ...] [--message ...]
  python3 -m src.product.progress <file> fail --message ... [--stopped]
  python3 -m src.product.progress <file> done [--message ...]
  <lệnh> | python3 -m src.product.progress <file> follow     # chép nguyên đầu ra, đổi bước theo dòng mốc
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

FORMAT_VERSION = 1
STAGE_IDS = ("images", "colmap", "train", "export")
STAGE_LABELS = {
    "colmap": "COLMAP: ước lượng camera",
    "train": "Train mô hình chính (mcmc)",
    "export": "Xuất mô hình web và thêm vào thư viện",
}
IMAGE_LABELS = {"video": "Tách khung ảnh từ video", "photos": "Chuẩn hóa ảnh", None: "Chuẩn bị ảnh"}
SOURCE_KINDS = ("video", "photos")
ACTIVE = ("queued", "running")
# Dòng mốc của scripts/pipeline/prepare_scene.sh -> (bước, loại nguồn)
MARKERS = (
    ("===== 1. Photos -> ", "images", "photos"),
    ("===== 1. Frames -> ", "images", "video"),
    ("===== 2. COLMAP", "colmap", None),
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stage_label(stage: str, source_kind: str | None = None) -> str:
    return IMAGE_LABELS.get(source_kind, IMAGE_LABELS[None]) if stage == "images" else STAGE_LABELS[stage]


def new(room: str, source_kind: str | None = None, status: str = "queued", job: str | None = None,
        log: str | None = None) -> dict:
    stamp = now()
    return {
        "format_version": FORMAT_VERSION,
        "room": room,
        "source_kind": source_kind,
        "status": status,
        "stage": None,
        "stages": [{"id": stage, "label": stage_label(stage, source_kind), "status": "waiting", "percent": None,
                    "message": None, "started_at": None, "finished_at": None} for stage in STAGE_IDS],
        "message": "Đang chờ GPU" if status == "queued" else None,
        "error": None,
        "job": job,
        "log": log,
        "created_at": stamp,
        "started_at": None if status == "queued" else stamp,
        "updated_at": stamp,
        "finished_at": None,
    }


def read(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("stages"), list) else None


def write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path) -> dict:
    data = read(path)
    if data is None:
        raise ValueError(f"{path}: no progress file (run 'start' first)")
    return data


def _stage(data: dict, stage: str) -> dict:
    for entry in data["stages"]:
        if entry["id"] == stage:
            return entry
    raise ValueError(f"unknown stage {stage!r} (stages: {', '.join(STAGE_IDS)})")


def _finish(entry: dict, status: str) -> None:
    """Bước kết thúc với `status`: ghi lúc kết thúc nếu bước đã chạy (bước chưa bắt đầu thì không có thời gian)."""
    was_running = entry["status"] == "running"
    entry["status"] = status
    if was_running and entry.get("started_at") and not entry.get("finished_at"):
        entry["finished_at"] = now()


def _set_source_kind(data: dict, source_kind: str | None) -> None:
    if source_kind:
        data["source_kind"] = source_kind
        _stage(data, "images")["label"] = stage_label("images", source_kind)


def queue(path: Path, room: str, source_kind: str | None = None, job: str | None = None) -> bool:
    """Ghi trạng thái "queued" cho một lần dựng sắp xếp hàng; không ghi đè lần dựng đang chờ / đang chạy."""
    current = read(path)
    if current is not None and current.get("status") in ACTIVE:
        return False
    write(path, new(room, source_kind, "queued", job))
    return True


def start(path: Path, room: str, source_kind: str | None = None, job: str | None = None,
          log: str | None = None) -> dict:
    """Bắt đầu một lần dựng (file mới); giữ created_at của trạng thái "queued" do `queue` ghi."""
    current = read(path)
    data = new(room, source_kind, "running", job, log)
    if current is not None and current.get("status") == "queued" and current.get("room") == room:
        data["created_at"] = current.get("created_at") or data["created_at"]
        data["job"] = job or current.get("job")
        if not source_kind:
            _set_source_kind(data, current.get("source_kind"))
    write(path, data)
    return data


def set_stage(path: Path, stage: str, message: str | None = None, source_kind: str | None = None) -> dict:
    """Bước `stage` bắt đầu chạy: các bước trước nó (đang chờ / đang chạy) coi như xong, các bước sau đang chờ."""
    data = _load(path)
    order = STAGE_IDS.index(_stage(data, stage)["id"])
    _set_source_kind(data, source_kind)
    for index, entry in enumerate(data["stages"]):
        if index < order and entry["status"] in ("waiting", "running"):
            _finish(entry, "done")
        elif index == order:
            entry.update(status="running", percent=None, message=message, started_at=now(), finished_at=None)
        elif index > order:
            entry.update(status="waiting", percent=None, message=None, started_at=None, finished_at=None)
    data.update(status="running", stage=stage, message=message or _stage(data, stage)["label"], error=None)
    write(path, data)
    return data


def skip(path: Path, stages: list[str], message: str | None = None) -> dict:
    data = _load(path)
    for stage in stages:
        _stage(data, stage).update(status="skipped", percent=None, message=message, started_at=None, finished_at=None)
    write(path, data)
    return data


def update(path: Path, stage: str, percent: float | None = None, message: str | None = None) -> dict | None:
    """Cập nhật phần trăm / dòng mô tả của một bước đang chạy (train_main.py gọi trong lúc train)."""
    data = read(path)
    if data is None:
        return None
    entry = _stage(data, stage)
    if percent is not None:
        entry["percent"] = round(max(0.0, min(100.0, float(percent))), 1)
    if message is not None:
        entry["message"] = message
        if data.get("stage") == stage:
            data["message"] = message
    write(path, data)
    return data


def fail(path: Path, message: str, stopped: bool = False) -> dict | None:
    data = read(path)
    if data is None:
        return None
    state = "stopped" if stopped else "failed"
    for entry in data["stages"]:
        if entry["status"] == "running":
            _finish(entry, state)
            data["stage"] = entry["id"]
    data.update(status=state, message=message, error=None if stopped else message, finished_at=now())
    write(path, data)
    return data


def done(path: Path, message: str | None = None) -> dict:
    data = _load(path)
    for entry in data["stages"]:
        if entry["status"] in ("waiting", "running"):
            _finish(entry, "done")
            if entry["id"] == "train" and entry["percent"] is not None:
                entry["percent"] = 100.0
    data.update(status="done", stage=None, message=message or "Xong", error=None, finished_at=now())
    write(path, data)
    return data


def follow(path: Path, stream_in=None, stream_out=None) -> int:
    """Chép nguyên đầu ra của scripts/pipeline/prepare_scene.sh; gặp dòng mốc thì đổi bước trong progress.json.

    Đọc / ghi dạng byte (đầu ra của COLMAP, ffmpeg có thể không phải UTF-8). Lỗi khi ghi progress.json chỉ in cảnh
    báo: không làm hỏng bước chuẩn bị ảnh đang chạy."""
    stream_in = stream_in or sys.stdin.buffer
    stream_out = stream_out or sys.stdout.buffer
    for raw in iter(stream_in.readline, b""):
        stream_out.write(raw)
        stream_out.flush()
        line = raw.decode("utf-8", errors="replace")
        for marker, stage, source_kind in MARKERS:
            if line.startswith(marker):
                try:
                    set_stage(path, stage, source_kind=source_kind)
                except (OSError, ValueError) as error:
                    stream_out.write(f"WARN: progress file not updated: {error}\n".encode())
                    stream_out.flush()
                break
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiến độ dựng một phòng (progress.json cho web)")
    parser.add_argument("file", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("queue", "start"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--room", required=True)
        cmd.add_argument("--source-kind", choices=SOURCE_KINDS)
        cmd.add_argument("--job")
        if name == "start":
            cmd.add_argument("--log")
    cmd = sub.add_parser("stage")
    cmd.add_argument("stage", choices=STAGE_IDS)
    cmd.add_argument("--message")
    cmd = sub.add_parser("skip")
    cmd.add_argument("stages", nargs="+", choices=STAGE_IDS)
    cmd.add_argument("--message")
    cmd = sub.add_parser("fail")
    cmd.add_argument("--message", required=True)
    cmd.add_argument("--stopped", action="store_true")
    cmd = sub.add_parser("done")
    cmd.add_argument("--message")
    sub.add_parser("follow")
    args = parser.parse_args(argv)
    try:
        if args.command == "queue":
            if not queue(args.file, args.room, args.source_kind, args.job):
                print(f"progress: {args.file} already belongs to a queued or running build; left unchanged")
        elif args.command == "start":
            start(args.file, args.room, args.source_kind, args.job, args.log)
        elif args.command == "stage":
            set_stage(args.file, args.stage, args.message)
        elif args.command == "skip":
            skip(args.file, args.stages, args.message)
        elif args.command == "fail":
            fail(args.file, args.message, args.stopped)
        elif args.command == "done":
            done(args.file, args.message)
        else:
            return follow(args.file)
    except (OSError, ValueError) as error:
        print(f"ERROR: progress: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
