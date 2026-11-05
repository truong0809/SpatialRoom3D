"""Tải thư viện của trình xem 3D (three.js, Spark) và phông chữ của trang (Be Vietnam Pro) đúng phiên bản ghim vào
app/web/static/vendor/.

Phiên bản, địa chỉ và mã kiểm tra (integrity SHA-512 của gói npm) nằm ở configs/environment/web_viewer.lock.json.
Thư mục vendor/ không đưa lên Git: chạy lệnh này một lần trên mỗi máy (pod, hoặc laptop dùng để xem offline):
  python3 -m app.web.vendor                    # tải từ registry.npmjs.org (cần Internet một lần)
  python3 -m app.web.vendor --from <thư mục>   # dùng các file .tgz đã tải sẵn (máy không có Internet)
  python3 -m app.web.vendor --check            # chỉ kiểm tra: exit 0 khi đã đủ file, đúng phiên bản
Gói tải về sai mã kiểm tra thì dừng, không ghi gì. Chỉ dùng thư viện chuẩn của Python.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "configs" / "environment" / "web_viewer.lock.json"
DEST = Path(__file__).resolve().parent / "static" / "vendor"
MARKER = "VERSIONS.json"


class VendorError(RuntimeError):
    pass


def read_lock(path: Path = LOCK) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def integrity_of(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


def installed(dest: Path = DEST, lock: dict | None = None) -> bool:
    lock = lock or read_lock()
    try:
        marker = json.loads((dest / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for name, package in lock["packages"].items():
        if marker.get(name) != {"version": package["version"], "integrity": package["integrity"]}:
            return False
        if not all((dest / target).is_file() for target in package["files"].values()):
            return False
    return True


def fetch_tarball(package: dict, offline_dir: Path | None) -> bytes:
    filename = package["tarball"].rsplit("/", 1)[-1]
    if offline_dir is not None:
        path = Path(offline_dir) / filename
        if not path.is_file():
            raise VendorError(f"không thấy {path}")
        return path.read_bytes()
    with urllib.request.urlopen(package["tarball"], timeout=180) as response:  # noqa: S310 - địa chỉ cố định trong lock
        return response.read()


def install(dest: Path = DEST, lock: dict | None = None, offline_dir: Path | None = None) -> list[str]:
    """Tải, kiểm tra mã, rồi chép đúng các file khai trong lock vào dest. Trả về các file đã ghi."""
    lock = lock or read_lock()
    extracted: dict[str, bytes] = {}
    for name, package in lock["packages"].items():
        data = fetch_tarball(package, offline_dir)
        if integrity_of(data) != package["integrity"]:
            raise VendorError(f"{name} {package['version']}: sai mã kiểm tra (integrity), không dùng gói này")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member_name, target in package["files"].items():
                if target.startswith("/") or ".." in Path(target).parts:
                    raise VendorError(f"đường dẫn đích không hợp lệ trong lock: {target}")
                member = archive.getmember(member_name)
                handle = archive.extractfile(member) if member.isfile() else None
                if handle is None:
                    raise VendorError(f"{name}: không có file {member_name} trong gói")
                extracted[target] = handle.read()
    written = []
    for target, content in extracted.items():
        path = dest / target
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        written.append(target)
    marker = {name: {"version": p["version"], "integrity": p["integrity"]} for name, p in lock["packages"].items()}
    (dest / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tải thư viện của trình xem 3D (three.js, Spark) vào app/web/static/vendor/")
    parser.add_argument("--check", action="store_true", help="chỉ kiểm tra; exit 0 khi đã đủ và đúng phiên bản")
    parser.add_argument("--from", dest="offline_dir", type=Path, default=None,
                        help="thư mục chứa các file .tgz đã tải sẵn (tên như cuối địa chỉ tarball trong lock)")
    args = parser.parse_args(argv)
    lock = read_lock()
    versions = ", ".join(f"{name} {p['version']}" for name, p in lock["packages"].items())
    if installed(DEST, lock):
        print(f"OK: thư viện xem 3D đã có ({versions})")
        return 0
    if args.check:
        print(f"MISSING: chưa có hoặc khác phiên bản ({versions}): chạy python3 -m app.web.vendor")
        return 1
    try:
        written = install(DEST, lock, args.offline_dir)
    except (VendorError, OSError, tarfile.TarError, KeyError) as error:
        print(f"ERROR: không tải được thư viện xem 3D: {error}")
        return 1
    print(f"OK: {versions} -> {DEST} ({len(written)} file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
