"""Tải bộ dữ liệu benchmark về volume theo mục `download` của configs/<luồng>.yaml. Chỉ chạy khi được gọi: các lệnh
train không bao giờ tải gì.

    python3 -m src.datasets.fetch --dataset deep_blending
    python3 -m src.datasets.fetch --dataset mipnerf360 --scenes room counter kitchen bonsai [--dry-run]
    (thường gọi qua: bash scripts/pod/fetch_data.sh <luồng> [cảnh ...])

Với mỗi cảnh được yêu cầu (mặc định: mọi cảnh của luồng):
  * đã có đủ theo `layout` của luồng: bỏ qua;
  * có thư mục nhưng thiếu phần nào đó: dừng, không sửa và không ghi đè dữ liệu đã có (tự xem lại hoặc dời đi);
  * chưa có: tải file zip chứa cảnh (curl, tự thử lại khi lỗi mạng), kiểm SHA-256 khi cấu hình có, giải nén CHỈ các thư
    mục `keep` của cảnh vào thư mục tạm `<root>/.fetch_tmp_<cảnh>`, kiểm tra bố cục, rồi đổi tên thành `<root>/<cảnh>`.
File zip nằm ở `<WS>/data/downloads/` và bị xóa sau khi giải nén xong (`--keep-archive` để giữ). Nguồn, kích thước,
SHA-256 của file đã tải và số file của mỗi cảnh được ghi vào `<root>/.sr3d_fetch.json` ngay khi cảnh đó vào chỗ (cảnh
sau lỗi thì cảnh trước vẫn có bản ghi), và vào `<WS>/outputs/reports/fetch_<luồng>_<thời gian>.json` (cả khi lỗi, kèm
lỗi). curl tự thử lại khi lỗi mạng, và khi tải chậm dưới 100 KB/s suốt 5 phút (kết nối treo).

File zip tải về là dữ liệu không tin cậy: chỉ đọc các mục nằm dưới `<cảnh>/<keep>/`; bỏ qua mục là liên kết (symlink)
và file ẩn (tên bắt đầu bằng '.', vd. `._*` của macOS: gsplat ghép ảnh theo mọi file trong thư mục); tên có `..` hoặc
`\\` thì dừng; mỗi file được tạo mới (không ghi đè) trong thư mục tạm; zipfile kiểm CRC của từng file khi đọc.
Mã thoát: 0 khi mọi cảnh được yêu cầu đã đủ; 1 khi tải / giải nén / kiểm tra lỗi; 2 khi cấu hình hoặc tham số sai.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urlparse

from src.datasets.registry import CONFIG_DIR, check_scene, load_registry

MANIFEST = ".sr3d_fetch.json"
STAGING_PREFIX = ".fetch_tmp_"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CHUNK = 1 << 20
CORRUPT = (zipfile.BadZipFile, EOFError, zlib.error)  # file zip hỏng (tải lỗi): xóa để lần sau tải lại
STALL = ["--speed-limit", "102400", "--speed-time", "300"]  # dưới 100 KB/s suốt 5 phút: curl ngắt, --retry thử lại


class FetchError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _gb(size: int) -> str:
    return f"{size / 1e9:.2f} GB"


def archive_name(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.zip", name):
        raise ValueError(f"a download url must end with a .zip file name: {url}")
    return name


def download_entries(name: str, entry: dict) -> list[dict]:
    """Mục `download` của luồng, đã kiểm tra: mỗi mục có `url` (.zip), `scenes`, `keep`; `folder`, `sha256` tùy chọn."""
    items = entry.get("download")
    if not isinstance(items, list) or not items:
        raise ValueError(f"dataset {name} has no download section in its configs/<dataset>.yaml")
    result = []
    for index, item in enumerate(items):
        where = f"{name}: download[{index}]"
        if not isinstance(item, dict) or not item.get("url") or not item.get("scenes") or not item.get("keep"):
            raise ValueError(f"{where} needs url, scenes and keep")
        folder = str(item.get("folder") or "").strip("/")
        names = [*(folder.split("/") if folder else []), *map(str, item["scenes"]), *map(str, item["keep"])]
        bad = [part for part in names if not NAME.fullmatch(part)]
        if bad:
            raise ValueError(f"{where}: invalid folder / scene / keep name(s) {bad}")
        sha = item.get("sha256")
        if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            raise ValueError(f"{where}: sha256 must be 64 lowercase hex digits")
        result.append({**item, "folder": folder, "archive": archive_name(str(item["url"])),
                       "scenes": [str(scene) for scene in item["scenes"]], "keep": [str(k) for k in item["keep"]]})
    return result


def zip_members(zf: zipfile.ZipFile, folder: str, scene: str, keep: list[str]) -> list[tuple[zipfile.ZipInfo, str]]:
    """Các file của cảnh trong zip: (mục, đường dẫn trong thư mục cảnh), chỉ dưới các thư mục `keep`.

    Kiểm tra trước khi giải nén bất cứ thứ gì: mọi thư mục `keep` phải có ít nhất một file."""
    prefix = f"{folder}/{scene}/" if folder else f"{scene}/"
    members, found, skipped = [], set(), 0
    for info in zf.infolist():
        name = info.filename
        if not name.startswith(prefix) or info.is_dir():
            continue
        parts = [part for part in name[len(prefix):].split("/") if part not in ("", ".")]
        if not parts or parts[0] not in keep:
            continue
        if "\\" in name or ".." in parts:
            raise FetchError(f"unsafe path in the archive: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16) or any(part.startswith(".") for part in parts):
            skipped += 1
            continue
        members.append((info, "/".join(parts)))
        found.add(parts[0])
    if skipped:
        print(f"[fetch] {scene}: skipped {skipped} symlink / hidden entr{'y' if skipped == 1 else 'ies'}")
    missing = [k for k in keep if k not in found]
    if missing:
        top = sorted({info.filename.split("/")[0] for info in zf.infolist()})[:12]
        raise FetchError(f"{prefix}{{{','.join(missing)}}}/ not found in the archive (top-level entries: {top})")
    return members


def extract(zf: zipfile.ZipFile, members: list[tuple[zipfile.ZipInfo, str]], staging: Path) -> tuple[int, int]:
    """Giải nén các mục vào thư mục tạm (mỗi file tạo mới, không ghi đè); trả về (số file, số byte)."""
    files = size = 0
    for info, rel in members:
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as source, open(target, "xb") as handle:
            shutil.copyfileobj(source, handle, CHUNK)  # đọc tới hết file thì zipfile kiểm CRC
        files += 1
        size += info.file_size
    return files, size


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def curl_version(curl: str) -> tuple[int, ...]:
    try:
        first = subprocess.run([curl, "--version"], capture_output=True, text=True, check=True).stdout.split("\n")[0]
    except (OSError, subprocess.CalledProcessError) as error:
        raise FetchError(f"curl is not available ({error})") from error
    match = re.match(r"curl (\d+)\.(\d+)", first)
    return tuple(int(value) for value in match.groups()) if match else (0, 0)


def download(url: str, target: Path, curl: str = "curl") -> None:
    """Tải vào `<file>.part` rồi đổi tên: file mang tên cuối chỉ có khi curl tải xong trọn vẹn."""
    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)
    cmd = [curl, "-fL", "--retry", "5", "--retry-delay", "10", "--connect-timeout", "30", *STALL]
    if curl_version(curl) >= (7, 71):
        cmd.append("--retry-all-errors")  # thử lại cả khi kết nối bị ngắt giữa chừng
    cmd += ["-o", str(part), url]
    print(f"[fetch] downloading {url}\n        -> {target}", flush=True)
    code = subprocess.run(cmd).returncode
    if code != 0:
        part.unlink(missing_ok=True)
        raise FetchError(f"curl exited with code {code} for {url}")
    os.replace(part, target)


def _read_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"fetches": []}
    return data if isinstance(data, dict) and isinstance(data.get("fetches"), list) else {"fetches": []}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _lock(root: Path):
    """Mỗi lúc một lần tải cho mỗi bộ dữ liệu (file khóa `<root>/.fetch.lock`, giữ lại sau khi xong); hệ thống file
    không hỗ trợ flock thì chạy không khóa."""
    handle = open(root / ".fetch.lock", "a", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[fetch] another fetch is writing {root}: waiting for it to finish...", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
    except OSError as error:
        print(f"WARN: cannot lock {root} ({error}); continuing without the lock")
    return handle


def _save_record(root: Path, name: str, record: dict) -> None:
    """Ghi bản ghi của một lần tải (một file zip) vào manifest, hoặc cập nhật nếu đã ghi (cùng `id`): gọi sau mỗi cảnh
    vừa vào chỗ, để cảnh nào đã có trong `<root>` cũng có nguồn và SHA-256 kèm theo."""
    path = root / MANIFEST
    manifest = _read_manifest(path)
    entry = {"dataset": name, **record}
    for index, old in enumerate(manifest["fetches"]):
        if old.get("id") == record["id"]:
            manifest["fetches"][index] = entry
            break
    else:
        manifest["fetches"].append(entry)
    _write_json(path, manifest)


def fetch_archive(name: str, item: dict, scenes: list[str], root: Path, layout: list[str], archive: Path, curl: str,
                  keep_archive: bool, records: list[dict]) -> dict:
    """Tải (nếu chưa có) một file zip và giải nén các cảnh `scenes` của nó. Bản ghi (nguồn, SHA-256, các cảnh đã vào
    chỗ) được thêm vào `records` (báo cáo) ngay khi kiểm xong SHA-256, và vào manifest sau mỗi cảnh."""
    if archive.is_file():
        print(f"[fetch] reusing {archive} (downloaded earlier)")
    else:
        download(str(item["url"]), archive, curl)
    size = archive.stat().st_size
    print(f"[fetch] {archive.name}: {_gb(size)}; computing SHA-256...", flush=True)
    digest = sha256_file(archive)
    expected = item.get("sha256")
    if expected and digest != expected:
        archive.unlink()
        raise FetchError(f"SHA-256 of {archive.name} is {digest}, expected {expected}: the file was removed, "
                         "run again to download it again")
    print(f"[fetch] SHA-256 {digest}" + (" (matches configs)" if expected else " (no published value; recorded)"))
    record = {"id": uuid.uuid4().hex, "url": item["url"], "archive": archive.name, "bytes": size, "sha256": digest,
              "sha256_expected": expected, "folder": item["folder"], "keep": item["keep"],
              "fetched": _now(), "scenes": {}}
    records.append(record)
    try:
        with zipfile.ZipFile(archive) as zf:
            plans = {scene: zip_members(zf, item["folder"], scene, item["keep"]) for scene in scenes}
            for scene in scenes:
                staging = root / f"{STAGING_PREFIX}{scene}"
                shutil.rmtree(staging, ignore_errors=True)  # dở dang của một lần trước (thư mục riêng của bước này)
                staging.mkdir(parents=True)
                try:
                    files, nbytes = extract(zf, plans[scene], staging)
                    state = check_scene(staging, layout, name=scene)
                    if state["status"] != "OK":
                        raise FetchError(f"{scene}: unexpected layout after extraction (missing {state['missing']})")
                    os.rename(staging, root / scene)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
                record["scenes"][scene] = {"files": files, "bytes": nbytes}
                _save_record(root, name, record)
                print(f"[fetch] {scene}: {files} files ({_gb(nbytes)}) -> {root / scene}", flush=True)
    except CORRUPT as error:
        archive.unlink(missing_ok=True)
        raise FetchError(f"{archive.name} is damaged ({error}): the file was removed, run again to download it "
                         "again") from error
    if not keep_archive:
        archive.unlink()
        print(f"[fetch] removed {archive} (KEEP_ARCHIVE=1 / --keep-archive keeps it)")
    return record


def plan(name: str, entry: dict, wanted: list[str], sources: dict) -> dict[str, tuple[dict, list[str]]]:
    """Trạng thái các cảnh được yêu cầu (in ra) và các file zip cần tải: {tên zip: (mục download, cảnh)}."""
    root = Path(entry["root"])
    layout = list(entry.get("layout") or [])
    states = {scene: check_scene(root / scene, layout) for scene in wanted}
    print(f"[fetch] {name}: {root}")
    for scene in wanted:
        state = states[scene]
        detail = "" if state["status"] != "INCOMPLETE" else f" (missing {', '.join(state['missing'])})"
        print(f"   {scene:<12} {state['status']}{detail}")
    broken = [scene for scene in wanted if states[scene]["status"] == "INCOMPLETE"]
    if broken:
        raise FetchError(f"{', '.join(broken)} already exist(s) in {root} but are incomplete: not touching existing "
                         "data; fix or move those folders away, then run again")
    missing = [scene for scene in wanted if states[scene]["status"] == "MISSING"]
    unsourced = [scene for scene in missing if scene not in sources]
    if unsourced:
        raise FetchError(f"no download source for {unsourced} in the download section of {name}")
    groups: dict[str, tuple[dict, list[str]]] = {}
    for scene in missing:
        item = sources[scene]
        groups.setdefault(item["archive"], (item, []))[1].append(scene)
    for item, here in groups.values():
        print(f"[fetch] {item['archive']} ({item['url']}): {', '.join(here)}; folders {', '.join(item['keep'])}")
    return groups


def fetch(name: str, entry: dict, ws: Path, scenes: list[str] | None = None, download_dir: Path | None = None,
          keep_archive: bool = False, dry_run: bool = False, curl: str = "curl") -> int:
    root = Path(entry["root"])
    layout = list(entry.get("layout") or [])
    known = [str(scene) for scene in entry.get("scenes") or []]
    wanted = list(scenes) if scenes else known
    unknown = [scene for scene in wanted if scene not in known]
    if unknown:
        raise ValueError(f"unknown scene(s) {unknown} for {name}; known: {known}")
    sources: dict[str, dict] = {}
    for item in download_entries(name, entry):
        for scene in item["scenes"]:
            sources.setdefault(scene, item)
    download_dir = download_dir or ws / "data" / "downloads"
    if dry_run:
        if plan(name, entry, wanted, sources):
            print(f"[fetch] --dry-run: nothing downloaded (archives would go to {download_dir})")
        else:
            print("[fetch] every requested scene is already there: nothing to download")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    lock = _lock(root)  # kiểm tra lại trạng thái sau khi có khóa: một lần tải khác có thể vừa xong
    report = {"dataset": name, "root": str(root), "requested": wanted, "fetched": [], "started": _now()}
    try:
        groups = plan(name, entry, wanted, sources)
        if not groups:
            print("[fetch] every requested scene is already there: nothing to download")
            return 0
        report["already_there"] = [scene for scene in wanted
                                   if not any(scene in here for _, here in groups.values())]
        try:
            download_dir.mkdir(parents=True, exist_ok=True)
            for archive_file, (item, here) in groups.items():
                fetch_archive(name, item, here, root, layout, download_dir / archive_file, curl, keep_archive,
                              report["fetched"])
        except (FetchError, OSError) as error:
            report["error"] = str(error)
            raise
        finally:  # báo cáo cả khi lỗi: cảnh nào đã vào chỗ, lỗi gì
            states = {scene: check_scene(root / scene, layout) for scene in wanted}
            report["result"] = {scene: state["status"] for scene, state in states.items()}
            report["finished"] = _now()
            reports = ws / "outputs" / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            out = reports / f"fetch_{name}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
            _write_json(out, report)
            print(f"[fetch] record: {root / MANIFEST} and {out}")
    finally:
        lock.close()
        try:
            download_dir.rmdir()  # chỉ khi trống
        except OSError:
            pass
    bad = [scene for scene, state in states.items() if state["status"] != "OK"]
    if bad:
        print(f"ERROR: still not complete after fetching: {bad}", file=sys.stderr)
        return 1
    print(f"[fetch] OK: {', '.join(wanted)} ready in {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=CONFIG_DIR, help="thư mục cấu hình (mặc định configs/ của repo)")
    parser.add_argument("--ws", default=os.environ.get("WS") or "/workspace", help="thư mục gốc trên pod (WS)")
    parser.add_argument("--dataset", required=True, help="tên luồng có mục download (deep_blending, mipnerf360)")
    parser.add_argument("--scenes", nargs="*", default=None, help="chỉ các cảnh này (mặc định: mọi cảnh của luồng)")
    parser.add_argument("--download-dir", type=Path, help="nơi để file zip (mặc định <WS>/data/downloads)")
    parser.add_argument("--keep-archive", action="store_true", help="giữ file zip sau khi giải nén")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in sẽ tải gì, không tải, không ghi")
    parser.add_argument("--curl", default="curl", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    ws = Path(args.ws.rstrip("/") or "/")
    try:
        registry = load_registry(args.configs, str(ws))
        datasets = registry.get("datasets", {})
        if args.dataset not in datasets:
            raise ValueError(f"unknown dataset {args.dataset!r}; known: {sorted(datasets)}")
        return fetch(args.dataset, datasets[args.dataset], ws, args.scenes, args.download_dir, args.keep_archive,
                     args.dry_run, args.curl)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (FetchError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
