"""Test bước tải dữ liệu benchmark (src/datasets/fetch.py, scripts/pod/fetch_data.sh) với curl giả: curl chép một file
zip tổng hợp trong tmp_path thay vì tải từ Internet. File zip mô phỏng bố cục của 360_v2.zip (các cảnh ở gốc) và
tandt_db.zip (các cảnh trong db/); nội dung file là dữ liệu giả.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile

import pytest
import yaml

from shell_env import REPO, ShellEnv
from src.datasets.fetch import main

FAKE_CURL = r'''#!/usr/bin/env bash
# curl giả: --version in phiên bản; còn lại chép $FAKE_CURL_SRC/<tên file trong URL> vào file sau -o
if [ "${1:-}" = "--version" ]; then echo "curl 7.81.0 (x86_64-pc-linux-gnu)"; exit 0; fi
echo "curl $*" >> "$FAKE_CURL_LOG"
out=""
prev=""
for arg in "$@"; do
  [ "$prev" = "-o" ] && out="$arg"
  prev="$arg"
  url="$arg"
done
src="$FAKE_CURL_SRC/$(basename "$url")"
[ -f "$src" ] || exit 22
cp "$src" "$out"
'''

MODEL = ("cameras.bin", "images.bin", "points3D.bin")


def add_scene(zf, prefix, extra_folders=("images_2", "images_4", "images_8")):
    for folder in ("images", *extra_folders):
        for i in range(2):
            zf.writestr(f"{prefix}{folder}/DSC{i:04d}.JPG", f"{prefix}{folder}{i}")
    for name in MODEL:
        zf.writestr(f"{prefix}sparse/0/{name}", b"model")
    zf.writestr(f"{prefix}poses_bounds.npy", b"poses")


def make_360(path, scenes=("room", "counter", "bicycle"), extras=None):
    with zipfile.ZipFile(path, "w") as zf:
        for scene in scenes:
            add_scene(zf, f"{scene}/")
        zf.writestr("room/images/._DSC0000.JPG", b"macos")  # file ẩn: không được giải nén
        link = zipfile.ZipInfo("room/images/link.JPG")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16  # mục symlink: bỏ qua
        zf.writestr(link, "/etc/passwd")
        for name, data in (extras or {}).items():
            zf.writestr(name, data)
    return path


def make_db(path):
    with zipfile.ZipFile(path, "w") as zf:
        for scene in ("drjohnson", "playroom"):
            add_scene(zf, f"db/{scene}/", extra_folders=())
        add_scene(zf, "tandt/train/", extra_folders=())
    return path


@pytest.fixture
def fake(tmp_path, monkeypatch):
    src = tmp_path / "server"
    src.mkdir()
    curl = tmp_path / "bin" / "curl"
    curl.parent.mkdir()
    curl.write_text(FAKE_CURL, encoding="utf-8")
    curl.chmod(0o755)
    log = tmp_path / "curl.log"
    monkeypatch.setenv("FAKE_CURL_SRC", str(src))
    monkeypatch.setenv("FAKE_CURL_LOG", str(log))
    ws = tmp_path / "ws"
    ws.mkdir()
    return {"src": src, "curl": curl, "log": log, "ws": ws, "tmp": tmp_path}


def run_fetch(fake, *args, configs=None):
    argv = ["--ws", str(fake["ws"]), "--curl", str(fake["curl"]), *args]
    if configs is not None:
        argv = ["--configs", str(configs), *argv]
    return main(argv)


def curl_calls(fake):
    return fake["log"].read_text(encoding="utf-8").splitlines() if fake["log"].is_file() else []


def m360_root(fake):
    return fake["ws"] / "data" / "benchmarks" / "mipnerf360"


def test_fetches_only_the_missing_scenes_and_the_kept_folders(fake, capsys):
    make_360(fake["src"] / "360_v2.zip")
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room", "counter") == 0
    root = m360_root(fake)
    for scene in ("room", "counter"):
        assert sorted(p.name for p in (root / scene).iterdir()) == ["images", "images_2", "images_4", "sparse"]
        assert sorted(p.name for p in (root / scene / "images").iterdir()) == ["DSC0000.JPG", "DSC0001.JPG"]
        assert (root / scene / "sparse" / "0" / "cameras.bin").read_bytes() == b"model"
    assert not (root / "bicycle").exists()  # cảnh không được yêu cầu
    assert not list(root.glob(".fetch_tmp_*"))  # không còn thư mục tạm
    assert not (fake["ws"] / "data" / "downloads").exists()  # file zip đã xóa, thư mục trống cũng bỏ
    (call,) = curl_calls(fake)
    assert "--retry-all-errors" in call and call.endswith("https://storage.googleapis.com/gresearch/refraw360/360_v2.zip")
    assert "--speed-limit 102400 --speed-time 300" in call  # kết nối treo thì ngắt và thử lại, không chờ mãi
    assert f"-o {fake['ws'] / 'data' / 'downloads' / '360_v2.zip.part'} " in call
    manifest = json.loads((root / ".sr3d_fetch.json").read_text(encoding="utf-8"))
    (record,) = manifest["fetches"]
    digest = hashlib.sha256((fake["src"] / "360_v2.zip").read_bytes()).hexdigest()
    assert record["sha256"] == digest and record["sha256_expected"] is None
    assert record["scenes"]["room"]["files"] == 2 * 3 + 3  # 3 thư mục ảnh × 2 + 3 file mô hình
    (report,) = (fake["ws"] / "outputs" / "reports").glob("fetch_mipnerf360_*.json")
    assert json.loads(report.read_text(encoding="utf-8"))["result"] == {"room": "OK", "counter": "OK"}
    out = capsys.readouterr().out
    assert "skipped 2 symlink / hidden entries" in out and "no published value; recorded" in out

    # chạy lại: đủ rồi thì không tải nữa; thêm cảnh thì chỉ tải cho cảnh còn thiếu
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room", "counter") == 0
    assert len(curl_calls(fake)) == 1 and "nothing to download" in capsys.readouterr().out
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room", "bicycle") == 0
    assert len(curl_calls(fake)) == 2 and (root / "bicycle" / "images").is_dir()
    assert len(json.loads((root / ".sr3d_fetch.json").read_text(encoding="utf-8"))["fetches"]) == 2


def test_deep_blending_checks_the_configured_sha256(fake, tmp_path):
    make_db(fake["src"] / "tandt_db.zip")
    # SHA-256 trong configs/ là của file thật: file giả khác SHA thì bị xóa và dừng, không giải nén gì
    assert run_fetch(fake, "--dataset", "deep_blending", "--keep-archive") == 1
    root = fake["ws"] / "data" / "benchmarks" / "deep_blending"
    assert not (root / "drjohnson").exists() and not (fake["ws"] / "data" / "downloads" / "tandt_db.zip").exists()

    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    db = yaml.safe_load((configs / "deep_blending.yaml").read_text(encoding="utf-8"))
    digest = hashlib.sha256((fake["src"] / "tandt_db.zip").read_bytes()).hexdigest()
    text = (configs / "deep_blending.yaml").read_text(encoding="utf-8")
    (configs / "deep_blending.yaml").write_text(text.replace(db["download"][0]["sha256"], digest), encoding="utf-8")
    assert run_fetch(fake, "--dataset", "deep_blending", "--keep-archive", configs=configs) == 0
    for scene in ("drjohnson", "playroom"):
        assert sorted(p.name for p in (root / scene).iterdir()) == ["images", "sparse"]  # chỉ thư mục keep
    assert not (root / "train").exists() and not (root / "tandt").exists()
    assert (fake["ws"] / "data" / "downloads" / "tandt_db.zip").is_file()  # --keep-archive
    record = json.loads((root / ".sr3d_fetch.json").read_text(encoding="utf-8"))["fetches"][0]
    assert record["sha256"] == record["sha256_expected"] == digest


def test_existing_incomplete_scene_is_never_touched(fake, capsys):
    room = m360_root(fake) / "room" / "images"
    room.mkdir(parents=True)
    (room / "mine.JPG").write_bytes(b"user data")
    make_360(fake["src"] / "360_v2.zip")
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room") == 1
    assert "already exist(s)" in capsys.readouterr().err
    assert sorted(p.name for p in room.iterdir()) == ["mine.JPG"] and not curl_calls(fake)


def test_unsafe_or_missing_entries_stop_before_anything_is_written(fake, capsys):
    make_360(fake["src"] / "360_v2.zip", extras={"room/images/../../escape.txt": "x"})
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "counter", "room") == 1
    assert "unsafe path in the archive" in capsys.readouterr().err
    root = m360_root(fake)
    assert not (root / "counter").exists() and not (root / "room").exists()  # kiểm tra mọi cảnh trước khi giải nén
    assert not list(fake["tmp"].rglob("escape.txt"))
    assert (fake["ws"] / "data" / "downloads" / "360_v2.zip").is_file()  # zip không hỏng: giữ để chạy lại

    (fake["ws"] / "data" / "downloads" / "360_v2.zip").unlink()
    with zipfile.ZipFile(fake["src"] / "360_v2.zip", "w") as zf:
        add_scene(zf, "room/", extra_folders=("images_2",))  # thiếu images_4
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room") == 1
    err = capsys.readouterr().err
    assert "room/{images_4}/ not found in the archive" in err and "top-level entries: ['room']" in err
    assert not (root / "room").exists()


def test_a_scene_already_in_place_keeps_its_record_when_a_later_scene_fails(fake, capsys):
    with zipfile.ZipFile(fake["src"] / "360_v2.zip", "w") as zf:
        add_scene(zf, "room/")
        add_scene(zf, "kitchen/")
        for folder in ("images", "images_2", "images_4"):
            zf.writestr(f"counter/{folder}/DSC0000.JPG", b"x")
        zf.writestr("counter/sparse/cameras.bin", b"model")  # thiếu sparse/0/: bố cục sai sau khi giải nén
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room", "counter", "kitchen") == 1
    assert "counter: unexpected layout after extraction" in capsys.readouterr().err
    root = m360_root(fake)
    assert (root / "room" / "sparse" / "0").is_dir() and not (root / "counter").exists()
    (record,) = json.loads((root / ".sr3d_fetch.json").read_text(encoding="utf-8"))["fetches"]
    assert list(record["scenes"]) == ["room"] and record["sha256"]  # cảnh đã vào chỗ có nguồn và SHA-256
    (report,) = (fake["ws"] / "outputs" / "reports").glob("fetch_mipnerf360_*.json")
    report = json.loads(report.read_text(encoding="utf-8"))
    assert report["result"] == {"room": "OK", "counter": "MISSING", "kitchen": "MISSING"}
    assert "unexpected layout" in report["error"] and list(report["fetched"][0]["scenes"]) == ["room"]
    assert (fake["ws"] / "data" / "downloads" / "360_v2.zip").is_file()  # zip không hỏng: giữ để chạy lại

    (fake["ws"] / "data" / "downloads" / "360_v2.zip").unlink()
    make_360(fake["src"] / "360_v2.zip", scenes=("room", "counter", "kitchen"))
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room", "counter", "kitchen") == 0
    records = json.loads((root / ".sr3d_fetch.json").read_text(encoding="utf-8"))["fetches"]
    assert [list(r["scenes"]) for r in records] == [["room"], ["counter", "kitchen"]]


def test_damaged_archive_is_removed_so_the_next_run_downloads_again(fake, capsys):
    (fake["src"] / "360_v2.zip").write_bytes(b"not a zip at all")
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room") == 1
    assert "is damaged" in capsys.readouterr().err
    assert not (fake["ws"] / "data" / "downloads" / "360_v2.zip").exists()
    make_360(fake["src"] / "360_v2.zip")
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room") == 0
    assert len(curl_calls(fake)) == 2


def test_failed_download_leaves_no_partial_file(fake, capsys):
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "room") == 1  # server không có file: curl exit 22
    assert "curl exited with code 22" in capsys.readouterr().err
    downloads = fake["ws"] / "data" / "downloads"
    assert not downloads.exists() or not list(downloads.iterdir())


def test_dry_run_and_bad_requests(fake, capsys):
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "kitchen", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "kitchen      MISSING" in out and "360_v2.zip" in out and "--dry-run: nothing downloaded" in out
    assert not (fake["ws"] / "data").exists() and not curl_calls(fake)
    assert run_fetch(fake, "--dataset", "mipnerf360", "--scenes", "nope") == 2
    assert run_fetch(fake, "--dataset", "custom_scene") == 2  # không có mục download
    assert run_fetch(fake, "--dataset", "nope") == 2
    assert "unknown dataset" in capsys.readouterr().err


def test_every_benchmark_scene_has_a_download_source():
    for name in ("deep_blending", "mipnerf360"):
        entry = yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        sources = [scene for item in entry["download"] for scene in item["scenes"]]
        assert sorted(sources) == sorted(entry["scenes"]), name
        assert all(item["url"].startswith("https://") and item["url"].endswith(".zip") for item in entry["download"])


def test_fetch_data_sh_passes_scenes_and_options(fake):
    make_360(fake["src"] / "360_v2.zip")
    sh = ShellEnv(fake["tmp"])
    sh.ws = fake["ws"]
    env = {"WS": str(fake["ws"]), "PATH": f"{fake['curl'].parent}{os.pathsep}{sh.env['PATH']}",
           "FAKE_CURL_SRC": str(fake["src"]), "FAKE_CURL_LOG": str(fake["log"]), "KEEP_ARCHIVE": "1"}
    script = str(REPO / "scripts" / "pod" / "fetch_data.sh")
    out = sh.run(script, "mipnerf360", "room", env=env, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (m360_root(fake) / "room" / "images_4").is_dir() and not (m360_root(fake) / "counter").exists()
    assert (fake["ws"] / "data" / "downloads" / "360_v2.zip").is_file()  # KEEP_ARCHIVE=1
    out = sh.run(script, "mipnerf360", "counter", "--dry-run", env=env, timeout=60)
    assert out.returncode == 0 and "counter      MISSING" in out.stdout and not (m360_root(fake) / "counter").exists()
    assert sh.run(script, "mipnerf360", "bad name", env=env).returncode == 2
    assert sh.run(script, "Bad-Name", env=env).returncode == 2
    assert sh.run(script, "mipnerf360", "--force", env=env).returncode == 2
    assert sh.run(script, env=env).returncode == 2
    assert subprocess.run(["bash", script, "--help"], capture_output=True, text=True).returncode == 2
