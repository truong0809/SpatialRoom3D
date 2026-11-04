"""Test trang web trong trình duyệt thật (Chromium headless của Playwright, WebGL2 bằng SwiftShader trên CPU).

Chỉ chạy khi đặt SR3D_BROWSER=1 và có Playwright (chậm: SwiftShader vẽ vài giây một khung):
  SR3D_BROWSER=1 python -m pytest tests/browser -q            # SR3D_BROWSER_SHOTS=<thư mục>: lưu ảnh chụp màn hình

Kiểm tra trọn luồng của người dùng với một bản dựng giả (không GPU): đăng nhập -> thư viện -> dựng phòng mới từ ảnh
(tải lên từng phần thật) -> trang tiến độ đi qua các bước -> xem 3D. Và kiểm tra camera của trình xem khớp mô hình:
các điểm mốc của phòng tổng hợp (tests/helpers/synthetic_room.py) phải hiện đúng pixel mà cameras.json tính ra.
"""

import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SR3D_BROWSER") != "1", reason="đặt SR3D_BROWSER=1 để chạy test trình duyệt")
playwright_api = pytest.importorskip("playwright.sync_api") if os.environ.get("SR3D_BROWSER") == "1" else None

import numpy as np  # noqa: E402

from app.web import server as web_server  # noqa: E402
from synthetic_room import make_room  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TOKEN = "browser-test"
SHOTS = Path(os.environ["SR3D_BROWSER_SHOTS"]) if os.environ.get("SR3D_BROWSER_SHOTS") else None

# Bản dựng giả: build_room.sh --background trả lời ngay, một tiến trình nền ghi progress.json qua đủ các bước như
# luồng thật (src.product.progress), rồi đưa một bản sao của phòng tổng hợp vào thư viện.
FAKE_BUILD = r'''#!/usr/bin/env bash
room="$2"; name="$4"
setsid nohup "$FAKE_PYTHON" "$FAKE_WORKER" "$room" "$name" "$1" > "$FAKE_LOG_DIR/worker_$room.log" 2>&1 < /dev/null &
echo "Đã chạy nền 'room_$room'."
'''
FAKE_WORKER = r'''
import json, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["SR3D_REPO"])
from src.product import progress, export_web
room, name, source = sys.argv[1], sys.argv[2], sys.argv[3]
ws = Path(os.environ["WS"])
root = ws / "outputs" / "product"
file = root / room / "progress.json"
log = ws / "outputs" / "logs" / f"build_room_{room}.log"
log.parent.mkdir(parents=True, exist_ok=True)
kind = "photos" if Path(source).is_dir() else "video"
progress.queue(file, room, kind, f"room_{room}")
time.sleep(0.5)
progress.start(file, room, kind, f"room_{room}", str(log))
for stage in ("images", "colmap", "train"):
    progress.set_stage(file, stage)
    log.open("a").write(f"===== {stage}\n")
    time.sleep(0.6)
for pct in (25, 50, 75, 100):
    progress.update(file, "train", pct, f"Vòng {pct * 300}/30.000")
    time.sleep(0.3)
progress.set_stage(file, "export")
src = root / "web" / "rooms" / "phong_tong_hop"
dst = root / "web" / "rooms" / room
shutil.copytree(src, dst)
entry = json.loads((dst / "room.json").read_text())
entry.update(id=room, name=name, files={"model": f"rooms/{room}/model.splat", "cameras": f"rooms/{room}/cameras.json",
                                         "thumbnail": None})
entry["source"]["kind"] = kind
entry["model"]["trained_at"] = progress.now()
(dst / "room.json").write_text(json.dumps(entry, ensure_ascii=False))
export_web.rebuild_manifest(root / "web")
progress.done(file, "Xong: phòng đã có trong thư viện")
'''


def wait_stable(page, frame_index, attempts=8, pause=4.0):
    """Ảnh khung xem sau khi đổi camera: Spark sắp xếp lại Gaussian bất đồng bộ (SwiftShader chậm), nên chờ tới khi
    hai ảnh liên tiếp giống nhau."""
    from PIL import Image

    page.evaluate(f"window.__sr3dViewer.viewFrame({frame_index})")
    previous = None
    for _ in range(attempts):
        time.sleep(pause)
        url = page.evaluate("window.__sr3dViewer.capture()")
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")).astype(int)
        if previous is not None and np.abs(image - previous).mean() < 0.5:
            return image
        previous = image
    return previous


def project(cameras, index, point):
    frame = cameras["frames"][index]
    position, forward, up = (np.array(frame[key], dtype=float) for key in ("position", "forward", "up"))
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    d = np.asarray(point) - position
    z = d @ forward
    k = cameras["intrinsics"]
    return None if z <= 0 else (k["fx"] * (d @ right) / z + k["cx"], k["fy"] * -(d @ up) / z + k["cy"])


@pytest.fixture
def site(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    root = ws / "outputs" / "product"
    web_dir = root / "web"
    room = make_room(web_dir, gaussians=int(os.environ.get("SR3D_BROWSER_GAUSSIANS", "20000")))
    build = tmp_path / "fake_build.sh"
    build.write_text(FAKE_BUILD, encoding="utf-8")
    build.chmod(0o755)
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    for key, value in {"FAKE_PYTHON": sys.executable, "FAKE_WORKER": str(worker), "FAKE_LOG_DIR": str(tmp_path),
                       "SR3D_REPO": str(REPO), "WS": str(ws), "JOB_LOCK": str(tmp_path / "gpu.lock")}.items():
        monkeypatch.setenv(key, value)
    settings = web_server.Settings(ws=ws, web_dir=web_dir, product_root=root, token=TOKEN, build_script=build)
    httpd = web_server.make_server(settings, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield {"url": f"http://127.0.0.1:{httpd.server_address[1]}", "room": room, "web": web_dir, "tmp": tmp_path}
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def page(site):
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-swiftshader"])
        context = browser.new_context(viewport={"width": 1000, "height": 700}, device_scale_factor=1, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.errors = errors
        page.goto(site["url"] + "/")
        page.wait_for_selector("#login-dialog[open]", timeout=20000)
        page.fill("#login-token", TOKEN)
        page.click("#login-form button[type=submit]")
        page.wait_for_selector(".card", timeout=20000)
        yield page
        browser.close()


def shot(page, name):
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / f"{name}.png"), timeout=180000)


def test_upload_photos_follow_the_progress_then_view_the_room(page, site):
    from PIL import Image

    assert "Phòng tổng hợp" in page.inner_text("#lib-body")
    shot(page, "1_library")
    page.click("a[href='#/new']")
    page.click("[data-kind=photos]")
    photos = site["tmp"] / "photos"
    photos.mkdir()
    for i in range(12):
        Image.new("RGB", (64, 48), (20 * i, 100, 200)).save(photos / f"IMG_{i:04d}.jpg")
    page.set_input_files("#up-input", [str(p) for p in sorted(photos.iterdir())])
    assert "12 ảnh" in page.inner_text("#up-files") and "nên từ 30 ảnh" in page.inner_text("#up-files")
    page.fill("#up-name", "Phòng thử web")
    assert page.input_value("#up-room") == "phong_thu_web"
    shot(page, "2_upload")
    page.click("#up-start")
    page.wait_for_url("**/#/room/phong_thu_web", timeout=60000)
    page.wait_for_selector("#job-steps li", timeout=20000)
    page.wait_for_function("document.querySelector('#job-status').textContent.includes('đã xong')", timeout=60000)
    steps = page.eval_on_selector_all("#job-steps li", "items => items.map(li => li.className)")
    assert steps == ["done"] * 4
    assert "giây" in page.inner_text("#job-steps")  # thời gian từng bước
    page.wait_for_timeout(1000)  # thanh tiến độ có hiệu ứng chuyển 0,4 giây
    ratio = page.evaluate("(() => { const bar = document.querySelector('#job-bar');"
                          " return bar.firstElementChild.getBoundingClientRect().width / bar.getBoundingClientRect().width; })()")
    assert ratio > 0.99, ratio
    shot(page, "3_progress_done")
    uploaded = sorted(p.name for p in (site["tmp"] / "ws" / "data" / "uploads" / "phong_thu_web").iterdir())
    assert uploaded == [".sr3d_upload.json"] + [f"IMG_{i:04d}.jpg" for i in range(12)]  # + tên đặt lúc tải lên
    page.click("a[href='#/view/phong_thu_web']")
    page.wait_for_function("window.__sr3dViewer && window.__sr3dViewer.ready", timeout=180000)
    assert page.inner_text("h1") == "Phòng thử web"
    assert not page.errors, page.errors


def test_the_viewer_camera_matches_the_model(page, site):
    """Điểm mốc (magenta) hiện đúng pixel mà cameras.json tính ra: camera của trình xem, hệ tọa độ của Spark và file
    xuất của sản phẩm khớp nhau (sai lệch ≤ 3 pixel)."""
    room = site["room"]
    cameras = json.loads((site["web"] / "rooms" / room["room"] / "cameras.json").read_text(encoding="utf-8"))
    page.goto(f"{page.url.split('#')[0]}#/view/{room['room']}")
    page.wait_for_function("window.__sr3dViewer && window.__sr3dViewer.ready", timeout=180000)
    errors = []
    for index in (12, 24, 36):
        image = wait_stable(page, index)
        if SHOTS:
            from PIL import Image

            SHOTS.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image.astype(np.uint8)).save(SHOTS / f"frame_{index}.png")
        height, width = image.shape[:2]
        scale = height / cameras["intrinsics"]["height"]
        magenta = (image[..., 0] > 170) & (image[..., 2] > 170) & (image[..., 1] < 110)
        ys, xs = np.nonzero(magenta)
        for marker in room["markers"]:
            uv = project(cameras, index, marker)
            if uv is None:
                continue
            u = width / 2 + (uv[0] - cameras["intrinsics"]["cx"]) * scale
            v = height / 2 + (uv[1] - cameras["intrinsics"]["cy"]) * scale
            if not (50 < u < width - 50 and 50 < v < height - 50):
                continue
            near = (xs - u) ** 2 + (ys - v) ** 2 < 45 ** 2
            assert near.sum() > 30, f"frame {index}: marker not visible near ({u:.0f}, {v:.0f})"
            errors.append(float(np.hypot(xs[near].mean() - u, ys[near].mean() - v)))
    # camera sai (hướng, góc nhìn, hệ trục) làm điểm mốc lệch hàng chục pixel hoặc biến mất; vài pixel là do điểm mốc bị
    # che một phần (Gaussian của tường vẽ đè khi thứ tự sắp xếp chưa cập nhật xong)
    assert len(errors) >= 4 and max(errors) <= 8.0 and float(np.median(errors)) <= 3.0, errors
    for mode in ("walk", "tour", "orbit"):
        page.click(f"[data-mode={mode}]")
        time.sleep(2)
        assert page.evaluate("window.__sr3dViewer.cameraState().mode") == mode
    shot(page, "4_viewer")
    with page.expect_download(timeout=120000) as download:
        page.click("#tb-shot")
    assert download.value.suggested_filename.endswith(".png")
    assert not page.errors, page.errors
