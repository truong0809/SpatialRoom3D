"""Huấn luyện một cảnh bằng gsplat theo một chế độ trong configs/base.yaml (train_modes), rồi đánh giá và tóm tắt.

Chạy bằng Python của env gs-gsplat (trên pod có GPU):

    python -m src.training.run --mode default --scene /workspace/outputs/custom_scenes/phong01/colmap \\
        --scene-name phong01 --out /workspace/outputs/custom_scenes/phong01/default_s42
    python -m src.training.run --mode mcmc --scene /workspace/outputs/custom_scenes/phong01/colmap \\
        --scene-name phong01 --out /workspace/outputs/custom_scenes/phong01/mcmc_s42 \\
        --cap-max-from /workspace/outputs/custom_scenes/phong01/default_s42

Các bước: kiểm tra cảnh → chia train/test (split.json) → huấn luyện qua gsplat_launcher → đánh giá
checkpoint cuối (lệnh riêng, như benchmark của gsplat) → summary.json. Lệnh đã chạy, commit, GPU, thời gian
và mọi sai khác so với protocol được ghi vào run_meta.json.

Benchmark dùng ảnh thu nhỏ có sẵn (Mip-NeRF 360: `--dataset-protocol data_factor=4`): gsplat đọc `images_<N>/` và tự
ghi thêm `images_<N>_png/` vào thư mục dữ liệu nó được đưa. Vì vậy gsplat được đưa một bản xem dữ liệu `<out>/data`
(liên kết tới cảnh gốc), còn `images_<N>_png/` nằm trong thư mục kết quả của cảnh
(`<out>/../.images_<N>_png_<dấu mô hình COLMAP>`, dùng chung cho mọi chế độ và seed; dấu "đủ" nằm cạnh, không nằm
trong thư mục vì gsplat ghép ảnh theo thứ tự tên mọi file trong đó): bộ dữ liệu gốc không bị ghi thêm gì.

Lượt có biến thể có code (`sr3d.kr` / `sr3d.mono`, src/training/trainer_hooks.py) train bằng bản sao có vá của
simple_trainer.py (launcher `--hook`); bước đánh giá vẫn dùng file gốc. Lượt có `sr3d.mono` có thêm bước `mono` trước
bước train (src.training.mono_depth, log mono_depth.log): depth prior của cảnh nằm trong
`<out>/../.mono_depth_<dấu mô hình COLMAP>` (dùng chung cho mọi chế độ / seed), kết quả kiểm ghi vào mono_depth.json.

`--dry-run` chỉ in kế hoạch (không ghi file, không cần GPU). Một thư mục kết quả đã có lượt chạy OK chỉ được
bỏ qua khi yêu cầu mới giống hệt: cùng cảnh và cùng model COLMAP (dấu SHA-256 của sparse/0), cùng chế độ,
protocol (cả protocol gốc), tham số, tham số của hook, seed, ngân sách Gaussian và tập train/test. Đường dẫn không được
so: dời thư mục dữ liệu hay kết quả không làm train lại. Lượt dở dang được chạy lại với
`--rerun-incomplete`, trừ khi tiến trình đang chạy lượt đó vẫn còn sống.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from src.evaluation.gsplat_results import load_run, load_summary, read_json_or_none
from src.utils.gpu_lock import inherited_lock_fds
from src.reconstruction.colmap_model import model_exists, read_cameras_binary, read_images_binary
from src.training.profiles import (
    eval_args,
    hook_names,
    load_config,
    parse_assignments,
    resolve_cap_max,
    resolve_mode,
    trainer_args,
)
from src.training.mono_depth import WEIGHTS_FILE
from src.training.split import make_split, make_train_view, write_split
from src.training.trainer_hooks import hook_parameters

REPO_DIR = Path(__file__).resolve().parents[2]  # gốc repo: package `src` import được từ đây
DEFAULT_CONFIG = REPO_DIR / "configs" / "base.yaml"  # mục train_modes
UNDISTORTED_MODELS = ("PINHOLE", "SIMPLE_PINHOLE")
INRIA_MAX_WIDTH = 1600  # train.py của Inria tự thu nhỏ ảnh rộng hơn 1600 px (khi --resolution mặc định)
PROGRESS_INTERVAL = 30.0  # giây giữa hai dòng tiến độ tqdm được ghi vào log


class SceneError(RuntimeError):
    pass


def scene_fingerprint(scene_dir: Path) -> str:
    """SHA-256 của model COLMAP `sparse/0` (cameras, images, points3D): đổi khi cảnh được dựng lại."""
    digest = hashlib.sha256()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        digest.update(name.encode("ascii") + b"\0")
        with open(Path(scene_dir) / "sparse" / "0" / name, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def inspect_scene(scene_dir: Path, data_factor: int = 1) -> dict:
    """Kiểm tra cảnh có `images/` và `sparse/0/` dùng được (và `images_<N>/` khi data_factor N > 1)."""
    if data_factor < 1:
        raise SceneError(f"data_factor must be >= 1, got {data_factor}")
    images_dir = scene_dir / "images"
    model_dir = scene_dir / "sparse" / "0"
    if not images_dir.is_dir():
        raise SceneError(f"{images_dir} not found")
    if not model_exists(model_dir):
        raise SceneError(f"{model_dir} must contain cameras.bin, images.bin and points3D.bin")
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    if not images:
        raise SceneError(f"no registered images in {model_dir}")
    names = sorted(image.name for image in images.values())
    missing = [name for name in names if not (images_dir / name).is_file()]
    if missing:
        raise SceneError(f"{len(missing)} registered image(s) are missing in {images_dir}, e.g. {missing[:3]}")

    models = sorted({camera.model for camera in cameras.values()})
    sizes = sorted({(camera.width, camera.height) for camera in cameras.values()})
    factor_info = {}
    if data_factor > 1:
        factor_dir = scene_dir / f"images_{data_factor}"
        if not factor_dir.is_dir():
            raise SceneError(f"{factor_dir} not found: data_factor {data_factor} needs the dataset's downscaled "
                             f"images (images_{data_factor}/)")
        count_full = sum(1 for path in images_dir.iterdir() if path.is_file())
        count_factor = sum(1 for path in factor_dir.iterdir() if path.is_file())
        if count_factor != count_full:
            raise SceneError(f"{factor_dir} has {count_factor} files but {images_dir} has {count_full}: gsplat pairs "
                             "the two folders by sorted file name")
        sizes = sorted({(width // data_factor, height // data_factor) for width, height in sizes})
        factor_info = {"data_factor": data_factor, "factor_image_dir": str(factor_dir)}
    warnings = []
    if any(model not in UNDISTORTED_MODELS for model in models):
        warnings.append(f"camera model(s) {models} are not undistorted: gsplat undistorts internally, the Inria "
                        "baseline needs PINHOLE images (use colmap_runner, which runs image_undistorter)")
    if max(width for width, _ in sizes) > INRIA_MAX_WIDTH:
        warnings.append(f"images are wider than {INRIA_MAX_WIDTH} px: Inria train.py downsamples them, gsplat with "
                        "data_factor 1 does not, so resolutions would differ from B0 (extract frames with "
                        f"--max-side {INRIA_MAX_WIDTH})")
    image_files = [path for path in images_dir.iterdir() if path.is_file()]
    return {
        "fingerprint": scene_fingerprint(scene_dir),
        "num_registered": len(names),
        "num_image_files": len(image_files),
        "num_cameras": len(cameras),
        "camera_models": models,
        "image_sizes": [list(size) for size in sizes],
        **factor_info,
        "warnings": warnings,
    }


def git_state(path: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def gpu_name(index: str) -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", index, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.CalledProcessError):
        return None


def torch_info() -> dict:
    try:
        import torch  # có trong env gs-gsplat; không có trong môi trường test CPU
    except ImportError:
        return {}
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda}


def start_gpu_monitor(csv_path: Path, index: str):
    """Ghi memory.used mỗi 2 giây, cùng định dạng với scripts/pipeline/train_inria_b0.sh."""
    if shutil.which("nvidia-smi") is None:
        return None
    handle = open(csv_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        ["nvidia-smi", "-i", index, "--query-gpu=timestamp,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-l", "2"],
        stdout=handle, stderr=subprocess.DEVNULL)
    process.log_handle = handle  # đóng cùng lúc với tiến trình
    return process


def stop_gpu_monitor(process) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    process.log_handle.close()


def run_command(cmd: list[str], cwd: Path, env: dict, log_path: Path,
                progress_interval: float = PROGRESS_INTERVAL) -> int:
    """Chạy lệnh, in ra màn hình và ghi log; trả về mã thoát.

    Thanh tiến độ tqdm của gsplat cập nhật mỗi vòng lặp bằng ký tự '\\r'. Các đoạn kết thúc bằng '\\r'
    chỉ được ghi tối đa một lần mỗi `progress_interval` giây; dòng kết thúc bằng '\\n' luôn được ghi.
    """
    last_progress = float("-inf")
    with open(log_path, "a", encoding="utf-8") as log:
        def emit(raw: bytes, newline: bool) -> None:
            nonlocal last_progress
            if not newline:
                if not raw.strip():
                    return  # đoạn rỗng trước '\r' đầu tiên của tqdm
                now = time.monotonic()
                if now - last_progress < progress_interval:
                    return
                last_progress = now
            text = raw.decode("utf-8", errors="replace") + "\n"
            sys.stdout.write(text)
            log.write(text)

        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        # tiến trình huấn luyện giữ luôn khóa GPU của script cha (xem src.utils.gpu_lock)
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   pass_fds=inherited_lock_fds())
        assert process.stdout is not None
        buffer = b""
        while True:
            chunk = process.stdout.read1(65536)
            buffer += chunk
            while buffer:
                cut = min((i for i in (buffer.find(b"\n"), buffer.find(b"\r")) if i >= 0), default=-1)
                if cut < 0:
                    break
                if buffer[cut:cut + 1] == b"\r":
                    if cut + 1 == len(buffer) and chunk:
                        break  # chờ thêm dữ liệu để biết có phải "\r\n" không
                    if buffer[cut + 1:cut + 2] == b"\n":
                        emit(buffer[:cut], True)
                        buffer = buffer[cut + 2:]
                        continue
                    emit(buffer[:cut], False)
                else:
                    emit(buffer[:cut], True)
                buffer = buffer[cut + 1:]
            if not chunk:
                break
        if buffer:
            emit(buffer, True)
        sys.stdout.flush()
        return process.wait()


def png_cache_dir(out: Path, data_factor: int, fingerprint: str) -> Path:
    """Nơi gsplat ghi ảnh PNG thu nhỏ của cảnh: trong thư mục kết quả của cảnh, dùng chung cho mọi chế độ / seed.

    Tên có dấu của mô hình COLMAP: hai cảnh có thư mục kết quả chung cha không bao giờ dùng chung một cache.
    """
    return out.parent / f".images_{data_factor}_png_{fingerprint[:12]}"


def mono_cache_dir(out: Path, fingerprint: str) -> Path:
    """Nơi src.training.mono_depth lưu depth prior của cảnh: cạnh các lượt của cảnh, dùng chung cho mọi chế độ / seed."""
    return out.parent / f".mono_depth_{fingerprint[:12]}"


def mono_summary(record: dict | None) -> dict | None:
    """Phần chính của mono_depth.json (kết quả kiểm depth prior) để ghi vào run_meta.json."""
    if not isinstance(record, dict):
        return None
    keys = ("status", "train_images", "used", "unchecked", "flipped", "median_corr", "cache", "manifest_sha256")
    return {key: record.get(key) for key in keys} | {"low_corr": len(record.get("low_corr") or [])}


def cache_marker(cache_dir: Path) -> Path:
    """Dấu "cache đủ" nằm CẠNH thư mục cache, không nằm trong: gsplat (examples/datasets/colmap.py @6e8c837) liệt kê
    mọi file trong images_<N>_png rồi ghép với images/ theo thứ tự tên, nên một file lạ trong đó làm lệch cặp ảnh."""
    return cache_dir.with_name(cache_dir.name + ".complete")


def _rel_files(folder: Path) -> set[str]:
    """Đường dẫn tương đối của mọi file trong thư mục (đệ quy), như `_get_rel_paths` của gsplat."""
    return {os.path.relpath(os.path.join(root, name), folder)
            for root, _dirs, names in os.walk(folder) for name in names}


def expected_png_files(scene_dir: Path) -> set[str]:
    """Các file PNG mà gsplat ghi cho cảnh: `<tên ảnh trong images/ bỏ đuôi>.png` (như `_resize_image_folder`)."""
    return {os.path.splitext(name)[0] + ".png" for name in _rel_files(scene_dir / "images")}


def gsplat_converts(scene_dir: Path, data_factor: int) -> bool:
    """gsplat (colmap.py @6e8c837) chỉ thu nhỏ lại `images/` thành PNG khi file đầu tiên (theo tên) trong
    `images_<N>/` có đuôi .jpg; nếu không, nó đọc thẳng `images_<N>/` và không dùng cache."""
    files = sorted(_rel_files(scene_dir / f"images_{data_factor}"))
    return bool(files) and os.path.splitext(files[0])[1].lower() == ".jpg"


def cache_is_complete(cache_dir: Path, scene_dir: Path, fingerprint: str) -> bool:
    """Cache dùng lại được: có dấu hoàn tất của đúng mô hình COLMAP và chứa đúng, đủ các file gsplat cần."""
    marker = read_json_or_none(cache_marker(cache_dir))
    if not isinstance(marker, dict) or marker.get("fingerprint") != fingerprint or not cache_dir.is_dir():
        return False
    return _rel_files(cache_dir) == expected_png_files(scene_dir)


def add_factor_links(scene_dir: Path, view_dir: Path, data_factor: int, cache_dir: Path, fingerprint: str) -> dict:
    """Thêm vào bản xem dữ liệu: `images_<N>` (liên kết tới cảnh gốc) và `images_<N>_png` (liên kết tới cache).

    Cache chỉ được dùng lại khi đã đủ (`cache_is_complete`, dấu ghi sau một lượt huấn luyện xong). Cache dở dang
    (lượt bị dừng lúc gsplat đang ghi PNG) hoặc có file lạ bị xóa để gsplat ghi lại từ đầu, vì gsplat bỏ qua mọi
    file PNG đã có và ghép ảnh theo thứ tự tên.
    """
    view_dir.mkdir(parents=True, exist_ok=True)
    for name in ("poses_bounds.npy", "ext_metadata.json"):  # gsplat đọc thêm nếu có
        if (scene_dir / name).is_file() and not (view_dir / name).exists():
            (view_dir / name).symlink_to((scene_dir / name).resolve())
    (view_dir / f"images_{data_factor}").symlink_to((scene_dir / f"images_{data_factor}").resolve(),
                                                    target_is_directory=True)
    reused = cache_is_complete(cache_dir, scene_dir, fingerprint)
    if not reused:
        cache_marker(cache_dir).unlink(missing_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (view_dir / f"images_{data_factor}_png").symlink_to(cache_dir.resolve(), target_is_directory=True)
    return {"data_factor": data_factor, "png_cache": str(cache_dir), "png_cache_reused": reused,
            "png_cache_used": gsplat_converts(scene_dir, data_factor)}


def make_factor_view(scene_dir: Path, view_dir: Path, data_factor: int, cache_dir: Path, fingerprint: str) -> dict:
    """Bản xem dữ liệu cho data_factor > 1 khi không có danh sách test riêng: mọi thứ đều là liên kết."""
    view_dir.mkdir(parents=True, exist_ok=True)
    (view_dir / "images").symlink_to((scene_dir / "images").resolve(), target_is_directory=True)
    (view_dir / "sparse").symlink_to((scene_dir / "sparse").resolve(), target_is_directory=True)
    return add_factor_links(scene_dir, view_dir, data_factor, cache_dir, fingerprint)


def mark_cache_complete(cache_dir: Path, scene_dir: Path, fingerprint: str) -> bool:
    """Ghi dấu hoàn tất (cạnh thư mục cache) khi cache có đúng, đủ các file PNG gsplat cần; False nếu chưa đủ."""
    files = _rel_files(cache_dir)
    if files != expected_png_files(scene_dir):
        return False
    _write_json(cache_marker(cache_dir), {"fingerprint": fingerprint, "files": len(files), "written": _now()})
    return True


def request_differences(meta: dict, spec, seed: int, cap_max: int | None, split_digest: str,
                        scene: str | None = None, fingerprint: str | None = None) -> list[str]:
    """Các khác biệt giữa một lượt chạy đã có (run_meta.json) và yêu cầu hiện tại."""
    previous = meta.get("spec", {})
    pairs = {
        "scene": (meta.get("scene"), scene),
        "scene_model": ((meta.get("scene_info") or {}).get("fingerprint"), fingerprint),
        "preset": (previous.get("preset"), spec.preset),
        "protocol": (previous.get("protocol"), spec.protocol),
        # protocol gốc (gồm quy ước của benchmark, vd. data_factor của Mip-NeRF 360): cùng protocol cuối nhưng một
        # bên là sai khác (--protocol) và một bên là quy ước (--dataset-protocol) thì không phải cùng một lượt
        "base_protocol": (previous.get("base_protocol"), spec.base_protocol),
        "args": (previous.get("args"), spec.args),
        # hằng số của biến thể có code (src/training/trainer_hooks.py); lượt không có hook: None ở cả hai bên
        "hooks": (meta.get("hooks"), hook_parameters(hook_names(spec))),
        "seed": (meta.get("seed"), seed),
        "cap_max": ((meta.get("cap_max") or {}).get("value"), cap_max),
        "split": (meta.get("split", {}).get("sha256"), split_digest),
    }
    return [name for name, (old, new) in pairs.items() if old != new]


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    """Ghi qua file tạm rồi đổi tên: pod tắt giữa chừng không để lại JSON hỏng."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _owner() -> dict:
    return {"host": socket.gethostname(), "pid": os.getpid()}


def owner_alive(meta: dict) -> bool:
    """Tiến trình đã ghi run_meta.json (lượt đang RUNNING) còn chạy trên máy này không."""
    owner = meta.get("owner") or {}
    pid = owner.get("pid")
    if owner.get("host") != socket.gethostname() or not isinstance(pid, int) or pid == os.getpid():
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "src.training.run" in cmdline


def backup_name(out: Path) -> Path:
    """Tên cho thư mục cũ trước khi làm lại: <tên>.prev_<thời gian>[_<số>] (không ghi đè, không xóa)."""
    stamp = f"{dt.datetime.now():%Y%m%d_%H%M%S}"
    backup, counter = out.with_name(f"{out.name}.prev_{stamp}"), 1
    while backup.exists():
        backup, counter = out.with_name(f"{out.name}.prev_{stamp}_{counter}"), counter + 1
    return backup


def is_stale(meta: dict, differences: list[str], budget_from_baseline: bool) -> bool:
    """Lượt OK cũ làm từ dữ liệu cũ (dùng cho --replace-stale), không phải do người chạy chọn cấu hình khác.

    - Mô hình COLMAP của cảnh đã đổi (`scene_model`): cả lượt đã lỗi thời, dù cấu hình khác hay không.
    - Chỉ ngân sách Gaussian khác, cả lượt cũ lẫn yêu cầu mới đều lấy ngân sách từ lượt baseline
      (`match_baseline`): lượt baseline đã train lại.
    Tập train/test khác mà mô hình không đổi chỉ có thể do --test-list: là cấu hình người chạy chọn, không thay.
    """
    if "scene" in differences:
        return False  # thư mục của cảnh khác: không bao giờ thay
    if "scene_model" in differences:
        return True
    old_policy = (meta.get("cap_max") or {}).get("policy")
    return set(differences) == {"cap_max"} and budget_from_baseline and old_policy == "match_baseline"


def check_previous_run(out: Path, differences_for, force: bool, rerun_incomplete: bool,
                       stale_check=None) -> str:
    """Quyết định với thư mục kết quả đã có: 'new' | 'skip' | 'replace'. Lỗi -> ValueError.

    `differences_for(meta)` trả về danh sách khác biệt giữa lượt cũ và yêu cầu hiện tại.
    `stale_check(meta, differences)` (với --replace-stale, xem `is_stale`): lượt OK làm từ dữ liệu cũ thì được
    thay, không báo lỗi.
    """
    if not (out.exists() and any(out.iterdir())):
        return "new"
    meta = read_json_or_none(out / "run_meta.json")
    if meta is None:
        meta = {}
        status = "UNREADABLE" if (out / "run_meta.json").exists() else "MISSING"
    else:
        status = load_summary(out).status
    if meta.get("status") == "RUNNING" and owner_alive(meta):
        raise ValueError(f"{out} is being written by a run that is still going (pid {meta['owner']['pid']}); "
                         "wait for it or stop it first")
    differences = differences_for(meta)
    if status == "OK" and not force:
        if not differences:
            return "skip"
        if stale_check is not None and stale_check(meta, differences):
            print(f"[run] {out} holds an OK run made from older inputs ({', '.join(differences)} changed); "
                  "it will be replaced (the old folder is renamed, not deleted)")
            return "replace"
        raise ValueError(f"{out} holds an OK run with a different configuration ({', '.join(differences)}); "
                         "use another --out, or --force to replace it (the old folder is renamed)")
    if not (force or rerun_incomplete):
        raise ValueError(f"{out} exists but is not a finished run (status {status}); "
                         "use --rerun-incomplete or --force (the old folder is renamed)")
    return "replace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", required=True, help="tên chế độ trong file cấu hình (default, mcmc, ...)")
    parser.add_argument("--scene", required=True, type=Path, help="thư mục cảnh có images/ và sparse/0/")
    parser.add_argument("--scene-name", help="tên cảnh ghi vào kết quả (mặc định: tên thư mục cảnh)")
    parser.add_argument("--out", required=True, type=Path, help="thư mục kết quả của lượt chạy này")
    parser.add_argument("--gsplat-dir", type=Path, default=Path(os.environ.get("WS", "/workspace")) / "code" / "gsplat")
    parser.add_argument("--depth-anything-dir", type=Path, default=None,
                        help="code Depth Anything V2 cho lượt có sr3d.mono (mặc định: <gsplat-dir>/../depth_anything_v2)")
    parser.add_argument("--seed", type=int, default=None, help="mặc định: default_seed trong file cấu hình")
    parser.add_argument("--test-list", type=Path, help="file danh sách ảnh test (thay cho mỗi ảnh thứ test_every)")
    cap = parser.add_mutually_exclusive_group()
    cap.add_argument("--cap-max", type=int, help="giới hạn số Gaussian cho MCMC (ghi đè chính sách trong cấu hình)")
    cap.add_argument("--cap-max-from", type=Path, help="thư mục lượt chạy baseline cùng cảnh để lấy số Gaussian")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="tham số riêng của chế độ, vd. --set opacity_reg=0.001 (lặp lại được)")
    parser.add_argument("--protocol", action="append", default=[], metavar="KEY=VALUE",
                        help="đổi protocol (bị ghi là sai khác, vd. max_steps=3000 để chạy thử)")
    parser.add_argument("--dataset-protocol", action="append", default=[], metavar="KEY=VALUE",
                        help="quy ước của benchmark, chỉ data_factor (vd. data_factor=4 cho cảnh ngoài trời của "
                             "Mip-NeRF 360): thay protocol gốc, không bị ghi là sai khác")
    parser.add_argument("--tag", help="tên biến thể của chế độ (mục variants của kế hoạch thực nghiệm), chỉ ghi vào "
                                      "run_meta.json; thư mục kết quả do --out quyết định")
    parser.add_argument("--gpu", default="0", help="chỉ số GPU khi CUDA_VISIBLE_DEVICES chưa đặt")
    parser.add_argument("--python", default=sys.executable, help="Python của env gs-gsplat")
    parser.add_argument("--force", action="store_true", help="chạy lại; thư mục cũ được đổi tên, không xóa")
    parser.add_argument("--replace-stale", action="store_true",
                        help="thay lượt OK cũ nếu nó làm từ dữ liệu cũ: mô hình COLMAP của cảnh đã đổi, hoặc chỉ "
                             "ngân sách Gaussian lấy từ baseline đã đổi (thư mục cũ đổi tên, không xóa)")
    parser.add_argument("--rerun-incomplete", action="store_true",
                        help="chạy lại nếu thư mục kết quả là lượt dở dang / lỗi (không đụng tới lượt OK)")
    parser.add_argument("--allow-gsplat-mismatch", action="store_true",
                        help="cho phép checkout gsplat khác commit trong cấu hình (ghi vào run_meta)")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None, runner: Callable[..., int] = run_command) -> int:
    args = build_parser().parse_args(argv)
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        print(f"ERROR: invalid --tag {args.tag!r} (letters, digits, '_' and '-')", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        protocol_overrides = parse_assignments(args.protocol)
        spec = resolve_mode(config, args.mode, parse_assignments(args.set), protocol_overrides,
                            parse_assignments(args.dataset_protocol))
        scene_dir = args.scene.resolve()
        scene_name = args.scene_name or scene_dir.name
        data_factor = int(spec.protocol["data_factor"])
        scene_info = inspect_scene(scene_dir, data_factor)
        split = make_split(scene_dir, spec.protocol["test_every"], args.test_list)
        seed = args.seed if args.seed is not None else int(config.get("default_seed", 42))
        baseline = load_summary(args.cap_max_from).to_dict() if args.cap_max_from else None
        cap_max, cap_source = resolve_cap_max(spec, args.cap_max, baseline, seed,
                                              scene=scene_name, scene_fingerprint=scene_info["fingerprint"])
    except (ValueError, SceneError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if baseline is not None and baseline.get("split_sha256") != split.digest():
        print("ERROR: the baseline run used a different train/test split; the Gaussian budget would not be "
              "comparable", file=sys.stderr)
        return 2

    out = args.out.resolve()
    examples = args.gsplat_dir.resolve() / "examples"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and "," in visible:
        print("ERROR: CUDA_VISIBLE_DEVICES lists several GPUs; training must use exactly one", file=sys.stderr)
        return 2
    gpu_index = visible if visible not in (None, "") else str(args.gpu)

    hooks = hook_names(spec)
    hook_info = hook_parameters(hooks)
    mono_cache = mono_cache_dir(out, scene_info["fingerprint"]) if "mono" in hooks else None
    depth_anything = (args.depth_anything_dir or args.gsplat_dir.parent / "depth_anything_v2").resolve()

    def launcher(record: str, resolved: str, trainer: list[str], with_hooks: bool = False) -> list[str]:
        cmd = [args.python, "-m", "src.training.gsplat_launcher", "--examples-dir", str(examples),
               "--seed", str(seed), "--record-split", str(out / record), "--resolved-config", str(out / resolved)]
        if split.rule == "test_list":
            cmd += ["--split-file", str(out / "split.json")]
        if with_hooks:  # chỉ bước train; bước đánh giá chạy simple_trainer.py gốc (hook không đổi gì khi đánh giá)
            for name in hooks:
                cmd += ["--hook", name]
            if mono_cache is not None:
                cmd += ["--mono-dir", str(mono_cache)]
        return cmd + ["--", *trainer]

    # Với danh sách test riêng, gsplat đọc một bản dữ liệu đã lọc điểm SfM (xem split.make_train_view); với
    # data_factor > 1, một bản xem dữ liệu để gsplat không ghi images_<N>_png/ vào bộ dữ liệu gốc.
    data_dir = out / "data" if split.rule == "test_list" or data_factor > 1 else scene_dir
    train_cmd = launcher("split_applied.json", "resolved_config.json", trainer_args(spec, data_dir, out, cap_max),
                         with_hooks=True)
    eval_cmd = launcher("split_applied_eval.json", "resolved_config_eval.json",
                        eval_args(spec, data_dir, out, cap_max))
    mono_cmd = None
    if mono_cache is not None:  # depth prior đọc ảnh và mô hình COLMAP của cảnh gốc (không qua bản xem dữ liệu)
        mono_cmd = [args.python, "-m", "src.training.mono_depth", "--scene", str(scene_dir),
                    "--fingerprint", scene_info["fingerprint"], "--cache", str(mono_cache),
                    "--split", str(out / "split.json"), "--record", str(out / "mono_depth.json"),
                    "--depth-anything-dir", str(depth_anything)]
    warnings = list(scene_info["warnings"])
    if spec.deviations:
        warnings.append(f"protocol deviations: {sorted(spec.deviations)} - not comparable with protocol runs")

    if args.dry_run:
        plan = {"spec": spec.to_dict(), "seed": seed, **({"tag": args.tag} if args.tag else {}),
                "cap_max": cap_source, "scene_info": scene_info, "data_dir": str(data_dir),
                **({"png_cache": str(png_cache_dir(out, data_factor, scene_info["fingerprint"]))}
                   if data_factor > 1 else {}),
                "split": {k: v for k, v in split.to_dict().items() if k not in ("train", "test")},
                **({"hooks": hook_info} if hooks else {}),
                **({"mono": shlex.join(mono_cmd)} if mono_cmd else {}),
                "train": shlex.join(train_cmd), "eval": shlex.join(eval_cmd), "warnings": warnings}
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not (examples / "simple_trainer.py").is_file():
        print(f"ERROR: {examples / 'simple_trainer.py'} not found (run scripts/setup/setup_gsplat.sh)", file=sys.stderr)
        return 2
    stale = None
    if args.replace_stale:
        stale = functools.partial(
            is_stale, budget_from_baseline=(spec.cap_max_policy or {}).get("policy") == "match_baseline")
    try:
        action = check_previous_run(
            out, lambda meta: request_differences(meta, spec, seed, cap_max, split.digest(),
                                                  scene_name, scene_info["fingerprint"]),
            args.force, args.rerun_incomplete, stale)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if action == "skip":
        print(f"[run] {out} already has an OK run with the same configuration; skipping")
        return 0

    # Mọi kiểm tra có thể thất bại chạy TRƯỚC khi đụng tới thư mục cũ, để lượt OK cũ không bị dời đi vô ích.
    gsplat_commit, gsplat_dirty = git_state(args.gsplat_dir)
    expected = config.get("trainer", {}).get("commit")
    if expected and gsplat_commit != expected:
        message = f"gsplat checkout is at {gsplat_commit}, configuration expects {expected}"
        if not args.allow_gsplat_mismatch:
            print(f"ERROR: {message} (run scripts/setup/setup_gsplat.sh or pass --allow-gsplat-mismatch)", file=sys.stderr)
            return 2
        warnings.append(message)
    if mono_cmd is not None and not (depth_anything / WEIGHTS_FILE).is_file():
        print(f"ERROR: {depth_anything / WEIGHTS_FILE} not found: the mono variant needs Depth Anything V2 Small "
              "(run bash scripts/setup/setup_depth_anything.sh)", file=sys.stderr)
        return 2
    train_view = None
    factor_view = None
    staged_view = out.with_name(f".{out.name}.data_tmp")
    if split.rule == "test_list":
        shutil.rmtree(staged_view, ignore_errors=True)
        try:
            train_view = make_train_view(scene_dir, staged_view, split)
        except ValueError as error:
            shutil.rmtree(staged_view, ignore_errors=True)
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        train_view["data_dir"] = str(data_dir)
        print(f"[run] test list: {train_view['points_kept']}/{train_view['points_total']} SfM points are seen by "
              f">= {train_view['min_train_views']} training images and kept for initialisation")
    if data_factor > 1:
        cache_dir = png_cache_dir(out, data_factor, scene_info["fingerprint"])
        if train_view is None:
            shutil.rmtree(staged_view, ignore_errors=True)
            factor_view = make_factor_view(scene_dir, staged_view, data_factor, cache_dir, scene_info["fingerprint"])
        else:
            factor_view = add_factor_links(scene_dir, staged_view, data_factor, cache_dir, scene_info["fingerprint"])
        print(f"[run] data_factor {data_factor}: gsplat reads {data_dir} (links to {scene_dir}); downscaled PNGs in "
              f"{factor_view['png_cache']}" + (" (reused)" if factor_view["png_cache_reused"] else ""))

    if action == "replace":
        backup = backup_name(out)
        out.rename(backup)
        print(f"[run] previous results moved to {backup}")
    out.mkdir(parents=True, exist_ok=True)
    if train_view is not None or factor_view is not None:
        staged_view.rename(data_dir)
    write_split(split, out / "split.json")

    repo_commit, repo_dirty = git_state(REPO_DIR)
    for message in warnings:
        print(f"WARN: {message}")

    meta = {
        "scene": scene_name,
        "scene_dir": str(scene_dir),
        "mode": spec.mode,
        "seed": seed,
        **({"tag": args.tag} if args.tag else {}),
        "status": "RUNNING",
        "failed_step": None,
        "spec": spec.to_dict(),
        "cap_max": cap_source,
        "split": {k: v for k, v in split.to_dict().items() if k not in ("train", "test")},
        "scene_info": scene_info,
        "data_dir": str(data_dir),
        "train_view": train_view,
        **({"factor_view": factor_view} if factor_view is not None else {}),
        **({"hooks": hook_info} if hooks else {}),
        "commands": {**({"mono": mono_cmd} if mono_cmd else {}), "train": train_cmd, "eval": eval_cmd},
        "gsplat_dir": str(args.gsplat_dir.resolve()),
        "gsplat_commit": gsplat_commit,
        "gsplat_dirty": gsplat_dirty,
        "gsplat_commit_expected": expected,
        "repo_commit": repo_commit,
        "repo_dirty": repo_dirty,
        "gpu": gpu_name(gpu_index),
        "cuda_visible_devices": gpu_index,
        "python": sys.version.split()[0],
        **torch_info(),
        "warnings": warnings,
        "owner": _owner(),
        "started": _now(),
    }
    meta_path = out / "run_meta.json"
    _write_json(meta_path, meta)

    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # cùng thứ tự với chỉ số mà nvidia-smi -i dùng
    env["CUDA_VISIBLE_DEVICES"] = gpu_index
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"  # Jupyter đặt backend inline; env gs-gsplat không có gói đó (matplotlib báo lỗi)
    env["PYTHONPATH"] = str(REPO_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    steps = [("train", train_cmd, "train.log", examples), ("eval", eval_cmd, "eval.log", examples)]
    if mono_cmd is not None:
        steps.insert(0, ("mono", mono_cmd, "mono_depth.log", REPO_DIR))
    for step, cmd, log_name, cwd in steps:
        print(f"\n===== [{dt.datetime.now():%H:%M:%S}] {step}: {spec.mode} / {meta['scene']} =====")
        monitor = start_gpu_monitor(out / "gpu_log.csv", gpu_index) if step == "train" else None
        started = time.monotonic()
        try:
            code = runner(cmd, cwd, env, out / log_name)
        finally:
            stop_gpu_monitor(monitor)
        meta[f"{step}_seconds"] = round(time.monotonic() - started, 1)
        if step == "mono":
            meta["mono_depth"] = mono_summary(read_json_or_none(out / "mono_depth.json"))
        if code == 0 and step == "train" and factor_view is not None and factor_view["png_cache_used"] \
                and not mark_cache_complete(Path(factor_view["png_cache"]), scene_dir, scene_info["fingerprint"]):
            warnings.append(f"PNG cache {factor_view['png_cache']} does not hold exactly one PNG per image after "
                            "training: not marked complete, the next run rebuilds it")
            print(f"WARN: {warnings[-1]}")
        if code != 0:
            meta.update(status="FAILED", failed_step=step, finished=_now(), exit_code=code)
            _write_json(meta_path, meta)
            print(f"ERROR: {step} failed with exit code {code}; see {out / log_name}", file=sys.stderr)
            return 1
        _write_json(meta_path, meta)

    meta["finished"] = _now()
    meta["status"] = "DONE"
    _write_json(meta_path, meta)
    result = load_run(out)
    if result.status != "OK":
        meta["status"] = "FAILED"
        meta["failed_step"] = "collect"
        _write_json(meta_path, meta)
        result = load_run(out)
    _write_json(out / "summary.json", result.to_dict())
    metrics = result.metrics or {}
    print(f"\n[run] {result.status} | {spec.mode} / {meta['scene']} | PSNR {metrics.get('PSNR', float('nan')):.3f} "
          f"SSIM {metrics.get('SSIM', float('nan')):.4f} LPIPS {metrics.get('LPIPS', float('nan')):.4f} | "
          f"Gaussians {result.num_gaussians} | train {meta['train_seconds'] / 60:.1f} min | {out / 'summary.json'}")
    if result.split_applied_ok is False:
        print("WARN: the images gsplat used differ from split.json", file=sys.stderr)
    return 0 if result.status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
