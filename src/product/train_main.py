"""Huấn luyện MÔ HÌNH CHÍNH của sản phẩm (3DGS-MCMC trên gsplat) cho một phòng đã chuẩn bị (ảnh + COLMAP).

File riêng của luồng sản phẩm, không dùng code thực nghiệm (src/training/): chạy thẳng examples/simple_trainer.py của
thư viện gsplat (commit ghim trong configs/product.yaml), preset mcmc, đúng cấu hình của configs/product.yaml. Thuật toán
(rasterization, MCMCStrategy) là của gsplat; file này dựng lệnh train, kiểm tra điều kiện, chạy và ghi kết quả. Không chấm
điểm, không render video: kết quả là mô hình để src/product/export_web.py xuất cho web.

So với lượt mcmc của thực nghiệm chính (cùng preset, cùng tham số, cùng điều kiện train), có chủ ý khác hai điểm:
  - số Gaussian tối đa cố định (model.cap_max) thay cho "bằng số Gaussian cuối của lượt default cùng cảnh": không phải
    train default trước, và biết trước dung lượng file web;
  - training.test_every rất lớn: mọi ảnh trừ ảnh số 0 dùng để train (thực nghiệm giữ 1/8 số ảnh để chấm điểm).

Các bước của main(), theo thứ tự các phần trong file:
  1. đọc cấu hình        product_config.load         configs/product.yaml
  2. dựng lệnh train     trainer_command              preset mcmc + mục training + model.cap_max
  3. kiểm tra            check_scene, is_current      phòng đã có COLMAP; đã train khớp thì dừng ở đây, không train lại
                         prepare_training             đúng gsplat (commit, gói đã cài), GPU dùng được, thư mục kết quả
  4. train               run_trainer, TrainerOutput   chạy simple_trainer.py; log, % train cho progress.json
  5. ghi kết quả         finish                       train_meta.json: OK / FAILED (STOPPED khi bị dừng ở bước 4)

Cách dùng (Python của env gs-gsplat; thường do scripts/build_room.sh gọi):
  python -m src.product.train_main --scene <phòng>/colmap --out <phòng>/model [--progress <phòng>/progress.json]
                                   [--config configs/product.yaml] [--force] [--dry-run]
  --dry-run: chỉ in lệnh train (không đọc gì trên đĩa ngoài file cấu hình, không train).
  Đã có mô hình train xong từ đúng mô hình COLMAP này với đúng cấu hình này: không train lại (--force: train lại).
  Thư mục --out cũ (dở dang, khác cấu hình, khác COLMAP, hoặc --force) được đổi tên <out>.prev_<thời gian>, không xóa.

Kết quả trong --out:
  ply/point_cloud_<max_steps-1>.ply         mô hình (PLY chuẩn của 3DGS: vị trí, SH bậc 0-3, opacity, scale, quaternion)
  ckpts/ckpt_<max_steps-1>_rank0.pt         checkpoint của gsplat
  stats/train_step<max_steps-1>_rank0.json  bộ nhớ GPU đỉnh, thời gian, số Gaussian (gsplat ghi)
  train_meta.json                           trạng thái, cấu hình đã dùng, lệnh, thời gian, số Gaussian
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

from src.product import config as product_config
from src.product import ply as ply_io
from src.product import progress as room_progress

META_FILE = "train_meta.json"
META_VERSION = 1
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
# File của gói gsplat phải trùng với bản checkout (v1.5.3 và commit được ghim cùng báo phiên bản "1.5.3")
GUARDED_FILES = ("strategy/default.py", "strategy/mcmc.py", "strategy/ops.py", "rendering.py")
# Chạy thử fused-ssim trên GPU: bản build chỉ chạy trên kiến trúc GPU lúc cài (scripts/setup/setup_gsplat.sh)
GPU_CHECK = ("import torch, fused_ssim; from gsplat import csrc; "
             "x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)")
TQDM_STEP = re.compile(r"(\d+)/(\d+) \[")         # thanh tiến độ của simple_trainer: "... 3600/30000 [01:10<08:30, ..."
GAUSSIANS = re.compile(r"Now having (\d+) GSs")   # MCMCStrategy(verbose=True) sau mỗi lần thêm Gaussian
PROGRESS_EVERY_S = 5.0                            # ghi progress.json tối đa mỗi 5 giây
PRINT_EVERY_PERCENT = 5                           # in thanh tiến độ vào log mỗi 5 %
TAIL_LINES = 40                                   # số dòng cuối của trainer giữ lại để ghi lỗi


class TrainError(RuntimeError):
    """Không train được; thông điệp chỉ rõ cách sửa."""


class Stopped(Exception):
    """Nhận SIGTERM (vd. scripts/pod/job.sh stop)."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def say(text: str) -> None:
    print(text, flush=True)


def report(progress_file: Path | None, percent: float, message: str) -> None:
    """Cập nhật bước train trong progress.json; lỗi ghi file chỉ là cảnh báo (không làm hỏng lần train)."""
    if progress_file is None:
        return
    try:
        room_progress.update(progress_file, "train", percent, message)
    except (OSError, ValueError) as error:
        say(f"WARN: progress file not updated: {error}")


def _number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _millions(value: int) -> str:
    return f"{value / 1e6:.2f}".replace(".", ",") + " triệu"


# ======================================================================================= 1-2. lệnh train
def final_step(cfg: dict) -> int:
    return cfg["training"]["max_steps"] - 1  # simple_trainer lưu mô hình ở vòng max_steps - 1


def examples_dir(cfg: dict) -> Path:
    return Path(cfg["trainer"]["dir"]) / "examples"


def ply_path(out: Path, cfg: dict) -> Path:
    return Path(out) / "ply" / f"point_cloud_{final_step(cfg)}.ply"


def ckpt_path(out: Path, cfg: dict) -> Path:
    return Path(out) / "ckpts" / f"ckpt_{final_step(cfg)}_rank0.pt"


def stats_path(out: Path, cfg: dict) -> Path:
    return Path(out) / "stats" / f"train_step{final_step(cfg):04d}_rank0.json"


def cli_tokens(flag: str, value) -> list[str]:
    """(tham số, giá trị) -> token dòng lệnh theo quy ước của tyro (bool False -> --no-<tên>)."""
    if isinstance(value, bool):
        if value:
            return [f"--{flag}"]
        prefix, _, leaf = flag.rpartition(".")
        return [f"--{prefix}.no-{leaf}" if prefix else f"--no-{leaf}"]
    if isinstance(value, (list, tuple)):
        if not value:
            raise product_config.ConfigError(f"empty list for --{flag}")
        return [f"--{flag}", *(str(item) for item in value)]
    return [f"--{flag}", str(value)]


def trainer_command(cfg: dict, scene: Path, out: Path, python: str | None = None) -> list[str]:
    """Lệnh train mô hình chính: toàn bộ cấu hình nằm ở đây, mọi giá trị lấy từ configs/product.yaml."""
    training, model = cfg["training"], cfg["model"]
    steps = training["max_steps"]
    argv = [python or sys.executable, str(examples_dir(cfg) / "simple_trainer.py"), model["preset"]]
    argv += cli_tokens("data-dir", str(scene))
    argv += cli_tokens("result-dir", str(out))
    argv += cli_tokens("data-factor", training["data_factor"])
    argv += cli_tokens("test-every", training["test_every"])
    argv += cli_tokens("max-steps", steps)
    argv += cli_tokens("antialiased", training["antialiased"])
    argv += cli_tokens("init-type", training["init_type"])
    argv += cli_tokens("lpips-net", training["lpips_net"])
    if model["preset"] == "mcmc":
        argv += cli_tokens("strategy.cap-max", model["cap_max"])
    argv += cli_tokens("eval-steps", [-1])  # không chấm điểm (và không render video quỹ đạo) trong lúc train
    argv += cli_tokens("save-steps", [steps])
    argv += cli_tokens("ply-steps", [steps])
    argv += cli_tokens("save-ply", True)
    argv += cli_tokens("disable-viewer", True)  # có viewer thì simple_trainer chờ vô hạn sau khi train xong
    for key, value in model["args"].items():
        argv += cli_tokens(product_config.flag_name(key), value)
    return argv


def training_key(cfg: dict) -> dict:
    """Mọi thứ quyết định mô hình, ghi vào train_meta.json. Khác với lần train trước thì mô hình cũ không còn là mô
    hình chính: train lại. Đổi cấu trúc dict này là các phòng đã dựng sẽ bị train lại."""
    return {
        "model": {key: cfg["model"][key] for key in ("name", "preset", "args", "cap_max")},
        "trainer": {"commit": cfg["trainer"]["commit"], "seed": cfg["trainer"]["seed"]},
        "training": dict(cfg["training"]),
    }


# ============================================================================ 3. kiểm tra trước khi train
def check_scene(scene: Path) -> dict:
    """Phòng đã chuẩn bị: có mô hình COLMAP (sparse/0) và ảnh (images/). Trả về số ảnh đã định vị."""
    sparse = Path(scene) / "sparse" / "0"
    missing = [name for name in SPARSE_FILES if not (sparse / name).is_file()]
    if missing:
        raise TrainError(f"{scene}: no COLMAP model ({', '.join(missing)} missing in sparse/0); "
                         "prepare the room first (scripts/build_room.sh)")
    images = Path(scene) / "images"
    if not images.is_dir() or not any(images.iterdir()):
        raise TrainError(f"{images} is missing or empty")
    with open(sparse / "images.bin", "rb") as handle:  # COLMAP: số ảnh đã định vị nằm ở 8 byte đầu (uint64)
        head = handle.read(8)
    if len(head) != 8:
        raise TrainError(f"{sparse / 'images.bin'} is truncated")
    registered = struct.unpack("<Q", head)[0]
    if registered < 2:
        raise TrainError(f"{scene}: only {registered} image(s) in the COLMAP model")
    return {"registered_images": registered}


def train_image_count(images: int, test_every: int) -> int:
    """Số ảnh gsplat dùng để train: ảnh i (theo thứ tự tên) là ảnh kiểm tra khi i % test_every == 0."""
    return images - (images + test_every - 1) // test_every


def scene_fingerprint(scene: Path) -> str:
    """SHA-256 của mô hình COLMAP (sparse/0): COLMAP làm lại thì mô hình cũ không khớp camera mới."""
    digest = hashlib.sha256()
    for name in SPARSE_FILES:
        digest.update(name.encode() + b"\0")
        with open(Path(scene) / "sparse" / "0" / name, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def check_trainer(cfg: dict) -> dict:
    """Checkout gsplat đúng commit ghim, và gói gsplat đang cài trùng mã nguồn với checkout đó."""
    gsplat_dir = Path(cfg["trainer"]["dir"])
    script = gsplat_dir / "examples" / "simple_trainer.py"
    if not script.is_file():
        raise TrainError(f"{script} not found: install gsplat with scripts/setup/setup_gsplat.sh")
    head, expected = git_head(gsplat_dir), cfg["trainer"]["commit"]
    if head != expected:
        raise TrainError(f"{gsplat_dir} is at commit {head or 'unknown'}, configs/product.yaml expects {expected}: "
                         "re-run scripts/setup/setup_gsplat.sh")
    spec = importlib.util.find_spec("gsplat")  # không import (không nạp torch)
    if spec is None or not spec.submodule_search_locations:
        raise TrainError(f"the gsplat package is not installed for {sys.executable} "
                         f"(activate the {cfg['trainer']['env']} env)")
    installed = Path(list(spec.submodule_search_locations)[0]).resolve()
    checkout = gsplat_dir / "gsplat"
    differing = [name for name in GUARDED_FILES if _sha256(installed / name) != _sha256(checkout / name)]
    if differing:
        raise TrainError(f"the installed gsplat ({installed}) differs from {checkout} in {differing}: "
                         "reinstall it with scripts/setup/setup_gsplat.sh (FORCE_REBUILD=1)")
    return {"dir": str(gsplat_dir), "commit": head, "installed": str(installed)}


def check_examples_imports(examples: Path) -> None:
    """simple_trainer.py import `datasets.colmap` từ examples/; examples/datasets không có __init__.py (namespace
    package) nên một gói `datasets` thường cài trong env (vd. của Hugging Face) sẽ được import thay vào."""
    spec = importlib.machinery.PathFinder.find_spec("datasets", [str(examples), *sys.path])
    locations = {Path(item).resolve() for item in (spec.submodule_search_locations or [])} if spec else set()
    if (Path(examples) / "datasets").resolve() not in locations:
        origin = getattr(spec, "origin", None)
        raise TrainError(f"'import datasets' would load {origin or 'something else'} instead of {examples}/datasets: "
                         "remove that package from this env (pip uninstall datasets)")


def trainer_env() -> dict:
    """Biến môi trường của trainer: một GPU, log không bị đệm, matplotlib không cần màn hình."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # MPLBACKEND của Jupyter làm matplotlib báo lỗi trong env gs-gsplat
    env["PYTHONUNBUFFERED"] = "1"  # dòng print của trainer tới log ngay, đúng thứ tự với thanh tiến độ
    if env.get("CUDA_VISIBLE_DEVICES") is None:
        env["CUDA_VISIBLE_DEVICES"] = "0"  # nhiều GPU thì gsplat tự train phân tán: khác điều kiện đã đo
    elif "," in env["CUDA_VISIBLE_DEVICES"]:
        raise TrainError("CUDA_VISIBLE_DEVICES lists several GPUs; the main model is trained on one GPU "
                         "(set CUDA_VISIBLE_DEVICES=0)")
    return env


def check_gpu(cfg: dict, env: dict) -> None:
    """GPU dùng được với env hiện tại (torch, fused-ssim, phần CUDA của gsplat); chỉ kiểm khi thật sự train."""
    try:
        result = subprocess.run([sys.executable, "-c", GPU_CHECK], env=env, capture_output=True, text=True,
                                timeout=600)
    except subprocess.TimeoutExpired:
        raise TrainError("the GPU check did not finish in 10 minutes") from None
    if result.returncode != 0:
        last = (result.stderr.strip().splitlines() or ["no error message"])[-1]
        raise TrainError(f"the {cfg['trainer']['env']} env is not ready for this GPU ({last}): run "
                         "scripts/setup/setup_gsplat.sh (it rebuilds when the GPU type changes)")


def prepare_training(cfg: dict, out: Path) -> tuple[dict, dict]:
    """Mọi kiểm tra cần trước khi train, rồi dọn thư mục kết quả. Trả về (thông tin gsplat, biến môi trường)."""
    trainer = check_trainer(cfg)
    check_examples_imports(examples_dir(cfg))
    env = trainer_env()
    check_gpu(cfg, env)
    moved = set_aside(out)
    if moved is not None:
        say(f"[train_main] previous results kept in {moved}")
    Path(out).mkdir(parents=True)
    return trainer, env


# ============================================================================ trạng thái (train_meta.json)
def read_meta(out: Path) -> dict | None:
    try:
        data = json.loads((Path(out) / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_meta(out: Path, meta: dict) -> None:
    path = Path(out) / META_FILE
    tmp = path.with_name(f".{META_FILE}.{os.getpid()}.tmp")  # ghi file tạm rồi đổi tên: không ai đọc file viết dở
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_current(meta: dict | None, out: Path, cfg: dict, fingerprint: str) -> bool:
    """Đã có mô hình train xong từ đúng mô hình COLMAP này với đúng cấu hình này."""
    return (bool(meta) and meta.get("status") == "OK" and meta.get("scene_fingerprint") == fingerprint
            and meta.get("config") == training_key(cfg) and ply_path(out, cfg).is_file())


def set_aside(out: Path) -> Path | None:
    """Đổi tên thư mục kết quả cũ thành <out>.prev_<thời gian> (không xóa); thư mục rỗng thì bỏ."""
    out = Path(out)
    if not out.exists():
        return None
    if out.is_dir() and not any(out.iterdir()):
        out.rmdir()
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target, index = out.with_name(f"{out.name}.prev_{stamp}"), 1
    while target.exists():
        target, index = out.with_name(f"{out.name}.prev_{stamp}_{index}"), index + 1
    out.rename(target)
    return target


def new_meta(cfg: dict, config_file: Path, scene: Path, out: Path, info: dict, fingerprint: str, trainer: dict,
             cmd: list[str], env: dict) -> dict:
    """train_meta.json lúc bắt đầu train (status RUNNING); finish() điền phần kết quả."""
    return {
        "format_version": META_VERSION,
        "status": "RUNNING",
        "model": cfg["model"]["name"],
        "scene": str(scene),
        "scene_fingerprint": fingerprint,
        "registered_images": info["registered_images"],
        "train_images": train_image_count(info["registered_images"], cfg["training"]["test_every"]),
        "config": training_key(cfg),
        "config_file": str(Path(config_file).resolve()),
        "trainer": trainer,
        "command": cmd,
        "cwd": str(examples_dir(cfg)),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "started_at": now(),
        "finished_at": None,
        "duration_s": None,
        "num_gaussians": None,
        "gsplat_stats": None,
        "files": {"ply": str(ply_path(out, cfg).relative_to(out)), "ckpt": str(ckpt_path(out, cfg).relative_to(out))},
        "error": None,
    }


# ================================================================================================ 4. train
class TrainerOutput:
    """Đầu ra của simple_trainer.py, từng dòng (thanh tqdm xuống dòng bằng \\r, các dòng khác bằng \\n):
      - thanh tiến độ của vòng train: in vào log mỗi 5 %, ghi % vào progress.json tối đa mỗi 5 giây;
      - "Now having N GSs" của MCMCStrategy: số Gaussian hiện tại, ghi kèm vào dòng tiến độ;
      - các dòng khác: in nguyên vào log, giữ 40 dòng cuối để ghi lỗi vào train_meta.json."""

    def __init__(self, max_steps: int, progress_file: Path | None = None):
        self.max_steps = max_steps
        self.progress_file = progress_file
        self.tail: collections.deque[str] = collections.deque(maxlen=TAIL_LINES)
        self.gaussians: int | None = None
        self._printed_bucket = -1
        self._written_at: float | None = None

    def line(self, text: str) -> None:
        text = text.rstrip()
        if not text:
            return
        match = TQDM_STEP.search(text)
        if match and int(match.group(2)) == self.max_steps:  # thanh của vòng train (không phải thanh tải file...)
            self._step(int(match.group(1)), text)
            return
        found = GAUSSIANS.search(text)
        if found:
            self.gaussians = int(found.group(1))
        self.tail.append(text)
        say(text)

    def _step(self, step: int, text: str) -> None:
        percent = 100.0 * step / self.max_steps
        bucket = int(percent // PRINT_EVERY_PERCENT)
        if bucket > self._printed_bucket:
            self._printed_bucket = bucket
            say(text)
        clock = time.monotonic()
        if step == self.max_steps or self._written_at is None or clock - self._written_at >= PROGRESS_EVERY_S:
            self._written_at = clock
            message = f"Vòng {_number(step)}/{_number(self.max_steps)}"
            if self.gaussians:
                message += f" · {_millions(self.gaussians)} Gaussian"
            report(self.progress_file, percent, message)


def _on_sigterm(signum, frame) -> None:
    raise Stopped()


def run_trainer(cmd: list[str], cwd: Path, env: dict, output: TrainerOutput) -> int:
    """Chạy simple_trainer.py, đưa đầu ra (stdout + stderr) qua `output` từng dòng; trả về mã thoát. Bị dừng
    (SIGTERM, Ctrl+C) thì dừng luôn trainer rồi báo tiếp lên (Stopped / KeyboardInterrupt)."""
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    pending = b""
    try:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            *lines, pending = re.split(rb"[\r\n]", pending + chunk)
            for raw in lines:
                output.line(raw.decode("utf-8", errors="replace"))
        output.line(pending.decode("utf-8", errors="replace"))
        return process.wait()
    except (KeyboardInterrupt, Stopped):
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


# ========================================================================================= 5. ghi kết quả
def finish(meta: dict, out: Path, cfg: dict, code: int, tail: list[str], duration_s: float,
           progress_file: Path | None) -> int:
    """Ghi kết quả vào train_meta.json: OK khi trainer thoát 0 và PLY cuối đọc được, FAILED nếu không."""
    meta.update(finished_at=now(), duration_s=round(duration_s, 1))
    model_file = ply_path(out, cfg)
    problem = None
    if code != 0:
        problem = f"the trainer exited with code {code}"
    elif not model_file.is_file():
        problem = f"the trainer finished without writing {model_file}"
    else:
        try:
            meta["num_gaussians"] = ply_io.vertex_count(model_file)
        except (OSError, ply_io.PlyError) as error:
            problem = f"cannot read {model_file}: {error}"
    if problem is not None:
        meta.update(status="FAILED", error=problem, output_tail=tail[-20:])
        write_meta(out, meta)
        say(f"ERROR: training failed: {problem} (see the output above; {Path(out) / META_FILE})")
        return 1
    try:
        meta["gsplat_stats"] = json.loads(stats_path(out, cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta["gsplat_stats"] = None
    meta["status"] = "OK"
    write_meta(out, meta)
    say(f"[train_main] OK: {meta['num_gaussians']} Gaussians in {meta['duration_s'] / 60:.1f} min -> {model_file}")
    report(progress_file, 100.0, f"Train xong: {_millions(meta['num_gaussians'])} Gaussian")
    return 0


# ================================================================================================== main
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mô hình chính của sản phẩm (configs/product.yaml) cho một phòng")
    parser.add_argument("--scene", type=Path, required=True, help="thư mục phòng đã chuẩn bị: images/, sparse/0/")
    parser.add_argument("--out", type=Path, required=True, help="thư mục mô hình, vd. <phòng>/model")
    parser.add_argument("--config", type=Path, default=product_config.DEFAULT_CONFIG)
    parser.add_argument("--ws", default=None, help="thư mục gốc trên pod (mặc định: biến WS, rồi /workspace)")
    parser.add_argument("--progress", type=Path, default=None, help="progress.json của phòng (cập nhật % train)")
    parser.add_argument("--force", action="store_true", help="train lại dù đã có mô hình khớp")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in lệnh train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # 1. cấu hình
    try:
        cfg = product_config.load(args.config, args.ws)
    except product_config.ConfigError as error:
        say(f"ERROR: {args.config}: {error}")
        return 2
    # 2. lệnh train (đường dẫn tuyệt đối: trainer chạy với thư mục hiện tại = examples/ của gsplat)
    scene, out = args.scene.resolve(), args.out.resolve()
    cmd = trainer_command(cfg, scene, out)
    if args.dry_run:
        say(shlex.join(cmd))
        say(f"(cwd: {examples_dir(cfg)})")
        return 0
    # 3. kiểm tra
    try:
        info = check_scene(scene)
        fingerprint = scene_fingerprint(scene)
        old = read_meta(out)
        if not args.force and is_current(old, out, cfg, fingerprint):
            say(f"[train_main] {out}: already trained from this COLMAP model with this configuration "
                f"({old.get('num_gaussians')} Gaussians); not training again (--force retrains)")
            report(args.progress, 100.0, "Đã có mô hình khớp, không train lại")
            return 0
        trainer, env = prepare_training(cfg, out)
    except (TrainError, OSError) as error:
        say(f"ERROR: {error}")
        return 1
    # 4. train
    meta = new_meta(cfg, args.config, scene, out, info, fingerprint, trainer, cmd, env)
    write_meta(out, meta)
    say(f"[train_main] {cfg['model']['name']} | {info['registered_images']} images | out {out}")
    say(f"[train_main] {shlex.join(cmd)}")
    output = TrainerOutput(cfg["training"]["max_steps"], args.progress)
    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        code = run_trainer(cmd, examples_dir(cfg), env, output)
    except (KeyboardInterrupt, Stopped):
        meta.update(status="STOPPED", finished_at=now(), duration_s=round(time.monotonic() - started, 1),
                    error="stopped before the end of training")
        write_meta(out, meta)
        say("[train_main] stopped")
        return 143
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    # 5. ghi kết quả
    return finish(meta, out, cfg, code, list(output.tail), time.monotonic() - started, args.progress)


if __name__ == "__main__":
    sys.exit(main())
