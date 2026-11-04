"""Bản giả của gsplat cho test luồng sản phẩm (src/product/): simple_trainer.py, Parser, một checkout có kho git.

FAKE_TRAINER ghi đúng các file simple_trainer.py @6e8c837 ghi khi train xong (ply/point_cloud_<N>.ply, ckpts/, stats/)
và in thanh tiến độ như tqdm. FAKE_PARSER thay examples/datasets/colmap.py: đọc camera từ <cảnh>/fake_parser.json.
"""

import subprocess
from pathlib import Path

FAKE_TRAINER = r'''
import json, os, struct, sys, time
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
out, steps = value("--result-dir"), int(value("--max-steps"))
if os.environ.get("FAKE_TRAINER_RECORD"):
    with open(os.environ["FAKE_TRAINER_RECORD"], "w") as handle:
        json.dump({"argv": sys.argv, "cwd": os.getcwd(), "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "unbuffered": os.environ.get("PYTHONUNBUFFERED"), "mpl": os.environ.get("MPLBACKEND")}, handle)
time.sleep(float(os.environ.get("FAKE_TRAINER_SLEEP", "0")))
for step in range(steps + 1):
    sys.stderr.write(f"\rloss=0.100| sh degree=0| : {100 * step // steps:3d}%|#| {step}/{steps} [00:00<00:00, 999it/s]")
    sys.stderr.flush()
    if step == steps // 2:
        print(f"Step {step}: Added 5 GSs. Now having 1234 GSs.")
sys.stderr.write("\n")
print("Downloading: 100%|#| 528M/528M [00:05<00:00, 100MB/s]")
code = int(os.environ.get("FAKE_TRAINER_EXIT", "0"))
if code:
    print("RuntimeError: CUDA out of memory (fake)")
    sys.exit(code)
last = steps - 1
for sub in ("ply", "ckpts", "stats"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
n = int(os.environ.get("FAKE_TRAINER_GAUSSIANS", "7"))
props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]
header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
          + "".join(f"property float {p}\n" for p in props) + "end_header\n")
if os.environ.get("FAKE_TRAINER_NO_PLY") != "1":
    with open(os.path.join(out, "ply", f"point_cloud_{last}.ply"), "wb") as handle:
        handle.write(header.encode() + struct.pack(f"<{n * len(props)}f", *([0.5] * (n * len(props)))))
with open(os.path.join(out, "ckpts", f"ckpt_{last}_rank0.pt"), "wb") as handle:
    handle.write(b"ckpt")
with open(os.path.join(out, "stats", f"train_step{last:04d}_rank0.json"), "w") as handle:
    json.dump({"mem": 1.5, "ellipse_time": 12.0, "num_GS": n}, handle)
print("Step: ", last, {"num_GS": n})
'''

FAKE_PARSER = '''
import json, os
import numpy as np


class Parser:
    def __init__(self, data_dir, factor=1, normalize=False, test_every=8):
        spec = json.load(open(os.path.join(data_dir, "fake_parser.json")))
        json.dump({"factor": factor, "normalize": normalize, "test_every": test_every},
                  open(os.path.join(data_dir, "parser_args.json"), "w"))
        if spec.get("raise"):
            raise RuntimeError(spec["raise"])
        self.image_names = spec["names"]
        self.camtoworlds = np.array(spec["c2w"], dtype=float)
        self.camera_ids = [1] * len(self.image_names)
        self.Ks_dict = {1: np.array(spec["K"], dtype=float)}
        self.imsize_dict = {1: tuple(spec["size"])}
'''


GUARDED_FILES = ("strategy/default.py", "strategy/mcmc.py", "strategy/ops.py", "rendering.py")


def write_fake_gpu_modules(site: Path, error: str | None = None) -> None:
    """torch, fused_ssim, gsplat.csrc giả cho bước kiểm tra GPU của train_main (GPU_CHECK); `error`: torch.rand báo lỗi
    như khi bản build không chạy được trên GPU này."""
    (site / "torch").mkdir(parents=True, exist_ok=True)
    body = f"raise RuntimeError({error!r})" if error else "return 0"
    (site / "torch" / "__init__.py").write_text(f"def rand(*args, **kwargs):\n    {body}\n", encoding="utf-8")
    (site / "fused_ssim.py").write_text("def fused_ssim(a, b):\n    return 0\n", encoding="utf-8")
    (site / "gsplat").mkdir(parents=True, exist_ok=True)
    (site / "gsplat" / "csrc.py").write_text("", encoding="utf-8")


def make_gsplat_checkout(root: Path) -> tuple[Path, Path, str]:
    """Checkout gsplat giả (kho git, có examples/ và gói gsplat/) và một bản 'đã cài' giống hệt ở site/, kèm torch,
    fused_ssim giả để bước kiểm tra GPU chạy được."""
    checkout, site = root / "gsplat", root / "site"
    for base in (checkout, site):
        for name in GUARDED_FILES + ("__init__.py",):
            path = base / "gsplat" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {name}\n", encoding="utf-8")
    (checkout / "examples" / "datasets").mkdir(parents=True)
    (checkout / "examples" / "simple_trainer.py").write_text(FAKE_TRAINER, encoding="utf-8")
    write_fake_gpu_modules(site)
    (checkout / "examples" / "datasets" / "colmap.py").write_text(FAKE_PARSER, encoding="utf-8")
    git = ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fake gsplat"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return checkout, site / "gsplat", head
