"""Bản giả cho test luồng sản phẩm (src/product/, scripts/build_room.sh, trang web app/web/).

- gsplat: FAKE_TRAINER ghi đúng các file simple_trainer.py @6e8c837 ghi khi train xong (ply/point_cloud_<N>.ply,
  ckpts/, stats/) và in thanh tiến độ như tqdm; FAKE_PARSER thay examples/datasets/colmap.py (đọc camera từ
  <cảnh>/fake_parser.json); make_gsplat_checkout: một checkout có kho git.
- build_room.sh: make_build_env dựng repo giả (script thật, prepare_scene.sh giả), python giả (train_main /
  export_web giả, trừ khi REAL_STEPS=1), conda giả, một thư mục ảnh và một video.
"""

import subprocess
from pathlib import Path

from shell_env import ShellEnv, make_fake_repo, write_stub

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


# ============================================================================== scripts/build_room.sh
FAKE_PREPARE = r'''
echo "prepare_scene $* OVERWRITE=${OVERWRITE:-} OUT_ROOT=${OUT_ROOT:-} HELD=${SR3D_GPU_LOCK_HELD:-}" >> "$CALLS"
scene="$OUT_ROOT/$2/colmap"
if [ -d "$1" ]; then kind=images; echo "===== 1. Photos -> $scene/input (no frame extraction) ====="
else kind=video; echo "===== 1. Frames -> $scene/input ====="; fi
mkdir -p "$scene/sparse/0" "$scene/images"
printf '{"source": "%s", "source_type": "%s", "selected": 12}\n' "$1" "$kind" > "$scene/frames_report.json"
echo
echo "===== 2. COLMAP ====="
for f in cameras images points3D; do echo "$f" > "$scene/sparse/0/$f.bin"; done
printf '{"verdict": "%s", "hints": ["Quay chậm hơn"]}\n' "${FAKE_VERDICT:-PASS}" > "$scene/sfm_report.json"
exit "${FAKE_PREP_EXIT:-0}"
'''

# Chuẩn bị ảnh giả cho lượt chạy với train_main / export_web thật: mô hình COLMAP đọc được số ảnh, ảnh, camera cho
# Parser giả (product_fakes.FAKE_PARSER), các báo cáo
REAL_PREPARE = r'''
echo "prepare_scene $*" >> "$CALLS"
scene="$OUT_ROOT/$2/colmap"
echo "===== 1. Frames -> $scene/input ====="
mkdir -p "$scene/sparse/0" "$scene/images"
python3 - "$scene" "$1" <<'PY'
import json, struct, sys
from pathlib import Path
scene, source = Path(sys.argv[1]), sys.argv[2]
names = [f"{i:05d}.jpg" for i in range(1, 4)]
(scene / "sparse" / "0" / "images.bin").write_bytes(struct.pack("<Q", len(names)))
(scene / "sparse" / "0" / "cameras.bin").write_bytes(b"c")
(scene / "sparse" / "0" / "points3D.bin").write_bytes(b"p")
for name in names:
    (scene / "images" / name).write_bytes(b"\xff\xd8 fake \xff\xd9")
c2w = [[[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for x in (-1, 0, 1)]
(scene / "fake_parser.json").write_text(json.dumps({"names": names, "c2w": c2w, "size": [100, 80],
                                                    "K": [[100, 0, 50], [0, 100, 40], [0, 0, 1]]}))
(scene / "frames_report.json").write_text(json.dumps({"source": source, "source_type": "video", "selected": 3}))
(scene / "sfm_report.json").write_text(json.dumps({"verdict": "PASS", "input_images": 3,
                                                   "model": {"registered_images": 3}}))
PY
echo "===== 2. COLMAP ====="
'''

FAKE_PYTHON = r'''#!/usr/bin/env bash
case "${1:-}" in
  -c) echo "python -c" >> "$CALLS"; exit 0 ;;  # kiểm tra fused-ssim / gsplat trên GPU
  -m)
    [ "${REAL_STEPS:-0}" = 1 ] || case "${2:-}" in
      src.product.train_main)
        printf 'train_main %s\n' "$(printf '%q ' "${@:3}")" >> "$CALLS"
        out="" prev=""
        for arg in "$@"; do [ "$prev" = --out ] && out="$arg"; prev="$arg"; done
        mkdir -p "$out" && echo '{"status": "OK"}' > "$out/train_meta.json"
        exit "${FAKE_TRAIN_EXIT:-0}" ;;
      src.product.export_web)
        printf 'export_web %s\n' "$(printf '%q ' "${@:3}")" >> "$CALLS"
        exit "${FAKE_EXPORT_EXIT:-0}" ;;
    esac ;;
esac
exec python3 "$@"
'''


def make_build_env(tmp_path: Path) -> ShellEnv:
    """Môi trường chạy scripts/build_room.sh thật trong tmp_path: prepare_scene.sh, conda, GPU và (mặc định)
    train_main / export_web là giả; kiểm tra nguồn, cấu hình, tiến độ, khóa GPU và job.sh là thật."""
    repo = make_fake_repo(tmp_path, ["build_room.sh"], with_src=True, with_configs=True)
    write_stub(repo, "scripts/pipeline/prepare_scene.sh", FAKE_PREPARE)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(FAKE_PYTHON, encoding="utf-8")
    (bin_dir / "python").chmod(0o755)
    env.env["PATH"] = f"{bin_dir}:{env.env['PATH']}"
    conda = env.ws / "miniforge3" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { echo \"conda $*\" >> \"$CALLS\"; }\n", encoding="utf-8")
    photos = tmp_path / "data" / "phong1"
    photos.mkdir(parents=True)
    for index in range(12):
        (photos / f"IMG_{index:04d}.jpg").write_bytes(b"jpg")
    video = tmp_path / "data" / "video8.mp4"
    video.write_bytes(b"mp4")
    env.photos, env.video = photos, video
    env.room = env.ws / "outputs" / "product"
    return env
