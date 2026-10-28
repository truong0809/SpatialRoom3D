"""Sinh các notebook trong app/notebooks/ (không sửa tay các file .ipynb: sửa ở đây rồi chạy lại):

- runpod.ipynb: code tổng, mọi việc trên pod (cài môi trường, thực nghiệm chính, benchmark, cảnh tự chụp, kết quả);
- b0_inria.ipynb, gsplat_default.ipynb, gsplat_mcmc.ipynb: mỗi notebook chạy riêng một phương pháp của thực nghiệm chính
  (scripts/pod/run_experiment.sh main --only <phương pháp>), cùng dữ liệu, protocol và thư mục kết quả với code tổng.

Các ô chung (kiểm tra máy, khóa deploy, lấy code, cài môi trường, theo dõi, gói kết quả, dừng) lấy từ cùng một nguồn nên
giống hệt nhau giữa các notebook (tests/unit/test_notebooks.py kiểm).

    python3 app/notebooks/build_notebooks.py           # ghi lại các notebook
    python3 app/notebooks/build_notebooks.py --check   # chỉ so: notebook nào khác bản sinh ra thì in tên, exit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = "/workspace/SpatialRoom3D"
REPO_SSH = "git@github.com:truong0809/SpatialRoom3D.git"
MAIN_NAME = "main"  # tên bản của code tổng (cả kế hoạch main): việc nền, báo cáo, gói kết quả main_<thời gian>
SSH_CMD = ('SSH_CMD="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=/workspace/.ssh/known_hosts '
           '-o StrictHostKeyChecking=accept-new"')


class Notebook:
    def __init__(self, filename: str):
        self.filename = filename
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

    def code(self, text: str) -> None:
        text = text.replace("@CODE@", CODE).replace("@REPO_SSH@", REPO_SSH)
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": text.strip("\n").splitlines(keepends=True)})

    def text(self) -> str:
        cells = [dict(cell, id=f"cell-{index:02d}") for index, cell in enumerate(self.cells)]
        notebook = {"cells": cells,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5}
        return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def fill(text: str, **numbers) -> str:
    """Thay {tên} bằng số ô của notebook (không dùng str.format: nội dung có dấu ngoặc nhọn khác)."""
    for name, value in numbers.items():
        text = text.replace("{" + name + "}", str(value))
    return text


# ------------------------------------------------------------------------------------------------ ô 1–5 (chung)
def machine(nb: Notebook) -> None:
    nb.md("## 1. Kiểm tra máy")
    nb.code(r"""
%%bash
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]; }
gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1)
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')
vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
drv=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n 1)
nv=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
gccv=$(gcc -dumpversion 2>/dev/null)
fst=$(findmnt -n -o FSTYPE --target /workspace 2>/dev/null)
echo "GPU:         ${gpu:-KHÔNG THẤY GPU} (compute capability ${cc:-?})"
echo "Driver CUDA: ${drv:-?} | nvcc: ${nv:-không có} | gcc: ${gccv:-?} | /workspace: ${fst:-?}"
echo "----------------------------------------"
ok=1
if [ -z "$drv" ]; then echo "✗ Không thấy GPU hoặc driver: deploy lại pod"; ok=0
elif ver_ge "$drv" 12.8; then echo "✓ Driver hỗ trợ CUDA $drv: dùng PyTorch cu128"
elif ver_ge "$drv" 12.6; then echo "! Driver chỉ hỗ trợ CUDA $drv: ở ô 4 đặt TORCH_CUDA_OVERRIDE=\"cu126\""
else echo "✗ Driver CUDA $drv quá cũ: Terminate rồi deploy máy khác"; ok=0; fi
case "$cc" in
  8.0|8.6|8.9|9.0) echo "✓ GPU có trong danh sách kiến trúc build sẵn (8.0/8.6/8.9/9.0)" ;;
  *) echo "✗ GPU compute capability ${cc:-?} không được hỗ trợ: dùng RTX 4090, A6000, A40, L40S, A100 hoặc H100"; ok=0 ;;
esac
if [ "${nv%%.*}" = "12" ]; then echo "✓ nvcc $nv"; else echo "✗ nvcc ${nv:-không có}: sai template, cần image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"; ok=0; fi
if [ -n "$gccv" ] && [ "${gccv%%.*}" -le 12 ]; then echo "✓ gcc $gccv"; else echo "✗ gcc ${gccv:-?}: cần gcc 12 trở xuống (template Ubuntu 22.04)"; ok=0; fi
if [ -n "$drv" ] && ! ver_ge "$drv" 12.9; then echo "! Driver CUDA $drv dưới 12.9: COLMAP dùng bản CPU (chậm hơn nhưng chạy được, tự chọn)"; fi
if [ "${vram:-0}" -ge 23000 ] 2>/dev/null; then echo "✓ VRAM ${vram} MiB"; else echo "! VRAM ${vram:-?} MiB dưới 24 GB: có thể thiếu bộ nhớ"; fi
case "$fst" in
  fuse*) echo "✓ /workspace là network volume" ;;
  *) echo "! /workspace là ổ '${fst:-?}', không phải network volume: Terminate pod sẽ mất dữ liệu" ;;
esac
if python3 -c 'import yaml' 2>/dev/null; then echo "✓ python3 có PyYAML (đọc configs/*.yaml)"
else echo "! python3 chưa có PyYAML: chạy 'python3 -m pip install pyyaml', hoặc chờ ô 4 cài xong env gs-gsplat (có sẵn PyYAML)"; fi
command -v git > /dev/null && command -v ssh-keygen > /dev/null && echo "✓ git, ssh có sẵn" || { echo "✗ thiếu git hoặc ssh"; ok=0; }
[ -d /workspace/miniforge3 ] && echo "! volume đã có /workspace/miniforge3 (không phải volume trống?)" || echo "✓ volume trống: ô 4 sẽ cài môi trường"
if [ "$ok" = 1 ]; then echo "=> ĐẠT: chạy tiếp ô 2"; else echo "=> CHƯA ĐẠT: sửa các dòng ✗ trước"; fi
""")


def deploy_key(nb: Notebook) -> None:
    nb.md("""
## 2. Khóa deploy cho repo private (một lần cho mỗi volume)

Repo `SpatialRoom3D` là private. Pod đọc repo bằng một **deploy key chỉ đọc**: khóa chỉ mở được đúng repo này, không push được.

1. Chạy ô này: tạo khóa trong `/workspace/.ssh` (nếu chưa có) và in **khóa công khai**.
2. Trên GitHub: repo `SpatialRoom3D` → **Settings → Deploy keys → Add deploy key**, dán dòng khóa công khai, **không** tick "Allow write access".
3. Chạy lại ô này: dòng cuối phải là `✓ GitHub nhận khóa`.

Khóa riêng nằm trên volume (còn khi Terminate pod), không bao giờ được in hay gửi đi.
""")
    nb.code(r"""
%%bash
KEY=/workspace/.ssh/github_deploy
mkdir -p /workspace/.ssh && chmod 700 /workspace/.ssh
if [ -f "$KEY" ]; then
  echo "Khóa đã có: $KEY (không tạo lại)"
else
  ssh-keygen -t ed25519 -N "" -C "runpod-spatialroom3d" -f "$KEY" -q && echo "Đã tạo khóa mới: $KEY"
fi
chmod 600 "$KEY"
echo
echo "Khóa công khai, dán vào GitHub (SpatialRoom3D → Settings → Deploy keys → Add deploy key, KHÔNG tick 'Allow write access'):"
cat "$KEY.pub"
echo
@SSH_CMD@
if $SSH_CMD -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
  echo "✓ GitHub nhận khóa: chạy tiếp ô 3"
else
  echo "✗ GitHub chưa nhận khóa: thêm khóa công khai ở trên vào Deploy keys của repo SpatialRoom3D, rồi chạy lại ô này"
fi
""".replace("@SSH_CMD@", SSH_CMD))


def fetch_code(nb: Notebook) -> None:
    nb.md("""
## 3. Lấy code vào `/workspace/SpatialRoom3D`

- Chưa có: clone nhánh `BRANCH` từ GitHub bằng khóa của ô 2.
- Đã có: `git pull` (chỉ tiến lên, không ghi đè). Code trên pod có thay đổi chưa commit thì **không cập nhật** và in danh sách thay đổi; ô này không xóa hay cất gì.
- Không chạy khi đang có việc nền.
""")
    nb.code(r"""
%%bash
REPO_SSH="@REPO_SSH@"
BRANCH="main"
CODE=@CODE@
KEY=/workspace/.ssh/github_deploy
@SSH_CMD@
if [ -f "$CODE/scripts/pod/job.sh" ] && bash "$CODE/scripts/pod/job.sh" busy; then
  echo "Đang có việc chạy nền: không đổi code lúc này (xem ô 5)"; exit 0
fi
if [ -d "$CODE/.git" ]; then
  changes="$(git -C "$CODE" status --porcelain)"
  if [ -n "$changes" ]; then
    echo "Code trên pod có thay đổi chưa commit: KHÔNG cập nhật (không xóa, không cất gì)."
    echo "$changes" | head -n 20
    echo "Xem: git -C $CODE status"
    exit 1
  fi
  GIT_SSH_COMMAND="$SSH_CMD" git -C "$CODE" pull --ff-only origin "$BRANCH" || { echo "Không cập nhật được (xem lỗi ở trên)"; exit 1; }
elif [ -e "$CODE" ]; then
  echo "$CODE đã có nhưng không phải repo Git: đổi tên hoặc xem lại, rồi chạy lại ô này"; exit 1
else
  GIT_SSH_COMMAND="$SSH_CMD" git clone -q -b "$BRANCH" "$REPO_SSH" "$CODE" \
    || { echo "Không clone được: chạy lại ô 2 (khóa deploy) và kiểm tra tên repo"; exit 1; }
  git -C "$CODE" config core.sshCommand "$SSH_CMD"
  echo "Đã clone $REPO_SSH (nhánh $BRANCH) vào $CODE"
fi
mkdir -p /workspace/data/custom_scenes /workspace/outputs
echo "Commit: $(git -C "$CODE" log --oneline -1)"
""".replace("@SSH_CMD@", SSH_CMD))


def setup(nb: Notebook) -> None:
    nb.md("""
## 4. Cài môi trường (nền, không train)

`scripts/pod/run_all.sh`: kiểm tra máy → Miniforge + env `gs-inria` (3DGS gốc của Inria) → env `gs-tools` (COLMAP, ffmpeg) → env `gs-gsplat` (kèm smoke test) → gói log. Phiên bản ghim trong `configs/environment/`. **Chỉ cài môi trường, không train.**

Lần đầu trên volume trống mất khoảng 80 phút (đo trên RTX 4090 ngày 17/10/2026: `gs-inria` 29 phút, `gs-gsplat` 32 phút, `gs-tools` 18 phút); pod vẫn tính tiền trong lúc cài. Chạy lại khi env đã có thì chỉ kiểm tra lại. Ô này xong ngay, việc chạy tiếp ở nền (theo dõi ở ô 5).
""")
    nb.code(r"""
%%bash
TORCH_CUDA_OVERRIDE=""   # "cu126" nếu ô 1 báo driver chỉ hỗ trợ CUDA 12.6
MAX_JOBS=""              # "4" nếu log báo Killed khi build gsplat (hết RAM)
cd @CODE@ || { echo "Chưa có @CODE@: chạy ô 3"; exit 0; }
[ -n "$TORCH_CUDA_OVERRIDE" ] && export TORCH_CUDA_OVERRIDE
[ -n "$MAX_JOBS" ] && export MAX_JOBS
bash scripts/pod/run_all.sh --background
""")


def monitor(nb: Notebook) -> None:
    nb.md("""
## 5. Theo dõi

Cập nhật mỗi 15 giây: trạng thái các việc và cuối log của việc đang chạy; tự dừng khi không còn việc nào. Bấm nút ■ (Interrupt) để thôi theo dõi: việc nền vẫn chạy.
""")
    nb.code(r"""
import subprocess, time
from IPython.display import clear_output

JOB = "@CODE@/scripts/pod/job.sh"

def job(*args):
    return subprocess.run(["bash", JOB, *args], capture_output=True, text=True, errors="replace")

while True:
    status = job("status").stdout
    running = [line.split()[0] for line in status.splitlines()
               if line and not line[0].isspace() and "ĐANG CHẠY" in line]
    tail = job("log", running[0], "20").stdout if running else ""
    clear_output(wait=True)
    print(time.strftime("%H:%M:%S"), "(cập nhật mỗi 15 giây; bấm ■ để thôi theo dõi, việc nền vẫn chạy)\n")
    print(status)
    if running:
        print(f"---------- cuối log của {running[0]}")
        print(tail)
    if job("busy").returncode != 0:
        print("=> Không còn việc nào chạy hoặc xếp hàng.")
        break
    time.sleep(15)
""")


# ------------------------------------------------------------------------------------------- ô cuối (chung)
def pack(nb: Notebook, number: int, name: str) -> None:
    """Ô xem mọi kết quả và đóng gói; gói mang tên bản của notebook: ket_qua_<tên bản>_<thời gian>.tar.gz."""
    nb.md(f"## {number}. Mọi kết quả (đọc file) và gói kết quả nhẹ `ket_qua_{name}_<thời gian>.tar.gz` để tải về "
          "(không có checkpoint, PLY hay video)\n\nGói mang tên bản của notebook này nhưng gồm mọi kết quả nhẹ trên "
          "volume (báo cáo, log, JSON của mọi lần chạy), không chỉ của bản này.")
    nb.code(fr"""
!bash @CODE@/scripts/pod/results.sh
!bash @CODE@/scripts/pod/pack_results.sh --name {name}
""")


def stop(nb: Notebook, number: int) -> None:
    nb.md(f"""
## {number}. Dừng mọi việc đang chạy

Đặt `DUNG=1` rồi chạy ô: dừng các việc chạy nền (kể cả việc đang xếp hàng) và mọi tiến trình con. Chạy lại ô tương ứng sau đó: phần đã xong tự bỏ qua, lượt train dở được chạy lại từ đầu.
""")
    nb.code(r"""
%%bash
DUNG=0   # 1: dừng mọi việc chạy nền
if [ "$DUNG" != 1 ]; then echo "Chưa dừng gì: đặt DUNG=1 rồi chạy lại ô này"; exit 0; fi
[ -f @CODE@/scripts/pod/job.sh ] && bash @CODE@/scripts/pod/job.sh stop --all
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
""")


# ------------------------------------------------------------------------------------------------- khi gặp lỗi
COMMON_ROWS = [
    "| `✗ GitHub chưa nhận khóa`, `Permission denied (publickey)` (ô 2, 3) | Khóa công khai chưa được thêm vào Deploy keys của **đúng** repo `SpatialRoom3D`: thêm rồi chạy lại ô 2 |",
    "| `Code trên pod có thay đổi chưa commit` (ô 3) | Có file trong `/workspace/SpatialRoom3D` bị sửa (vd. mở notebook trong repo): xem `git -C /workspace/SpatialRoom3D status`, tự quyết định giữ hay bỏ |",
    "| `no Python with PyYAML` | `python3 -m pip install pyyaml`, hoặc chờ ô 4 cài xong env gs-gsplat |",
    '| `driver supports CUDA 12.6 < 12.8` | Ô 4: đặt `TORCH_CUDA_OVERRIDE="cu126"`, chạy lại ô 4 |',
    '| `Killed` khi build gsplat | Ô 4: đặt `MAX_JOBS="4"`, chạy lại ô 4 |',
    "| `compute capability ... is not in TORCH_CUDA_ARCH_LIST` | GPU không được hỗ trợ (vd. T4, RTX 50xx): deploy GPU khác |",
    "| `another job is using the GPU` | Đang có việc khác chạy: xem ô 5; muốn dừng thì ô {stop} |",
    "| `MẤT` ở ô 5 | Pod tắt hoặc việc bị giết giữa chừng: chạy lại ô vừa dùng |",
]
SPACE_ROW = ("| {space_msg} (ô {run}{space_where}) | Volume không đủ chỗ: tăng dung lượng network volume trên trang Runpod, "
             "đặt `VOLUME_GB` bằng dung lượng mới ở ô {run} rồi chạy lại (phần đã xong tự bỏ qua) |")
SEED_SPACE = "`seed ... would not fit in a ... GB volume`, `may not fit`, `không đủ chỗ trống cho seed`"
B0_SPACE = "`B0 (about ... GB more) would not fit in a ... GB volume`"
DOWNLOAD_ROWS = [
    "| `SHA-256 of ... expected`, `is damaged` (tải dữ liệu) | File tải về hỏng hoặc khác bản gốc: file đó đã bị xóa, chạy lại ô {run} để tải lại |",
    "| `already exist(s) ... but are incomplete` (tải dữ liệu) | Thư mục cảnh đã có nhưng thiếu phần (vd. tự đưa lên dở dang): xem lại hoặc dời thư mục đó đi, rồi chạy lại; bước tải không bao giờ sửa dữ liệu đã có |",
]
SCENE_ROWS = [
    "| `prepared must be <results.gsplat>/<cảnh>/colmap` | `configs/local.yaml` chỉ đổi một trong hai khóa của `custom_scene`: đổi cả hai |",
    "| `Chỉ có 1 ảnh`, `Chỉ có N ảnh dùng được` | Chụp thêm (ít nhất 10, nên 30–100 ảnh), tải vào cùng thư mục, chạy lại ô {scene} |",
    "| `ảnh khác kích thước` | Lẫn ảnh dọc và ngang, hoặc khác ống kính: bỏ các ảnh ở nhóm ít hơn (tên in trong thông báo) hoặc chụp lại cùng chiều máy |",
    "| `không ảnh nào có tiêu cự trong EXIF` | Ảnh đã mất EXIF (gửi qua ứng dụng nhắn tin, chụp màn hình, chỉnh sửa): dùng file gốc của máy nếu có |",
    "| `đã được dựng từ ..., không phải ...` | Tên cảnh đã dùng cho nguồn khác: đặt `SCENE` mới |",
    "| `COLMAP verdict FAIL` | Xem các dòng `gợi ý` ở ô {results}, chụp / quay lại theo `docs/data/capture_protocol.md` và chạy ô {scene} với tên cảnh mới |",
    "| `góc nhìn: THIẾU` (ô {results}), `VIEW COVERAGE: LOW` (log) | Chỉ là cảnh báo: có hướng chưa ảnh nào nhìn tới, hoặc camera gần như đứng yên |",
    "| `not a finished run` (ô {results}) | Lượt `RUN` chưa train xong hoặc tên sai: xem tên các lượt trong bảng của ô {results} |",
    "| `COLMAP model ... changed after this run was trained` (ô {results}) | Cảnh đã làm lại COLMAP sau lượt này: chạy ô {scene} với `SOURCE = \"-\"` (lượt cũ tự được train lại), rồi render lượt mới; `XUAT_PLY` của lượt cũ vẫn được |",
]
OTHER_ROW = "| Lỗi khác | Chạy ô {pack} và gửi gói `ket_qua_*.tar.gz` |"


def troubleshooting(nb: Notebook, rows: list[str], **numbers) -> None:
    table = "\n".join(fill(row, **numbers) for row in rows)
    nb.md(f"""
## Khi gặp lỗi

| Thấy trong log / ô | Cách xử lý |
|---|---|
{table}
""")


# ------------------------------------------------------------------------------------------------- code tổng
def runpod() -> Notebook:
    nb = Notebook("runpod.ipynb")
    nb.md("""
# SpatialRoom3D: tái tạo không gian nội thất 3D từ ảnh / video, chạy trên Runpod

Notebook này nằm trong repo (`app/notebooks/runpod.ipynb`), dùng cho **pod mới gắn network volume mới (trống)**.

**Tên bản:** code tổng là `main` (thực nghiệm chính, ô 7 `EXP = "main"`: việc nền `main`, báo cáo `main_<thời gian>.md`, gói kết quả `ket_qua_main_<thời gian>.tar.gz`); dò tham số là `tune_mcmc`. Ba notebook chạy riêng từng phương pháp: `b0_inria`, `gsplat_default`, `gsplat_mcmc` (cùng tên cho việc nền, báo cáo, gói kết quả). PLY xuất ra mang tên bản của lượt, cảnh và seed: `gsplat_<chế độ>_<cảnh>_s<seed>.ply`.

**Cách dùng:** tải file notebook này lên `/workspace` (kéo thả vào thanh file bên trái), mở bản ở `/workspace`, chạy từng ô bằng **Shift+Enter**. Không mở thẳng bản trong repo: Jupyter ghi kết quả chạy vào file, làm repo trên pod có thay đổi và ô 3 sẽ không cập nhật code. Chép lại khi cần: `cp /workspace/SpatialRoom3D/app/notebooks/runpod.ipynb /workspace/`.

**Bố cục trên volume:**
```
/workspace/
├── SpatialRoom3D/        code (repo Git, clone từ GitHub)
├── data/                 chỉ dữ liệu gốc (pipeline không ghi vào)
│   ├── custom_scenes/<cảnh>/        ảnh / video tự chụp
│   └── benchmarks/{deep_blending,mipnerf360}/   bộ benchmark (bạn tự đưa lên khi cần)
├── outputs/              mọi thứ sinh ra
│   ├── custom_scenes/<cảnh>/        colmap/ + các lượt train <chế_độ>_s<seed>/
│   ├── benchmarks/<benchmark>/      inria_b0/, gsplat/, gsplat_quick/
│   ├── checkpoints/                 PLY đã xuất để tải về / xem 3D (gsplat_<chế độ>_<cảnh>_s<seed>.ply)
│   ├── panels/                      ảnh so sánh các lượt, video xếp chồng (ô 7, ô 9)
│   ├── logs/, reports/, jobs/       log, báo cáo (<tên bản>_<thời gian>.md), trạng thái việc chạy nền
├── miniforge3/, code/    env conda, mã nguồn 3DGS Inria và gsplat (ô 4 cài)
└── .ssh/                 khóa deploy chỉ đọc (ô 2); không bao giờ gửi thư mục này cho ai
```

| Lúc nào | Chạy các ô |
|---|---|
| Lần đầu trên pod mới, volume trống: chỉ cài môi trường | 1 → 2 → 3 → 4 (theo dõi ở ô 5) |
| Thực nghiệm chính: tự tải dữ liệu, so sánh `default` / `mcmc`, 3 seed (tốn GPU nhiều giờ; gửi được ngay sau ô 4, tự chạy khi cài xong) | 7 (`WHAT = "experiment"`, `EXP = "main"`, `CHAY = True`), theo dõi ở ô 5 |
| Dò tham số mcmc trên `room` (tốn GPU khoảng 8 giờ), rồi vòng ghép khi đã có file kế hoạch | 7 (`WHAT = "experiment"`, `EXP = "tune_mcmc"`, sau đó `EXP = "tune_mcmc_combo"`, `CHAY = True`), theo dõi ở ô 5 |
| Đưa dữ liệu của bạn lên (trong lúc ô 4 chạy cũng được), rồi kiểm tra | 6 |
| Benchmark bạn chọn: tải dữ liệu, Deep Blending / Mip-NeRF 360 (tốn GPU) | 7 (`CHAY = True`), theo dõi ở ô 5 |
| Cảnh tự chụp: thư mục ảnh hoặc video (tốn GPU) | 8 (`CHAY = True`), theo dõi ở ô 5, kết quả ở ô 9 |
| Ảnh so sánh các lượt trên cùng ảnh test (vùng cắt phóng to, bản đồ sai số) và video xếp chồng (không cần GPU) | 7 (`WHAT = "panels"`) hoặc 9 (`SO_SANH = True`) |
| Xem mọi kết quả, đóng gói để tải về | 10 |
| Dừng mọi việc đang chạy | 11 (`DUNG=1`) |
| Chạy riêng một phương pháp của thực nghiệm chính (B0, `default` hoặc `mcmc`): cùng dữ liệu, cùng thư mục kết quả, lượt đã xong ở đây hay ở đó đều được dùng lại | notebook `b0_inria.ipynb`, `gsplat_default.ipynb`, `gsplat_mcmc.ipynb` (`app/notebooks/` của repo; sau ô 3 cũng có ở `/workspace/SpatialRoom3D/app/notebooks/`: tải lên hoặc chép vào `/workspace` như notebook này) |

- Notebook không tự chạy train hay tự tải dữ liệu: bạn chọn ở ô 7, ô 8. Dữ liệu benchmark tải được thẳng về pod (ô 7); ảnh / video của bạn thì bạn tự đưa lên (ô 6).
- Việc dài đều chạy nền: đóng trình duyệt không sao, nhưng **pod vẫn tính tiền** tới khi Stop / Terminate.
- Mỗi lúc chỉ một việc dùng GPU; việc gửi sau tự xếp hàng.
- Xong việc: tải gói `ket_qua_main_<thời gian>.tar.gz` (ô 10 in đường dẫn), rồi **Terminate pod**. Dữ liệu trên volume vẫn còn.
- Cùng các lệnh này bằng dòng lệnh: `cd /workspace/SpatialRoom3D && python3 -m app.cli --help` (hướng dẫn: `docs/USAGE_GUIDE.md`).
""")
    machine(nb)
    deploy_key(nb)
    fetch_code(nb)
    setup(nb)
    monitor(nb)
    nb.md("""
## 6. Đưa dữ liệu lên, kiểm tra (chỉ đọc, không cần GPU)

Bạn tự đưa dữ liệu lên (kéo thả vào thanh file của Jupyter, hoặc cách khác), đúng chỗ dưới đây, rồi chạy ô này để kiểm tra. Benchmark (Deep Blending, Mip-NeRF 360) không cần đưa lên: ô 7 tải thẳng về pod (`WHAT = "fetch"`, hoặc `"experiment"` tự tải).

| Dữ liệu | Đặt ở |
|---|---|
| Ảnh tự chụp | `/workspace/data/custom_scenes/<cảnh>/`: các ảnh JPG/JPEG/PNG nằm thẳng trong thư mục (ô 8) |
| Video tự quay | `/workspace/data/custom_scenes/<cảnh>/<video>.mp4` (ô 8) |
| Deep Blending | `/workspace/data/benchmarks/deep_blending/drjohnson/`, `.../playroom/`: mỗi cảnh có `images/` và `sparse/0/` (ô 7 tải được) |
| Mip-NeRF 360 | `/workspace/data/benchmarks/mipnerf360/<cảnh>/`: `images/`, `images_2/`, `images_4/`, `sparse/0/` (ô 7 tải được) |

- Tên cảnh: chữ không dấu, số và `. _ -`, không dấu cách. Không đặt dữ liệu trong `@CODE@` (thư mục code).
- Dữ liệu nằm chỗ khác (vd. Mip-NeRF 360 đặt tên `MipNeRF360`): tạo `@CODE@/configs/local.yaml` với nội dung `datasets: {mipnerf360: {root: "${WS}/data/MipNeRF360"}}`, rồi chạy lại ô này.

Ô này in theo `configs/*.yaml`: thư mục gốc của từng bộ dữ liệu, từng cảnh có `images/` và `sparse/0/` không, có bao nhiêu lượt train. Benchmark chưa đưa lên thì báo thiếu: không sao nếu bạn không định chạy nó.
""".replace("@CODE@", CODE))
    nb.code(r"""
import subprocess
repo = "@CODE@"
out = subprocess.run(["bash", f"{repo}/scripts/pod/check_datasets.sh"], capture_output=True, text=True,
                     errors="replace", cwd=repo)
print(out.stdout + out.stderr)
""")
    nb.md("""
## 7. Benchmark: thực nghiệm (chính, dò tham số), tải dữ liệu, Deep Blending hoặc Mip-NeRF 360

Chọn `WHAT` (và `BENCH` với các việc của một benchmark). Việc chạy nền (`experiment`, `fetch`, `gsplat`, `b0`) chỉ chạy khi `CHAY = True`, xếp hàng sau việc khác; theo dõi ở ô 5. Lượt đã có cùng cấu hình tự bỏ qua. Protocol từng benchmark: `experiments/<benchmark>/README.md`.

| `WHAT` | Việc | Ghi chú |
|---|---|---|
| `"experiment"` | Một thực nghiệm theo kế hoạch `EXP` trong `configs/experiments/` (`scripts/pod/run_experiment.sh`). `EXP = "main"`: **thực nghiệm chính**, tải dữ liệu còn thiếu → B0 trên Deep Blending → `default`, `mcmc` với seed 42, 43, 44 trên Deep Blending (2 cảnh) và 4 cảnh trong nhà của Mip-NeRF 360 → bảng so sánh → gói kết quả. `EXP = "tune_mcmc"`: **dò tham số mcmc** trên `room`, seed 42: `default`, `mcmc` và 16 biến thể của `mcmc` (depth prior, Kill-then-Relocate, bilateral grid của notebook v2 và gói v2, chạy trước; 12 tham số), lọc Gaussian sau khi train, bảng so với `default`, với `mcmc` và trước / sau khi lọc; tự tải Depth Anything V2 Small cho depth prior (không tải được thì chỉ bỏ hai lượt dùng nó). `EXP = "tune_mcmc_combo"`: **vòng ghép**, 2 seed mới trên `room`, sau `main` (file kế hoạch viết khi có kết quả `tune_mcmc`) | Một việc nền. Ước lượng: `main` khoảng 12 giờ GPU, thêm khoảng 56 GB trên volume; `tune_mcmc` khoảng 8 giờ GPU (khoảng 6,7 giờ khi chạy sau `main`), thêm khoảng 35 GB; vòng ghép khoảng 1–3,5 giờ GPU, thêm khoảng 10–20 GB (giữ mọi file). Đặt `VOLUME_GB` đúng dung lượng volume. `CHAY = False`: chỉ in kế hoạch và dữ liệu đã có. Luôn chạy đầy đủ; `BENCH`, `SEEDS`, `SCENES` không dùng |
| `"check"` | Kiểm tra dữ liệu: bố cục từng cảnh (`scripts/prepare_data.sh <benchmark>`) | Không dùng GPU, chạy ngay |
| `"fetch"` | Tải `BENCH` về volume (`scripts/pod/fetch_data.sh`): chỉ cảnh còn thiếu (chọn bằng `SCENES`), chỉ các thư mục cần dùng; xóa file zip sau khi giải nén | Không dùng GPU nhưng chạy nền, xếp hàng như việc khác. Deep Blending: `tandt_db.zip` của Inria (650 MB). Mip-NeRF 360: tải cả `360_v2.zip` (7 cảnh) dù chỉ chọn vài cảnh |
| `"gsplat"` | `default` rồi `mcmc` cùng số Gaussian, 30.000 vòng, bảng so sánh (`scripts/run_train.sh <benchmark> gsplat`) | Tốn GPU. `QUICK = True`: chạy thử 3.000 vòng (thư mục riêng). `SEEDS = "42 43 44"` để đủ 3 seed (cần trước khi nói "cải thiện") |
| `"b0"` | 3DGS gốc của Inria, đối chiếu số liệu công bố (`scripts/run_train.sh <benchmark> b0`) | Tốn GPU. Mốc đối chiếu cho `default` |
| `"evaluate"` | Lập lại bảng so sánh từ kết quả đã có, mọi chế độ đã chạy (`scripts/run_evaluate.sh`); `SCENES` chọn cảnh, `TAGS` đưa thêm các biến thể có tag (vd. 16 biến thể của `tune_mcmc`: `"mono kr bilagrid v2 or0001 or0005 or005 sr0001 sr005 nlr250k nlr1m mo002 rbg depth rs30k sh2"`) | Không dùng GPU, chạy ngay |
| `"panels"` | Ảnh so sánh các lượt của từng cảnh trong `SCENES` trên cùng ảnh test (ảnh gốc, render, vùng cắt phóng to, bản đồ sai số) và video quỹ đạo của các lượt xếp chồng (`scripts/run_panels.sh`); `PANEL_RUNS` chọn lượt, để trống = các lượt seed 42 không có tag | Không dùng GPU, chạy ngay. Kết quả: `/workspace/outputs/panels/<benchmark>[_quick]_<cảnh>_<thời gian>/` (không vào gói kết quả nhẹ) |

- Thực nghiệm chính: khoảng 56 GB (giữ mọi file) cộng môi trường conda khoảng 15–20 GB, nên volume 100 GB đủ theo ước lượng. Chạy cả `main`, `tune_mcmc`, vòng ghép trên cùng volume, rồi thêm chế độ mới vào `main`: nên dùng 170 GB. Tăng volume trên trang Runpod (tăng được, không giảm được) rồi đặt `VOLUME_GB` bằng dung lượng mới. Trước khi chạy và trước mỗi seed, việc kiểm tra chỗ trống; không đủ thì dừng train, các lượt đã xong vẫn giữ.
- `tune_mcmc`: một cảnh, một seed, chỉ để chọn ứng viên cho vòng ghép; vòng ghép (2 seed mới trên `room`) chọn tham số cho chế độ mới `mcmc_indoor`. Cả hai chưa phải kết quả "cải thiện" (`docs/design/experiments.md` mục 3, 3.6).
- `gsplat`, `b0`: dữ liệu phải có sẵn; trước khi xếp hàng, ô này kiểm tra dữ liệu (không cần GPU), thiếu thì báo ngay và không xếp hàng. Muốn bảng đối chiếu đủ thì chạy `b0` trước `gsplat`: chưa có B0, việc `gsplat` (30.000 vòng) kết thúc với exit 1 vì phần đối chiếu B0 là INCOMPLETE; các lượt train vẫn được giữ.
- Thời gian đo trên RTX 4090 ngày 17/10/2026 (repo trước, Deep Blending, seed 42): chạy thử 3.000 vòng 17 phút, B0 khoảng 50 phút, gsplat đầy đủ (`default` + `mcmc`) 74 phút.
- Chạy một phần cảnh: `SCENES = "playroom"` (Deep Blending), `SCENES = "room counter"` (Mip-NeRF 360).
""")
    nb.code(r"""
WHAT = "experiment"       # "experiment", "check", "fetch", "gsplat", "b0", "evaluate" hoặc "panels"
CHAY = False              # True: cho phép việc chạy nền (experiment, fetch, gsplat, b0)
VOLUME_GB = 100           # experiment: dung lượng thật của network volume (GB)
AUTO_STOP = False         # experiment: True = dừng pod khi việc cuối cùng xong (chưa thử trên pod)
EXP = "main"              # experiment: "main" (thực nghiệm chính), "tune_mcmc" (dò tham số mcmc trên room), "tune_mcmc_combo" (vòng ghép)
BENCH = "deep_blending"   # check, fetch, gsplat, b0, evaluate: "deep_blending" hoặc "mipnerf360"
QUICK = False             # True: chạy thử 3.000 vòng (gsplat, evaluate, panels)
SEEDS = "42"              # vd. "42 43 44" (gsplat)
SCENES = ""               # để trống = mọi cảnh của benchmark (fetch, gsplat, b0, evaluate); panels: phải chọn cảnh
TAGS = ""                 # evaluate: tag các biến thể đưa thêm vào bảng, vd. "or0001 rbg" (tune_mcmc)
PANEL_RUNS = ""           # panels: lượt đem so, vd. "mcmc_s42 mcmc_indoor_s42" (trống = seed 42, không có tag)

import os, subprocess
repo = "@CODE@"
env = dict(os.environ, SEEDS=SEEDS, QUICK="1" if QUICK else "0")
if SCENES.strip():
    env["SCENES"] = SCENES
if TAGS.strip():
    env["TAGS"] = TAGS

def run(cmd, tail=6000):
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=repo, env=env)
    print(out.stdout[-tail:] + out.stderr[-3000:])
    return out.returncode

if WHAT == "experiment":
    # kế hoạch trong configs/experiments/ quyết định chế độ, seed, cảnh; không phải biến của ô này
    env = dict(os.environ, VOLUME_GB=str(VOLUME_GB), AUTO_STOP="1" if AUTO_STOP else "0")
    if QUICK:
        print("experiment luôn chạy đầy đủ (30.000 vòng): đặt QUICK = False.")
    elif not CHAY:
        run(["bash", f"{repo}/scripts/pod/run_experiment.sh", EXP, "--plan"], tail=12000)
        print("=> Chưa chạy: đây là kế hoạch. Đặt CHAY = True để xếp hàng (tốn GPU nhiều giờ).")
    else:
        run(["bash", f"{repo}/scripts/pod/run_experiment.sh", EXP, "--background"])
elif WHAT == "check":
    run(["bash", f"{repo}/scripts/prepare_data.sh", BENCH])
elif WHAT == "evaluate":
    run(["bash", f"{repo}/scripts/run_evaluate.sh", BENCH])
elif WHAT == "panels":
    env = dict(env, VIDEO="1")
    if not SCENES.split():
        print('panels: đặt SCENES, vd. SCENES = "counter" (mỗi cảnh một bộ ảnh so sánh).')
    for scene in SCENES.split():
        run(["bash", f"{repo}/scripts/run_panels.sh", BENCH, scene, *PANEL_RUNS.split()])
elif WHAT in ("fetch", "gsplat", "b0") and not CHAY:
    print(f"Chưa chạy {WHAT} trên {BENCH}: đặt CHAY = True ({'tải dữ liệu' if WHAT == 'fetch' else 'tốn GPU'}).")
elif WHAT == "fetch":
    run(["bash", f"{repo}/scripts/pod/job.sh", "start", f"fetch_{BENCH}", "--",
         "bash", f"{repo}/scripts/pod/fetch_data.sh", BENCH, *SCENES.split()])
elif WHAT in ("gsplat", "b0"):
    # kiểm tra dữ liệu trước (không cần GPU): thiếu thì không xếp hàng
    if run(["bash", f"{repo}/scripts/prepare_data.sh", BENCH], tail=2000) != 0:
        print(f"=> Không xếp hàng {WHAT}: dữ liệu {BENCH} chưa có hoặc sai bố cục (tải: WHAT = \"fetch\"; chỗ đặt: ô 6).")
    else:
        job = f"{WHAT}_quick_{BENCH}" if WHAT == "gsplat" and QUICK else f"{WHAT}_{BENCH}"
        run(["bash", f"{repo}/scripts/pod/job.sh", "start", job, "--",
             "bash", f"{repo}/scripts/run_train.sh", BENCH, WHAT])
else:
    print(f"WHAT = {WHAT!r}: chọn \"experiment\", \"check\", \"fetch\", \"gsplat\", \"b0\", \"evaluate\" hoặc \"panels\".")
""")
    nb.md("""
## 8. Cảnh tự chụp (thư mục ảnh hoặc video) → COLMAP → train `default` → train `mcmc`

**Ảnh chụp** (JPG/JPEG/PNG, ít nhất 10 ảnh, nên 30–100, cùng một căn phòng, cùng chiều máy, dùng file ảnh gốc của máy để giữ EXIF; cách chụp: `docs/data/capture_protocol.md` mục 8):
1. Trong thanh file, vào `/workspace/data/custom_scenes`, bấm nút thư mục mới (New Folder), đặt tên cảnh, ví dụ `phong1`.
2. Mở thư mục đó, bấm Upload và chọn **nhiều ảnh cùng lúc** (hoặc kéo thả nhiều file).
3. Đặt `SOURCE = "/workspace/data/custom_scenes/phong1"`.

**Video:** tạo thư mục như trên, tải video vào đó, đặt `SOURCE` là đường dẫn file video (vd. `/workspace/data/custom_scenes/phong2/phong2.mp4`).

Đặt `CHAY = True` rồi chạy ô (tốn GPU: COLMAP rồi hai lượt train, chạy nền). `SCENE` để trống thì lấy tên thư mục chứa ảnh / video. Kết quả: `/workspace/outputs/custom_scenes/<cảnh>/` (`colmap/` và các lượt train, mỗi lượt có ảnh test, video quỹ đạo và file PLY); thư mục ảnh gốc không bị sửa.

Một ảnh, dưới 10 ảnh, ảnh khác kích thước, hay tên cảnh đã dùng cho nguồn khác thì ô này báo lỗi ngay, không xếp hàng. COLMAP báo `FAIL` thì không train; lý do và cách chụp lại ở ô 9 (`gợi ý`, `góc nhìn`).

- Mỗi nguồn một tên cảnh (một job). Cảnh đã có COLMAP thì chỉ train (không đọc lại nguồn); `SOURCE = "-"` cũng là chỉ train (khi đó phải đặt `SCENE`).
- `REDO_COLMAP = True`: làm lại bước ảnh đầu vào + COLMAP; lượt train cũ được giữ dưới tên `*.prev_<thời gian>`.
- `RETRAIN = True`: train lại cả hai chế độ dù đã có kết quả.
""")
    nb.code(scene_code("COLMAP + train default, mcmc; tốn GPU", modes=None))
    nb.md("""
## 9. Kết quả một cảnh, xuất PLY, render lại, lọc Gaussian, ảnh so sánh

Ô này in: ảnh đầu vào, kết quả COLMAP, gợi ý, độ phủ góc nhìn, các lượt train (PSNR / SSIM / LPIPS, số Gaussian), video quỹ đạo và file PLY.

- `XUAT_PLY = True`: đưa file PLY của lượt `RUN` vào `/workspace/outputs/checkpoints/custom_scene/<cảnh>/gsplat_<chế độ>_<cảnh>_s<seed>.ply` (vd. lượt `mcmc_s42` của `phong1`: `gsplat_mcmc_phong1_s42.ply`, tên bản như notebook `gsplat_mcmc.ipynb`) để tải về và mở bằng trình xem 3D Gaussian Splatting (không cần GPU, không tốn thêm chỗ).
- `RENDER_LAI = True`: render lại ảnh test và video quỹ đạo `TRAJ` (`interp`, `ellipse`, `spiral`) từ checkpoint của lượt `RUN`, vào thư mục mới trong lượt đó (chạy nền, cần GPU; không train, không ghi đè lượt gốc).
- `SO_SANH = True`: ảnh so sánh các lượt seed 42 của cảnh trên cùng ảnh test (ảnh gốc, render, vùng cắt phóng to, bản đồ sai số) và video quỹ đạo của các lượt xếp chồng (`scripts/run_panels.sh`, không cần GPU). Kết quả trong `/workspace/outputs/panels/custom_scene_<cảnh>_<thời gian>/` (tải qua thanh file của Jupyter).
- `LOC = True`: lọc Gaussian của lượt `RUN` (bỏ Gaussian gần như trong suốt và Gaussian quá lớn) rồi chấm lại trên đúng ảnh test (chạy nền, cần GPU; `scripts/run_filter.sh`). Kết quả nằm trong `<lượt>/filtered/` (PLY đã lọc ở `filtered/ply/`), lượt gốc không đổi. Chạy lại ô này sau khi việc nền xong để xem chỉ số trước / sau khi lọc.
""")
    nb.code(r"""
SCENE = "phong1"
RUN = "default_s42"      # lượt: <chế_độ>_s<seed>, vd. mcmc_s42
XUAT_PLY = False         # True: xuất PLY của lượt RUN (không cần GPU)
RENDER_LAI = False       # True: render lại ảnh test + video từ checkpoint (chạy nền, cần GPU)
TRAJ = "interp"          # quỹ đạo video: "interp", "ellipse" hoặc "spiral"
LOC = False              # True: lọc Gaussian của lượt RUN rồi chấm lại (chạy nền, cần GPU)
SO_SANH = False          # True: ảnh so sánh các lượt seed 42 của cảnh, video xếp chồng (không cần GPU)

import os, subprocess
repo = "@CODE@"
out = subprocess.run(["bash", f"{repo}/scripts/pod/results.sh", SCENE], capture_output=True, text=True,
                     errors="replace", cwd=repo)
print(out.stdout + out.stderr)
if XUAT_PLY:
    out = subprocess.run(["bash", f"{repo}/scripts/run_render.sh", "custom", SCENE, RUN], capture_output=True,
                         text=True, errors="replace", cwd=repo, env=dict(os.environ, RENDER="0"))
    print(out.stdout + out.stderr)
if RENDER_LAI:
    out = subprocess.run(["bash", f"{repo}/scripts/pod/job.sh", "start", f"render_{SCENE}_{RUN}", "--",
                          "bash", f"{repo}/scripts/run_render.sh", "custom", SCENE, RUN], capture_output=True,
                         text=True, errors="replace", cwd=repo, env=dict(os.environ, TRAJ=TRAJ))
    print(out.stdout + out.stderr)
if LOC:
    out = subprocess.run(["bash", f"{repo}/scripts/pod/job.sh", "start", f"filter_{SCENE}_{RUN}", "--",
                          "bash", f"{repo}/scripts/run_filter.sh", "custom", SCENE, RUN], capture_output=True,
                         text=True, errors="replace", cwd=repo)
    print(out.stdout + out.stderr)
if SO_SANH:
    out = subprocess.run(["bash", f"{repo}/scripts/run_panels.sh", "custom", SCENE], capture_output=True,
                         text=True, errors="replace", cwd=repo, env=dict(os.environ, VIDEO="1"))
    print(out.stdout + out.stderr)
""")
    pack(nb, 10, MAIN_NAME)
    stop(nb, 11)
    rows = COMMON_ROWS + [
        "| `dataset not found` (ô 7) | Dữ liệu benchmark chưa có ở chỗ `configs/<benchmark>.yaml` ghi: tải bằng ô 7 (`WHAT = \"fetch\"`), hoặc tự đưa lên đúng chỗ (bảng ở ô 6), hoặc ghi đường dẫn vào `configs/local.yaml` |",
        SPACE_ROW] + DOWNLOAD_ROWS + SCENE_ROWS + [OTHER_ROW]
    troubleshooting(nb, rows, space_msg=SEED_SPACE, stop=11, run=7, space_where=", `experiment`", scene=8, results=9,
                    pack=10)
    return nb


def scene_code(what: str, modes: str | None) -> str:
    """Ô cảnh tự chụp (scripts/pod/run_scene.sh); `modes`: MODES của các chế độ train (None: mặc định default, mcmc)."""
    env = "" if modes is None else f',\n                         env=dict(os.environ, MODES="{modes}")'
    return fr"""
CHAY = False               # True: chạy ({what})
SOURCE = "/workspace/data/custom_scenes/phong1"  # thư mục ảnh hoặc file video; "-" = chỉ train cảnh đã có
SCENE = ""                 # để trống: tên thư mục chứa ảnh / video; chỉ chữ, số và . _ - (không dấu cách)
REDO_COLMAP = False        # True: làm lại bước ảnh đầu vào + COLMAP từ đầu
RETRAIN = False            # True: train lại mọi chế độ dù đã có kết quả
FRAMES_ARGS = ""           # tham số thêm, vd. "--target 300" (video) hoặc "--min-images 8" (ảnh)

import os, shlex, subprocess
if os.path.isdir(SOURCE):
    default = os.path.basename(SOURCE.rstrip("/"))
else:  # video: tên thư mục chứa nó; video đặt thẳng trong thư mục chung thì lấy tên file
    default = os.path.basename(os.path.dirname(os.path.abspath(SOURCE)))
    if default in ("custom_scenes", "data", "workspace", ""):
        default = os.path.splitext(os.path.basename(SOURCE))[0]
scene = SCENE.strip() or default
if SOURCE == "-" and not SCENE.strip():
    print("SOURCE = \"-\" (chỉ train) cần SCENE")
elif not CHAY:
    print(f"Chưa chạy cảnh {{scene}}: đặt CHAY = True (COLMAP + train, tốn GPU).")
else:
    cmd = ["bash", "@CODE@/scripts/pod/run_scene.sh", SOURCE, scene, "--background"]
    if REDO_COLMAP:
        cmd.append("--redo-colmap")
    if RETRAIN:
        cmd.append("--retrain")
    if FRAMES_ARGS.strip():
        cmd += ["--", *shlex.split(FRAMES_ARGS)]
    print(f"cảnh: {{scene}}")
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd="@CODE@"{env})
    print(out.stdout + out.stderr)
"""


# ------------------------------------------------------------------------------------- một phương pháp mỗi notebook
METHODS = {
    "b0": {
        "file": "b0_inria.ipynb",
        "name": "B0",
        "long": "3DGS gốc của Inria",
        "what": ("**B0**: 3DGS gốc (Kerbl et al., SIGGRAPH 2023) bằng code chính thức của Inria "
                 "(`graphdeco-inria/gaussian-splatting` @54c035f, env `gs-inria`), tham số như `full_eval.py`. Là mốc "
                 "đối chiếu: `default` của gsplat chỉ được dùng làm baseline khi khớp B0 trên Deep Blending."),
        "part": "phần B0 của thực nghiệm chính: Deep Blending (`drjohnson`, `playroom`), 30.000 vòng",
        "cost_short": "tốn GPU khoảng 1 giờ",
        "cost": "khoảng 1 giờ GPU (số đo ngày 17/10/2026 của repo trước: 49,5 phút, gồm cả tải dữ liệu), thêm khoảng 5 GB",
        "steps": ("kiểm tra môi trường (env `gs-inria`) → kiểm tra chỗ trống → tải Deep Blending nếu chưa có → B0 trên "
                  "`drjohnson`, `playroom` → bảng B0 so với số liệu Inria công bố → gói kết quả nhẹ"),
        "report": "các bước và bảng B0 so với số liệu Inria công bố (PSNR / SSIM / LPIPS từng cảnh, trung bình, chênh lệch, kết luận)",
    },
    "default": {
        "file": "gsplat_default.ipynb",
        "name": "`default`",
        "long": "3DGS với DefaultStrategy của gsplat",
        "what": ("**`default`**: 3DGS (Kerbl et al., 2023) cài đặt lại trong gsplat @6e8c837 (`examples/simple_trainer.py`, "
                 "DefaultStrategy: thêm / bớt Gaussian theo heuristic của 3DGS gốc; env `gs-gsplat`). Là baseline của "
                 "mọi bảng so sánh."),
        "part": ("phần `default` của thực nghiệm chính: Deep Blending (2 cảnh) và 4 cảnh trong nhà của Mip-NeRF 360, "
                 "seed 42, 43, 44, 30.000 vòng"),
        "cost_short": "tốn GPU khoảng 6 giờ",
        "cost": "khoảng 6 giờ GPU, thêm khoảng 30 GB",
        "steps": ("kiểm tra môi trường (env `gs-gsplat`, smoke test) → kiểm tra chỗ trống → tải dữ liệu còn thiếu → "
                  "`default` theo từng seed trên mọi cảnh → bảng so sánh → gói kết quả nhẹ"),
        "report": "các bước và bảng của từng benchmark (PSNR / SSIM / LPIPS, số Gaussian, đối chiếu số Inria công bố và B0 khi đã có B0)",
        "modes": "default",
        "scene_title": "train `default`",
        "scene_what": "COLMAP + train default; tốn GPU",
        "run": "default_s42",
    },
    "mcmc": {
        "file": "gsplat_mcmc.ipynb",
        "name": "`mcmc`",
        "long": "3DGS-MCMC của gsplat",
        "what": ("**`mcmc`**: 3DGS-MCMC (Kheradmand et al., NeurIPS 2024) qua MCMCStrategy của gsplat @6e8c837 "
                 "(env `gs-gsplat`), cùng số Gaussian tối đa với lượt `default` cùng cảnh, cùng seed."),
        "part": ("phần `mcmc` của thực nghiệm chính: Deep Blending (2 cảnh) và 4 cảnh trong nhà của Mip-NeRF 360, "
                 "seed 42, 43, 44, 30.000 vòng; lượt `default` cùng cảnh, cùng seed đã xong thì dùng lại, chưa có thì "
                 "train trước (`mcmc` lấy số Gaussian tối đa từ nó)"),
        "cost_short": "tốn GPU khoảng 12 giờ khi chưa có lượt `default`, khoảng 6 giờ khi đã có",
        "cost": ("khoảng 12 giờ GPU và thêm khoảng 54 GB khi chưa có lượt `default` nào; khoảng một nửa khi `default` "
                 "đã xong, ở `gsplat_default.ipynb` hay code tổng"),
        "steps": ("kiểm tra môi trường (env `gs-gsplat`, smoke test) → kiểm tra chỗ trống → tải dữ liệu còn thiếu → "
                  "`default` (nếu chưa có) rồi `mcmc` theo từng seed trên mọi cảnh → bảng so với `default` → gói kết quả nhẹ"),
        "report": "các bước và bảng `mcmc` so với `default` của từng benchmark (PSNR / SSIM / LPIPS, số Gaussian, số cặp tốt hơn)",
        "modes": "default mcmc",
        "scene_title": "train `default` (nếu chưa có) → train `mcmc`",
        "scene_what": "COLMAP + train default nếu chưa có, rồi mcmc; tốn GPU",
        "run": "mcmc_s42",
    },
}


def method_notebook(method: str) -> Notebook:
    info = METHODS[method]
    gsplat = method != "b0"
    nb = Notebook(info["file"])
    label = Path(info["file"]).stem  # tên bản = tên notebook (src.training.experiment.run_name("main", method))
    numbers = ({"run": 6, "scene": 7, "report": 8, "results": 9, "pack": 10, "stop": 11} if gsplat
               else {"run": 6, "report": 7, "pack": 8, "stop": 9})
    when = [
        ("Lần đầu trên pod mới, volume trống: cài môi trường (cả 3 env, dùng chung với các notebook khác)",
         "1 → 2 → 3 → 4 (theo dõi ở ô 5)"),
        (f"Chạy riêng {info['name']} trên benchmark, tự tải dữ liệu còn thiếu ({info['cost_short']})",
         f"{numbers['run']} (`CHAY = True`), theo dõi ở ô 5"),
    ]
    if gsplat:
        when.append((f"Cảnh tự chụp: thư mục ảnh hoặc video → COLMAP → {info['scene_title']} (tốn GPU)",
                     f"{numbers['scene']} (`CHAY = True`), theo dõi ở ô 5, kết quả ở ô {numbers['results']}"))
    when += [("Bảng của lần chạy benchmark gần nhất", str(numbers["report"]))]
    if gsplat:
        when.append(("Kết quả một cảnh tự chụp, xuất PLY", str(numbers["results"])))
    when += [("Xem mọi kết quả, đóng gói để tải về", str(numbers["pack"])),
             ("Dừng mọi việc đang chạy", f"{numbers['stop']} (`DUNG=1`)")]
    table = "\n".join(f"| {left} | {right} |" for left, right in when)
    b0_note = ("- B0 chỉ chạy trên benchmark: repo chưa có luồng B0 cho cảnh tự chụp. B0 trên Mip-NeRF 360 (không thuộc "
               "thực nghiệm chính): `runpod.ipynb`, ô 7, `WHAT = \"b0\"`, `BENCH = \"mipnerf360\"`.\n" if not gsplat else
               "- Đối chiếu `default` với B0 tự chạy chỉ khi mọi cảnh Deep Blending đã có B0 (`b0_inria.ipynb` hoặc code "
               f"tổng); chưa có thì không đối chiếu, không tính là lỗi (kế hoạch ở ô {numbers['run']} và báo cáo ở ô "
               f"{numbers['report']} ghi rõ). Muốn có đối chiếu ngay thì chạy `b0_inria.ipynb` trước; chạy B0 sau thì "
               f"chạy lại ô {numbers['run']} (lượt đã xong không train lại, bảng lập lại có B0).\n")
    nb.md(f"""
# SpatialRoom3D: chạy riêng {info['name']} ({info['long']}) trên Runpod

Notebook này nằm trong repo (`app/notebooks/{info['file']}`). Nó chạy riêng {info['what']}

Đây là đúng {info['part']}, cùng thư mục kết quả với code tổng (`runpod.ipynb`, ô 7, `EXP = "main"`; `scripts/pod/run_experiment.sh main --only {method}`). Lượt đã xong ở đây được code tổng dùng lại, và ngược lại: không train hai lần.

**Tên bản: `{label}`.** Việc nền, báo cáo (`/workspace/outputs/reports/{label}_<thời gian>.md`), log và gói kết quả (`ket_qua_{label}_<thời gian>.tar.gz`) của notebook này đều mang tên đó{f", file PLY xuất ra cũng vậy (`{label}_<cảnh>_s<seed>.ply`)" if gsplat else ""}; code tổng là `main`. Ghi chú, báo cáo đồ án: gọi đúng tên này.

**Cách dùng:** tải file notebook này lên `/workspace` (kéo thả vào thanh file bên trái), mở bản ở `/workspace`, chạy từng ô bằng **Shift+Enter**. Không mở thẳng bản trong repo: Jupyter ghi kết quả chạy vào file, làm repo trên pod có thay đổi và ô 3 sẽ không cập nhật code. Chép lại khi cần: `cp /workspace/SpatialRoom3D/app/notebooks/{info['file']} /workspace/`.

| Lúc nào | Chạy các ô |
|---|---|
{table}

- Ô 1–5 giống hệt `runpod.ipynb`: đã cài môi trường bằng notebook khác thì ô 4 chỉ kiểm tra lại.
{b0_note}- Việc dài đều chạy nền: đóng trình duyệt không sao, nhưng **pod vẫn tính tiền** tới khi Stop / Terminate. Mỗi lúc chỉ một việc dùng GPU; việc gửi sau tự xếp hàng.
- Xong việc: tải gói `ket_qua_{label}_<thời gian>.tar.gz` (ô {numbers['pack']} in đường dẫn), rồi **Terminate pod**. Dữ liệu trên volume vẫn còn.
""")
    machine(nb)
    deploy_key(nb)
    fetch_code(nb)
    setup(nb)
    monitor(nb)
    nb.md(f"""
## {numbers['run']}. Chạy {info['name']} trên benchmark

`scripts/pod/run_experiment.sh main --only {method}`, trong việc nền `{label}`: {info['steps']}.

- `CHAY = False`: chỉ in kế hoạch, ước lượng và dữ liệu đã có (không tải, không train). `CHAY = True`: xếp hàng; ước lượng trên RTX 4090, chưa phải số đo: {info['cost']}.
- Đặt `VOLUME_GB` đúng dung lượng volume: trước khi chạy, việc kiểm tra chỗ trống; không đủ thì dừng, lượt đã xong vẫn giữ.
- Luôn chạy đầy đủ (30.000 vòng). Lượt đã xong cùng cấu hình tự bỏ qua.
""")
    nb.code(fr"""
CHAY = False              # True: xếp hàng việc nền (tốn GPU)
VOLUME_GB = 100           # dung lượng thật của network volume (GB)
AUTO_STOP = False         # True = dừng pod khi việc cuối cùng xong (chưa thử trên pod)

import os, subprocess
repo = "@CODE@"
env = dict(os.environ, VOLUME_GB=str(VOLUME_GB), AUTO_STOP="1" if AUTO_STOP else "0")
if not os.path.isdir(repo):
    print(f"Chưa có {{repo}}: chạy ô 3.")
else:
    cmd = ["bash", f"{{repo}}/scripts/pod/run_experiment.sh", "main", "--only", "{method}",
           "--background" if CHAY else "--plan"]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=repo, env=env)
    print(out.stdout[-12000:] + out.stderr[-3000:])
    if not CHAY:
        print("=> Chưa chạy: đây là kế hoạch. Đặt CHAY = True để xếp hàng (tốn GPU).")
""")
    if gsplat:
        nb.md(f"""
## {numbers['scene']}. Cảnh tự chụp (thư mục ảnh hoặc video) → COLMAP → {info['scene_title']}

Ảnh (JPG/JPEG/PNG, ít nhất 10, nên 30–100, cùng một căn phòng, cùng chiều máy, file ảnh gốc của máy để giữ EXIF; cách chụp: `docs/data/capture_protocol.md` mục 8) hoặc video: tạo thư mục `/workspace/data/custom_scenes/<cảnh>/` trong thanh file của Jupyter (New Folder), tải ảnh hoặc video vào đó (Upload), đặt `SOURCE` là thư mục ảnh hoặc file video, `CHAY = True` rồi chạy ô (chạy nền, tốn GPU). Hướng dẫn đủ: ô 8 của `runpod.ipynb`.

- Cảnh đã có COLMAP thì chỉ train (không đọc lại nguồn); `SOURCE = "-"` cũng là chỉ train (khi đó phải đặt `SCENE`). Lượt đã xong ở đây hay ở `runpod.ipynb` tự bỏ qua.
- Một ảnh, dưới 10 ảnh, ảnh khác kích thước, hay tên cảnh đã dùng cho nguồn khác thì ô này báo lỗi ngay, không xếp hàng. COLMAP báo `FAIL` thì không train; lý do và cách chụp lại ở ô {numbers['results']}.
- Kết quả: `/workspace/outputs/custom_scenes/<cảnh>/` (`colmap/` và các lượt train); thư mục ảnh gốc không bị sửa.
""")
        nb.code(scene_code(info["scene_what"], modes=info["modes"]))
    nb.md(f"""
## {numbers['report']}. Bảng của lần chạy benchmark gần nhất (đọc file, không cần GPU)

In báo cáo mới nhất của ô {numbers['run']} (`/workspace/outputs/reports/{label}_<thời gian>.md`): {info['report']}. Việc nền chưa xong thì báo cáo chưa có hoặc là của lần trước.
""")
    nb.code(fr"""
import glob, os, re
# chỉ báo cáo tổng của lần chạy riêng này (không lấy bảng từng benchmark, không lấy chế độ khác có tên bắt đầu như vậy)
reports = sorted(path for path in glob.glob("/workspace/outputs/reports/{label}_*.md")
                 if re.fullmatch(r"{label}_\d{{8}}_\d{{6}}\.md", os.path.basename(path)))
if reports:
    print(reports[-1])
    print(open(reports[-1], encoding="utf-8").read())
else:
    print("Chưa có báo cáo: chạy ô {numbers['run']} với CHAY = True, chờ việc nền xong (ô 5).")
""")
    if gsplat:
        nb.md(f"""
## {numbers['results']}. Kết quả một cảnh tự chụp, xuất PLY

Ô này in: ảnh đầu vào, kết quả COLMAP, gợi ý, độ phủ góc nhìn, các lượt train (PSNR / SSIM / LPIPS, số Gaussian), video quỹ đạo và file PLY. `XUAT_PLY = True`: đưa file PLY của lượt `RUN` vào `/workspace/outputs/checkpoints/custom_scene/<cảnh>/{label}_<cảnh>_s<seed>.ply` (vd. `{label}_phong1_s42.ply`) để tải về và mở bằng trình xem 3D Gaussian Splatting (không cần GPU). Render lại, lọc Gaussian, ảnh so sánh: ô 9 của `runpod.ipynb`.
""")
        nb.code(fr"""
SCENE = "phong1"
{f'RUN = "{info["run"]}"'.ljust(25)}# lượt: <chế_độ>_s<seed>
XUAT_PLY = False         # True: xuất PLY của lượt RUN (không cần GPU)

import os, subprocess
repo = "@CODE@"
out = subprocess.run(["bash", f"{{repo}}/scripts/pod/results.sh", SCENE], capture_output=True, text=True,
                     errors="replace", cwd=repo)
print(out.stdout + out.stderr)
if XUAT_PLY:
    out = subprocess.run(["bash", f"{{repo}}/scripts/run_render.sh", "custom", SCENE, RUN], capture_output=True,
                         text=True, errors="replace", cwd=repo, env=dict(os.environ, RENDER="0"))
    print(out.stdout + out.stderr)
""")
    pack(nb, numbers["pack"], label)
    stop(nb, numbers["stop"])
    rows = COMMON_ROWS + [SPACE_ROW] + DOWNLOAD_ROWS
    if gsplat:
        rows.append("| `no own Inria B0 run yet ... no B0 check` | Chỉ là ghi chú: chưa có B0 để đối chiếu. Chạy `b0_inria.ipynb` "
                    f"(hoặc code tổng), rồi chạy lại ô {numbers['run']} (lượt đã xong không train lại) hoặc `runpod.ipynb` ô 7 "
                    "với `WHAT = \"evaluate\"` để lập lại bảng có B0 |")
        rows += SCENE_ROWS
    else:
        rows.append("| `Kết luận: **FAIL**` ở ô 7, `B0 does NOT match the Inria reference` (log) | B0 lệch số liệu Inria công bố quá ngưỡng (0,5 dB / 0,01 / 0,01): xem log của việc nền (ô 5) và `docs/ENVIRONMENT_AUDIT.md`; chưa dùng `default` làm baseline khi chưa rõ lý do |")
    rows.append(OTHER_ROW)
    troubleshooting(nb, rows, space_msg=SEED_SPACE if gsplat else B0_SPACE, space_where="", **numbers)
    return nb


def notebooks() -> list[Notebook]:
    return [runpod()] + [method_notebook(method) for method in METHODS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="chỉ so với file trong app/notebooks/, không ghi")
    parser.add_argument("--out", type=Path, default=HERE, help="thư mục ghi (mặc định app/notebooks/)")
    args = parser.parse_args(argv)
    stale = []
    for nb in notebooks():
        path = args.out / nb.filename
        text = nb.text()
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(nb.filename)
            continue
        path.write_text(text, encoding="utf-8")
        print(f"{path} ({len(nb.cells)} ô)")
    if stale:
        print("khác bản sinh ra (chạy python3 app/notebooks/build_notebooks.py): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
