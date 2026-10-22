"""Giao diện dòng lệnh của SpatialRoom3D: một lệnh cho mọi bước, gọi đúng script trong scripts/ (không có logic riêng).

Luồng chính: Ảnh/Video -> kiểm tra đầu vào -> frame (video) -> COLMAP -> chuẩn bị dữ liệu -> train 3DGS -> (lọc
Gaussian, tùy chọn) -> render, xuất PLY -> đánh giá (PSNR / SSIM / LPIPS).

    python -m app.cli prepare custom /workspace/data/custom_scenes/phong1 phong1   # = bash scripts/prepare_data.sh ...
    python -m app.cli train custom phong1                                          # = bash scripts/run_train.sh ...
    python -m app.cli render custom phong1 default_s42                             # = bash scripts/run_render.sh ...
    python -m app.cli filter custom phong1 mcmc_s42                                # = bash scripts/run_filter.sh ...
    python -m app.cli evaluate custom phong1                                       # = bash scripts/run_evaluate.sh ...
    python -m app.cli scene /workspace/data/custom_scenes/phong1 phong1 --background   # cả luồng, chạy nền
    python -m app.cli --dry-run train deep_blending gsplat                         # chỉ in lệnh bash sẽ chạy

Danh sách lệnh: `python -m app.cli --help`. Biến môi trường (WS, SEEDS, QUICK, ...) được truyền nguyên cho script.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]

# lệnh -> (script, tham số cố định đặt trước, mô tả)
COMMANDS: dict[str, tuple[str, list[str], str]] = {
    "check": ("scripts/pod/check_datasets.sh", [], "kiểm tra dữ liệu và kết quả của mọi luồng (chỉ đọc)"),
    "prepare": ("scripts/prepare_data.sh", [], "chuẩn bị dữ liệu: custom <ảnh|video> <cảnh> | <benchmark>"),
    "train": ("scripts/run_train.sh", [], "huấn luyện: custom <cảnh> [chế độ ...] | <benchmark> [gsplat|b0]"),
    "render": ("scripts/run_render.sh", [], "render ảnh / video, xuất PLY: custom <cảnh> [lượt] | <benchmark> <cảnh> [lượt]"),
    "filter": ("scripts/run_filter.sh", [],
               "lọc Gaussian rồi chấm lại: custom <cảnh> [lượt ...] | <benchmark> <cảnh> [lượt ...]"),
    "evaluate": ("scripts/run_evaluate.sh", [], "đánh giá, so sánh các chế độ và B0: <benchmark> | custom [cảnh ...]"),
    "scene": ("scripts/pod/run_scene.sh", [], "cả luồng cho một cảnh tự chụp: <ảnh|video|-> <cảnh> [--background]"),
    "fetch": ("scripts/pod/fetch_data.sh", [], "tải dữ liệu benchmark về volume: <benchmark> [cảnh ...] [--dry-run]"),
    "experiment": ("scripts/pod/run_experiment.sh", [],
                   "một thực nghiệm trọn vẹn (configs/experiments/): <tên> --plan | --background"),
    "setup": ("scripts/pod/run_all.sh", ["--background"], "cài / kiểm tra môi trường trên pod (chạy nền, không train)"),
    "status": ("scripts/pod/job.sh", ["status"], "trạng thái các việc chạy nền"),
    "log": ("scripts/pod/job.sh", ["log"], "cuối log của một việc: log <tên> [số dòng]"),
    "stop": ("scripts/pod/job.sh", ["stop"], "dừng việc: stop <tên> | stop --all"),
    "results": ("scripts/pod/results.sh", [], "xem nhanh mọi kết quả [cảnh ...]"),
    "pack": ("scripts/pod/pack_results.sh", [], "đóng gói kết quả nhẹ để tải về (không có PLY / checkpoint)"),
}


def usage() -> str:
    lines = ["Cách dùng: python -m app.cli [--dry-run] <lệnh> [tham số của script]", "",
             "SpatialRoom3D: tái tạo không gian nội thất 3D từ ảnh / video. Các lệnh:"]
    for name, (script, fixed, help_text) in COMMANDS.items():
        lines.append(f"  {name:<10} {help_text}")
        lines.append(f"  {'':<10}   = bash {script} {' '.join(fixed)}".rstrip())
    lines += ["", "--dry-run: chỉ in lệnh bash sẽ chạy. Hướng dẫn: docs/USAGE_GUIDE.md"]
    return "\n".join(lines)


def command_for(name: str, args: list[str]) -> list[str]:
    script, fixed, _ = COMMANDS[name]
    return ["bash", str(REPO_DIR / script), *fixed, *args]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dry_run = bool(argv) and argv[0] == "--dry-run"
    if dry_run:
        argv = argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    if argv[0] not in COMMANDS:
        print(f"lệnh không có: {argv[0]}\n\n{usage()}", file=sys.stderr)
        return 2
    cmd = command_for(argv[0], argv[1:])  # tham số còn lại đi nguyên tới script (kể cả --background, -- ...)
    if dry_run:
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd).returncode  # giữ thư mục hiện tại: đường dẫn tương đối tính từ chỗ người dùng


if __name__ == "__main__":
    sys.exit(main())
