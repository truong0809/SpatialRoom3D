#!/usr/bin/env bash
# prepare_scene.sh - Canh tu chup: video hoac thu muc anh -> anh dau vao -> COLMAP -> bao cao chat luong.
#
# Cach chay (tren pod, sau scripts/setup/setup_3dgs_inria.sh va setup_data_tools.sh):
#   bash scripts/prepare_data.sh custom <video.mp4 | thu_muc_anh> <ten_canh> [tham so them cho frames.py]
# Vi du:
#   bash scripts/prepare_data.sh custom /workspace/data/custom_scenes/phong1 phong1        # thu muc anh JPG/JPEG/PNG
#   bash scripts/prepare_data.sh custom /workspace/data/custom_scenes/phong2/phong2.mp4 phong2 --target 300
# Thuong goi qua scripts/pod/run_scene.sh (chuan bi + huan luyen, chay nen, xep hang GPU).
# Ket qua: $OUT_ROOT/<ten_canh>/colmap/ (input/, images/, sparse/0/, frames_report.json, sfm_report.json);
# nguon (anh / video goc) khong bi sua.
#
# Video: trich va chon frame net. Thu muc anh: moi anh dung duoc deu vao input/ (khong chon frame);
# mot anh, hoac it hon --min-images anh (mac dinh 10), bi tu choi truoc khi tao gi.
# Mac dinh: anh co canh dai <= 1600 px (cung anh ma 3DGS goc cua Inria dung); COLMAP tu chon
# GPU/CPU theo ban da cai (ban CPU khi driver < CUDA 12.9).
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-inria  TOOLS_ENV=gs-tools
#   OUT_ROOT=...   mac dinh results.gsplat trong configs/custom_scene.yaml ($WS/outputs/custom_scenes)
#   COLMAP_ARGS="--matcher sequential --loop-detection"   tham so them cho src.reconstruction.colmap_runner
#   OVERWRITE=1    xoa moi ket qua cu cua canh (frame, COLMAP, bao cao) roi lam lai

set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,21p' "$0"
  exit 1
fi
SRC="$1"
NAME="$2"
shift 2

die() { echo "ERROR: $*"; exit 1; }
# Kiem tra truoc khi tao thu muc hay log nao
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "scene name must start with a letter or digit and contain only letters, digits, '.', '_' and '-'"
[ -e "$SRC" ] || die "source not found: $SRC"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
# Loai nguon (video / thu muc anh) va kiem tra nhanh truoc khi tao gi; chi can thu vien chuan cua Python
PY3="$(command -v python3 || echo "$WS/miniforge3/envs/gs-inria/bin/python")"
KIND="$(PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY3" -m src.input.sources check "$SRC" "$@")" \
  || die "${KIND#ERROR: }"
ENV_NAME="${ENV_NAME:-gs-inria}"
TOOLS_ENV="${TOOLS_ENV:-gs-tools}"
CONDA_DIR="$WS/miniforge3"
TOOLS_BIN="$CONDA_DIR/envs/$TOOLS_ENV/bin"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
if [ -z "${OUT_ROOT:-}" ]; then
  OUT_ROOT="$(registry_value custom_scene results.gsplat)" || die "cannot read the output folder from the registry (see above)"
fi
SCENE="$OUT_ROOT/$NAME/colmap"
LOG_DIR="$WS/outputs/logs"
[ -x "$TOOLS_BIN/colmap" ] && [ -x "$TOOLS_BIN/ffmpeg" ] || die "run scripts/setup/setup_data_tools.sh first"
[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ] || die "Miniforge not found in $CONDA_DIR - run scripts/setup/setup_3dgs_inria.sh first"

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "prepare_scene $NAME"   # COLMAP ban CUDA dung GPU; khong chay chen voi viec khac

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$SCENE"
LOG="$LOG_DIR/prepare_scene_${NAME}_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

# shellcheck source=/dev/null
. "$CONDA_DIR/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM=offscreen  # pod khong co man hinh

OVERWRITE_FLAG=()
if [ "${OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_FLAG=(--overwrite)
fi

if [ "$KIND" = photos ]; then
  echo "===== 1. Photos -> $SCENE/input (no frame extraction) ====="
  python -m src.input.frames --images "$SRC" --scene-dir "$SCENE" "${OVERWRITE_FLAG[@]}" "$@"
else
  echo "===== 1. Frames -> $SCENE/input ====="
  python -m src.input.frames --video "$SRC" --scene-dir "$SCENE" --ffmpeg "$TOOLS_BIN/ffmpeg" \
    "${OVERWRITE_FLAG[@]}" "$@"
fi

echo
echo "===== 2. COLMAP ====="
read -r -a EXTRA_COLMAP_ARGS <<< "${COLMAP_ARGS:-}"
# frames.py da xoa (OVERWRITE=1) hoac da kiem tra khong con ket qua COLMAP cu cua canh
python -m src.reconstruction.colmap_runner --scene "$SCENE" --colmap "$TOOLS_BIN/colmap" "${EXTRA_COLMAP_ARGS[@]}"

echo
echo "DONE. Scene: $SCENE | reports: $SCENE/frames_report.json, $SCENE/sfm_report.json | log: $LOG"
echo "Next: bash scripts/run_train.sh custom $NAME   (default, then mcmc)"
echo "      (or everything at once in the background: bash scripts/pod/run_scene.sh - $NAME --background)"
