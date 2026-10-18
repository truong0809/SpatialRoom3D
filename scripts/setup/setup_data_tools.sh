#!/usr/bin/env bash
# setup_data_tools.sh - Cai COLMAP va ffmpeg vao conda env rieng "gs-tools" trong /workspace.
# Tach rieng khoi env gs-inria de tranh xung dot thu vien.
# Chay lai an toan: env da co se duoc giu nguyen.
#
# Cach chay (tren pod, sau setup_3dgs_inria.sh vi can Miniforge da cai):
#   bash scripts/setup/setup_data_tools.sh
# Bien tuy chon: WS=/workspace  TOOLS_ENV=gs-tools  COLMAP_BUILD_OVERRIDE=cpu|cuda (bo qua viec tu chon)
# Mac dinh dung ban CUDA; neu driver ho tro CUDA thap hon COLMAP_CUDA_MIN (gs-tools.conf) thi tu dung ban CPU.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONF="${CONF:-$REPO_DIR/configs/environment/gs-tools.conf}"
[ -r "$CONF" ] || { echo "ERROR: missing $CONF" >&2; exit 1; }
# shellcheck source=/dev/null
. "$CONF"
COLMAP_BUILD="${COLMAP_BUILD_OVERRIDE:-$COLMAP_BUILD}"

WS="${WS:-/workspace}"
TOOLS_ENV="${TOOLS_ENV:-gs-tools}"
CONDA_DIR="$WS/miniforge3"
LOG_DIR="$WS/outputs/logs"
REPORT_DIR="$WS/outputs/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
LOG="$LOG_DIR/setup_data_tools_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { echo "ERROR: $*"; exit 1; }
warn() { echo "WARN: $*"; }
# ver_ge A B -> true if version A >= version B
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]; }

# Moi luc chi mot viec dung GPU / env: buoc 1 co the xoa va cai lai gs-tools khi mot canh dang chay COLMAP
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"

step "0. Preflight"
gpu_lock_acquire "$(basename "$0")"
[ -x "$CONDA_DIR/bin/conda" ] || die "Miniforge not found in $CONDA_DIR - run scripts/setup/setup_3dgs_inria.sh first"
case "$COLMAP_BUILD" in
  cuda|cpu) ;;
  *) die "COLMAP_BUILD must be 'cuda' or 'cpu' (got '$COLMAP_BUILD')" ;;
esac
DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n 1 || true)"
echo "driver supports CUDA: ${DRIVER_CUDA:-?} | CUDA build of colmap $COLMAP_VERSION needs >= ${COLMAP_CUDA_MIN:-?}"
if [ "$COLMAP_BUILD" = "cuda" ] && [ -z "${COLMAP_BUILD_OVERRIDE:-}" ] && [ -n "${COLMAP_CUDA_MIN:-}" ]; then
  if [ -z "$DRIVER_CUDA" ] || ! ver_ge "$DRIVER_CUDA" "$COLMAP_CUDA_MIN"; then
    warn "driver CUDA ${DRIVER_CUDA:-unknown} < $COLMAP_CUDA_MIN: using the CPU build of colmap (feature extraction and matching on CPU)"
    COLMAP_BUILD="cpu"
  fi
fi
# shellcheck source=/dev/null
. "$CONDA_DIR/etc/profile.d/conda.sh"

ENV_BIN="$CONDA_DIR/envs/$TOOLS_ENV/bin"
# Ban COLMAP dang cai: dong dau cua `colmap -h` ghi "with CUDA" hoac "without GPU support" (COLMAP 4.x;
# 3.x ghi "without CUDA" o dong thu hai). /workspace dung chung cho nhieu pod, nen env cai tren pod co
# driver moi (ban CUDA) co the bi dung lai tren pod driver cu: khi do phai cai lai dung ban.
installed_build() {
  local head
  [ -x "$ENV_BIN/colmap" ] || { echo none; return; }
  head="$(QT_QPA_PLATFORM=offscreen "$ENV_BIN/colmap" -h 2>&1 | head -n 5 || true)"
  head="${head,,}"
  case "$head" in
    *"without cuda"*|*"without gpu"*) echo cpu ;;
    *"with cuda"*) echo cuda ;;
    *) echo unknown ;;
  esac
}

step "1. Conda env $TOOLS_ENV: colmap=$COLMAP_VERSION ($COLMAP_BUILD build) + ffmpeg"
CURRENT_BUILD="$(installed_build)"
if [ -d "$CONDA_DIR/envs/$TOOLS_ENV/conda-meta" ] && [ "$CURRENT_BUILD" = "$COLMAP_BUILD" ] \
   && [ -x "$ENV_BIN/ffmpeg" ] && [ -x "$ENV_BIN/ffprobe" ]; then
  echo "env already exists with the $COLMAP_BUILD build of colmap"
else
  if [ -d "$CONDA_DIR/envs/$TOOLS_ENV" ]; then
    warn "env $TOOLS_ENV has colmap build '$CURRENT_BUILD', need '$COLMAP_BUILD': reinstalling it"
    conda env remove -y -n "$TOOLS_ENV" || rm -rf "${CONDA_DIR:?}/envs/$TOOLS_ENV"
  fi
  conda create -y -n "$TOOLS_ENV" -c conda-forge "colmap=$COLMAP_VERSION=${COLMAP_BUILD}*" ffmpeg \
    || die "conda could not install colmap ($COLMAP_BUILD build). If the driver is too old for the CUDA build, rerun with COLMAP_BUILD_OVERRIDE=cpu"
fi

step "2. Verify"
[ -x "$ENV_BIN/colmap" ] || die "colmap not found in $ENV_BIN"
[ -x "$ENV_BIN/ffmpeg" ] || die "ffmpeg not found in $ENV_BIN"
[ -x "$ENV_BIN/ffprobe" ] || die "ffprobe not found in $ENV_BIN"
COLMAP_HEADER="$(QT_QPA_PLATFORM=offscreen "$ENV_BIN/colmap" -h 2>&1 | grep -m 1 -i 'colmap' || true)"
echo "colmap: $COLMAP_HEADER"
echo "ffmpeg: $("$ENV_BIN/ffmpeg" -version | head -n 1 || true)"
CURRENT_BUILD="$(installed_build)"
[ "$CURRENT_BUILD" = "$COLMAP_BUILD" ] || warn "installed colmap reports build '$CURRENT_BUILD', expected '$COLMAP_BUILD'"
# `|| true`: head dong pipe som thi grep nhan SIGPIPE, voi pipefail + set -e script se dung giua chung
GPU_OPT="$(QT_QPA_PLATFORM=offscreen "$ENV_BIN/colmap" feature_extractor -h 2>&1 | grep -oE '(Feature|Sift)Extraction\.use_gpu' | head -n 1 || true)"
echo "GPU option name: ${GPU_OPT:-?}"

step "3. Record environment"
conda env export -n "$TOOLS_ENV" --no-builds > "$REPORT_DIR/conda_env_${TOOLS_ENV}_$STAMP.yml"
{
  echo "date: $(date -Is)"
  echo "colmap: $COLMAP_HEADER"
  echo "ffmpeg: $("$ENV_BIN/ffmpeg" -version | head -n 1 || true)"
  echo "colmap build: $CURRENT_BUILD"
  echo "conf: $CONF (COLMAP_VERSION=$COLMAP_VERSION, COLMAP_BUILD=$COLMAP_BUILD, COLMAP_CUDA_MIN=${COLMAP_CUDA_MIN:-})"
  echo "driver CUDA: ${DRIVER_CUDA:-?}"
} | tee "$REPORT_DIR/setup_data_tools_$STAMP.txt"

echo
echo "DONE. colmap: $ENV_BIN/colmap | ffmpeg: $ENV_BIN/ffmpeg | log: $LOG"
