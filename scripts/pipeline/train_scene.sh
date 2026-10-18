#!/usr/bin/env bash
# train_scene.sh - Huan luyen mot canh bang gsplat theo mot che do trong configs/base.yaml (muc train_modes).
#
# Cach chay (tren pod, sau scripts/setup/setup_gsplat.sh va scripts/prepare_data.sh custom ...):
#   bash scripts/pipeline/train_scene.sh <ten_canh | thu_muc_canh> <che_do> [tham so them cho src.training.run]
#   ten_canh: canh tu chup da chuan bi trong $OUT_ROOT/<ten_canh>/colmap
#   Thuong goi qua: bash scripts/run_train.sh custom <ten_canh> [che_do ...] [-- tham so them]
# Vi du:
#   bash scripts/pipeline/train_scene.sh phong01 default      # baseline
#   bash scripts/pipeline/train_scene.sh phong01 mcmc         # tu lay so Gaussian cuoi cua lan chay default cung canh
#   bash scripts/pipeline/train_scene.sh phong01 mcmc_demo    # tham so cua notebook demo (khong vao so sanh cong bang)
#   TAG=thu bash scripts/pipeline/train_scene.sh phong01 default --protocol max_steps=3000   # chay thu nhanh
#
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-gsplat  SEED=42
#   OUT_ROOT=...     mac dinh results.gsplat trong configs/custom_scene.yaml ($WS/outputs/custom_scenes)
#   TAG=<ten>        bat buoc khi co --protocol / --set, de khong ghi vao thu muc cua luot chuan
#   BASELINE_TAG=    TAG cua luot default dung lam ngan sach cho mcmc (mac dinh = TAG; de trong = luot chuan)
# Ket qua: $OUT_ROOT/<canh>/<che_do>[_<TAG>]_s<seed>/ (summary.json, run_meta.json, train.log, eval.log, ...)

set -euo pipefail

if [ "$#" -lt 2 ]; then
  sed -n '2,19p' "$0"
  exit 1
fi
SCENE_ARG="$1"
MODE="$2"
shift 2
export MPLBACKEND=Agg  # xem setup_gsplat.sh: MPLBACKEND cua Jupyter lam matplotlib bao loi trong env gs-gsplat

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
if [ -z "${OUT_ROOT:-}" ]; then
  OUT_ROOT="$(registry_value custom_scene results.gsplat)" || { echo "ERROR: cannot read the output folder from the registry (see above)"; exit 1; }
fi
SEED="${SEED:-42}"
TAG="${TAG:-}"
BASELINE_TAG="${BASELINE_TAG-$TAG}"
LOG_DIR="$WS/outputs/logs"

die() { echo "ERROR: $*"; exit 1; }

if [[ "$SCENE_ARG" == */* ]] || [ "$SCENE_ARG" = . ] || [ "$SCENE_ARG" = .. ]; then
  [ -d "$SCENE_ARG" ] || die "scene not found: $SCENE_ARG"
  SCENE_DIR="$(cd "$SCENE_ARG" && pwd)"
  # thu muc ket qua cua canh tu chup (<OUT_ROOT>/<ten_canh>): canh da chuan bi nam trong colmap/
  if [ ! -d "$SCENE_DIR/sparse" ] && [ -d "$SCENE_DIR/colmap/sparse" ]; then SCENE_DIR="$SCENE_DIR/colmap"; fi
  NAME="$(basename "$SCENE_DIR")"
  # canh tu chup: <OUT_ROOT>/<ten_canh>/colmap -> ten canh la thu muc cha
  [ "$NAME" != colmap ] || NAME="$(basename "$(dirname "$SCENE_DIR")")"
else
  NAME="$SCENE_ARG"
  SCENE_DIR="$OUT_ROOT/$NAME/colmap"
  [ -d "$SCENE_DIR" ] || die "scene $NAME not found: $SCENE_DIR (prepare it with scripts/prepare_data.sh custom or scripts/pod/run_scene.sh)"
fi
[[ "$MODE" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid mode name: $MODE"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"
[[ "$TAG$BASELINE_TAG" =~ ^[A-Za-z0-9_-]*$ ]] || die "TAG may only contain letters, digits, '_' and '-'"
if [ -z "$TAG" ] && [[ " $* " == *" --protocol"* || " $* " == *" --set"* ]]; then
  die "--protocol / --set change the configuration: set TAG=<name> so the standard run is not overwritten"
fi
OUT="$OUT_ROOT/$NAME/${MODE}${TAG:+_$TAG}_s$SEED"

gpu_lock_acquire "train_scene $NAME $MODE"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_${NAME}_${MODE}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
# chay thu fused-ssim tren GPU: ban 328dc98 chi duoc build cho kien truc GPU luc cai dat
python -c "import torch, fused_ssim; from gsplat import csrc; x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)" 2>/dev/null \
  || die "conda env $ENV_NAME is not ready for this GPU - run scripts/setup/setup_gsplat.sh (it rebuilds when the GPU type changes)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=("$@")
needs_budget="$(python - "$REPO_DIR/configs/base.yaml" "$MODE" <<'PY'
import sys
from src.training.profiles import load_config  # muc train_modes cua configs/base.yaml (PyYAML co trong env gs-gsplat)
mode = load_config(sys.argv[1])["modes"].get(sys.argv[2], {})
print("yes" if (mode.get("cap_max") or {}).get("policy") == "match_baseline" else "no")
PY
)"
if [ "$needs_budget" = "yes" ] && [[ " ${EXTRA[*]-} " != *" --cap-max"* ]]; then
  BASELINE="$OUT_ROOT/$NAME/default${BASELINE_TAG:+_$BASELINE_TAG}_s$SEED"
  [ -f "$BASELINE/summary.json" ] || die "mode $MODE needs a finished default run first: $BASELINE"
  EXTRA+=(--cap-max-from "$BASELINE")
fi

echo "scene: $SCENE_DIR | mode: $MODE | seed: $SEED | out: $OUT"
if [[ " ${EXTRA[*]-} " != *" --test-list"* ]]; then
  echo "NOTE: test images are every 8th frame of the same video (interpolation between near-identical views);"
  echo "      for a held-out trajectory pass --test-list <file> (docs/design/training_modes.md, 4.2)"
fi
python -m src.training.run --mode "$MODE" --scene "$SCENE_DIR" --scene-name "$NAME" --out "$OUT" \
  --seed "$SEED" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete "${EXTRA[@]}"
echo "log: $LOG"
