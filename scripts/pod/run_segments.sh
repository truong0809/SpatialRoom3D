#!/usr/bin/env bash
# run_segments.sh - Danh gia canh tu quay voi tap test la cac doan lien cua chinh video chinh (khi khong co doan
# quay kiem tra rieng): chia anh (python -m src.training.split segments: cac doan test, bo vai khung moi ben doan),
# train default roi mcmc voi tung seed tren cung tap chia, roi lap bang so sanh (scripts/run_evaluate.sh).
#
# Cach chay (tren pod; canh da co mo hinh COLMAP tu scripts/pod/run_scene.sh hoac scripts/prepare_data.sh custom):
#   bash scripts/pod/job.sh start seg_phong_nhom -- bash scripts/pod/run_segments.sh phong_nhom   # chay nen
#   DRY_RUN=1 bash scripts/pod/run_segments.sh phong_nhom      # chi in cac lenh, khong chay gi
#
# - Doc mo hinh COLMAP cua canh o <results.gsplat cua configs/custom_scene.yaml>/<canh>/colmap (khong sua gi o do).
# - Ket qua de rieng o $WS/outputs/custom_segments/<canh>/: split/ (test.txt, exclude.txt, segments.json) va
#   <che_do>_s<seed>/; khong dung toi cac luot chuan (moi anh thu 8 lam test) cua canh.
# - Luot da xong voi dung tap chia thi bo qua (src.training.run --rerun-incomplete); doi BLOCKS / FRACTION / GAP
#   thi tap chia khac: luot cu bao loi, khong bi ghi de.
# - Mot luot loi thi dung, khong train tiep.
# Bien tuy chon: WS=/workspace  SEEDS="42 43 44"  MODES="default mcmc"  BLOCKS=6  FRACTION=0.125  GAP=3
#   SEG_ROOT=$WS/outputs/custom_segments   ENV_NAME=gs-gsplat
# Bao cao: $WS/outputs/reports/segments_<canh>_<thoi gian>.{md,json}

set -euo pipefail

usage() { sed -n '2,20p' "$0"; exit 2; }
[ "$#" -eq 1 ] || usage
NAME="$1"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid scene name: $NAME"; exit 2; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
export WS
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
SRC_ROOT="$(registry_value custom_scene results.gsplat)" || { echo "ERROR: cannot read results.gsplat of configs/custom_scene.yaml"; exit 1; }
SEG_ROOT="${SEG_ROOT:-$WS/outputs/custom_segments}"
SEEDS="${SEEDS:-42 43 44}"
MODES="${MODES:-default mcmc}"
BLOCKS="${BLOCKS:-6}"
FRACTION="${FRACTION:-0.125}"
GAP="${GAP:-3}"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a MODE_LIST <<< "$MODES"
[ "${#SEED_LIST[@]}" -gt 0 ] || { echo "ERROR: SEEDS is empty"; exit 2; }
for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid seed in SEEDS: $seed"; exit 2; }
done
[ "${MODE_LIST[0]:-}" = default ] || { echo "ERROR: MODES must start with default (mcmc takes its Gaussian budget from it)"; exit 2; }
for mode in "${MODE_LIST[@]}"; do
  [[ "$mode" =~ ^[A-Za-z0-9_]+$ ]] || { echo "ERROR: invalid mode in MODES: $mode"; exit 2; }
done
[[ "$BLOCKS" =~ ^[0-9]+$ && "$GAP" =~ ^[0-9]+$ ]] || { echo "ERROR: BLOCKS and GAP must be whole numbers"; exit 2; }
[[ "$FRACTION" =~ ^0?\.[0-9]+$ ]] || { echo "ERROR: FRACTION must be a number between 0 and 0.5, e.g. 0.125"; exit 2; }

SCENE_DIR="$SRC_ROOT/$NAME/colmap"
[ -f "$SCENE_DIR/sparse/0/images.bin" ] \
  || { echo "ERROR: no COLMAP model for $NAME in $SCENE_DIR (prepare it with scripts/pod/run_scene.sh or scripts/prepare_data.sh custom)"; exit 1; }
SPLIT_DIR="$SEG_ROOT/$NAME/split"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$WS/outputs/reports/segments_${NAME}_$STAMP"

# Python co numpy de doc mo hinh COLMAP: python cua env gs-gsplat, khong co thi python3 cua may
PY=""
for py in "$WS/miniforge3/envs/${ENV_NAME:-gs-gsplat}/bin/python" python3; do
  if command -v "$py" > /dev/null 2>&1 && "$py" -c 'import numpy' 2> /dev/null; then
    PY="$py"
    break
  fi
done

SPLIT_CMD=("${PY:-python3}" -m src.training.split segments --scene "$SCENE_DIR" --out "$SPLIT_DIR"
           --blocks "$BLOCKS" --test-fraction "$FRACTION" --gap "$GAP")
TRAIN_ARGS=(--test-list "$SPLIT_DIR/test.txt" --exclude-list "$SPLIT_DIR/exclude.txt")

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "# scene: $SCENE_DIR -> $SEG_ROOT/$NAME"
  echo "PYTHONPATH=$REPO_DIR ${SPLIT_CMD[*]}"
  for seed in "${SEED_LIST[@]}"; do
    for mode in "${MODE_LIST[@]}"; do
      echo "OUT_ROOT=$SEG_ROOT SEED=$seed TAG= bash $REPO_DIR/scripts/pipeline/train_scene.sh $SCENE_DIR $mode ${TRAIN_ARGS[*]}"
    done
  done
  echo "CUSTOM_ROOT=$SEG_ROOT REPORT=$REPORT bash $REPO_DIR/scripts/run_evaluate.sh custom $NAME"
  exit 0
fi

[ -n "$PY" ] || { echo "ERROR: no Python with numpy (install the environment first: bash scripts/pod/run_all.sh)"; exit 1; }
echo "===== 1. Split: $NAME"
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "${SPLIT_CMD[@]}"

step=2
for seed in "${SEED_LIST[@]}"; do
  for mode in "${MODE_LIST[@]}"; do
    echo
    echo "===== $step. Train $mode, seed $seed"
    OUT_ROOT="$SEG_ROOT" SEED="$seed" TAG="" BASELINE_TAG="" \
      bash "$REPO_DIR/scripts/pipeline/train_scene.sh" "$SCENE_DIR" "$mode" "${TRAIN_ARGS[@]}" \
      || { echo "ERROR: $mode seed $seed failed (see the log above); later runs are not started"; exit 1; }
    step=$((step + 1))
  done
done

echo
echo "===== $step. Compare"
CUSTOM_ROOT="$SEG_ROOT" REPORT="$REPORT" bash "$REPO_DIR/scripts/run_evaluate.sh" custom "$NAME"
echo "DONE. Split: $SPLIT_DIR/segments.json | report: $REPORT.md"
