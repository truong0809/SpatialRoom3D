#!/usr/bin/env bash
# run_evaluate.sh - Danh gia chat luong (PSNR / SSIM / LPIPS) tu cac luot da chay: bang so sanh default (DefaultStrategy)
# voi mcmc, doi chieu B0 Inria va so lieu cong bo. CHI DOC ket qua, khong train, khong dung GPU.
# Goi src.evaluation.compare_modes voi dung thu muc ket qua cua tung luong (configs/<luong>.yaml); bang kiem tra dieu
# kien cong bang (cung canh / mo hinh COLMAP / tap test / protocol / so Gaussian / so seed) nam trong bao cao.
#
# Cach chay (tu thu muc goc cua repo):
#   bash scripts/run_evaluate.sh deep_blending           # outputs/benchmarks/deep_blending/gsplat, doi chieu Inria + B0
#   bash scripts/run_evaluate.sh mipnerf360              # outputs/benchmarks/mipnerf360/gsplat
#   QUICK=1 bash scripts/run_evaluate.sh deep_blending   # cac luot chay thu 3000 vong (gsplat_quick)
#   bash scripts/run_evaluate.sh custom phong1 phong2    # canh tu chup (outputs/custom_scenes/<canh>)
#   bash scripts/run_evaluate.sh custom                  # moi canh tu chup ("custom_scene", "app" cung duoc)
# Bao cao: $WS/outputs/reports/compare_<ten>_<thoi gian>.{md,json}; cau hinh benchmark da dung: $WS/outputs/reports/configs/.
# Bien tuy chon: WS=/workspace  MODES="default mcmc"

set -euo pipefail

usage() { sed -n '2,14p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
WHAT="$1"
shift
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 1
read -r -a MODE_LIST <<< "${MODES:-default mcmc}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WS/outputs/reports"
ARGS=()

case "$WHAT" in
  custom|custom_scene|app)
    ROOT="$(registry_value custom_scene results.gsplat)"
    if [ "$#" -eq 0 ]; then
      [ -d "$ROOT" ] || { echo "ERROR: no results of your own scenes in $ROOT"; exit 1; }
      ARGS+=(--root "$ROOT")
    else
      for scene in "$@"; do
        [[ "$scene" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid scene name: $scene"; exit 2; }
        [ -d "$ROOT/$scene" ] || { echo "ERROR: no results for scene $scene in $ROOT"; exit 1; }
        ARGS+=(--root "$ROOT/$scene")
      done
    fi
    NAME="custom"
    ;;
  *)
    [ "$#" -eq 0 ] || usage
    [[ "$WHAT" =~ ^[a-z0-9_]+$ ]] || usage
    TAG="$(registry_value "$WHAT" benchmark.tag 2> /dev/null)" \
      || { echo "ERROR: unknown benchmark $WHAT (no benchmark section in configs/$WHAT.yaml)"; exit 2; }
    ROOT="$(registry_value "$WHAT" results.gsplat)"
    NAME="$TAG"
    if [ "${QUICK:-0}" = 1 ]; then
      ROOT="$(registry_value "$WHAT" results.gsplat_quick)"
      NAME="${TAG}_quick"
      ARGS+=(--baseline "default{max_steps=3000}" --include-deviating)
    fi
    [ -d "$ROOT" ] || { echo "ERROR: no results in $ROOT (run bash scripts/run_train.sh $WHAT gsplat first)"; exit 1; }
    CONFIG="$WS/outputs/reports/configs/${WHAT}_$STAMP.json"  # cau hinh benchmark da dung, de doi chieu sau nay
    config_export_benchmark "$WHAT" "$CONFIG" || exit 2
    ARGS+=(--root "$ROOT" --reference-config "$CONFIG")
    B0_ROOT="$(registry_value "$WHAT" results.inria_b0)"
    if [ -d "$B0_ROOT" ]; then
      ARGS+=(--b0-root "$B0_ROOT")
    else
      echo "NOTE: no own Inria B0 run in $B0_ROOT: the per-scene B0 check is skipped"
    fi
    ;;
esac

REPORT="$WS/outputs/reports/compare_${NAME}_$STAMP"
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.evaluation.compare_modes \
  --modes "${MODE_LIST[@]}" "${ARGS[@]}" --out "$REPORT.json" --markdown "$REPORT.md"
echo "Report: $REPORT.md"
