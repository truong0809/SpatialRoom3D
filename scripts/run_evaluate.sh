#!/usr/bin/env bash
# run_evaluate.sh - Danh gia chat luong (PSNR / SSIM / LPIPS) tu cac luot da chay: bang so sanh default (DefaultStrategy)
# voi cac che do khac (mcmc, che do tu them...), doi chieu B0 Inria va so lieu cong bo. Khong train, khong dung GPU, khong
# sua file nao cua luot: chi ghi them <luot>/splat_stats.json (thong ke PLY cuoi cho bang cau truc mo hinh,
# src.evaluation.splat_stats; luot da co va PLY khong doi thi bo qua).
# Goi src.evaluation.compare_modes voi dung thu muc ket qua cua tung luong (configs/<luong>.yaml); bang kiem tra dieu
# kien cong bang (cung canh / mo hinh COLMAP / tap test / protocol / so Gaussian / so seed) nam trong bao cao.
#
# Cach chay (tu thu muc goc cua repo):
#   bash scripts/run_evaluate.sh deep_blending           # outputs/benchmarks/deep_blending/gsplat, doi chieu Inria + B0
#   bash scripts/run_evaluate.sh mipnerf360              # outputs/benchmarks/mipnerf360/gsplat
#   QUICK=1 bash scripts/run_evaluate.sh deep_blending   # cac luot chay thu 3000 vong (gsplat_quick)
#   bash scripts/run_evaluate.sh custom phong1 phong2    # canh tu chup (outputs/custom_scenes/<canh>)
#   bash scripts/run_evaluate.sh custom                  # moi canh tu chup ("custom_scene", "app" cung duoc)
#   BASELINE=mcmc MODES="mcmc <che do X>" bash scripts/run_evaluate.sh deep_blending   # so che do X voi mcmc
# Bao cao: $WS/outputs/reports/compare_<ten>_<thoi gian>.{md,json}; cau hinh benchmark da dung: $WS/outputs/reports/configs/.
# Bieu do (src.evaluation.charts, can matplotlib): compare_<ten>_<thoi gian>_charts/ (doi chieu Inria, chenh lech tong
# so voi moc; anh PNG va index.md: cach doc, bang so); bao cao .md co them muc "Bieu do" dan toi cac anh do.
# Bien tuy chon: WS=/workspace
#   MODES="default mcmc"   chi cac che do nay (mac dinh: moi che do co luot chay)
#   SCENES="room counter"  benchmark: chi cac canh nay; mot phan cac canh thi doi chieu so cong bo ghi SKIPPED
#   BASELINE=default       che do lam moc so sanh; khac default thi khong doi chieu voi Inria (chi danh cho default)
#   TAGS="or0001 rbg"      dua ca cac luot co tag nay vao bang (bien the cua mot ke hoach thuc nghiem, vd. tune_mcmc);
#                          mac dinh luot co tag bi de ngoai, bang chi co cac luot chuan
#   REPORT=<duong dan>     ten file bao cao, khong co duoi (mac dinh nhu tren)
#   SPLAT_STATS=0          khong tinh thong ke PLY (bang cau truc mo hinh chi co cac luot da tinh truoc do);
#   SPLAT_STATS=force      tinh lai ca khi da co (lan dau doc moi PLY cuoi: vai phut voi thuc nghiem chinh)
#   ENV_NAME=gs-gsplat     env co numpy de tinh thong ke PLY (khong co thi python3 cua may neu co numpy)
#   CHARTS=0               khong ve bieu do (mac dinh ve khi co Python co matplotlib: env gs-gsplat, hoac python3)

set -euo pipefail

usage() { sed -n '2,29p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
WHAT="$1"
shift
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 1
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WS/outputs/reports"
BASELINE="${BASELINE:-default}"
[[ "$BASELINE" =~ ^[A-Za-z0-9_]+$ ]] || { echo "ERROR: BASELINE must be a mode name, e.g. BASELINE=mcmc"; exit 2; }
if [ "$BASELINE" != default ] && [ "${QUICK:-0}" = 1 ]; then
  echo "ERROR: BASELINE is for the full runs only (QUICK=1 always compares against the quick default runs)"
  exit 2
fi
ARGS=()
ROOTS=()
STATS_ROOTS=()  # thong ke PLY: cung thu muc voi bang, hoac chi cac canh trong SCENES
if [ -n "${MODES:-}" ]; then
  read -r -a MODE_LIST <<< "$MODES"
  ARGS+=(--modes "${MODE_LIST[@]}")
fi
if [ "$BASELINE" != default ]; then
  ARGS+=(--baseline "$BASELINE")
fi
if [ -n "${TAGS:-}" ]; then
  read -r -a TAG_LIST <<< "$TAGS"
  for tag in "${TAG_LIST[@]}"; do
    [[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: invalid tag in TAGS: $tag"; exit 2; }
  done
  ARGS+=(--tags "${TAG_LIST[@]}")
fi

case "$WHAT" in
  custom|custom_scene|app)
    ROOT="$(registry_value custom_scene results.gsplat)"
    if [ "$#" -eq 0 ]; then
      [ -d "$ROOT" ] || { echo "ERROR: no results of your own scenes in $ROOT"; exit 1; }
      ROOTS+=("$ROOT")
      STATS_ROOTS+=("$ROOT")
    else
      for scene in "$@"; do
        [[ "$scene" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid scene name: $scene"; exit 2; }
        [ -d "$ROOT/$scene" ] || { echo "ERROR: no results for scene $scene in $ROOT"; exit 1; }
        ROOTS+=("$ROOT/$scene")
        STATS_ROOTS+=("$ROOT/$scene")
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
    ROOTS+=("$ROOT")
    if [ -n "${SCENES:-}" ]; then
      read -r -a SCENE_LIST <<< "$SCENES"
      ARGS+=(--scenes "${SCENE_LIST[@]}")
      for scene in "${SCENE_LIST[@]}"; do
        [ ! -d "$ROOT/$scene" ] || STATS_ROOTS+=("$ROOT/$scene")
      done
    else
      STATS_ROOTS+=("$ROOT")
    fi
    if [ "$BASELINE" = default ]; then
      CONFIG="$WS/outputs/reports/configs/${WHAT}_$STAMP.json"  # cau hinh benchmark da dung, de doi chieu sau nay
      config_export_benchmark "$WHAT" "$CONFIG" || exit 2
      ARGS+=(--reference-config "$CONFIG")
      B0_ROOT="$(registry_value "$WHAT" results.inria_b0)"
      if [ -d "$B0_ROOT" ]; then
        ARGS+=(--b0-root "$B0_ROOT")
      else
        echo "NOTE: no own Inria B0 run in $B0_ROOT: the per-scene B0 check is skipped"
      fi
    else
      echo "NOTE: baseline $BASELINE: no check against Inria (that check is for the default baseline only)"
    fi
    ;;
esac

for root in "${ROOTS[@]}"; do ARGS+=(--root "$root"); done

if [ "${SPLAT_STATS:-1}" != 0 ] && [ "${#STATS_ROOTS[@]}" -gt 0 ]; then
  # thong ke PLY cuoi cua moi luot (can numpy): python cua env gs-gsplat, khong co thi python3 cua may
  STATS_PY=""
  for py in "$WS/miniforge3/envs/${ENV_NAME:-gs-gsplat}/bin/python" "$PY"; do
    if [ -x "$py" ] && "$py" -c 'import numpy' 2> /dev/null; then
      STATS_PY="$py"
      break
    fi
  done
  if [ -z "$STATS_PY" ]; then
    echo "NOTE: no Python with numpy: model-structure statistics skipped (src.evaluation.splat_stats)"
  else
    STATS_ARGS=()
    for root in "${STATS_ROOTS[@]}"; do STATS_ARGS+=(--root "$root"); done
    [ "${SPLAT_STATS:-1}" != force ] || STATS_ARGS+=(--force)
    PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$STATS_PY" -m src.evaluation.splat_stats "${STATS_ARGS[@]}" \
      || echo "WARN: some PLY files could not be read (see above): their model-structure rows are missing"
  fi
fi

REPORT="${REPORT:-$WS/outputs/reports/compare_${NAME}_$STAMP}"
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.evaluation.compare_modes \
  "${ARGS[@]}" --out "$REPORT.json" --markdown "$REPORT.md"
make_charts "$REPORT" "$REPORT.json"
echo "Report: $REPORT.md"
