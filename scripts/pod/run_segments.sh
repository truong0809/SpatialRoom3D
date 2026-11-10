#!/usr/bin/env bash
# run_segments.sh - Danh gia canh tu quay voi tap test la cac doan lien cua chinh video chinh (khi khong co doan
# quay kiem tra rieng): chia anh (python -m src.training.split segments: cac doan test, bo vai khung moi ben doan),
# train default roi mcmc (va cac bien the trong VARIANTS) voi tung seed tren cung tap chia, roi lap bang so sanh
# (scripts/run_evaluate.sh).
#
# Cach chay (tren pod; canh da co mo hinh COLMAP tu scripts/pod/run_scene.sh hoac scripts/prepare_data.sh custom):
#   bash scripts/pod/job.sh start seg_phong_nhom -- bash scripts/pod/run_segments.sh phong_nhom   # chay nen
#   DRY_RUN=1 bash scripts/pod/run_segments.sh phong_nhom      # chi in cac lenh, khong chay gi
#   SRC_ROOT=/workspace/outputs/product bash scripts/pod/run_segments.sh video10   # phong dung bang scripts/build_room.sh
#   VARIANTS="mcmc:mono:sr3d.mono=true mcmc:kr:sr3d.kr=true mcmc:rbg:random_bkgd=true" \
#     bash scripts/pod/run_segments.sh phong_nhom              # them bien the cua mcmc (ablation tren phong that)
#
# - Doc mo hinh COLMAP cua canh o $SRC_ROOT/<canh>/colmap (khong sua gi o do). Mac dinh SRC_ROOT la results.gsplat
#   cua configs/custom_scene.yaml; SRC_ROOT=$WS/outputs/product dung lai mo hinh COLMAP cua luong san pham.
# - Ket qua de rieng o $SEG_ROOT/<canh>/: split/ (test.txt, exclude.txt, segments.json) va <che_do>[_<tag>]_s<seed>/;
#   khong dung toi cac luot chuan (moi anh thu 8 lam test) cua canh.
# - VARIANTS: <che_do>:<tag>:<khoa>=<gia_tri>[,...] nhu scripts/pipeline/train_gsplat_modes.sh. Moi seed chay cac che
#   do roi cac bien the, cung tap chia, cung so Gaussian voi default cua seed do. sr3d.mono=true can Depth Anything V2
#   Small (bash scripts/setup/setup_depth_anything.sh). Co bien the cua mcmc thi them bang so voi mcmc (_vs_mcmc).
# - Luot da xong voi dung tap chia thi bo qua (src.training.run --rerun-incomplete); doi BLOCKS / FRACTION / GAP
#   thi tap chia khac: luot cu bao loi, khong bi ghi de.
# - Mot che do (MODES) loi thi dung, khong train tiep. Mot bien the loi thi chay tiep cac luot sau, cuoi cung exit 1.
# Bien tuy chon: WS=/workspace  SEEDS="42 43 44"  MODES="default mcmc"  VARIANTS=  BLOCKS=6  FRACTION=0.125  GAP=3
#   SRC_ROOT=<results.gsplat cua configs/custom_scene.yaml>  SEG_ROOT=$WS/outputs/custom_segments  ENV_NAME=gs-gsplat
# Bao cao: $WS/outputs/reports/segments_<canh>_<thoi gian>.{md,json}; co bien the cua mcmc: them ..._vs_mcmc.{md,json}

set -euo pipefail

usage() { sed -n '2,27p' "$0"; exit 2; }
[ "$#" -eq 1 ] || usage
NAME="$1"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid scene name: $NAME"; exit 2; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
export WS
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
if [ -z "${SRC_ROOT:-}" ]; then
  SRC_ROOT="$(registry_value custom_scene results.gsplat)" || { echo "ERROR: cannot read results.gsplat of configs/custom_scene.yaml"; exit 1; }
fi
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
[ "$(realpath -m "$SEG_ROOT")" != "$(realpath -m "$SRC_ROOT")" ] \
  || { echo "ERROR: SEG_ROOT must differ from SRC_ROOT ($SRC_ROOT): segment runs must not mix with the runs there"; exit 2; }

# bien the: <che_do>:<tag>:<khoa>=<gia_tri>[,...], cung quy tac voi scripts/pipeline/train_gsplat_modes.sh
read -r -a VARIANT_ARGS <<< "${VARIANTS:-}"
VARIANT_MODE=() VARIANT_TAG=() VARIANT_SETS=() MCMC_TAGS=()
NEEDS_MONO=0
for entry in "${VARIANT_ARGS[@]}"; do
  IFS=: read -r vmode vtag vsets vrest <<< "$entry"
  if ! [[ "$vmode" =~ ^[A-Za-z0-9_]+$ && "$vtag" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ && -n "$vsets" && -z "$vrest" ]]; then
    echo "ERROR: invalid entry in VARIANTS: $entry (expected <mode>:<tag>:<key>=<value>[,<key>=<value>...])"; exit 2
  fi
  [ "$vmode" != default ] || { echo "ERROR: VARIANTS: $entry is a variant of default, the baseline (not supported)"; exit 2; }
  [[ " ${MODE_LIST[*]} " == *" $vmode "* ]] || { echo "ERROR: VARIANTS: the mode of $entry is not in MODES (${MODE_LIST[*]})"; exit 2; }
  [[ " ${MODE_LIST[*]} " != *" ${vmode}_$vtag "* ]] || { echo "ERROR: VARIANTS: ${vmode}_$vtag is also a mode in MODES"; exit 2; }
  for other in "${!VARIANT_MODE[@]}"; do
    [ "${VARIANT_MODE[$other]}_${VARIANT_TAG[$other]}" != "${vmode}_$vtag" ] || { echo "ERROR: VARIANTS: ${vmode}_$vtag is listed twice"; exit 2; }
  done
  IFS=, read -r -a pairs <<< "$vsets"
  for pair in "${pairs[@]}"; do
    [[ "$pair" =~ ^[A-Za-z0-9_.-]+=[A-Za-z0-9_.+-]+$ ]] || { echo "ERROR: VARIANTS: invalid parameter '$pair' in $entry"; exit 2; }
    [ "${pair,,}" != sr3d.mono=true ] || NEEDS_MONO=1
  done
  VARIANT_MODE+=("$vmode")
  VARIANT_TAG+=("$vtag")
  VARIANT_SETS+=("$vsets")
  [ "$vmode" != mcmc ] || MCMC_TAGS+=("$vtag")
done

SCENE_DIR="$SRC_ROOT/$NAME/colmap"
[ -f "$SCENE_DIR/sparse/0/images.bin" ] \
  || { echo "ERROR: no COLMAP model for $NAME in $SCENE_DIR (prepare it with scripts/pod/run_scene.sh or scripts/prepare_data.sh custom; a room built by scripts/build_room.sh: SRC_ROOT=$WS/outputs/product)"; exit 1; }
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

# bien the sr3d.mono: kiem Depth Anything V2 Small truoc khi train cac luot khac (src.training.run cung kiem tung luot)
DA2_MISSING=""
if [ "$NEEDS_MONO" = 1 ]; then
  DA2_FILE="$( . "$REPO_DIR/configs/environment/gs-gsplat.conf" && printf '%s' "${DA2_WEIGHTS_FILE:-}")"
  DA2_WEIGHTS="$WS/code/depth_anything_v2/$DA2_FILE"
  if [ -z "$DA2_FILE" ] || [ ! -f "$DA2_WEIGHTS" ]; then
    DA2_MISSING="Depth Anything V2 Small not found ($DA2_WEIGHTS): the sr3d.mono variant needs it (bash scripts/setup/setup_depth_anything.sh)"
  fi
fi

variant_sets() {  # variant_sets <chi so bien the>: --set <khoa>=<gia_tri> ... vao mang SETS
  local pair pairs
  SETS=()
  IFS=, read -r -a pairs <<< "${VARIANT_SETS[$1]}"
  for pair in "${pairs[@]}"; do
    SETS+=(--set "$pair")
  done
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "# scene: $SCENE_DIR -> $SEG_ROOT/$NAME"
  [ -z "$DA2_MISSING" ] || echo "# WARN: $DA2_MISSING"
  echo "PYTHONPATH=$REPO_DIR ${SPLIT_CMD[*]}"
  for seed in "${SEED_LIST[@]}"; do
    for mode in "${MODE_LIST[@]}"; do
      echo "OUT_ROOT=$SEG_ROOT SEED=$seed TAG= BASELINE_TAG= bash $REPO_DIR/scripts/pipeline/train_scene.sh $SCENE_DIR $mode ${TRAIN_ARGS[*]}"
    done
    for v in "${!VARIANT_MODE[@]}"; do
      variant_sets "$v"
      echo "OUT_ROOT=$SEG_ROOT SEED=$seed TAG=${VARIANT_TAG[$v]} BASELINE_TAG= bash $REPO_DIR/scripts/pipeline/train_scene.sh $SCENE_DIR ${VARIANT_MODE[$v]} ${TRAIN_ARGS[*]} ${SETS[*]}"
    done
  done
  echo "CUSTOM_ROOT=$SEG_ROOT REPORT=$REPORT${VARIANT_TAG[*]:+ TAGS=\"${VARIANT_TAG[*]}\"} bash $REPO_DIR/scripts/run_evaluate.sh custom $NAME"
  if [ "${#MCMC_TAGS[@]}" -gt 0 ]; then
    echo "CUSTOM_ROOT=$SEG_ROOT REPORT=${REPORT}_vs_mcmc BASELINE=mcmc MODES=mcmc TAGS=\"${MCMC_TAGS[*]}\" bash $REPO_DIR/scripts/run_evaluate.sh custom $NAME"
  fi
  exit 0
fi

[ -n "$PY" ] || { echo "ERROR: no Python with numpy (install the environment first: bash scripts/pod/run_all.sh)"; exit 1; }
[ -z "$DA2_MISSING" ] || { echo "ERROR: $DA2_MISSING"; exit 1; }
echo "scene: $SCENE_DIR -> $SEG_ROOT/$NAME | seeds: ${SEED_LIST[*]} | modes: ${MODE_LIST[*]}${VARIANT_ARGS[*]:+ | variants: ${VARIANT_ARGS[*]}}"
echo "===== 1. Split: $NAME"
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "${SPLIT_CMD[@]}"

FAILED=()
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
  for v in "${!VARIANT_MODE[@]}"; do
    variant_sets "$v"
    echo
    echo "===== $step. Train ${VARIANT_MODE[$v]}, variant ${VARIANT_TAG[$v]} (${VARIANT_SETS[$v]}), seed $seed"
    # BASELINE_TAG= : ngan sach Gaussian lay tu luot default (khong tag) cua cung seed tren cung tap chia
    if ! OUT_ROOT="$SEG_ROOT" SEED="$seed" TAG="${VARIANT_TAG[$v]}" BASELINE_TAG="" \
        bash "$REPO_DIR/scripts/pipeline/train_scene.sh" "$SCENE_DIR" "${VARIANT_MODE[$v]}" "${TRAIN_ARGS[@]}" "${SETS[@]}"; then
      echo "WARN: variant ${VARIANT_TAG[$v]} seed $seed failed (see the log above); continuing with the next runs"
      FAILED+=("${VARIANT_MODE[$v]}_${VARIANT_TAG[$v]}_s$seed")
    fi
    step=$((step + 1))
  done
done

echo
echo "===== $step. Compare"
COMPARE_FAILED=0
CUSTOM_ROOT="$SEG_ROOT" REPORT="$REPORT" TAGS="${VARIANT_TAG[*]-}" \
  bash "$REPO_DIR/scripts/run_evaluate.sh" custom "$NAME" || COMPARE_FAILED=1
if [ "${#MCMC_TAGS[@]}" -gt 0 ]; then
  echo
  echo "===== $((step + 1)). Compare the variants with mcmc"
  CUSTOM_ROOT="$SEG_ROOT" REPORT="${REPORT}_vs_mcmc" BASELINE=mcmc MODES=mcmc TAGS="${MCMC_TAGS[*]}" \
    bash "$REPO_DIR/scripts/run_evaluate.sh" custom "$NAME" || COMPARE_FAILED=1
fi
[ "${#FAILED[@]}" -eq 0 ] || { echo "ERROR: failed variant runs: ${FAILED[*]} (the reports above list them as incomplete, without numbers)"; exit 1; }
[ "$COMPARE_FAILED" = 0 ] || { echo "ERROR: building a comparison table failed (see above)"; exit 1; }
echo "DONE. Split: $SPLIT_DIR/segments.json | report: $REPORT.md${MCMC_TAGS[*]:+ | vs mcmc: ${REPORT}_vs_mcmc.md}"
