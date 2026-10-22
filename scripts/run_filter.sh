#!/usr/bin/env bash
# run_filter.sh - Loc Gaussian cua cac luot da train roi cham lai ban da loc tren dung anh test (src/training/prune.py).
# Khong train, khong ghi de luot goc: ket qua vao <luot>/filtered/ (checkpoint va PLY da loc, chi so, anh, video,
# filter_meta.json). Bang truoc / sau khi loc: bash scripts/run_evaluate.sh <luong> (doc filtered/ tu dong).
#
#   bash scripts/run_filter.sh custom <ten_canh> [<luot> ...]     vd. bash scripts/run_filter.sh custom phong1 mcmc_s42
#   bash scripts/run_filter.sh <benchmark> <canh> [<luot> ...]    vd. bash scripts/run_filter.sh mipnerf360 room mcmc_s42
#   <luot>: thu muc <che_do>[_<TAG>]_s<seed> trong thu muc ket qua cua canh; mac dinh moi luot da train xong cua canh
#
# Bien tuy chon:
#   MIN_OPACITY=0.02   bo Gaussian co opacity nho hon (0 = tat)
#   MAX_SCALE=0.1      bo Gaussian co truc lon nhat > MAX_SCALE x scene_scale cua gsplat (0 = tat)
#   FORCE=1            loc lai ca khi da loc voi cung nguong (thu muc filtered/ cu doi ten, khong xoa)
#   QUICK=1            benchmark: luot chay thu (gsplat_quick)
#   WS=/workspace  ENV_NAME=gs-gsplat
# Can GPU (cham lai bang gsplat) va env gs-gsplat (bash scripts/pod/run_all.sh). Log: $WS/outputs/logs/filter_*.log

set -euo pipefail

usage() { sed -n '2,16p' "$0"; exit 2; }
die() { echo "ERROR: $*"; exit 1; }
[ "$#" -ge 2 ] || usage
WHAT="$1"
SCENE="$2"
shift 2
for name in "$SCENE" "$@"; do
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid name: $name"
done
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"

case "$WHAT" in
  custom|custom_scene) FLOW="custom_scene"; KEY="gsplat" ;;
  -h|--help) usage ;;
  *)
    [[ "$WHAT" =~ ^[a-z0-9_]+$ ]] || usage
    FLOW="$WHAT"
    KEY="gsplat"
    [ "${QUICK:-0}" != 1 ] || KEY="gsplat_quick"
    ;;
esac
ROOT="$(registry_value "$FLOW" "results.$KEY")" || exit 2
[ -d "$ROOT/$SCENE" ] || die "no results for scene $SCENE in $ROOT"
RUNS=()
if [ "$#" -gt 0 ]; then
  for name in "$@"; do
    [ -f "$ROOT/$SCENE/$name/run_meta.json" ] || die "no run at $ROOT/$SCENE/$name (runs of $SCENE: ls $ROOT/$SCENE)"
    RUNS+=("$ROOT/$SCENE/$name")
  done
else
  for dir in "$ROOT/$SCENE"/*_s[0-9]*/; do
    [ -f "$dir/summary.json" ] || continue
    [[ "$dir" != *.prev_* ]] || continue  # thu muc cu do --force / --rerun-incomplete doi ten
    RUNS+=("${dir%/}")
  done
  [ "${#RUNS[@]}" -gt 0 ] || die "no finished run of $SCENE in $ROOT"
fi
ARGS=(--min-opacity "${MIN_OPACITY:-0.02}" --max-scale "${MAX_SCALE:-0.1}")
[ "${FORCE:-0}" != 1 ] || ARGS+=(--force)
[ "$#" -gt 0 ] || ARGS+=(--skip-unfinished)

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_filter $FLOW $SCENE"
mkdir -p "$WS/outputs/logs"
LOG="$WS/outputs/logs/filter_${SCENE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "runs: ${RUNS[*]##*/} | thresholds: ${ARGS[*]} | log: $LOG"
command -v nvidia-smi > /dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
[ -f "$WS/miniforge3/etc/profile.d/conda.sh" ] || die "Miniforge not found in $WS/miniforge3 - run bash scripts/pod/run_all.sh"
# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME" || die "conda env $ENV_NAME not found - run scripts/setup/setup_gsplat.sh"
set -u
export MPLBACKEND=Agg  # xem setup_gsplat.sh: MPLBACKEND cua Jupyter lam matplotlib bao loi trong env gs-gsplat
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m src.training.prune "${ARGS[@]}" "${RUNS[@]}"
