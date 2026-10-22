#!/usr/bin/env bash
# run_panels.sh - Anh so sanh cac luot tren cung anh test (anh goc, render, vung cat phong to, ban do sai so) va video quy
# dao cac luot xep chong (src/evaluation/panels.py). Khong train, khong dung GPU: chi doc renders/ va videos/ cua luot.
#
#   bash scripts/run_panels.sh <benchmark> <canh> [<luot> ...]   vd. bash scripts/run_panels.sh mipnerf360 counter
#   bash scripts/run_panels.sh custom <ten_canh> [<luot> ...]
#   <luot>: thu muc <che_do>[_<TAG>]_s<seed> trong thu muc ket qua cua canh, hoac <luot>/filtered (ban da loc Gaussian).
#           Mac dinh: moi luot da xong, khong co tag, cua seed SEED, co renders/ (default, mcmc truoc, roi theo ten)
#
# Bien tuy chon:
#   IMAGES="0 12 24"   chi so anh test (mac dinh: anh dau, giua, cuoi)
#   CROP=x,y,w,h       vung cat theo ti le, vd. 0.5,0.4,0.25,0.25 (mac dinh: o 1/4 x 1/4 co tong sai so, cong tren moi
#                      luot, lon nhat: thuong la cho hong nang nhat cua luot kem nhat)
#   VIDEO=1            ghep them video quy dao (videos/traj_<N>.mp4) cua cac luot, moi luot mot hang
#   SEED=42            seed cua cac luot mac dinh
#   QUICK=1            benchmark: cac luot chay thu (gsplat_quick)
#   WS=/workspace  ENV_NAME=gs-gsplat   python co numpy va opencv: cua env nay, khong co thi python3 cua may
# Ket qua: $WS/outputs/panels/<luong>_<canh>_<thoi gian>/ (panel_<i>.png, panels.json, panels.md, traj_compare.mp4);
# khong nam trong goi ket qua nhe (scripts/pod/pack_results.sh chi gom panels.json), tai ve qua thanh file cua Jupyter.

set -euo pipefail

usage() { sed -n '2,19p' "$0"; exit 2; }
die() { echo "ERROR: $*"; exit 1; }
[ "$#" -ge 2 ] || usage
WHAT="$1"
SCENE="$2"
shift 2
[[ "$SCENE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid scene name: $SCENE"
for name in "$@"; do
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(/filtered)?$ ]] || die "invalid run name: $name"
done
SEED="${SEED:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a number: $SEED"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"

KEY="gsplat"
case "$WHAT" in
  custom|custom_scene) FLOW="custom_scene" ;;
  *)
    [[ "$WHAT" =~ ^[a-z0-9_]+$ ]] || usage
    FLOW="$WHAT"
    [ "${QUICK:-0}" != 1 ] || KEY="gsplat_quick"
    ;;
esac
ROOT="$(registry_value "$FLOW" "results.$KEY")" || exit 2
DIR="$ROOT/$SCENE"
[ -d "$DIR" ] || die "no results for $SCENE in $ROOT"

PY=""
for candidate in "$WS/miniforge3/envs/$ENV_NAME/bin/python" "$(command -v python3 || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ] && "$candidate" -c 'import cv2, numpy' 2> /dev/null; then
    PY="$candidate"
    break
  fi
done
[ -n "$PY" ] || die "no Python with numpy and opencv (env $ENV_NAME: bash scripts/pod/run_all.sh)"

NAME="$FLOW"
[ "$KEY" != gsplat_quick ] || NAME="${FLOW}_quick"
OUT="$WS/outputs/panels/${NAME}_${SCENE}_$(date +%Y%m%d_%H%M%S)"
ARGS=(--out "$OUT")
if [ "$#" -gt 0 ]; then
  RUNS=()
  for name in "$@"; do
    if [[ "$name" == */filtered ]]; then
      [ -f "$DIR/$name/filter_meta.json" ] || die "no filtered model at $DIR/$name (bash scripts/run_filter.sh first)"
    else
      [ -f "$DIR/$name/run_meta.json" ] || die "no run $name in $DIR (runs: ls $DIR)"
    fi
    RUNS+=("$DIR/$name")
  done
  ARGS+=(--runs "${RUNS[@]}")
else
  ARGS+=(--scene-dir "$DIR" --seed "$SEED")
fi
if [ -n "${IMAGES:-}" ]; then
  read -r -a IMAGE_LIST <<< "$IMAGES"
  for index in "${IMAGE_LIST[@]}"; do
    [[ "$index" =~ ^[0-9]+$ ]] || die "IMAGES must be test image numbers: $IMAGES"
  done
  ARGS+=(--images "${IMAGE_LIST[@]}")
fi
[ -z "${CROP:-}" ] || ARGS+=(--crop "$CROP")
[ "${VIDEO:-0}" != 1 ] || ARGS+=(--video)

PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.evaluation.panels "${ARGS[@]}"
echo "Panels: $OUT/panels.md"
