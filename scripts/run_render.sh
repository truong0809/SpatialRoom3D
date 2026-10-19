#!/usr/bin/env bash
# run_render.sh - Render anh / video tu mot luot da train va xuat mo hinh 3D (PLY) (buoc "Export PLY/Checkpoint" cua
# luong chinh). Khong train, khong ghi de luot goc (src/rendering/render.py; chi dung che do --ckpt co san cua gsplat).
#
#   bash scripts/run_render.sh custom <ten_canh> [<luot>]     vd. bash scripts/run_render.sh custom phong1 mcmc_s42
#   bash scripts/run_render.sh <benchmark> <canh> [<luot>]    vd. bash scripts/run_render.sh deep_blending playroom
#   <luot>: thu muc <che_do>[_<TAG>]_s<seed> trong thu muc ket qua cua canh, mac dinh default_s42
#
# Bien tuy chon:
#   TRAJ=interp|ellipse|spiral   quy dao cua video (mac dinh interp: noi suy giua cac vi tri camera da chup)
#   EXPORT=1 (mac dinh) | 0      dua PLY cuoi vao $WS/outputs/checkpoints/<luong>/<canh>/<luot>.ply (lien ket cung
#                                neu duoc, khong ton them cho; ten da la PLY khac thi <luot>_<luc train xong>.ply)
#   RENDER=1 (mac dinh) | 0      0: chi xuat PLY (khong can GPU, khong can env)
#   QUICK=1                      benchmark: luot chay thu (gsplat_quick), PLY vao checkpoints/<luong>_quick/
#   WS=/workspace  ENV_NAME=gs-gsplat
# Ket qua: <luot>/render_<quy dao>_<thoi gian>/ (renders/, videos/, stats/, render_meta.json); log: $WS/outputs/logs/

set -euo pipefail

usage() { sed -n '2,16p' "$0"; exit 2; }
die() { echo "ERROR: $*"; exit 1; }
[ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage
WHAT="$1"
SCENE="$2"
RUN_NAME="${3:-default_s42}"
for name in "$SCENE" "$RUN_NAME"; do
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid name: $name"
done
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"

case "$WHAT" in
  custom|custom_scene) FLOW="custom_scene"; KEY="gsplat" ;;
  *)
    [[ "$WHAT" =~ ^[a-z0-9_]+$ ]] || usage
    FLOW="$WHAT"
    KEY="gsplat"
    [ "${QUICK:-0}" != 1 ] || KEY="gsplat_quick"
    ;;
esac
EXPORT_ROOT="$WS/outputs/checkpoints/$FLOW"
[ "$KEY" != gsplat_quick ] || EXPORT_ROOT="${EXPORT_ROOT}_quick"  # luot chay thu: khong lan voi PLY cua luot chuan
ROOT="$(registry_value "$FLOW" "results.$KEY")" || exit 2
RUN="$ROOT/$SCENE/$RUN_NAME"
[ -f "$RUN/run_meta.json" ] || die "no run at $RUN (finished runs of $SCENE: ls $ROOT/$SCENE)"

ARGS=(--run "$RUN" --traj "${TRAJ:-interp}")
[ "${EXPORT:-1}" = 0 ] || ARGS+=(--export "$EXPORT_ROOT")
if [ "${RENDER:-1}" = 0 ]; then
  [ "${EXPORT:-1}" != 0 ] || die "RENDER=0 and EXPORT=0: nothing to do"
  PY="$(command -v python3 || true)"
  [ -n "$PY" ] || die "python3 not found"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m src.rendering.render "${ARGS[@]}" --no-render
fi

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_render $FLOW $SCENE $RUN_NAME"
mkdir -p "$WS/outputs/logs"
LOG="$WS/outputs/logs/render_${SCENE}_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "run: $RUN | trajectory: ${TRAJ:-interp} | log: $LOG"
command -v nvidia-smi > /dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
[ -f "$WS/miniforge3/etc/profile.d/conda.sh" ] || die "Miniforge not found in $WS/miniforge3 - run bash scripts/pod/run_all.sh"
# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME" || die "conda env $ENV_NAME not found - run scripts/setup/setup_gsplat.sh"
set -u
export MPLBACKEND=Agg  # xem setup_gsplat.sh: MPLBACKEND cua Jupyter lam matplotlib bao loi trong env gs-gsplat
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m src.rendering.render "${ARGS[@]}"
