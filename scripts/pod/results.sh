#!/usr/bin/env bash
# results.sh - Xem nhanh ket qua tren pod (chi doc file, khong sua gi):
#   lan run_all gan nhat, B0 (Inria goc), cac che do gsplat tren Deep Blending, thuc nghiem gan nhat
#   (scripts/pod/run_experiment.sh), cac canh tu quay, goi ket qua.
#
# Cach chay (tu thu muc goc cua repo tren pod):
#   bash scripts/pod/results.sh              tat ca
#   bash scripts/pod/results.sh video8 ...   chi cac canh nay
# Bien tuy chon: WS=/workspace  OUT_ROOT=... (mac dinh theo configs/custom_scene.yaml, giong train_scene.sh / run_scene.sh)

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"

SCENE_ARGS=()
[ -z "${OUT_ROOT:-}" ] || SCENE_ARGS+=(--out-root "$OUT_ROOT")
for scene in "$@"; do
  [[ "$scene" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid scene name: $scene"; exit 2; }
  SCENE_ARGS+=(--scene "$scene")
done

if bash "$REPO_DIR/scripts/pod/job.sh" busy; then
  echo "Đang có việc chạy hoặc xếp hàng: kết quả dưới đây có thể chưa đủ (xem: bash scripts/pod/job.sh status)"
  echo
fi

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 2  # overview doc configs/custom_scene.yaml (PyYAML)
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.evaluation.overview --ws "$WS" "${SCENE_ARGS[@]}"
