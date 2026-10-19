#!/usr/bin/env bash
# fetch_data.sh - Tai bo du lieu benchmark ve volume theo muc download cua configs/<benchmark>.yaml
# (src/datasets/fetch.py): chi tai cac canh con thieu, chi giai nen cac thu muc can dung, xoa file zip sau khi giai nen.
# Khong dung GPU; khong sua, khong ghi de du lieu da co. Cac lenh train khong bao gio tu tai.
#
# Cach chay (tren pod, tu thu muc goc cua repo):
#   bash scripts/pod/fetch_data.sh deep_blending                           # drjohnson, playroom (tandt_db.zip cua Inria)
#   bash scripts/pod/fetch_data.sh mipnerf360 room counter kitchen bonsai   # 4 canh trong nha (tu 360_v2.zip)
#   bash scripts/pod/fetch_data.sh mipnerf360 room --dry-run                # chi in se tai gi
# Chay nen (xep hang nhu moi viec khac): bash scripts/pod/job.sh start fetch_mipnerf360 -- bash scripts/pod/fetch_data.sh ...
# Bien tuy chon: WS=/workspace  KEEP_ARCHIVE=1 (giu file zip trong $WS/data/downloads)
# Ma thoat: 0 = moi canh da du; 1 = tai / giai nen loi; 2 = sai tham so hoac cau hinh.

set -uo pipefail

usage() { sed -n '2,12p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
DATASET="$1"
shift
[[ "$DATASET" =~ ^[a-z0-9_]+$ ]] || usage
ARGS=(--dataset "$DATASET")
SCENE_LIST=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) ARGS+=(--dry-run) ;;
    -*) usage ;;
    *)
      [[ "$arg" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid scene name: $arg"; exit 2; }
      SCENE_LIST+=("$arg") ;;
  esac
done
[ "${#SCENE_LIST[@]}" -eq 0 ] || ARGS+=(--scenes "${SCENE_LIST[@]}")
[ "${KEEP_ARCHIVE:-0}" != 1 ] || ARGS+=(--keep-archive)

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 2
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m src.datasets.fetch \
  --configs "$REPO_DIR/configs" --ws "$WS" "${ARGS[@]}"
