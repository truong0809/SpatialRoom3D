#!/usr/bin/env bash
# prepare_benchmark.sh - Kiem tra du lieu mot benchmark (deep_blending | mipnerf360) da dua len, KHONG dung GPU,
# KHONG tai hay giai nen gi (du lieu do nguoi dung tu dua len):
#   1. du lieu phai nam o cho configs/<benchmark>.yaml ghi (root, ghi de duoc bang configs/local.yaml / SR3D_DATA_*);
#   2. kiem tra bo cuc tung canh: images/, sparse/0/ va images_<N>/ khi benchmark dung anh thu nho.
# Thuong goi qua: bash scripts/prepare_data.sh <benchmark>
# Bien tuy chon: WS=/workspace  SCENES="..." (chi kiem tra cac canh nay)

set -euo pipefail

usage() { sed -n '2,7p' "$0"; exit 2; }
[ "$#" -eq 1 ] || usage
BENCH="$1"
[[ "$BENCH" =~ ^[a-z0-9_]+$ ]] || usage
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
REPORT_DIR="$WS/outputs/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
if [ -z "${CONFIG:-}" ]; then  # muc benchmark cua configs/<benchmark>.yaml, chup thanh JSON trong reports/ (bang chung)
  CONFIG="$REPORT_DIR/configs/${BENCH}_$STAMP.json"
  config_export_benchmark "$BENCH" "$CONFIG" || { echo "ERROR: no benchmark $BENCH in configs/ (see above)"; exit 2; }
fi
[ -r "$CONFIG" ] || { echo "ERROR: missing config $CONFIG"; exit 2; }
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/benchmark.sh"
read -r -a SCENES <<< "${SCENES:-$(read_cfg scenes)}"

step "1. Dataset $BENCH"
locate_dataset
for scene in "${SCENES[@]}"; do
  check_scene_layout "$scene"
done
echo "OK: $BENCH is ready in $DATA_ROOT (${#SCENES[@]} scenes: ${SCENES[*]})"
