#!/usr/bin/env bash
# prepare_benchmark.sh - Chuan bi du lieu mot benchmark (deep_blending | mipnerf360), KHONG dung GPU:
#   1. tim du lieu o cho configs/<benchmark>.yaml ghi (root); chua co thi tai, kiem tra SHA-256 va giai nen khi cau
#      hinh cho phep (source.auto_download: Deep Blending) hoac khi dat DOWNLOAD=1;
#   2. kiem tra bo cuc tung canh: images/, sparse/0/ va images_<N>/ khi benchmark dung anh thu nho.
# Thuong goi qua: bash scripts/prepare_data.sh <benchmark>
# Bien tuy chon: WS=/workspace  DOWNLOAD=1  SCENES="..." (chi kiem tra cac canh nay)

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
