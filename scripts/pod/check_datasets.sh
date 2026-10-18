#!/usr/bin/env bash
# check_datasets.sh - Kiem tra du lieu ghi trong configs/*.yaml: Deep Blending, Mip-NeRF 360, canh tu chup va thu muc
# ket qua. CHI DOC: khong tai, khong sua, khong doi cho gi.
#
# Cach chay (tren pod, hoac sau khi tao pod moi gan cung network volume):
#   bash scripts/pod/check_datasets.sh                          # moi luong
#   bash scripts/pod/check_datasets.sh --dataset deep_blending
#   WS=/duong/dan/khac bash scripts/pod/check_datasets.sh       # volume gan o cho khac
# Ma thoat: 0 = du cac luong bat buoc (deep_blending); 1 = thieu; 2 = loi cau hinh.

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 2
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m src.datasets.registry \
  --configs "$REPO_DIR/configs" --ws "$WS" "$@"
