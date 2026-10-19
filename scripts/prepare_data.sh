#!/usr/bin/env bash
# prepare_data.sh - Chuan bi du lieu dau vao (luong chinh: Anh/Video -> Validate Input -> Extract Frames -> COLMAP
# -> Prepare Dataset). Moi luong du lieu mot nhanh; duong dan lay tu configs/<luong>.yaml.
#
#   bash scripts/prepare_data.sh custom <thu_muc_anh | file_video> <ten_canh> [tham so cho frames.py]
#       canh tu chup: kiem tra anh / video, chon frame (video), COLMAP, bao cao; ket qua
#       outputs/custom_scenes/<ten_canh>/colmap/ (scripts/pipeline/prepare_scene.sh; can GPU lock, env gs-tools)
#   bash scripts/prepare_data.sh deep_blending | mipnerf360
#       benchmark: kiem tra du lieu ban da dua len (bo cuc tung canh)
#       (scripts/pipeline/prepare_benchmark.sh; khong dung GPU)
#
# Kiem tra du lieu da co (chi doc): bash scripts/pod/check_datasets.sh

set -uo pipefail

usage() { sed -n '2,12p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="$1"
shift
case "$WHAT" in
  custom|custom_scene)
    [ "$#" -ge 2 ] || usage
    exec bash "$REPO_DIR/scripts/pipeline/prepare_scene.sh" "$@"
    ;;
  -h|--help) usage ;;
  *)
    [ "$#" -eq 0 ] || usage
    exec bash "$REPO_DIR/scripts/pipeline/prepare_benchmark.sh" "$WHAT"
    ;;
esac
