#!/usr/bin/env bash
# run_scene.sh - Mot canh tu quay tu dau den cuoi: video hoac thu muc anh -> anh dau vao + COLMAP
#                -> train default -> train mcmc -> tom tat ket qua.
#
# Cach chay (tu thu muc goc cua repo tren pod, sau run_all.sh hoac sau ca ba script setup):
#   bash scripts/pod/run_scene.sh <video | thu_muc_anh | -> <ten_canh> [--background] [--redo-colmap] [--retrain]
#                             [-- tham so frames.py]
# Vi du:
#   bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/phong1 phong1 --background   # nhieu anh JPG/JPEG/PNG
#   bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/video8/video8.mp4 video8 --background
#   bash scripts/pod/run_scene.sh - video8 --background            # canh da co COLMAP: chi train
#   bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/video8/video8.mp4 video8 --redo-colmap --background -- --target 300
#
# - Moi job mot ten canh. Nguon goc nam trong data/custom_scenes/<ten_canh>/ va khong bi sua; moi thu sinh ra
#   nam trong outputs/custom_scenes/<ten_canh>/: colmap/ (anh da chuan hoa, COLMAP, bao cao) va <che_do>_s<seed>/.
#   Nguon (video / thu muc anh) duoc kiem tra truoc khi xep hang: mot anh, hoac it hon --min-images anh
#   (mac dinh 10), thi dung ngay va bao ro. Ten canh da dung cho mot nguon khac thi dung (dat ten moi).
# - Canh chua co mo hinh COLMAP (sparse/0): chay scripts/pipeline/prepare_scene.sh; phan con lai cua lan chuan bi do dang
#   (frame, COLMAP loi) duoc lam lai tu dau. Da co mo hinh: dung lai, khong doc lai nguon.
# - --redo-colmap: xoa frame, COLMAP va bao cao cu cua canh roi lam lai.
# - Lan train OK cu duoc train lai khi no lam tu du lieu cu (src.training.run --replace-stale): mo hinh
#   COLMAP da doi, hoac (mcmc) lan default da train lai nen ngan sach Gaussian doi. Kiem tra theo noi dung
#   run_meta.json, nen dung ca khi lan truoc dung giua chung. Thu muc cu doi ten <thu_muc>.prev_<thoi_gian>.
#   Lan OK cu khac cau hinh do nguoi chay chon (vd. muc train_modes cua configs/base.yaml da doi) thi bao loi;
#   --retrain: train lai moi che do (thu muc cu doi ten, khong xoa).
# - sfm_report.json ghi FAIL: dung, khong train; in ly do va goi y (anh khac phong, it chong lap...).
# - --background: chay nen qua scripts/pod/job.sh (ten viec: scene_<ten_canh>); dang co viec khac thi xep hang.
# Bien tuy chon: WS=/workspace  SEED=42  MODES="default mcmc"
#   OUT_ROOT=...   mac dinh results.gsplat trong configs/custom_scene.yaml ($WS/outputs/custom_scenes)
#   COLMAP_ARGS="--loop-detection"   tham so them cho COLMAP (xem scripts/pipeline/prepare_scene.sh)

set -uo pipefail

usage() { sed -n '2,30p' "$0"; exit 2; }
die() { echo "ERROR: $*"; exit 1; }

[ "$#" -ge 2 ] || usage
SRC="$1"
NAME="$2"
shift 2
BACKGROUND=0
REDO=0
RETRAIN=0
FRAMES_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --background) BACKGROUND=1 ;;
    --redo-colmap) REDO=1 ;;
    --retrain) RETRAIN=1 ;;
    --) shift; FRAMES_ARGS=("$@"); break ;;
    *) echo "ERROR: unknown option '$1' (options for frames.py go after --)"; exit 2 ;;
  esac
  shift
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
SEED="${SEED:-42}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
if [ -z "${OUT_ROOT:-}" ]; then
  OUT_ROOT="$(registry_value custom_scene results.gsplat)" || die "cannot read the output folder from the registry (see above)"
fi
read -r -a MODE_LIST <<< "${MODES:-default mcmc}"

# Kiem tra truoc khi xep hang hay tao thu muc nao
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "scene name must start with a letter or digit and contain only letters, digits, '.', '_' and '-'"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"
[ "${#MODE_LIST[@]}" -gt 0 ] || die "MODES is empty"
for mode in "${MODE_LIST[@]}"; do
  [[ "$mode" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid mode name: $mode"
done

SCENE_DIR="$OUT_ROOT/$NAME/colmap"
PY="$(command -v python3 || true)"
[ -n "$PY" ] || PY="$WS/miniforge3/envs/gs-inria/bin/python"

has_model() {  # giong src.reconstruction.colmap_model.model_exists
  local f
  for f in cameras images points3D; do
    [ -f "$SCENE_DIR/sparse/0/$f.bin" ] || return 1
  done
}
report_field() {  # report_field <file json> <khoa>: in gia tri (danh sach: moi phan tu mot dong), loi thi in rong
  "$PY" - "$1" "$2" <<'PY' 2>/dev/null || true
import json
import os
import sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2])
except (OSError, ValueError, AttributeError):
    sys.exit(0)
if sys.argv[2] == "source":  # duong dan nguon da dung de chuan bi canh (chi so sanh duoc khi la duong dan tuyet doi)
    value = os.path.realpath(value) if isinstance(value, str) and os.path.isabs(value) else None
if isinstance(value, list):
    print("\n".join(f"  - {item}" for item in value))
elif value is not None:
    print(value)
PY
}
sfm_verdict() { report_field "$SCENE_DIR/sfm_report.json" verdict; }

if [ "$SRC" = "-" ]; then
  [ "$REDO" = 0 ] || die "--redo-colmap needs the video or the image folder, not '-'"
  [ "${#FRAMES_ARGS[@]}" -eq 0 ] || die "options for frames.py need the video or the image folder, not '-'"
else
  [ -e "$SRC" ] || die "source not found: $SRC"
  SRC="$(realpath "$SRC")"  # viec chay nen doi thu muc lam viec: duong dan tuong doi se sai
  # Nguon co dung duoc khong (mot anh, qua it anh, thu muc khong co anh): bao ngay, truoc khi xep hang
  check="$(PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.input.sources check "$SRC" \
    "${FRAMES_ARGS[@]}")" || die "${check#ERROR: }"
  # Moi job mot ten canh: ten da dung cho nguon khac thi khong dung lai mo hinh COLMAP cua nguon cu
  if [ "$REDO" = 0 ] && has_model; then
    old_src="$(report_field "$SCENE_DIR/frames_report.json" source)"
    if [ -n "$old_src" ] && [ "$old_src" != "$SRC" ]; then
      die "cảnh $NAME đã được dựng từ $old_src, không phải $SRC. Mỗi job một tên cảnh: đặt tên cảnh mới cho nguồn mới (dùng '-' để train lại cảnh cũ, hoặc --redo-colmap để dựng lại $NAME từ nguồn mới)"
    fi
  fi
fi

if [ "$BACKGROUND" = 1 ]; then
  ARGS=("$SRC" "$NAME")
  [ "$REDO" = 1 ] && ARGS+=(--redo-colmap)
  [ "$RETRAIN" = 1 ] && ARGS+=(--retrain)
  [ "${#FRAMES_ARGS[@]}" -gt 0 ] && ARGS+=(-- "${FRAMES_ARGS[@]}")
  exec bash "$REPO_DIR/scripts/pod/job.sh" start "scene_$NAME" -- bash "$REPO_DIR/scripts/pod/run_scene.sh" "${ARGS[@]}"
fi

export MPLBACKEND=Agg
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_scene $NAME"  # giu khoa ca luot: prepare_scene.sh / train_scene.sh ben trong dung chung

echo "===== run_scene $NAME | source: $SRC | seed $SEED | modes: ${MODE_LIST[*]} | $(date '+%F %T')"

# ---------------------------------------------------------------------------------- 1. frame + COLMAP
if [ "$REDO" = 1 ] || ! has_model; then
  [ "$SRC" != "-" ] || die "scene $NAME has no COLMAP model yet ($SCENE_DIR/sparse/0): give the video or the image folder instead of '-'"
  OVERWRITE=0
  if [ "$REDO" = 1 ]; then
    OVERWRITE=1
  elif [ -d "$SCENE_DIR" ] && [ -n "$(ls -A "$SCENE_DIR" 2>/dev/null)" ]; then
    echo "Scene $NAME has outputs of an unfinished preparation (no COLMAP model): redoing frames and COLMAP"
    OVERWRITE=1
  fi
  OVERWRITE="$OVERWRITE" OUT_ROOT="$OUT_ROOT" bash "$REPO_DIR/scripts/pipeline/prepare_scene.sh" "$SRC" "$NAME" "${FRAMES_ARGS[@]}"
  rc=$?
  if [ "$rc" != 0 ]; then
    verdict="$(sfm_verdict)"
    report_field "$SCENE_DIR/sfm_report.json" hints
    die "preparing the scene failed (exit $rc${verdict:+, COLMAP verdict $verdict}): see the log above and $SCENE_DIR/sfm_report.json"
  fi
else
  echo "Scene $NAME already has a COLMAP model: reusing it (the source is not read again; --redo-colmap redoes it)"
  [ "${#FRAMES_ARGS[@]}" -eq 0 ] || echo "WARN: frames.py options ignored because COLMAP is not redone: ${FRAMES_ARGS[*]}"
fi

VERDICT="$(sfm_verdict)"
case "$VERDICT" in
  FAIL)
    report_field "$SCENE_DIR/sfm_report.json" hints
    die "COLMAP verdict FAIL for $NAME: not training. See $SCENE_DIR/sfm_report.json (record a new video or more photos, or --redo-colmap with other options)" ;;
  "") echo "WARN: $SCENE_DIR/sfm_report.json missing or unreadable: COLMAP quality unknown" ;;
  *) echo "COLMAP verdict: $VERDICT" ;;
esac

# ----------------------------------------------------------------------------------------- 2. train
FAILED=0
REPLACE=(--replace-stale)
[ "$RETRAIN" = 1 ] && REPLACE=(--force)
for mode in "${MODE_LIST[@]}"; do
  echo
  echo "===== train $mode"
  # Ten canh tran (khong co "/"): train_scene.sh hieu la canh trong $OUT_ROOT, ke ca khi trung ten mot thu muc trong
  # repo (vd. "docs"); TAG / BASELINE_TAG rong: luot chuan (bien con sot trong Jupyter khong doi thu muc ket qua).
  if ! TAG="" BASELINE_TAG="" SEED="$SEED" OUT_ROOT="$OUT_ROOT" \
       bash "$REPO_DIR/scripts/pipeline/train_scene.sh" "$NAME" "$mode" "${REPLACE[@]}"; then
    FAILED=1
    if [ "$mode" = default ]; then
      echo "default failed: the other modes need it, stopping"
      break
    fi
  fi
done

# ---------------------------------------------------------------------------------------- 3. tom tat
echo
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.evaluation.overview --ws "$WS" \
  --out-root "$OUT_ROOT" --scene "$NAME" || true
if [ "$FAILED" = 1 ]; then
  echo "run_scene $NAME: at least one training run FAILED (see the log above)"
fi
exit "$FAILED"
