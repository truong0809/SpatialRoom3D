#!/usr/bin/env bash
# build_room.sh - SAN PHAM: dung mot phong cho web bang mo hinh chinh (configs/product.yaml: 3DGS-MCMC).
#   video HOAC thu muc anh -> anh dau vao + COLMAP (scripts/pipeline/prepare_scene.sh, dung chung voi thuc nghiem)
#   -> train mo hinh chinh (src/product/train_main.py) -> file cho web (src/product/export_web.py).
#
# Cach chay (tu thu muc goc cua repo tren pod, sau scripts/pod/run_all.sh hoac sau ca ba script setup):
#   bash scripts/build_room.sh <video | thu_muc_anh | -> <ten_phong> [--name "Ten hien thi"] [--background]
#                              [--redo-colmap] [--retrain] [-- tham so frames.py]
# Vi du:
#   bash scripts/build_room.sh /workspace/data/custom_scenes/phong1 phong1 --background        # thu muc anh
#   bash scripts/build_room.sh /workspace/data/custom_scenes/video8/video8.mp4 phong_khach --name "Phòng khách" --background
#   bash scripts/build_room.sh - phong1        # phong da co COLMAP: train (khi can) va xuat lai file web
#
# - Dau vao: mot file video (MP4, MOV...) hoac mot thu muc anh JPG/JPEG/PNG (it nhat 10 anh chup cung mot phong);
#   kiem tra truoc khi xep hang. Nguon khong bi sua. Moi thu sinh ra nam trong output_root cua configs/product.yaml
#   ($WS/outputs/product/<ten_phong>/): colmap/ (anh, COLMAP, bao cao), model/ (mo hinh), progress.json (tien do).
#   File cho web nam trong web.dir ($WS/outputs/product/web/): manifest.json, rooms/<ten_phong>/ (docs/design/product.md).
# - Phong da co mo hinh COLMAP: dung lai (khong doc lai nguon); --redo-colmap: lam lai anh va COLMAP.
#   Mo hinh da train tu dung COLMAP do voi dung cau hinh: khong train lai; --retrain: train lai.
#   Ket qua cu khong bi xoa (thu muc mo hinh cu doi ten model.prev_<thoi_gian>).
# - COLMAP danh gia FAIL (sfm_report.json): dung, khong train; in ly do va goi y.
# - --background: chay nen qua scripts/pod/job.sh (ten viec: room_<ten_phong>); dang co viec khac thi xep hang.
# Bien tuy chon: WS=/workspace  COLMAP_ARGS="..." (scripts/pipeline/prepare_scene.sh)
#   PRODUCT_CONFIG=<file>   mac dinh configs/product.yaml
#   DRY_RUN=1               chi kiem tra dau vao va in cac lenh se chay; khong tao, khong ghi gi
#
# Trinh tu (moi buoc mot ham ben duoi, goi theo thu tu o cuoi file):
#   read_args -> load_config -> check_source -> [--background: queue_job]
#   -> start_run -> step_images_colmap -> check_colmap -> step_train -> step_export -> finish_run
# Loi ve dau vao cua nguoi dung va chu hien tren web (progress.json): tieng Viet; log he thong: tieng Anh nhu ca repo.

set -uo pipefail

usage() { sed -n '2,30p' "$0"; exit 2; }
die() { echo "ERROR: $*"; exit 1; }

# =============================================================================== 0. tham so, cau hinh
read_args() {
  [ "$#" -ge 2 ] || usage
  SRC="$1"
  NAME="$2"
  shift 2
  BACKGROUND=0
  REDO=0
  RETRAIN=0
  DISPLAY_NAME=""
  FRAMES_ARGS=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --background) BACKGROUND=1 ;;
      --redo-colmap) REDO=1 ;;
      --retrain) RETRAIN=1 ;;
      --name)
        [ "$#" -ge 2 ] && [ -n "$2" ] || die "--name needs a value"
        DISPLAY_NAME="$2"
        shift ;;
      --) shift; FRAMES_ARGS=("$@"); break ;;
      *) echo "ERROR: unknown option '$1' (options for frames.py go after --)"; exit 2 ;;
    esac
    shift
  done
  [[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "room name must start with a letter or digit and contain only letters, digits, '.', '_' and '-'"
  case "$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]')" in  # thu muc con cua outputs/product/ (app/web/uploads.py)
    web|uploads|server) die "tên phòng '$NAME' trùng thư mục của hệ thống trong outputs/product/: đặt tên khác" ;;
  esac
}

load_config() {  # duong dan cua phong va cua thu muc web, doc tu configs/product.yaml
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  WS="${WS:-/workspace}"
  DRY_RUN="${DRY_RUN:-0}"
  CONFIG="${PRODUCT_CONFIG:-$REPO_DIR/configs/product.yaml}"
  [ -f "$CONFIG" ] || die "product config not found: $CONFIG"
  CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"  # viec chay nen doi thu muc lam viec
  # shellcheck source=/dev/null
  . "$REPO_DIR/scripts/lib/config.sh"
  CFG_PY="$(config_python)" || die "cannot read $CONFIG without PyYAML (see above)"
  ROOT="$(product_value output_root)" || die "cannot read $CONFIG (see above)"
  WEB_DIR="$(product_value web.dir)" || die "cannot read $CONFIG (see above)"
  TRAIN_ENV="$(product_value trainer.env)" || die "cannot read $CONFIG (see above)"
  ROOM_DIR="$ROOT/$NAME"
  SCENE_DIR="$ROOM_DIR/colmap"
  MODEL_DIR="$ROOM_DIR/model"
  PROGRESS="$ROOM_DIR/progress.json"
  PY="$(command -v python3 || true)"  # python3 cua may: chi can thu vien chuan (ghi lai truoc khi kich hoat env conda)
  [ -n "$PY" ] || PY="$WS/miniforge3/envs/$TRAIN_ENV/bin/python"
}

product_value() {
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$CFG_PY" -m src.product.config --config "$CONFIG" --ws "$WS" \
    --get "$1"
}

# ============================================================================================ tien ich
pyrun() { PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$@"; }
progress() {  # progress <lenh> [tham so]: cap nhat progress.json cho web; loi ghi file khong dung viec dung phong
  [ "$DRY_RUN" = 1 ] && return 0
  pyrun -m src.product.progress "$PROGRESS" "$@" || echo "WARN: $PROGRESS not updated"
}
fail() { progress fail --message "$1"; die "$1"; }  # loi sau khi da bat dau: ghi vao progress.json roi dung
show() { echo "+ $(printf '%q ' "$@")"; }
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
if sys.argv[2] == "source":  # duong dan nguon da dung de chuan bi phong (chi so sanh duoc khi la duong dan tuyet doi)
    value = os.path.realpath(value) if isinstance(value, str) and os.path.isabs(value) else None
if isinstance(value, list):
    print("\n".join(f"  - {item}" for item in value))
elif value is not None:
    print(value)
PY
}
sfm_verdict() { report_field "$SCENE_DIR/sfm_report.json" verdict; }

# ================================================================ 1. kiem tra nguon (truoc khi xep hang)
check_source() {  # khong tao gi; dat SOURCE_KIND (video | photos | rong) va KIND_ARGS cho progress.py
  SOURCE_KIND=""
  if [ "$SRC" = "-" ]; then
    [ "$REDO" = 0 ] || die "--redo-colmap needs the video or the image folder, not '-'"
    [ "${#FRAMES_ARGS[@]}" -eq 0 ] || die "options for frames.py need the video or the image folder, not '-'"
    has_model || die "phòng $NAME chưa có mô hình COLMAP ($SCENE_DIR/sparse/0): cần video hoặc thư mục ảnh thay cho '-'"
    case "$(report_field "$SCENE_DIR/frames_report.json" source_type)" in
      video) SOURCE_KIND=video ;;
      images) SOURCE_KIND=photos ;;
    esac
  else
    [ -e "$SRC" ] || die "source not found: $SRC"
    SRC="$(realpath "$SRC")"  # viec chay nen doi thu muc lam viec: duong dan tuong doi se sai
    # Video hay thu muc anh; mot anh, qua it anh, thu muc khong co anh: bao ngay
    SOURCE_KIND="$(pyrun -m src.input.sources check "$SRC" "${FRAMES_ARGS[@]}")" || die "${SOURCE_KIND#ERROR: }"
    if [ "$REDO" = 0 ] && has_model; then  # moi phong mot ten: ten da dung cho nguon khac thi khong dung lai COLMAP cu
      local old_src
      old_src="$(report_field "$SCENE_DIR/frames_report.json" source)"
      if [ -n "$old_src" ] && [ "$old_src" != "$SRC" ]; then
        die "phòng $NAME đã được dựng từ $old_src, không phải $SRC. Mỗi phòng một tên: đặt tên mới cho nguồn mới (dùng '-' để dựng lại phòng cũ, hoặc --redo-colmap để dựng lại $NAME từ nguồn mới)"
      fi
    fi
  fi
  KIND_ARGS=()
  [ -z "$SOURCE_KIND" ] || KIND_ARGS=(--source-kind "$SOURCE_KIND")
}

# =================================================================================== 2. chay nen
queue_job() {  # xep hang qua job.sh, chay lai chinh script nay (khong co --background) trong viec room_<phong>
  local args=("$SRC" "$NAME") job
  [ -z "$DISPLAY_NAME" ] || args+=(--name "$DISPLAY_NAME")
  [ "$REDO" = 0 ] || args+=(--redo-colmap)
  [ "$RETRAIN" = 0 ] || args+=(--retrain)
  [ "${#FRAMES_ARGS[@]}" -eq 0 ] || args+=(-- "${FRAMES_ARGS[@]}")
  job=(bash "$REPO_DIR/scripts/pod/job.sh" start "room_$NAME" -- bash "$REPO_DIR/scripts/build_room.sh" "${args[@]}")
  if [ "$DRY_RUN" = 1 ]; then
    show "${job[@]}"
    exit 0
  fi
  export PRODUCT_CONFIG="$CONFIG"
  progress queue --room "$NAME" "${KIND_ARGS[@]}" --job "room_$NAME"
  exec "${job[@]}"
}

# ======================================================================= 3. bat dau: khoa GPU, log, tien do
on_stop() {
  trap - TERM INT HUP
  progress fail --stopped --message "Đã dừng"
  echo "build_room $NAME: stopped"
  exit 143
}

start_run() {
  LOG="$WS/outputs/logs/build_room_${NAME}_$(date +%Y%m%d_%H%M%S).log"
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: room $NAME | source: $SRC${SOURCE_KIND:+ ($SOURCE_KIND)} | room dir: $ROOM_DIR | web: $WEB_DIR"
    return 0
  fi
  # shellcheck source=/dev/null
  . "$REPO_DIR/scripts/lib/gpu_lock.sh"
  gpu_lock_acquire "build_room $NAME"  # giu khoa ca luot: prepare_scene.sh ben trong dung chung
  mkdir -p "$(dirname "$LOG")" "$ROOM_DIR" || die "cannot create $ROOM_DIR"
  exec > >(tee -a "$LOG" {SR3D_GPU_LOCK_FD}>&-) 2>&1  # tee khong giu khoa GPU (scripts/lib/gpu_lock.sh)
  local job_args=()
  [ -z "${SR3D_JOB_NAME:-}" ] || job_args=(--job "$SR3D_JOB_NAME")
  progress start --room "$NAME" "${KIND_ARGS[@]}" "${job_args[@]}" --log "$LOG"
  trap on_stop TERM INT HUP
  echo "===== build_room $NAME | source: $SRC${SOURCE_KIND:+ ($SOURCE_KIND)} | config: $CONFIG | $(date '+%F %T')"
}

# ========================================================================== 4. anh dau vao + COLMAP
step_images_colmap() {
  if [ "$REDO" = 0 ] && has_model; then
    echo "Room $NAME already has a COLMAP model: reusing it (the source is not read again; --redo-colmap redoes it)"
    [ "${#FRAMES_ARGS[@]}" -eq 0 ] || echo "WARN: frames.py options ignored because COLMAP is not redone: ${FRAMES_ARGS[*]}"
    progress skip images colmap --message "Dùng lại ảnh và COLMAP đã có"
    return 0
  fi
  [ "$SRC" != "-" ] || fail "Phòng $NAME chưa có mô hình COLMAP ($SCENE_DIR/sparse/0): cần video hoặc thư mục ảnh thay cho '-'"
  local overwrite=0 rc verdict
  if [ "$REDO" = 1 ]; then
    overwrite=1
  elif [ -d "$SCENE_DIR" ] && [ -n "$(ls -A "$SCENE_DIR" 2>/dev/null)" ]; then
    echo "Room $NAME has outputs of an unfinished preparation (no COLMAP model): redoing images and COLMAP"
    overwrite=1
  fi
  local prepare=(env OVERWRITE="$overwrite" OUT_ROOT="$ROOT" bash "$REPO_DIR/scripts/pipeline/prepare_scene.sh" "$SRC"
                 "$NAME" "${FRAMES_ARGS[@]}")
  if [ "$DRY_RUN" = 1 ]; then
    echo "+ $(printf '%q ' "${prepare[@]}")| python3 -m src.product.progress $(printf '%q' "$PROGRESS") follow"
    return 0
  fi
  progress stage images
  # progress.py follow chep nguyen dau ra va doi buoc (anh -> COLMAP) theo dong moc cua prepare_scene.sh
  "${prepare[@]}" | pyrun -m src.product.progress "$PROGRESS" follow
  rc="${PIPESTATUS[0]}"
  if [ "$rc" != 0 ]; then
    verdict="$(sfm_verdict)"
    report_field "$SCENE_DIR/sfm_report.json" hints
    fail "Chuẩn bị ảnh / COLMAP lỗi (mã $rc${verdict:+, COLMAP $verdict}): xem log và $SCENE_DIR/sfm_report.json"
  fi
}

check_colmap() {  # danh gia COLMAP cua sfm_report.json: FAIL thi khong train
  [ "$DRY_RUN" != 1 ] || return 0
  local verdict
  verdict="$(sfm_verdict)"
  case "$verdict" in
    FAIL)
      report_field "$SCENE_DIR/sfm_report.json" hints
      fail "COLMAP đánh giá FAIL: không train. Quay lại video hoặc chụp thêm ảnh (xem $SCENE_DIR/sfm_report.json)" ;;
    "") echo "WARN: $SCENE_DIR/sfm_report.json missing or unreadable: COLMAP quality unknown" ;;
    *) echo "COLMAP verdict: $verdict" ;;
  esac
}

# ===================================================================== 5. train, 6. xuat cho web
activate_env() {  # env conda cua gsplat; train_main.py tu kiem tra GPU khi can train
  local conda_sh="$WS/miniforge3/etc/profile.d/conda.sh"
  [ -f "$conda_sh" ] || fail "Không thấy Miniforge ($conda_sh): chạy scripts/setup/setup_gsplat.sh"
  # shellcheck source=/dev/null
  . "$conda_sh"
  set +u
  conda activate "$TRAIN_ENV" || fail "Không kích hoạt được env conda $TRAIN_ENV: chạy scripts/setup/setup_gsplat.sh"
  set -u
  export MPLBACKEND=Agg  # MPLBACKEND cua Jupyter lam matplotlib bao loi trong env gs-gsplat
  export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
}

step_train() {
  local train=(python -m src.product.train_main --scene "$SCENE_DIR" --out "$MODEL_DIR" --config "$CONFIG" --ws "$WS"
               --progress "$PROGRESS")
  [ "$RETRAIN" = 0 ] || train+=(--force)
  if [ "$DRY_RUN" = 1 ]; then
    echo "+ conda activate $TRAIN_ENV"
    show "${train[@]}"
    return 0
  fi
  progress stage train
  activate_env
  "${train[@]}" || fail "Train mô hình chính lỗi: xem log $LOG"
}

step_export() {
  local export_cmd=(python -m src.product.export_web --room "$NAME" --model "$MODEL_DIR" --scene "$SCENE_DIR"
                    --web-dir "$WEB_DIR" --config "$CONFIG" --ws "$WS" --progress "$PROGRESS")
  [ -z "$DISPLAY_NAME" ] || export_cmd+=("--name=$DISPLAY_NAME")  # dang --name=: ten bat dau bang "-" van dung
  if [ "$DRY_RUN" = 1 ]; then
    show "${export_cmd[@]}"
    return 0
  fi
  progress stage export
  "${export_cmd[@]}" || fail "Xuất file web lỗi: xem log $LOG"
}

finish_run() {
  [ "$DRY_RUN" != 1 ] || return 0
  progress "done" --message "Xong: phòng đã có trong thư viện"
  echo
  echo "DONE. Room $NAME: $WEB_DIR/rooms/$NAME/ (model.splat, cameras.json, thumbnail.jpg) | library: $WEB_DIR/manifest.json"
  echo "      model: $MODEL_DIR | progress: $PROGRESS | log: $LOG"
}

# ============================================================================================ trinh tu
read_args "$@"
load_config
check_source
[ "$BACKGROUND" = 0 ] || queue_job
start_run
step_images_colmap
check_colmap
step_train
step_export
finish_run
