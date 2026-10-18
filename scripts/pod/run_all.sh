#!/usr/bin/env bash
# run_all.sh - Luot chay day du tren pod Runpod: kiem tra moi truong -> cai dat -> chay thu nhanh -> B0 (Inria)
#              -> gsplat default/mcmc day du tren Deep Blending -> tom tat -> dong goi ket qua.
#              --setup-only: chi cai dat / kiem tra moi truong (env da cai thi chi kiem tra lai), khong chay benchmark
#              nao (vd. pod moi: cai truoc, chay baseline sau khi duyet chi phi GPU).
#
# Cach chay (tu thu muc goc cua repo tren pod):
#   bash scripts/pod/run_all.sh --background                chay nen qua scripts/pod/job.sh (ten viec: run_all)
#   bash scripts/pod/run_all.sh --background --setup-only   chi cai dat / kiem tra moi truong
#   bash scripts/pod/run_all.sh                             chay truc tiep (dong terminal / dung o Jupyter la dung luot)
# Theo doi: bash scripts/pod/job.sh status | bash scripts/pod/job.sh log run_all | ket qua: bash scripts/pod/results.sh
#
# Cac buoc (chay lai an toan: phan da xong tu bo qua; buoc loi chi lam bo qua cac buoc can no):
#   env_probe     GPU, driver, mang; co [FAIL] thi dung ca luot      scripts/setup/env_probe.sh
#   setup_inria   env gs-inria + 3DGS goc cua Inria                   scripts/setup/setup_3dgs_inria.sh
#   data_tools    env gs-tools: COLMAP + ffmpeg (can Miniforge)        scripts/setup/setup_data_tools.sh
#   setup_gsplat  env gs-gsplat: gsplat (default / mcmc)               scripts/setup/setup_gsplat.sh
#   quick_test    gsplat 3000 vong tren Deep Blending, chi kiem tra luong chay      (can setup_gsplat)
#                 QUICK=1 scripts/run_train.sh deep_blending gsplat
#   b0_inria      B0: 3DGS goc tren Deep Blending, doi chieu so lieu Inria          (can setup_inria)
#                 scripts/run_train.sh deep_blending b0
#   modes_full    gsplat default + mcmc 30000 vong, doi chieu Inria       (can setup_gsplat, quick_test)
#                 scripts/run_train.sh deep_blending gsplat
#   Luot da xong (B0, gsplat cung cau hinh) tu bo qua: chay lai khong train lai.
#   pack          goi ket qua nhe de tai ve (scripts/pod/pack_results.sh), luon chay
#
# Bien tuy chon (dat truoc lenh, vd. MAX_JOBS=4 bash scripts/pod/run_all.sh --background):
#   TORCH_CUDA_OVERRIDE=cu126       driver cua pod chi ho tro CUDA 12.6
#   MAX_JOBS=4                      build gsplat bi "Killed" (thieu RAM)
#   COLMAP_BUILD_OVERRIDE=cpu|cuda  mac dinh tu chon theo driver
#   SEEDS="42 43 44"                cac seed cho modes_full (quick_test chi chay seed dau tien)
#   SKIP="quick_test modes_full"    bo qua cac buoc nay (buoc can chung van chay)
#   AUTO_STOP=1                     voi --background: dung pod khi viec cuoi cung xong (ke ca video xep hang
#                                   sau do; xem scripts/pod/job.sh; chua thu tren pod). AUTO_STOP=0: huy yeu cau

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"

BACKGROUND=0
SETUP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --background) BACKGROUND=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
    *) sed -n '2,35p' "$0"; exit 2 ;;
  esac
done
if [ "$BACKGROUND" = 1 ]; then
  ARGS=()
  [ "$SETUP_ONLY" = 1 ] && ARGS=(--setup-only)
  exec bash "$REPO_DIR/scripts/pod/job.sh" start run_all -- bash "$REPO_DIR/scripts/pod/run_all.sh" "${ARGS[@]}"
fi
if [ "$SETUP_ONLY" = 1 ]; then
  SKIP="${SKIP:-} quick_test b0_inria modes_full"  # chi cai dat / kiem tra moi truong
fi

cd "$REPO_DIR" || exit 1
export ALLOW_PROVISIONAL=1  # phien ban trong configs/environment/*.conf: da chay duoc tren pod nhung chua chot
export MPLBACKEND=Agg       # Jupyter dat backend inline ma cac env conda khong co
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_all"

STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="$WS/outputs/reports"
mkdir -p "$REPORT_DIR" "$WS/outputs/logs"
SUMMARY_TSV="$REPORT_DIR/run_all_$STAMP.tsv"
PKG="$WS/ket_qua_$(date +%m%d_%H%M%S).tar.gz"
if [ -z "${SR3D_JOB_NAME:-}" ]; then  # chay qua job.sh thi log da nam trong $WS/outputs/logs/jobs/run_all.log
  LOG="$WS/outputs/logs/run_all_$STAMP.log"
  exec > >(tee -a "$LOG") 2>&1
  echo "Log: $LOG"
fi

declare -A RC=()    # ma thoat cua tung buoc
declare -A NOTE=()  # ly do bo qua
declare -A SECS=()
STEPS=()

hms() { printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
banner() { printf '\n########## %s %s\n' "$(date '+%F %T')" "$*"; }
user_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
# buoc "dat": chay thanh cong, hoac nguoi dung bo qua bang SKIP (coi nhu da co san)
ok() { [ "${RC[$1]:-}" = 0 ] || [ "${NOTE[$1]:-}" = "SKIP" ]; }

run_step() {  # run_step <ten> <lenh...>
  local name="$1" t0
  shift
  STEPS+=("$name")
  if user_skipped "$name"; then
    NOTE[$name]="SKIP"
    banner "BỎ QUA: $name (SKIP)"
    return 0
  fi
  banner "BẮT ĐẦU: $name"
  t0="$(date +%s)"
  "$@" < /dev/null
  RC[$name]=$?
  SECS[$name]=$(($(date +%s) - t0))
  banner "XONG: $name (exit ${RC[$name]}, $(hms "${SECS[$name]}"))"
}

skip_step() {  # skip_step <ten> <cac buoc can>
  STEPS+=("$1")
  NOTE[$1]="cần $2 thành công"
  banner "BỎ QUA: $1 (cần $2 thành công)"
}

when() {  # when "<cac buoc can>" <ten> <lenh...>: chay neu moi buoc can deu dat
  local needs="$1" dep bad=""
  shift
  for dep in $needs; do
    ok "$dep" || bad="$bad $dep"
  done
  if [ -n "$bad" ]; then
    skip_step "$1" "${bad# }"
  else
    run_step "$@"
  fi
}

hint() {  # goi y khi mot buoc loi
  case "$1" in
    env_probe) echo "xem phần SUMMARY của env_probe ở đầu log" ;;
    setup_inria|setup_gsplat|data_tools) echo "xem dòng ERROR trong log của bước; bảng 'Khi gặp lỗi' trong notebook" ;;
    quick_test) echo "lỗi khi train thử gsplat: xem $WS/outputs/logs/modes_db_quick_*.log" ;;
    b0_inria) echo "exit 1 = B0 lệch số liệu Inria quá ngưỡng hoặc thiếu cảnh: xem reports/baseline_db_summary_*.json" ;;
    modes_full)
      if ok b0_inria; then
        echo "exit 1 = có lượt lỗi, hoặc gsplat default lệch Inria quá ngưỡng: xem reports/modes_db_*.md"
      else
        echo "exit 1 = thiếu B0 để đối chiếu (b0_inria không thành công) hoặc có lượt lỗi: xem reports/modes_db_*.md"
      fi ;;
    *) echo "xem log" ;;
  esac
}

finish() {
  local name status failed=0 repo_commit
  repo_commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  {
    printf '# run_all %s | repo %s%s\n' "$STAMP" "$repo_commit" \
      "$([ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null | head -c 1)" ] && echo ' (có thay đổi chưa commit)')"
    printf '# step\tstatus\texit\tseconds\tnote\n'
    for name in "${STEPS[@]}"; do
      if [ -n "${RC[$name]:-}" ]; then
        if [ "${RC[$name]}" = 0 ]; then status=OK; else status=FAIL; fi
      else
        status=SKIP
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "${RC[$name]:-}" "${SECS[$name]:-}" "${NOTE[$name]:-}"
    done
  } > "$SUMMARY_TSV"

  run_step pack bash "$REPO_DIR/scripts/pod/pack_results.sh" "$PKG"

  printf '\n========== TÓM TẮT run_all (%s)\n' "$STAMP"
  for name in "${STEPS[@]}"; do
    if [ -n "${RC[$name]:-}" ]; then
      if [ "${RC[$name]}" = 0 ]; then
        printf '%-13s OK        %s\n' "$name" "$(hms "${SECS[$name]}")"
      else
        failed=1
        printf '%-13s LỖI (exit %s)  %s - %s\n' "$name" "${RC[$name]}" "$(hms "${SECS[$name]}")" "$(hint "$name")"
      fi
    elif [ "${NOTE[$name]:-}" = SKIP ]; then
      printf '%-13s BỎ QUA (SKIP)\n' "$name"
    else
      failed=1
      printf '%-13s BỎ QUA (%s)\n' "$name" "${NOTE[$name]:-}"
    fi
  done
  echo "Tóm tắt: $SUMMARY_TSV"
  [ "${RC[pack]:-1}" = 0 ] && echo "Gói kết quả: $PKG"
  echo "Xem kết quả: bash scripts/pod/results.sh"
  exit "$failed"
}

read -r QUICK_SEED _ <<< "${SEEDS:-42}"

run_step env_probe bash scripts/setup/env_probe.sh
if ! ok env_probe; then
  echo "env_probe có kiểm tra [FAIL]: dừng cả lượt, không cài đặt gì."
  for name in setup_inria data_tools setup_gsplat quick_test b0_inria modes_full; do
    skip_step "$name" env_probe
  done
  finish
fi
run_step setup_inria bash scripts/setup/setup_3dgs_inria.sh
run_step data_tools bash scripts/setup/setup_data_tools.sh
run_step setup_gsplat bash scripts/setup/setup_gsplat.sh
when "setup_gsplat" quick_test env QUICK=1 SEEDS="$QUICK_SEED" bash scripts/run_train.sh deep_blending gsplat
when "setup_inria" b0_inria bash scripts/run_train.sh deep_blending b0
when "setup_gsplat quick_test" modes_full bash scripts/run_train.sh deep_blending gsplat
finish
