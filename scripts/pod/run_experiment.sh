#!/usr/bin/env bash
# run_experiment.sh - Mot thuc nghiem tron ven theo configs/experiments/<ten>.yaml, trong MOT viec dung GPU:
#   kiem tra moi truong -> kiem tra cho trong -> tai du lieu con thieu (scripts/pod/fetch_data.sh) -> B0 Inria cho
#   benchmark co b0: true -> gsplat cac che do (scripts/run_train.sh <benchmark> gsplat), xen ke benchmark theo tung
#   seed (du mot seed tren moi canh roi moi sang seed sau) -> bao cao so sanh cuoi (scripts/run_evaluate.sh)
#   -> goi ket qua nhe (scripts/pod/pack_results.sh). Moi file ket qua duoc giu (khong xoa gi).
# Luon chay day du (30.000 vong); khong co che do chay thu. Chay lai an toan: du lieu da co khong tai lai, luot gsplat
# da xong va canh B0 da co ket qua tu bo qua.
#
# Cach chay (tren pod, tu thu muc goc cua repo, sau scripts/pod/run_all.sh):
#   VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --background   chay nen (viec exp_main)
#   bash scripts/pod/run_experiment.sh main --plan     chi in ke hoach, uoc luong, du lieu da co (khong chay gi)
#   bash scripts/pod/run_experiment.sh main            chay truc tiep (dong terminal la dung luot)
# Theo doi: bash scripts/pod/job.sh status | bash scripts/pod/job.sh log exp_main
#
# Bien tuy chon:
#   VOLUME_GB=100   dung luong that cua network volume (df tren Runpod khong doc duoc quota). Cho can cho mot seed =
#                   tong cac luot gsplat chua xong cua seed do; moi luot tinh bang dung luong that cua luot cung canh,
#                   cung che do da xong o seed khac, chua co thi theo uoc luong cua ke hoach (luot da xong khong tinh
#                   lai, ke ca khi chay lai sau khi pod dung). Truoc khi chay: seed dau tien con luot chua xong khong
#                   du cho thi dung, phan con lai cua ke hoach khong du cho thi canh bao; truoc moi seed: khong du cho
#                   thi dung train (cac luot da xong giu nguyen; tang volume roi chay lai de lam not). SPACE_CHECK=0:
#                   bo qua ca hai
#   AUTO_STOP=1     voi --background: dung pod khi viec cuoi cung xong (scripts/pod/job.sh; chua thu tren pod)
#   WS=/workspace
# Ket qua: $WS/outputs/reports/experiment_<ten>_<thoi gian>.{md,tsv}, bang tung benchmark _<benchmark>[_vs_<moc>].{md,json};
#          goi $WS/ket_qua_*.tar.gz

set -uo pipefail

usage() { sed -n '2,27p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac
NAME="$1"
shift
[[ "$NAME" =~ ^[a-z0-9_]+$ ]] || usage
ACTION=run
for arg in "$@"; do
  case "$arg" in
    --background) ACTION=background ;;
    --plan) ACTION=plan ;;
    *) usage ;;
  esac
done
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 2
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------- ke hoach (configs/experiments/<ten>.yaml)
PLAN_TEXT="$("$PY" -m src.training.experiment --configs "$REPO_DIR/configs" --ws "$WS" --plan "$NAME")" || exit 2
DESCRIPTION="" MODES="" AGAINST="" SEEDS="" GPU_HOURS="" DISK_GB=""
BENCHES=() B0S=() SCENE_SETS=()
while IFS='|' read -r key v1 v2 v3; do
  case "$key" in
    description) DESCRIPTION="$v1" ;;
    modes) MODES="$v1" ;;
    also_against) AGAINST="$v1" ;;
    seeds) SEEDS="$v1" ;;
    gpu_hours) GPU_HOURS="$v1" ;;
    disk_gb) DISK_GB="$v1" ;;
    bench) BENCHES+=("$v1"); B0S+=("$v2"); SCENE_SETS+=("$v3") ;;
  esac
done <<< "$PLAN_TEXT"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a MODE_LIST <<< "$MODES"
read -r -a AGAINST_LIST <<< "$AGAINST"
OTHER_MODES="${MODE_LIST[*]:1}"  # cac che do ngoai baseline: cung bang so sanh voi tung moc trong also_against
NEEDS_B0=0
for flag in "${B0S[@]}"; do
  if [ "$flag" = 1 ]; then NEEDS_B0=1; fi
done
VOLUME="${VOLUME_GB:-100}"
if ! [[ "$VOLUME" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: VOLUME_GB must be a number of GB, e.g. VOLUME_GB=100"
  exit 2
fi

print_plan() {
  local i
  echo "Thực nghiệm $NAME: $DESCRIPTION"
  echo "  chế độ: $MODES | seed: $SEEDS (xen kẽ: đủ một seed trên mọi cảnh rồi mới sang seed sau) | 30.000 vòng"
  for i in "${!BENCHES[@]}"; do
    echo "  ${BENCHES[$i]}: ${SCENE_SETS[$i]} | B0 Inria: $([ "${B0S[$i]}" = 1 ] && echo có || echo không)"
  done
  echo "  bảng so sánh: với ${MODE_LIST[0]}${AGAINST:+; thêm $OTHER_MODES so với $AGAINST}"
  echo "  giữ mọi file kết quả (checkpoint, PLY, ảnh render, video) của mọi seed"
  echo "  ước lượng (RTX 4090, chưa phải số đo): khoảng $GPU_HOURS giờ GPU, thêm khoảng $DISK_GB GB trên volume" \
    "(VOLUME_GB=$VOLUME)"
}

if [ "$ACTION" = plan ]; then
  print_plan
  echo
  echo "Dữ liệu (chỉ kiểm tra, không tải):"
  for i in "${!BENCHES[@]}"; do
    read -r -a scenes <<< "${SCENE_SETS[$i]}"
    bash "$REPO_DIR/scripts/pod/fetch_data.sh" "${BENCHES[$i]}" "${scenes[@]}" --dry-run
  done
  echo
  echo "Các bước: env_check, space_check, ${BENCHES[*]/#/fetch_}"
  for i in "${!BENCHES[@]}"; do
    if [ "${B0S[$i]}" = 1 ]; then
      echo "  b0_${BENCHES[$i]}: SCENES=\"${SCENE_SETS[$i]}\" bash scripts/run_train.sh ${BENCHES[$i]} b0"
    fi
  done
  for seed in "${SEED_LIST[@]}"; do
    for i in "${!BENCHES[@]}"; do
      echo "  gsplat_${BENCHES[$i]}_s$seed: MODES=\"$MODES\" SEEDS=$seed SCENES=\"${SCENE_SETS[$i]}\"" \
        "bash scripts/run_train.sh ${BENCHES[$i]} gsplat"
    done
  done
  for bench in "${BENCHES[@]}"; do
    echo "  report_$bench: MODES=\"$MODES\" bash scripts/run_evaluate.sh $bench"
    for against in "${AGAINST_LIST[@]}"; do
      echo "  report_${bench}_vs_$against: BASELINE=$against MODES=\"$OTHER_MODES\" bash scripts/run_evaluate.sh $bench"
    done
  done
  echo "  pack"
  echo "Chạy: VOLUME_GB=$VOLUME bash scripts/pod/run_experiment.sh $NAME --background"
  exit 0
fi
if [ "$ACTION" = background ]; then
  print_plan
  exec bash "$REPO_DIR/scripts/pod/job.sh" start "exp_$NAME" -- bash "$REPO_DIR/scripts/pod/run_experiment.sh" "$NAME"
fi

# ---------------------------------------------------------------------------------------------------- chay
cd "$REPO_DIR" || exit 1
export MPLBACKEND=Agg
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_experiment $NAME"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="$WS/outputs/reports"
REPORT="$REPORT_DIR/experiment_${NAME}_$STAMP"
PKG="$WS/ket_qua_$(date +%m%d_%H%M%S).tar.gz"
mkdir -p "$REPORT_DIR/configs" "$WS/outputs/logs"
if [ -z "${SR3D_JOB_NAME:-}" ]; then  # chay qua job.sh thi log da nam trong $WS/outputs/logs/jobs/exp_<ten>.log
  LOG="$WS/outputs/logs/experiment_${NAME}_$STAMP.log"
  exec > >(tee -a "$LOG" {SR3D_GPU_LOCK_FD}>&-) 2>&1  # tee khong giu khoa GPU (scripts/lib/gpu_lock.sh)
  echo "Log: $LOG"
fi
PLAN_JSON="$REPORT_DIR/configs/experiment_${NAME}_$STAMP.json"  # ban chup ke hoach da dung (bang chung)
"$PY" -m src.training.experiment --configs "$REPO_DIR/configs" --ws "$WS" --export "$NAME" --out "$PLAN_JSON" > /dev/null \
  || { echo "ERROR: cannot write $PLAN_JSON"; exit 2; }
print_plan
echo "  kế hoạch đã dùng: $PLAN_JSON"

declare -A RC=()    # ma thoat cua tung buoc
declare -A NOTE=()  # ly do bo qua
declare -A SECS=()
STEPS=()
hms() { printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
banner() { printf '\n########## %s %s\n' "$(date '+%F %T')" "$*"; }
ok() { [ "${RC[$1]:-}" = 0 ]; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1e9 }'; }

run_step() {  # run_step <ten> <lenh...>
  local name="$1" t0
  shift
  STEPS+=("$name")
  banner "BẮT ĐẦU: $name"
  t0="$(date +%s)"
  "$@" < /dev/null
  RC[$name]=$?
  SECS[$name]=$(($(date +%s) - t0))
  banner "XONG: $name (exit ${RC[$name]}, $(hms "${SECS[$name]}"))"
}

skip_step() {  # skip_step <ten> <ly do>
  STEPS+=("$1")
  NOTE[$1]="$2"
  banner "BỎ QUA: $1 ($2)"
}

# shellcheck disable=SC2329  # goi qua check_envs
env_ready() {  # env_ready <env conda> <doan python kiem tra>
  # shellcheck source=/dev/null
  if (set +u; . "$WS/miniforge3/etc/profile.d/conda.sh" && conda activate "$1" && python -c "$2"); then
    echo "env $1: OK"
  else
    echo "ERROR: conda env $1 is not ready (run bash scripts/pod/run_all.sh and read its log)"
    return 1
  fi
}

# shellcheck disable=SC2329  # goi qua run_step
check_envs() {
  command -v nvidia-smi > /dev/null 2>&1 || { echo "ERROR: nvidia-smi not found (no GPU visible)"; return 1; }
  [ -f "$WS/miniforge3/etc/profile.d/conda.sh" ] \
    || { echo "ERROR: no Miniforge in $WS/miniforge3: run bash scripts/pod/run_all.sh first"; return 1; }
  [ -f "$WS/code/gsplat/examples/simple_trainer.py" ] \
    || { echo "ERROR: no gsplat checkout in $WS/code/gsplat: run bash scripts/pod/run_all.sh first"; return 1; }
  env_ready gs-gsplat "import torch, fused_ssim; from gsplat import csrc; x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)" \
    || return 1
  # simple_trainer.py that nhan dung tham so cua moi che do trong ke hoach (vai giay, khong train): sai thi dung
  # ngay, khong de cac luot cua che do do loi sau nhieu gio
  # shellcheck source=/dev/null
  if ! (set +u; . "$WS/miniforge3/etc/profile.d/conda.sh" && conda activate gs-gsplat \
        && python "$REPO_DIR/scripts/setup/smoke_test_gsplat.py" --examples-dir "$WS/code/gsplat/examples" \
           --modes "${MODE_LIST[@]}"); then
    echo "ERROR: gsplat simple_trainer.py does not accept the arguments of a mode in the plan (see above)"
    return 1
  fi
  if [ "$NEEDS_B0" = 1 ]; then
    [ -d "$WS/code/gaussian-splatting" ] \
      || { echo "ERROR: no Inria checkout in $WS/code/gaussian-splatting: run bash scripts/pod/run_all.sh first"; return 1; }
    env_ready gs-inria "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; assert torch.cuda.is_available()" \
      || return 1
  fi
}

# ------------------------------------------------------------------------------------------------ cho trong
# Dung luong = tong kich thuoc file (du -b): df tren network volume cua Runpod bao ca cum luu tru, khong phai quota.
MARGIN=2000000000  # chua them 2 GB cho log, bao cao, file tam
LIMIT=""           # byte, dat boi space_check khi do duoc; rong = khong kiem tra (SPACE_CHECK=0 hoac du loi)
BASE=0             # byte dang dung ngoai du lieu / ket qua cua thuc nghiem nay (env conda, code, viec khac)
OWN_DIRS=()        # du lieu va thu muc ket qua cua cac benchmark trong ke hoach
declare -A NEED=() FINISHED=()  # theo seed: byte con can cho cac luot chua xong; "xong/tong" (seed_space)

du_bytes() {  # du_bytes <thu muc...>: tong kich thuoc (byte) cac thu muc dang co; rong neu du loi
  local dirs=() d
  for d in "$@"; do [ -e "$d" ] && dirs+=("$d"); done
  [ "${#dirs[@]}" -gt 0 ] || { echo 0; return 0; }
  du -sbc "${dirs[@]}" 2> /dev/null | tail -n 1 | cut -f1
}

own_bytes() {  # own_bytes: dung luong du lieu / ket qua cua thuc nghiem (nhanh: vai nghin file); rong neu du loi
  local value
  value="$(du_bytes "${OWN_DIRS[@]}")"
  [[ "$value" =~ ^[0-9]+$ ]] && echo "$value"
}

seed_space() {  # seed_space [benchmark...]: dat NEED[seed], FINISHED[seed] theo cac luot da xong tren dia
  local text key seed need finished runs only=()  # mac dinh moi benchmark cua ke hoach
  [ "$#" -gt 0 ] && only=(--datasets "$@")
  text="$("$PY" -m src.training.experiment --configs "$REPO_DIR/configs" --ws "$WS" --space "$NAME" "${only[@]}")" \
    || return 1
  while IFS='|' read -r key seed need finished runs _; do
    [ "$key" = seed ] || continue
    NEED[$seed]="$need"
    FINISHED[$seed]="$finished/$runs"
  done <<< "$text"
  [ "${#NEED[@]}" -gt 0 ]
}

# shellcheck disable=SC2329  # goi qua run_step
space_check() {
  if [ "${SPACE_CHECK:-1}" = 0 ]; then
    echo "SPACE_CHECK=0: free space is not checked"
    return 0
  fi
  local total own bench key value seed first="" rest=0
  for bench in "${BENCHES[@]}"; do
    for key in root results.gsplat results.inria_b0; do
      value="$(registry_value "$bench" "$key")" || return 1
      OWN_DIRS+=("$value")
    done
  done
  echo "Measuring the space used in $WS (du; can take a few minutes on a network volume)..."
  total="$(du_bytes "$WS")"
  own="$(own_bytes)"
  if ! [[ "$total" =~ ^[0-9]+$ && "$own" =~ ^[0-9]+$ ]] || ! seed_space; then
    echo "WARN: could not measure the space used; free space is not checked (volume ${VOLUME} GB)"
    return 0
  fi
  BASE=$((total - own))
  LIMIT="$(awk -v gb="$VOLUME" 'BEGIN { printf "%.0f", gb * 1e9 }')"
  for seed in "${SEED_LIST[@]}"; do
    rest=$((rest + NEED[$seed]))
    if [ -z "$first" ] && [ "${NEED[$seed]}" -gt 0 ]; then first="$seed"; fi
  done
  echo "Đang dùng $(gb "$total") GB, trong đó dữ liệu / kết quả của thực nghiệm này $(gb "$own") GB."
  for seed in "${SEED_LIST[@]}"; do
    echo "  seed $seed: ${FINISHED[$seed]} lượt gsplat đã xong, còn cần khoảng $(gb "${NEED[$seed]}") GB"
  done
  if [ -z "$first" ]; then
    echo "Mọi lượt gsplat của kế hoạch đã xong: không cần thêm chỗ."
    return 0
  fi
  echo "Dự kiến khi xong: $(gb $((total + rest))) GB / volume $VOLUME GB (lượt chưa xong tính theo lượt cùng cảnh, cùng" \
    "chế độ đã xong ở seed khác; chưa có thì theo ước lượng $DISK_GB GB của kế hoạch, gồm cả dữ liệu, B0, ảnh thu nhỏ)."
  if [ $((total + NEED[$first] + MARGIN)) -gt "$LIMIT" ]; then
    echo "ERROR: seed $first (about $(gb "${NEED[$first]}") GB more) would not fit in a ${VOLUME} GB volume (2 GB kept"
    echo "       free): enlarge the volume on Runpod and set VOLUME_GB to its new size, or free some space"
    echo "       (SPACE_CHECK=0: start anyway)"
    return 1
  fi
  if [ $((total + rest + MARGIN)) -gt "$LIMIT" ]; then
    echo "WARN: the rest of the plan may not fit in ${VOLUME} GB. Training starts; before each seed the space is"
    echo "      checked again, and training stops when that seed would not fit (finished runs are kept)."
    echo "      Enlarging the volume now avoids that."
  fi
}

room_for_seed() {  # room_for_seed <seed>: con du cho cho cac luot chua xong cua seed nay (benchmark se train)
  local own used bench active=()
  [ -n "$LIMIT" ] || return 0
  for bench in "${BENCHES[@]}"; do
    if ok "fetch_$bench"; then active+=("$bench"); fi
  done
  [ "${#active[@]}" -gt 0 ] || return 0  # khong benchmark nao se train
  if ! own="$(own_bytes)" || ! seed_space "${active[@]}"; then
    echo "WARN: could not measure the space used; space not checked for seed $1"
    return 0
  fi
  if [ "${NEED[$1]:-0}" = 0 ]; then
    echo "Seed $1: ${FINISHED[$1]} lượt gsplat đã xong, không cần thêm chỗ"
    return 0
  fi
  used=$((BASE + own))
  echo "Seed $1: ${FINISHED[$1]} lượt đã xong, cần thêm khoảng $(gb "${NEED[$1]}") GB; đang dùng $(gb "$used") GB" \
    "/ volume $VOLUME GB"
  [ $((used + NEED[$1] + MARGIN)) -le "$LIMIT" ]
}

write_summary() {  # bang cac buoc: experiment_<ten>_<thoi gian>.tsv (doc boi src.evaluation.overview) va .md
  local name status repo_commit i
  repo_commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2> /dev/null || echo unknown)"
  {
    printf '# experiment %s %s | repo %s%s\n' "$NAME" "$STAMP" "$repo_commit" \
      "$([ -n "$(git -C "$REPO_DIR" status --porcelain 2> /dev/null | head -c 1)" ] && echo ' (có thay đổi chưa commit)')"
    printf '# step\tstatus\texit\tseconds\tnote\n'
    for name in "${STEPS[@]}"; do
      if [ -n "${RC[$name]:-}" ]; then
        if [ "${RC[$name]}" = 0 ]; then status=OK; else status=FAIL; fi
      else
        status=SKIP
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "${RC[$name]:-}" "${SECS[$name]:-}" "${NOTE[$name]:-}"
    done
  } > "$REPORT.tsv"
  {
    echo "# Thực nghiệm $NAME ($STAMP)"
    echo
    echo "- $DESCRIPTION"
    echo "- Kế hoạch: \`configs/experiments/$NAME.yaml\`, bản đã dùng: \`$PLAN_JSON\`; repo $repo_commit"
    echo "- Chế độ: $MODES; seed: $SEEDS"
    echo
    echo "| Bước | Kết quả | Thời gian |"
    echo "|---|---|---|"
    for name in "${STEPS[@]}"; do
      if [ -n "${RC[$name]:-}" ]; then
        if [ "${RC[$name]}" = 0 ]; then status=OK; else status="LỖI (exit ${RC[$name]})"; fi
        echo "| $name | $status | $(hms "${SECS[$name]}") |"
      else
        echo "| $name | BỎ QUA (${NOTE[$name]:-}) | |"
      fi
    done
    for i in "${!BENCHES[@]}"; do
      echo
      echo "## ${BENCHES[$i]}: ${SCENE_SETS[$i]}"
      echo
      if [ -f "${REPORT}_${BENCHES[$i]}.md" ]; then
        cat "${REPORT}_${BENCHES[$i]}.md"
      else
        echo "(chưa có bảng so sánh)"
      fi
      for against in "${AGAINST_LIST[@]}"; do
        echo
        echo "### $OTHER_MODES so với $against (${BENCHES[$i]})"
        echo
        if [ -f "${REPORT}_${BENCHES[$i]}_vs_$against.md" ]; then
          cat "${REPORT}_${BENCHES[$i]}_vs_$against.md"
        else
          echo "(chưa có bảng so sánh)"
        fi
      done
    done
  } > "$REPORT.md"
}

# ------------------------------------------------------------------------------------------------- cac buoc
run_step env_check check_envs
if ok env_check; then
  run_step space_check space_check
fi
if ! ok env_check || ! ok space_check; then
  if ok env_check; then reason="không đủ chỗ trống"; else reason="môi trường chưa sẵn sàng"; fi
  echo "Dừng: $reason; chưa tải, chưa train gì."
  for name in "${BENCHES[@]}"; do skip_step "fetch_$name" "cần env_check, space_check thành công"; done
else
  for i in "${!BENCHES[@]}"; do
    read -r -a scenes <<< "${SCENE_SETS[$i]}"
    run_step "fetch_${BENCHES[$i]}" bash "$REPO_DIR/scripts/pod/fetch_data.sh" "${BENCHES[$i]}" "${scenes[@]}"
  done
  # bien moi truong cua nguoi dung khong duoc doi ke hoach: thu muc ket qua, cau hinh, tham so mcmc, chay thu
  CLEAN=(env -u OUT_ROOT -u B0_ROOT -u CONFIG -u MCMC_SET -u FORCE QUICK=0)
  for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    [ "${B0S[$i]}" = 1 ] || continue
    if ok "fetch_$bench"; then
      run_step "b0_$bench" "${CLEAN[@]}" SCENES="${SCENE_SETS[$i]}" bash "$REPO_DIR/scripts/run_train.sh" "$bench" b0
    else
      skip_step "b0_$bench" "cần fetch_$bench thành công"
    fi
  done
  STOP=""
  for seed in "${SEED_LIST[@]}"; do
    if [ -z "$STOP" ] && ! room_for_seed "$seed"; then
      STOP="không đủ chỗ trống cho seed $seed (VOLUME_GB=$VOLUME)"
      echo "Dừng train: $STOP. Các lượt đã xong được giữ; tăng dung lượng volume, đặt VOLUME_GB rồi chạy lại để làm nốt."
    fi
    for i in "${!BENCHES[@]}"; do
      bench="${BENCHES[$i]}"
      if [ -n "$STOP" ]; then
        skip_step "gsplat_${bench}_s$seed" "$STOP"
      elif ok "fetch_$bench"; then
        run_step "gsplat_${bench}_s$seed" "${CLEAN[@]}" MODES="$MODES" SEEDS="$seed" SCENES="${SCENE_SETS[$i]}" \
          bash "$REPO_DIR/scripts/run_train.sh" "$bench" gsplat
      else
        skip_step "gsplat_${bench}_s$seed" "cần fetch_$bench thành công"
      fi
    done
  done
  for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    run_step "report_$bench" env -u QUICK -u BASELINE MODES="$MODES" SCENES="${SCENE_SETS[$i]}" \
      REPORT="${REPORT}_$bench" bash "$REPO_DIR/scripts/run_evaluate.sh" "$bench"
    for against in "${AGAINST_LIST[@]}"; do
      run_step "report_${bench}_vs_$against" env -u QUICK BASELINE="$against" MODES="$OTHER_MODES" \
        SCENES="${SCENE_SETS[$i]}" REPORT="${REPORT}_${bench}_vs_$against" bash "$REPO_DIR/scripts/run_evaluate.sh" "$bench"
    done
  done
fi

write_summary
run_step pack bash "$REPO_DIR/scripts/pod/pack_results.sh" "$PKG"

FAILED=0
printf '\n========== TÓM TẮT thực nghiệm %s (%s)\n' "$NAME" "$STAMP"
for name in "${STEPS[@]}"; do
  if [ -n "${RC[$name]:-}" ]; then
    if [ "${RC[$name]}" = 0 ]; then
      printf '%-28s OK        %s\n' "$name" "$(hms "${SECS[$name]}")"
    else
      FAILED=1
      printf '%-28s LỖI (exit %s)  %s\n' "$name" "${RC[$name]}" "$(hms "${SECS[$name]}")"
    fi
  else
    FAILED=1
    printf '%-28s BỎ QUA (%s)\n' "$name" "${NOTE[$name]:-}"
  fi
done
echo "Bảng so sánh và tóm tắt: $REPORT.md"
[ "${RC[pack]:-1}" = 0 ] && echo "Gói kết quả nhẹ để tải về: $PKG"
if [ "$FAILED" = 1 ]; then
  echo "Có bước lỗi hoặc bị bỏ qua: xem log của bước đó (gsplat_*: một lượt train lỗi, hoặc phần đối chiếu default với"
  echo "Inria chưa PASS). Chạy lại lệnh này sau khi sửa: các phần đã xong tự bỏ qua."
fi
exit "$FAILED"
