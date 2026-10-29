#!/usr/bin/env bash
# run_experiment.sh - Mot thuc nghiem tron ven theo configs/experiments/<ten>.yaml, trong MOT viec dung GPU:
#   (ke hoach co luot sr3d.mono: tai Depth Anything V2 Small neu chua co, scripts/setup/setup_depth_anything.sh; loi
#   thi chi bo cac luot do) -> kiem tra moi truong -> kiem tra cho trong -> tai du lieu con thieu (scripts/pod/fetch_data.sh)
#   -> kiem tra diem SfM khi ke hoach co luot dung loss do sau (src/training/depth_check.py; khong dat thi chi bo cac
#   luot do tren benchmark do) -> B0 Inria cho benchmark co b0: true -> gsplat
#   cac che do va bien the (scripts/run_train.sh <benchmark> gsplat), xen ke benchmark theo tung seed (du mot seed tren
#   moi canh roi moi sang seed sau), sau moi seed loc Gaussian roi cham lai khi ke hoach co post_filter
#   (src/training/prune.py) -> bao cao so sanh cuoi (scripts/run_evaluate.sh) -> goi ket qua nhe
#   (scripts/pod/pack_results.sh). Moi file ket qua duoc giu (khong xoa gi).
# Luon chay day du (30.000 vong); khong co che do chay thu. Chay lai an toan: du lieu da co khong tai lai, luot gsplat
# da xong, luot da loc voi cung nguong va canh B0 da co ket qua tu bo qua.
#
# Cach chay (tren pod, tu thu muc goc cua repo, sau scripts/pod/run_all.sh):
#   VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --background   chay nen (viec main)
#   bash scripts/pod/run_experiment.sh main --plan     chi in ke hoach, uoc luong, du lieu da co (khong chay gi)
#   bash scripts/pod/run_experiment.sh main            chay truc tiep (dong terminal la dung luot)
#   VOLUME_GB=100 bash scripts/pod/run_experiment.sh main --only mcmc --background   chi mot phuong phap (viec gsplat_mcmc)
# Theo doi: bash scripts/pod/job.sh status | bash scripts/pod/job.sh log main
#
# --only <phuong phap>: chi phan cua mot phuong phap trong ke hoach (src/training/experiment.py, select_method), cung
#   canh, seed, protocol va thu muc ket qua, nen chay rieng roi chay ca ke hoach (hay nguoc lai) khong train lai luot da
#   xong. b0: chi cac buoc B0 Inria (benchmark co b0: true), bang B0 so voi so lieu Inria cong bo. <che do>: che do do,
#   cung default khi che do khac default (luot default da xong duoc dung lai, chua co thi train truoc: cac che do khac
#   lay ngan sach Gaussian tu no); khong B0, khong bien the; doi chieu default voi B0 chi khi da co B0 (B0_CHECK=if_present;
#   ke hoach va bao cao ghi ro).
# Ten ban (src.training.experiment.run_name): ten viec nen, bao cao, log, goi ket qua cua mot lan chay: ca ke hoach
#   = ten ke hoach (main, tune_mcmc); mot phuong phap cua main = ten notebook (b0_inria, gsplat_default, gsplat_mcmc).
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
# Ket qua: $WS/outputs/reports/<ten ban>_<thoi gian>.{md,tsv}, bang tung benchmark _<benchmark>[_vs_<moc>].{md,json}
#          (bien the cua ke hoach vao bang qua TAGS; luot co tag cua ke hoach khac bi de ngoai);
#          luot loc Gaussian: <luot>/filtered/ (bang truoc / sau khi loc nam trong bang so sanh);
#          bieu do tong cua ca lan chay: <ten ban>_<thoi gian>_charts/ (doi chieu Inria, chenh lech tong so voi
#          tung moc, gop cac benchmark; src.evaluation.charts; CHARTS=0: khong ve);
#          goi $WS/ket_qua_<ten ban>_<thoi gian>.tar.gz

set -uo pipefail

usage() { sed -n '2,45p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac
NAME="$1"
shift
[[ "$NAME" =~ ^[a-z0-9_]+$ ]] || usage
ACTION=run
ONLY=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --background) ACTION=background ;;
    --plan) ACTION=plan ;;
    --only)
      [ "$#" -ge 2 ] && [[ "$2" =~ ^[a-z0-9_]+$ ]] || usage
      ONLY="$2"
      shift ;;
    *) usage ;;
  esac
  shift
done
ONLY_ARGS=()
if [ -n "$ONLY" ]; then ONLY_ARGS=(--only "$ONLY"); fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"
PY="$(config_python)" || exit 2
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------- ke hoach (configs/experiments/<ten>.yaml)
PLAN_TEXT="$("$PY" -m src.training.experiment --configs "$REPO_DIR/configs" --ws "$WS" --plan "$NAME" \
  "${ONLY_ARGS[@]}")" || exit 2
DESCRIPTION="" MODES="" AGAINST="" VARIANTS="" RUNS="" DEPTH_RUNS="" TEST_EVERY="" POST_FILTER="" SEEDS=""
MONO_RUNS="" GPU_HOURS="" DISK_GB="" ESTIMATE_SCOPE="" RUN_ESTIMATE="" B0_REFERENCE="" RUN_NAME=""
BENCHES=() B0S=() SCENE_SETS=() WARNINGS=()
while IFS='|' read -r key v1 v2 v3; do
  case "$key" in
    description) DESCRIPTION="$v1" ;;
    modes) MODES="$v1" ;;
    also_against) AGAINST="$v1" ;;
    variants) VARIANTS="$v1" ;;
    runs) RUNS="$v1" ;;
    depth_runs) DEPTH_RUNS="$v1" ;;
    depth_test_every) TEST_EVERY="$v1" ;;
    mono_runs) MONO_RUNS="$v1" ;;
    post_filter) POST_FILTER="$v1" ;;
    seeds) SEEDS="$v1" ;;
    gpu_hours) GPU_HOURS="$v1" ;;
    disk_gb) DISK_GB="$v1" ;;
    estimate_scope) ESTIMATE_SCOPE="$v1" ;;
    run_estimate_bytes) RUN_ESTIMATE="$v1" ;;
    b0_reference) B0_REFERENCE="$v1" ;;
    run_name) RUN_NAME="$v1" ;;  # ten ban: ten viec nen, bao cao, log, goi ket qua (main, gsplat_mcmc...)
    bench) BENCHES+=("$v1"); B0S+=("$v2"); SCENE_SETS+=("$v3") ;;
    warning) WARNINGS+=("$v1") ;;
  esac
done <<< "$PLAN_TEXT"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a MODE_LIST <<< "$MODES"
read -r -a AGAINST_LIST <<< "$AGAINST"
read -r -a VARIANT_LIST <<< "$VARIANTS"  # <che_do>:<tag>:<khoa>=<gia_tri>[,...] (VARIANTS cua train_gsplat_modes.sh)
read -r -a RUN_LIST <<< "$RUNS"          # ten thu muc luot (khong co _s<seed>): che do roi bien the
VARIANT_TAGS=""  # tag cua cac bien the: bang so sanh cua ke hoach nay dua ca cac luot do vao (TAGS cua run_evaluate.sh)
for entry in "${VARIANT_LIST[@]}"; do
  IFS=: read -r _ vtag _ <<< "$entry"
  VARIANT_TAGS="${VARIANT_TAGS:+$VARIANT_TAGS }$vtag"
done
read -r FILTER_OPACITY FILTER_SCALE <<< "$POST_FILTER"
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

b0_missing() {  # b0_missing <chi so benchmark>: canh chua co ket qua B0 du chi so (rong: moi canh da co)
  local root method scenes
  root="$(registry_value "${BENCHES[$1]}" results.inria_b0)" || return 1
  # nhu compare_reference va train_gsplat_modes.sh: khong khai bao thi la ours_30000
  method="$(registry_value "${BENCHES[$1]}" benchmark.method 2> /dev/null)" || method=ours_30000
  read -r -a scenes <<< "${SCENE_SETS[$1]}"
  b0_missing_scenes "$root" "$method" "${scenes[@]}"
}

b0_plan_text() {  # b0_plan_text <chi so benchmark>: cot "B0 Inria" cua ke hoach
  local missing
  if [ "${B0S[$1]}" = 1 ]; then
    echo có
  elif [[ " $B0_REFERENCE " != *" ${BENCHES[$1]} "* ]]; then
    echo không
  elif ! missing="$(b0_missing "$1")"; then
    echo "không chạy trong lần này (chạy bằng --only b0); không đọc được kết quả B0"
  elif [ -z "$missing" ]; then
    echo "không chạy trong lần này (chạy bằng --only b0); đã có cho mọi cảnh: ${MODE_LIST[0]} được đối chiếu với B0"
  else
    echo "không chạy trong lần này (chạy bằng --only b0); chưa có cho $missing: ${MODE_LIST[0]} chưa được đối chiếu với B0"
  fi
}

print_plan() {
  local i entry vmode vtag vsets dropped
  if [ -n "$ONLY" ]; then
    echo "Thực nghiệm $NAME (chỉ $ONLY), một phần của kế hoạch: $DESCRIPTION"
  else
    echo "Thực nghiệm $NAME: $DESCRIPTION"
  fi
  echo "  tên bản: $RUN_NAME (việc nền, báo cáo, log và gói kết quả mang tên này)"
  if [ "$ONLY" = b0 ]; then
    echo "  chỉ B0: 3DGS gốc của Inria trên các benchmark có B0 (cùng thư mục kết quả với cả kế hoạch); không có lượt gsplat"
  else
    echo "  chế độ: $MODES | seed: $SEEDS (xen kẽ: đủ một seed trên mọi cảnh rồi mới sang seed sau) | 30.000 vòng"
  fi
  if [ -n "$ONLY" ] && [ "$ONLY" != b0 ] && [ "${#MODE_LIST[@]}" -gt 1 ]; then
    echo "  ${MODE_LIST[0]}: lượt đã xong (của cả kế hoạch hay chạy riêng) được dùng lại; chưa có thì train trước," \
      "vì $ONLY lấy ngân sách Gaussian từ nó"
  fi
  if [ "${#VARIANT_LIST[@]}" -gt 0 ]; then
    echo "  biến thể (thư mục <chế_độ>_<tag>_s<seed>, cùng ngân sách Gaussian với default):"
    for entry in "${VARIANT_LIST[@]}"; do
      IFS=: read -r vmode vtag vsets <<< "$entry"
      echo "    ${vmode}_$vtag: $vmode, ${vsets//,/, }"
    done
  fi
  for i in "${!BENCHES[@]}"; do
    echo "  ${BENCHES[$i]}: ${SCENE_SETS[$i]} | B0 Inria: $(b0_plan_text "$i")"
  done
  if [ -n "$DEPTH_RUNS" ]; then
    echo "  loss độ sâu ($DEPTH_RUNS): kiểm tra điểm SfM của mọi ảnh train trước khi train (depth_check_<benchmark>);" \
      "không đạt thì bỏ các lượt đó trên benchmark đó, các lượt khác vẫn train"
  fi
  if [ -n "$MONO_RUNS" ]; then
    if [[ " $MONO_RUNS " == *" ${MODE_LIST[0]} "* ]]; then
      dropped="không tải được thì không train lượt gsplat nào: ${MODE_LIST[0]} dùng depth prior"
    else
      dropped="không tải được thì bỏ các lượt đó, các lượt khác vẫn train"
    fi
    echo "  depth prior ($MONO_RUNS): bước depth_anything tải Depth Anything V2 Small vào $WS/code/depth_anything_v2 nếu" \
      "chưa có ($dropped); mỗi cảnh tính depth prior một lần (bước mono của lượt), prior không khớp điểm SfM thì lượt" \
      "đó lỗi"
  fi
  if [ -n "$POST_FILTER" ]; then
    echo "  sau mỗi seed: lọc Gaussian (opacity < $FILTER_OPACITY, trục lớn nhất > $FILTER_SCALE × scene_scale)" \
      "rồi chấm lại; bảng trước / sau khi lọc trong báo cáo"
  fi
  if [ "${#MODE_LIST[@]}" -gt 0 ]; then
    echo "  bảng so sánh: với ${MODE_LIST[0]}${AGAINST:+; thêm một bảng so với $AGAINST}"
    echo "  giữ mọi file kết quả (checkpoint, PLY, ảnh render, video${POST_FILTER:+, bản đã lọc}) của mọi seed"
  else
    echo "  bảng: B0 so với số liệu Inria công bố"
  fi
  if [ -n "$ONLY" ] && [ "$ONLY" != b0 ]; then
    echo "  đối chiếu ${MODE_LIST[0]} với B0 tự chạy: chỉ khi đã có B0 (chạy riêng b0 hoặc cả kế hoạch); chưa có thì ghi chú"
  fi
  for warning in "${WARNINGS[@]}"; do
    echo "  CẢNH BÁO: $warning"
  done
  echo "  ước lượng (RTX 4090, chưa phải số đo): khoảng $GPU_HOURS giờ GPU, thêm khoảng $DISK_GB GB trên volume" \
    "(VOLUME_GB=$VOLUME)$([ "$ESTIMATE_SCOPE" = plan ] && [ -n "$ONLY" ] && echo '; đây là ước lượng của cả kế hoạch')"
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
  echo "Các bước: ${MONO_RUNS:+depth_anything (bash scripts/setup/setup_depth_anything.sh), }env_check, space_check," \
    "${BENCHES[*]/#/fetch_}"
  B0_ENV=""
  if [ -n "$ONLY" ] && [ "$ONLY" != b0 ]; then B0_ENV="B0_CHECK=if_present "; fi
  for i in "${!BENCHES[@]}"; do
    if [ -n "$DEPTH_RUNS" ]; then
      echo "  depth_check_${BENCHES[$i]}: python -m src.training.depth_check --test-every $TEST_EVERY" \
        "<cảnh: ${SCENE_SETS[$i]}> (env gs-gsplat, không GPU)"
    fi
    if [ "${B0S[$i]}" = 1 ]; then
      echo "  b0_${BENCHES[$i]}: SCENES=\"${SCENE_SETS[$i]}\" bash scripts/run_train.sh ${BENCHES[$i]} b0"
    fi
  done
  for seed in "${SEED_LIST[@]}"; do
    [ "${#MODE_LIST[@]}" -gt 0 ] || break  # chi B0: khong co luot gsplat
    for i in "${!BENCHES[@]}"; do
      echo "  gsplat_${BENCHES[$i]}_s$seed: ${B0_ENV}MODES=\"$MODES\"${VARIANTS:+ VARIANTS=\"$VARIANTS\"} SEEDS=$seed" \
        "SCENES=\"${SCENE_SETS[$i]}\" bash scripts/run_train.sh ${BENCHES[$i]} gsplat"
      if [ -n "$POST_FILTER" ]; then
        echo "  filter_${BENCHES[$i]}_s$seed: python -m src.training.prune --min-opacity $FILTER_OPACITY" \
          "--max-scale $FILTER_SCALE --skip-unfinished <$RUNS × ${SCENE_SETS[$i]}, seed $seed>"
      fi
    done
  done
  for bench in "${BENCHES[@]}"; do
    if [ "${#MODE_LIST[@]}" -eq 0 ]; then
      echo "  report_$bench: python -m src.evaluation.compare_reference --model-root <B0 của $bench> (B0 so với số Inria" \
        "công bố, không GPU)"
      continue
    fi
    echo "  report_$bench: MODES=\"$MODES\"${VARIANT_TAGS:+ TAGS=\"$VARIANT_TAGS\"} bash scripts/run_evaluate.sh $bench"
    for against in "${AGAINST_LIST[@]}"; do
      echo "  report_${bench}_vs_$against: BASELINE=$against MODES=\"$OTHER_MODES\"${VARIANT_TAGS:+ TAGS=\"$VARIANT_TAGS\"}" \
        "bash scripts/run_evaluate.sh $bench"
    done
  done
  echo "  pack"
  echo "Chạy: VOLUME_GB=$VOLUME bash scripts/pod/run_experiment.sh $NAME${ONLY:+ --only $ONLY} --background"
  exit 0
fi
if [ "$ACTION" = background ]; then
  print_plan
  exec bash "$REPO_DIR/scripts/pod/job.sh" start "$RUN_NAME" -- bash "$REPO_DIR/scripts/pod/run_experiment.sh" "$NAME" \
    "${ONLY_ARGS[@]}"
fi

# ---------------------------------------------------------------------------------------------------- chay
cd "$REPO_DIR" || exit 1
export MPLBACKEND=Agg
# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_experiment $RUN_NAME"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="$WS/outputs/reports"
REPORT="$REPORT_DIR/${RUN_NAME}_$STAMP"
PKG="$WS/ket_qua_${RUN_NAME}_$STAMP.tar.gz"
mkdir -p "$REPORT_DIR/configs" "$WS/outputs/logs"
if [ -z "${SR3D_JOB_NAME:-}" ]; then  # chay qua job.sh thi log da nam trong $WS/outputs/logs/jobs/<ten ban>.log
  LOG="$WS/outputs/logs/${RUN_NAME}_$STAMP.log"
  exec > >(tee -a "$LOG" {SR3D_GPU_LOCK_FD}>&-) 2>&1  # tee khong giu khoa GPU (scripts/lib/gpu_lock.sh)
  echo "Log: $LOG"
fi
PLAN_JSON="$REPORT_DIR/configs/${RUN_NAME}_$STAMP.json"  # ban chup ke hoach da dung (bang chung)
"$PY" -m src.training.experiment --configs "$REPO_DIR/configs" --ws "$WS" --export "$NAME" "${ONLY_ARGS[@]}" \
  --out "$PLAN_JSON" > /dev/null || { echo "ERROR: cannot write $PLAN_JSON"; exit 2; }
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
  if [ "${#MODE_LIST[@]}" -gt 0 ]; then  # chi B0 (--only b0): khong can env gs-gsplat
    check_gsplat_env || return 1
  fi
  if [ "$NEEDS_B0" = 1 ]; then
    [ -d "$WS/code/gaussian-splatting" ] \
      || { echo "ERROR: no Inria checkout in $WS/code/gaussian-splatting: run bash scripts/pod/run_all.sh first"; return 1; }
    env_ready gs-inria "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; assert torch.cuda.is_available()" \
      || return 1
  fi
}

# shellcheck disable=SC2329  # goi qua check_envs
check_gsplat_env() {
  [ -f "$WS/code/gsplat/examples/simple_trainer.py" ] \
    || { echo "ERROR: no gsplat checkout in $WS/code/gsplat: run bash scripts/pod/run_all.sh first"; return 1; }
  env_ready gs-gsplat "import torch, fused_ssim; from gsplat import csrc; x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)" \
    || return 1
  # simple_trainer.py that nhan dung tham so cua moi che do va bien the se train (vai giay, khong train; bien the co code:
  # ban sao co va cua trainer, KR va loss depth prior tren GPU; depth prior: Depth Anything V2 chay duoc tren GPU): sai
  # thi dung ngay, khong de cac luot do loi sau nhieu gio. Luot bi bo vi buoc depth_anything loi thi khong kiem.
  local smoke=(--modes) mode entry vmode vtag variants=()
  for mode in "${MODE_LIST[@]}"; do
    [[ " $MONO_DROPPED " == *" $mode "* ]] || smoke+=("$mode")
  done
  for entry in "${VARIANT_LIST[@]}"; do
    IFS=: read -r vmode vtag _ <<< "$entry"
    [[ " $MONO_DROPPED " == *" ${vmode}_$vtag "* ]] || variants+=("$entry")
  done
  if [ "${#variants[@]}" -gt 0 ]; then smoke+=(--variants "${variants[@]}"); fi
  if [ -n "$MONO_RUNS" ] && [ -z "$MONO_DROPPED" ]; then smoke+=(--depth-anything-dir "$WS/code/depth_anything_v2"); fi
  # shellcheck source=/dev/null
  if ! (set +u; . "$WS/miniforge3/etc/profile.d/conda.sh" && conda activate gs-gsplat \
        && python "$REPO_DIR/scripts/setup/smoke_test_gsplat.py" --examples-dir "$WS/code/gsplat/examples" \
           "${smoke[@]}"); then
    echo "ERROR: the gsplat smoke test of the plan failed (simple_trainer.py arguments of a mode or variant, the code"
    echo "       variants on the GPU, or Depth Anything V2; see the FAIL lines above)"
    return 1
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

declare -A DROPPED=()  # theo benchmark: cac luot co loss do sau bi bo vi depth_check_<benchmark> khong dat
MONO_DROPPED=""        # moi benchmark: cac luot dung depth prior bi bo vi buoc depth_anything loi

seed_space() {  # seed_space [benchmark...]: dat NEED[seed], FINISHED[seed] theo cac luot da xong tren dia
  local text key seed need finished runs bench run only=() without=()  # mac dinh moi benchmark cua ke hoach
  [ "$#" -gt 0 ] && only=(--datasets "$@")
  for bench in "${!DROPPED[@]}"; do  # luot bi bo vi kiem tra diem SfM: khong train, khong can cho
    for run in ${DROPPED[$bench]}; do without+=("$bench:$run"); done
  done
  for run in $MONO_DROPPED; do  # luot bi bo vi buoc depth_anything loi, tren moi benchmark
    for bench in "${BENCHES[@]}"; do without+=("$bench:$run"); done
  done
  if [ "${#without[@]}" -gt 0 ]; then only+=(--without "${without[@]}"); fi
  text="$("$PY" -m src.training.experiment --configs "$REPO_DIR/configs" --ws "$WS" --space "$NAME" "${ONLY_ARGS[@]}" \
    "${only[@]}")" || return 1
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
  if [ "${#MODE_LIST[@]}" -eq 0 ] && [[ "$total" =~ ^[0-9]+$ ]]; then  # chi B0: khong co luot gsplat de tinh theo seed
    b0_space "$total"
    return
  fi
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
    "chế độ đã xong ở seed khác; chưa có thì khoảng $(gb "$RUN_ESTIMATE") GB mỗi lượt: ước lượng dung lượng của cả kế" \
    "hoạch chia đều cho mọi lượt, gồm cả dữ liệu, B0, ảnh thu nhỏ)."
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
  local own used i active=()
  [ -n "$LIMIT" ] || return 0
  for i in "${!BENCHES[@]}"; do
    if ready "$i"; then active+=("${BENCHES[$i]}"); fi
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

# shellcheck disable=SC2329  # goi qua run_step
depth_check() {  # depth_check <chi so benchmark>: moi anh train cua moi canh co diem SfM cho loss do sau
  local i="$1" root scene scenes dirs=()
  root="$(registry_value "${BENCHES[$i]}" root)" || return 1
  read -r -a scenes <<< "${SCENE_SETS[$i]}"
  for scene in "${scenes[@]}"; do dirs+=("$root/$scene"); done
  # shellcheck source=/dev/null
  (set +u; . "$WS/miniforge3/etc/profile.d/conda.sh" && conda activate gs-gsplat \
    && python -m src.training.depth_check --examples-dir "$WS/code/gsplat/examples" --test-every "$TEST_EVERY" \
      "${dirs[@]}")
}

# shellcheck disable=SC2329  # goi qua run_step
post_filter() {  # post_filter <chi so benchmark> <seed>: loc Gaussian moi luot da train xong cua seed nay, cham lai
  local i="$1" seed="$2" root scene run scenes dirs=()
  root="$(registry_value "${BENCHES[$i]}" results.gsplat)" || return 1
  read -r -a scenes <<< "${SCENE_SETS[$i]}"
  for scene in "${scenes[@]}"; do
    for run in "${RUN_LIST[@]}"; do dirs+=("$root/$scene/${run}_s$seed"); done
  done
  # luot chua train xong (gsplat loi) duoc bo qua: buoc gsplat cua seed nay da bao loi
  # shellcheck source=/dev/null
  (set +u; . "$WS/miniforge3/etc/profile.d/conda.sh" && conda activate gs-gsplat \
    && python -m src.training.prune --min-opacity "$FILTER_OPACITY" --max-scale "$FILTER_SCALE" --skip-unfinished \
      "${dirs[@]}")
}


# shellcheck disable=SC2329  # goi qua run_step
b0_report() {  # b0_report <chi so benchmark>: bang B0 so voi so lieu Inria cong bo (chi B0, --only b0; khong GPU)
  local bench="${BENCHES[$1]}" config root text rc=0
  config="$REPORT_DIR/configs/${bench}_$STAMP.json"
  config_export_benchmark "$bench" "$config" || return 1
  root="$(registry_value "$bench" results.inria_b0)" || return 1
  text="$(PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m src.evaluation.compare_reference --config "$config" \
    --model-root "$root" --out "${REPORT}_$bench.json")" || rc=$?
  printf '%s\n' "$text" | sed '/^Đã ghi tóm tắt: /d' > "${REPORT}_$bench.md"  # bang, khong co dong ghi file JSON
  return "$rc"
}


# shellcheck disable=SC2329  # goi qua space_check
b0_space() {  # b0_space <byte dang dung>: chi B0 (--only b0); con canh B0 chua co ket qua thi can them DISK_GB
  local total="$1" need=0 limit i missing
  limit="$(awk -v gb="$VOLUME" 'BEGIN { printf "%.0f", gb * 1e9 }')"
  for i in "${!BENCHES[@]}"; do
    [ "${B0S[$i]}" = 1 ] || continue
    if ! missing="$(b0_missing "$i")"; then  # cung tieu chi voi bang so sanh: results.json hong = chua co
      echo "WARN: could not read the B0 results of ${BENCHES[$i]}; free space for B0 is not checked (volume ${VOLUME} GB)"
      return 0
    fi
    [ -z "$missing" ] || need="$(awk -v gb="$DISK_GB" 'BEGIN { printf "%.0f", gb * 1e9 }')"
  done
  if [ "$need" = 0 ]; then
    echo "Đang dùng $(gb "$total") GB; mọi cảnh B0 đã có kết quả: không cần thêm chỗ."
    return 0
  fi
  if [ "$ESTIMATE_SCOPE" = plan ]; then  # khong co estimate.only.b0: uoc luong cua ca ke hoach qua lon cho B0
    echo "Đang dùng $(gb "$total") GB / volume $VOLUME GB; kế hoạch không có ước lượng riêng cho B0 (estimate.only.b0):" \
      "không kiểm tra chỗ trống cho B0."
    return 0
  fi
  echo "Đang dùng $(gb "$total") GB; B0 còn cảnh chưa có kết quả: cần thêm khoảng $(gb "$need") GB (ước lượng của" \
    "kế hoạch) / volume $VOLUME GB."
  if [ $((total + need + MARGIN)) -gt "$limit" ]; then
    echo "ERROR: B0 (about $(gb "$need") GB more) would not fit in a ${VOLUME} GB volume (2 GB kept free): enlarge the"
    echo "       volume on Runpod and set VOLUME_GB to its new size, or free some space (SPACE_CHECK=0: start anyway)"
    return 1
  fi
}

drop_depth_runs() {  # drop_depth_runs <chi so benchmark>: depth_check khong dat thi bo cac luot co loss do sau
  local bench="${BENCHES[$1]}" mode entry vmode vtag dropped=""
  { [ -n "$DEPTH_RUNS" ] && [ -n "${RC[depth_check_$bench]:-}" ] && ! ok "depth_check_$bench"; } || return 0
  for mode in "${MODE_LIST[@]}"; do
    if [[ " $DEPTH_RUNS " == *" $mode "* ]]; then dropped+=" $mode"; fi
  done
  for entry in "${VARIANT_LIST[@]}"; do  # bien the cua mot che do bi bo cung bi bo
    IFS=: read -r vmode vtag _ <<< "$entry"
    if [[ " $DEPTH_RUNS$dropped " == *" ${vmode}_$vtag "* || " $dropped " == *" $vmode "* ]]; then
      dropped+=" ${vmode}_$vtag"
    fi
  done
  DROPPED[$bench]="${dropped# }"
  if [[ " ${DROPPED[$bench]} " == *" ${MODE_LIST[0]} "* ]]; then
    echo "depth_check_$bench không đạt và ${MODE_LIST[0]} dùng loss độ sâu: không train lượt gsplat nào trên $bench"
  else
    echo "depth_check_$bench không đạt: bỏ ${DROPPED[$bench]} trên $bench (lượt có loss độ sâu sẽ lỗi); các lượt khác" \
      "vẫn train"
  fi
}

bench_modes() {  # bench_modes <chi so benchmark>: MODES cua buoc gsplat tren benchmark nay
  local mode out=""
  for mode in "${MODE_LIST[@]}"; do
    [[ " ${DROPPED[${BENCHES[$1]}]:-} $MONO_DROPPED " == *" $mode "* ]] || out+="${out:+ }$mode"
  done
  echo "$out"
}

bench_variants() {  # bench_variants <chi so benchmark>: VARIANTS cua buoc gsplat tren benchmark nay
  local entry vmode vtag out=""
  for entry in "${VARIANT_LIST[@]}"; do
    IFS=: read -r vmode vtag _ <<< "$entry"
    [[ " ${DROPPED[${BENCHES[$1]}]:-} $MONO_DROPPED " == *" ${vmode}_$vtag "* ]] || out+="${out:+ }$entry"
  done
  echo "$out"
}

ready() {  # ready <chi so benchmark>: du lieu da tai, va baseline khong bi bo vi kiem tra diem SfM
  ok "fetch_${BENCHES[$1]}" && [[ " ${DROPPED[${BENCHES[$1]}]:-} " != *" ${MODE_LIST[0]} "* ]]
}

needs() {  # needs <chi so benchmark>: ly do bo qua khi chua san sang
  if ! ok "fetch_${BENCHES[$1]}"; then
    echo "cần fetch_${BENCHES[$1]} thành công"
  else
    echo "depth_check_${BENCHES[$1]} không đạt và ${MODE_LIST[0]} dùng loss độ sâu"
  fi
}

b0_summary_line() {  # b0_summary_line <chi so benchmark>: chay rieng mot che do: default co duoc doi chieu voi B0 khong
  local missing bench="${BENCHES[$1]}"
  if ! missing="$(b0_missing "$1")"; then
    echo "- $bench: không đọc được kết quả B0 tự chạy"
  elif [ -z "$missing" ]; then
    echo "- $bench: B0 tự chạy đã có cho mọi cảnh: \`${MODE_LIST[0]}\` được đối chiếu với B0 từng cảnh (bảng bên dưới)"
  else
    echo "- $bench: chưa có B0 tự chạy đủ cho $missing: \`${MODE_LIST[0]}\` chưa được đối chiếu đủ với B0 từng cảnh." \
      "Chạy \`--only b0\` (notebook \`b0_inria.ipynb\`), rồi chạy lại lệnh này để lập lại bảng (lượt đã xong không" \
      "train lại)"
  fi
}

write_summary() {  # bang cac buoc: <ten ban>_<thoi gian>.tsv (doc boi src.evaluation.overview) va .md
  local name status repo_commit i
  repo_commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2> /dev/null || echo unknown)"
  {
    printf '# experiment %s %s | repo %s%s\n' "$NAME${ONLY:+ --only $ONLY}" "$STAMP" "$repo_commit" \
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
    echo "# Thực nghiệm $NAME${ONLY:+ (chỉ $ONLY)} ($STAMP)"
    echo
    echo "- Tên bản: \`$RUN_NAME\` (việc nền, báo cáo, log và gói kết quả \`$(basename "$PKG")\` mang tên này)"
    echo "- ${ONLY:+Một phần của kế hoạch: }$DESCRIPTION"
    echo "- Kế hoạch: \`configs/experiments/$NAME.yaml\`${ONLY:+ (\`--only $ONLY\`)}, bản đã dùng: \`$PLAN_JSON\`; repo $repo_commit"
    if [ "${#MODE_LIST[@]}" -gt 0 ]; then
      echo "- Chế độ: $MODES; seed: $SEEDS"
    else
      echo "- Chỉ B0 (3DGS gốc của Inria), so với số liệu Inria công bố"
    fi
    if [ "${#VARIANT_LIST[@]}" -gt 0 ]; then
      echo "- Biến thể: ${VARIANT_LIST[*]}"
    fi
    if [ -n "$POST_FILTER" ]; then
      echo "- Lọc Gaussian sau khi train: opacity < $FILTER_OPACITY, trục lớn nhất > $FILTER_SCALE × scene_scale"
    fi
    if [ -n "$MONO_DROPPED" ]; then
      echo "- Depth Anything V2 chưa sẵn sàng (bước depth_anything lỗi): không train $MONO_DROPPED trên mọi benchmark"
    fi
    for i in "${!BENCHES[@]}"; do
      if [ -n "$ONLY" ] && [ "${#MODE_LIST[@]}" -gt 0 ] && [[ " $B0_REFERENCE " == *" ${BENCHES[$i]} "* ]]; then
        b0_summary_line "$i"
      fi
      if [ -n "${DROPPED[${BENCHES[$i]}]:-}" ] && ! ready "$i"; then
        echo "- ${BENCHES[$i]}: kiểm tra điểm SfM không đạt và ${MODE_LIST[0]} dùng loss độ sâu, không train lượt" \
          "gsplat nào"
      elif [ -n "${DROPPED[${BENCHES[$i]}]:-}" ]; then
        echo "- ${BENCHES[$i]}: kiểm tra điểm SfM không đạt, không train ${DROPPED[${BENCHES[$i]}]}"
      fi
    done
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
        echo "### So với $against (${BENCHES[$i]})"
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
if [ -n "$MONO_RUNS" ]; then  # truoc env_check: smoke test chay thu Depth Anything V2 tren GPU
  run_step depth_anything bash "$REPO_DIR/scripts/setup/setup_depth_anything.sh"
  if ! ok depth_anything; then
    if [[ " $MONO_RUNS " == *" ${MODE_LIST[0]} "* ]]; then
      echo "depth_anything lỗi và ${MODE_LIST[0]} dùng depth prior: không train lượt gsplat nào"
    else
      MONO_DROPPED="$MONO_RUNS"
      echo "depth_anything lỗi: bỏ $MONO_DROPPED trên mọi benchmark (cần Depth Anything V2); các lượt khác vẫn train"
    fi
  fi
fi
if [ -n "$MONO_RUNS" ] && ! ok depth_anything && [ -z "$MONO_DROPPED" ]; then
  skip_step env_check "depth_anything lỗi và ${MODE_LIST[0]} dùng depth prior"
else
  run_step env_check check_envs
fi
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
  if [ -n "$DEPTH_RUNS" ]; then  # truoc khi dung GPU: luot co loss do sau se loi (KeyError) neu co anh train khong co
    for i in "${!BENCHES[@]}"; do  # diem SfM nao; khong dat thi chi bo cac luot do tren benchmark nay
      if ok "fetch_${BENCHES[$i]}"; then
        run_step "depth_check_${BENCHES[$i]}" depth_check "$i"
        drop_depth_runs "$i"
      else
        skip_step "depth_check_${BENCHES[$i]}" "cần fetch_${BENCHES[$i]} thành công"
      fi
    done
  fi
  # bien moi truong cua nguoi dung khong duoc doi ke hoach: thu muc ket qua, cau hinh, tham so mcmc, bien the, chay thu;
  # CHARTS=0: bang cua cac buoc B0 / gsplat ben trong khong ve bieu do rieng (bieu do ve mot lan o cuoi)
  CLEAN=(env -u OUT_ROOT -u B0_ROOT -u CONFIG -u MCMC_SET -u VARIANTS -u FORCE -u B0_CHECK QUICK=0 CHARTS=0)
  if [ -n "$ONLY" ] && [ "$ONLY" != b0 ]; then  # chay rieng mot che do: B0 khong thuoc lan chay nay
    CLEAN+=(B0_CHECK=if_present)
  fi
  for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    [ "${B0S[$i]}" = 1 ] || continue
    if ok "fetch_$bench"; then  # B0 (Inria) khong dung loss do sau: khong phu thuoc depth_check
      run_step "b0_$bench" "${CLEAN[@]}" SCENES="${SCENE_SETS[$i]}" bash "$REPO_DIR/scripts/run_train.sh" "$bench" b0
    else
      skip_step "b0_$bench" "cần fetch_$bench thành công"
    fi
  done
  STOP=""
  for seed in "${SEED_LIST[@]}"; do
    [ "${#MODE_LIST[@]}" -gt 0 ] || break  # chi B0 (--only b0): khong co luot gsplat
    if [ -z "$STOP" ] && ! room_for_seed "$seed"; then
      STOP="không đủ chỗ trống cho seed $seed (VOLUME_GB=$VOLUME)"
      echo "Dừng train: $STOP. Các lượt đã xong được giữ; tăng dung lượng volume, đặt VOLUME_GB rồi chạy lại để làm nốt."
    fi
    for i in "${!BENCHES[@]}"; do
      bench="${BENCHES[$i]}"
      if [ -n "$STOP" ]; then
        skip_step "gsplat_${bench}_s$seed" "$STOP"
      elif ready "$i"; then
        run_step "gsplat_${bench}_s$seed" "${CLEAN[@]}" MODES="$(bench_modes "$i")" VARIANTS="$(bench_variants "$i")" \
          SEEDS="$seed" SCENES="${SCENE_SETS[$i]}" bash "$REPO_DIR/scripts/run_train.sh" "$bench" gsplat
      else
        skip_step "gsplat_${bench}_s$seed" "$(needs "$i")"
      fi
      [ -n "$POST_FILTER" ] || continue
      if [ -n "${RC[gsplat_${bench}_s$seed]:-}" ]; then  # buoc gsplat da chay (ke ca khi co luot loi): loc luot da xong
        run_step "filter_${bench}_s$seed" post_filter "$i" "$seed"
      else
        skip_step "filter_${bench}_s$seed" "gsplat_${bench}_s$seed không chạy"
      fi
    done
  done
  for i in "${!BENCHES[@]}"; do
    bench="${BENCHES[$i]}"
    if [ "${#MODE_LIST[@]}" -eq 0 ]; then  # chi B0: bang B0 so voi so lieu Inria cong bo
      if [ -n "${RC[b0_$bench]:-}" ]; then
        run_step "report_$bench" b0_report "$i"
      else
        skip_step "report_$bench" "b0_$bench không chạy"
      fi
      continue
    fi
    modes="$(bench_modes "$i")"  # che do bi bo vi kiem tra diem SfM khong vao bang (khong thi moi canh deu "thieu")
    others="${modes#"${MODE_LIST[0]}"}"  # cac che do ngoai baseline, cho bang so voi tung moc trong also_against
    others="${others# }"
    # CHARTS=0: bieu do ve mot lan o cuoi cho ca lan chay (gop cac benchmark), khong ve cho tung bang (nhu CLEAN)
    run_step "report_$bench" env -u QUICK -u BASELINE CHARTS=0 MODES="$modes" TAGS="$VARIANT_TAGS" \
      SCENES="${SCENE_SETS[$i]}" REPORT="${REPORT}_$bench" bash "$REPO_DIR/scripts/run_evaluate.sh" "$bench"
    for against in "${AGAINST_LIST[@]}"; do
      if [[ " ${DROPPED[$bench]:-} " == *" $against "* ]]; then
        skip_step "report_${bench}_vs_$against" "$against không train trên $bench (kiểm tra điểm SfM không đạt)"
        continue
      fi
      if [[ " $MONO_DROPPED " == *" $against "* ]]; then
        skip_step "report_${bench}_vs_$against" "$against không train (bước depth_anything lỗi)"
        continue
      fi
      run_step "report_${bench}_vs_$against" env -u QUICK CHARTS=0 BASELINE="$against" MODES="$others" \
        TAGS="$VARIANT_TAGS" SCENES="${SCENE_SETS[$i]}" REPORT="${REPORT}_${bench}_vs_$against" \
        bash "$REPO_DIR/scripts/run_evaluate.sh" "$bench"
    done
  done
fi

write_summary
# bieu do tong cua ca lan chay (moi bang _<benchmark>[_vs_<moc>].json; cac bang cung moc duoc gop), muc "Bieu do"
# o cuoi $REPORT.md; truoc buoc goi ket qua de goi co ca anh
make_charts "$REPORT" "$REPORT"_*.json
run_step pack bash "$REPO_DIR/scripts/pod/pack_results.sh" "$PKG"

FAILED=0
printf '\n========== TÓM TẮT %s: thực nghiệm %s (%s)\n' "$RUN_NAME" "$NAME${ONLY:+ --only $ONLY}" "$STAMP"
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
[ ! -d "${REPORT}_charts" ] || echo "Biểu đồ: ${REPORT}_charts/ (ảnh PNG; cách đọc, bảng số: index.md)"
[ "${RC[pack]:-1}" = 0 ] && echo "Gói kết quả nhẹ để tải về: $PKG"
if [ "$FAILED" = 1 ]; then
  echo "Có bước lỗi hoặc bị bỏ qua: xem log của bước đó (gsplat_*: một lượt train lỗi, hoặc phần đối chiếu default với"
  echo "Inria chưa PASS). Chạy lại lệnh này sau khi sửa: các phần đã xong tự bỏ qua."
fi
exit "$FAILED"
