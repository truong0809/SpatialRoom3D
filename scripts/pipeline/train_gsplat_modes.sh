#!/usr/bin/env bash
# train_gsplat_modes.sh - Cac che do huan luyen gsplat (configs/base.yaml, train_modes) tren mot benchmark
# (deep_blending | mipnerf360):
#   1. default (baseline B1) cho tung canh;
#   2. cac che do con lai (mac dinh: mcmc) voi cung so Gaussian cuoi cua lan chay default (giao thuc cua paper 3DGS-MCMC),
#      roi cac bien the tham so (VARIANTS), cung ngan sach Gaussian;
#   3. bang so sanh cac che do; doi chieu default voi so lieu Inria cong bo (muc benchmark cua configs/<benchmark>.yaml)
#      va voi luot Inria tu chay cua scripts/pipeline/train_inria_b0.sh (tung canh), neu da co.
# Chay lai an toan: lan chay nao da co summary.json OK voi cung cau hinh se duoc bo qua.
#
# Cach chay (tren pod, sau scripts/setup/setup_gsplat.sh; thuong goi qua scripts/run_train.sh <benchmark> gsplat):
#   QUICK=1 bash scripts/pipeline/train_gsplat_modes.sh deep_blending   # chay thu toan bo luong (3000 vong)
#   bash scripts/pipeline/train_gsplat_modes.sh deep_blending           # luot day du (30000 vong)
#   bash scripts/pipeline/train_gsplat_modes.sh mipnerf360              # du lieu phai co san (experiments/mipnerf360/)
#
# Bien tuy chon:
#   QUICK=1        3000 vong, ket qua vao thu muc gsplat_quick; chi kiem tra chay duoc (doi chieu Inria: N/A)
#   SEEDS="42"     vd. SEEDS="42 43 44" de chay lap (docs/PROJECT_PLAN.md muc 4.2)
#   MODES="default mcmc"   cac che do (ten trong configs/base.yaml); phai co default, default luon chay truoc
#   SCENES="..."   chi chay cac canh nay (mac dinh: moi canh trong configs/<benchmark>.yaml); khi chi la mot phan cac
#                  canh, doi chieu voi so cong bo (trung binh moi canh) ghi SKIPPED, B0 chi doi chieu khi co du canh
#   MCMC_SET=...   tham so them cho moi che do preset mcmc, mac dinh benchmark.gsplat.mcmc_set (Deep Blending: opacity_reg=0.001);
#                  mot cap <khoa>=<gia_tri>; che do tu dat khoa do trong configs/base.yaml (args) thi giu gia tri cua che do
#   VARIANTS="mcmc:or0001:opacity_reg=0.001 mcmc:rbg:random_bkgd=true"
#                  bien the tham so (thuong do scripts/pod/run_experiment.sh dat theo muc variants cua ke hoach):
#                  <che_do>:<tag>:<khoa>=<gia_tri>[,<khoa>=<gia_tri>...]; che do phai co trong MODES (khong phai
#                  default); tham so them sau MCMC_SET; ket qua vao <che_do>_<tag>_s<seed>
#   WS=/workspace  ENV_NAME=gs-gsplat  CONFIG=<file JSON>   (du lieu phai co san)
#   OUT_ROOT, B0_ROOT   mac dinh lay tu configs/<benchmark>.yaml (results.gsplat | gsplat_quick, results.inria_b0):
#                       $WS/outputs/benchmarks/<benchmark>/{gsplat,gsplat_quick,inria_b0}
# Ket qua: $OUT_ROOT/<canh>/<che_do>[_<tag>]_s<seed>/; log va bao cao: $WS/outputs/logs,
#          $WS/outputs/reports/modes_<tag benchmark>[_quick]_<thoi gian>.*

set -euo pipefail
export MPLBACKEND=Agg  # xem setup_gsplat.sh: MPLBACKEND cua Jupyter lam matplotlib bao loi trong env gs-gsplat

usage() { sed -n '2,32p' "$0"; exit 2; }
[ "$#" -eq 1 ] || usage
BENCH="$1"
[[ "$BENCH" =~ ^[a-z0-9_]+$ ]] || usage
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-gsplat}"
LOG_DIR="$WS/outputs/logs"
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
TAG="$(read_cfg tag)"
QUICK="${QUICK:-0}"
QUICK_STEPS=3000
RUN_ARGS=()
NAME_TAG="modes_$TAG"
RESULTS_KEY="gsplat"
BASELINE_VARIANT="default"
if [ "$QUICK" = "1" ]; then
  RUN_ARGS=(--protocol "max_steps=$QUICK_STEPS")  # ghi la sai khac protocol; khong bao gio tron voi luot chuan
  NAME_TAG="modes_${TAG}_quick"
  RESULTS_KEY="gsplat_quick"
  BASELINE_VARIANT="default{max_steps=$QUICK_STEPS}"  # ten bien the ma compare_modes gan cho luot default chay thu
fi
OUT_ROOT="${OUT_ROOT:-$(registry_results "$RESULTS_KEY")}"
B0_ROOT="${B0_ROOT:-$(registry_results inria_b0)}"
SEEDS="${SEEDS:-42}"
MCMC_SET="${MCMC_SET-$(read_cfg gsplat.mcmc_set "")}"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a MODE_ARGS <<< "${MODES:-default mcmc}"
MODE_LIST=(default)  # default (baseline) luon chay truoc: cac che do khac lay ngan sach Gaussian tu no
HAS_DEFAULT=0
for mode in "${MODE_ARGS[@]}"; do
  [[ "$mode" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid mode name in MODES: $mode"
  if [ "$mode" = default ]; then HAS_DEFAULT=1; continue; fi
  [[ " ${MODE_LIST[*]} " == *" $mode "* ]] || MODE_LIST+=("$mode")
done
[ "$HAS_DEFAULT" = 1 ] || die "MODES must include default (the baseline, and the Gaussian budget of the other modes)"
# bien the tham so: <che_do>:<tag>:<khoa>=<gia_tri>[,...] (scripts/pod/run_experiment.sh, muc variants cua ke hoach)
read -r -a VARIANT_ARGS <<< "${VARIANTS:-}"
VARIANT_MODE=() VARIANT_TAG=() VARIANT_SETS=()
for entry in "${VARIANT_ARGS[@]}"; do
  IFS=: read -r vmode vtag vsets vrest <<< "$entry"
  if ! [[ "$vmode" =~ ^[A-Za-z0-9_]+$ && "$vtag" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ && -n "$vsets" && -z "$vrest" ]]; then
    die "invalid entry in VARIANTS: $entry (expected <mode>:<tag>:<key>=<value>[,<key>=<value>...])"
  fi
  [ "$vmode" != default ] || die "VARIANTS: $entry is a variant of default, the baseline (not supported)"
  [[ " ${MODE_LIST[*]} " == *" $vmode "* ]] || die "VARIANTS: the mode of $entry is not in MODES (${MODE_LIST[*]})"
  [[ " ${MODE_LIST[*]} " != *" ${vmode}_$vtag "* ]] || die "VARIANTS: ${vmode}_$vtag is also a mode in MODES"
  for other in "${!VARIANT_MODE[@]}"; do
    [ "${VARIANT_MODE[$other]}_${VARIANT_TAG[$other]}" != "${vmode}_$vtag" ] || die "VARIANTS: ${vmode}_$vtag is listed twice"
  done
  IFS=, read -r -a pairs <<< "$vsets"
  for pair in "${pairs[@]}"; do
    [[ "$pair" =~ ^[A-Za-z0-9_.-]+=[A-Za-z0-9_.+-]+$ ]] || die "VARIANTS: invalid parameter '$pair' in $entry"
  done
  VARIANT_MODE+=("$vmode")
  VARIANT_TAG+=("$vtag")
  VARIANT_SETS+=("$vsets")
done
[ "${#SEED_LIST[@]}" -gt 0 ] || die "SEEDS is empty"
for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "SEEDS must be non-negative integers: $seed"
done

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_gsplat_modes $BENCH"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUT_ROOT"
LOG="$LOG_DIR/${NAME_TAG}_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------
step "0. Environment"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
# chay thu fused-ssim tren GPU: ban 328dc98 chi duoc build cho kien truc GPU luc cai dat
python -c "import torch, fused_ssim; from gsplat import csrc; x = torch.rand(1, 3, 16, 16, device='cuda'); fused_ssim.fused_ssim(x, x)" \
  || die "conda env $ENV_NAME is not ready for this GPU - run scripts/setup/setup_gsplat.sh (it rebuilds when the GPU type changes)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
# preset cua tung che do (configs/base.yaml) va tham so them cua benchmark cho no: MCMC_SET chi ap cho che do preset
# mcmc, tru khi chinh che do da dat khoa do (vd. mot che do da chon opacity_reg rieng); ten sai thi dung truoc khi train
PRESETS="$(python - "$REPO_DIR/configs/base.yaml" "$MCMC_SET" "${MODE_LIST[@]}" <<'PY'
import sys
from src.training.profiles import load_config, normalize_flag  # PyYAML co trong env gs-gsplat
modes = load_config(sys.argv[1])["modes"]
convention, names = sys.argv[2], sys.argv[3:]
if convention and ("=" not in convention or len(convention.split()) != 1):
    sys.exit(f"MCMC_SET must be one key=value pair, got {convention!r}")
key = normalize_flag(convention.split("=", 1)[0]) if convention else None
unknown = [name for name in names if name not in modes]
if unknown:
    sys.exit(f"unknown mode(s) {unknown} in MODES; configs/base.yaml has {sorted(modes)}")
for name in names:
    entry = modes[name]
    own = {normalize_flag(arg) for arg in entry.get("args") or {}}
    use = convention and entry.get("preset") == "mcmc" and key not in own
    print(name, entry.get("preset"), convention if use else "-")
PY
)" || die "cannot use MODES=${MODE_LIST[*]} (see above)"
declare -A PRESET=() MODE_EXTRA=()
while read -r name preset extra; do
  PRESET[$name]="$preset"
  MODE_EXTRA[$name]="$extra"  # "-": khong them gi
done <<< "$PRESETS"

read -r -a SCENES <<< "${SCENES:-$(read_cfg scenes)}"
read -r -a ALL_SCENES <<< "$(read_cfg scenes)"
SUBSET=0  # chi chay mot phan cac canh cua benchmark
for scene in "${ALL_SCENES[@]}"; do
  [[ " ${SCENES[*]} " == *" $scene "* ]] || SUBSET=1
done
MCMC_EXTRA=()
if [ -n "$MCMC_SET" ]; then
  MCMC_EXTRA=(--set "$MCMC_SET")
fi
echo "benchmark: $BENCH | scenes: ${SCENES[*]} | seeds: ${SEED_LIST[*]} | modes: ${MODE_LIST[*]} | mcmc extra: ${MCMC_EXTRA[*]-none}"
for mode in "${MODE_LIST[@]:1}"; do
  if [ "${PRESET[$mode]}" = mcmc ] && [ -n "$MCMC_SET" ] && [ "${MODE_EXTRA[$mode]}" = - ]; then
    echo "  $mode sets ${MCMC_SET%%=*} itself (configs/base.yaml): MCMC_SET is not added to it"
  fi
done
if [ "${#VARIANT_ARGS[@]}" -gt 0 ]; then
  echo "variants: ${VARIANT_ARGS[*]}"
fi
echo "output: $OUT_ROOT"
if [ "$QUICK" = "1" ]; then
  echo "QUICK mode: ${RUN_ARGS[*]} (smoke test of the whole flow, numbers are not results)"
fi

# ---------------------------------------------------------------------------
step "1. Dataset"
locate_dataset
echo "dataset root: $DATA_ROOT"
for scene in "${SCENES[@]}"; do
  check_scene_layout "$scene"
done

# ---------------------------------------------------------------------------
FAILED=0
for seed in "${SEED_LIST[@]}"; do
  for scene in "${SCENES[@]}"; do
    src="$DATA_ROOT/$scene"
    base="$OUT_ROOT/$scene/default_s$seed"
    factor="$(scene_factor "$scene")"
    SCENE_ARGS=()
    if [ "$factor" != 1 ]; then
      # quy uoc cua benchmark (khong phai sai khac protocol): gsplat doc anh thu nho images_<N> (data_factor N)
      SCENE_ARGS=(--dataset-protocol "data_factor=$factor")
    fi
    step "2. default | $scene | seed $seed"
    if ! python -m src.training.run --mode default --scene "$src" --scene-name "$scene" --out "$base" \
         --seed "$seed" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete "${SCENE_ARGS[@]}" "${RUN_ARGS[@]}"; then
      echo "default failed for $scene (seed $seed); skipping the other modes of this scene and seed"
      FAILED=1
      continue
    fi
    for mode in "${MODE_LIST[@]:1}"; do
      step "3. $mode (same Gaussian count) | $scene | seed $seed"
      EXTRA=()
      if [ "${MODE_EXTRA[$mode]}" != - ]; then
        EXTRA=(--set "${MODE_EXTRA[$mode]}")
      fi
      # --replace-stale: lan OK cu lay ngan sach tu mot lan default truoc (da train lai) thi chay lai
      python -m src.training.run --mode "$mode" --scene "$src" --scene-name "$scene" \
        --out "$OUT_ROOT/$scene/${mode}_s$seed" --seed "$seed" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete \
        --replace-stale --cap-max-from "$base" "${EXTRA[@]}" "${SCENE_ARGS[@]}" "${RUN_ARGS[@]}" || FAILED=1
    done
    for v in "${!VARIANT_MODE[@]}"; do
      mode="${VARIANT_MODE[$v]}"
      tag="${VARIANT_TAG[$v]}"
      step "3. $mode, variant $tag (same Gaussian count) | $scene | seed $seed"
      EXTRA=()
      if [ "${MODE_EXTRA[$mode]}" != - ]; then
        EXTRA=(--set "${MODE_EXTRA[$mode]}")
      fi
      IFS=, read -r -a pairs <<< "${VARIANT_SETS[$v]}"
      for pair in "${pairs[@]}"; do
        EXTRA+=(--set "$pair")  # sau MCMC_SET: tham so cua bien the duoc uu tien
      done
      python -m src.training.run --mode "$mode" --scene "$src" --scene-name "$scene" \
        --out "$OUT_ROOT/$scene/${mode}_${tag}_s$seed" --seed "$seed" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete \
        --replace-stale --cap-max-from "$base" --tag "$tag" "${EXTRA[@]}" "${SCENE_ARGS[@]}" "${RUN_ARGS[@]}" \
        || FAILED=1
    done
  done
done

# ---------------------------------------------------------------------------
REPORT="$REPORT_DIR/${NAME_TAG}_$STAMP"
CHECK_ARGS=(--reference-config "$CONFIG" --scenes "${SCENES[@]}")
if [ "${#VARIANT_TAG[@]}" -gt 0 ]; then
  CHECK_ARGS+=(--tags "${VARIANT_TAG[@]}")  # cac bien the vua train; luot co tag khac (ke hoach khac) bi de ngoai
fi
if [ "$SUBSET" = 0 ]; then
  [ -d "$B0_ROOT" ] || echo "WARN: no own Inria run in $B0_ROOT (run bash scripts/run_train.sh $BENCH b0): the B0 check will be INCOMPLETE"
  CHECK_ARGS+=(--b0-root "$B0_ROOT")
else
  B0_MISSING=()
  for scene in "${SCENES[@]}"; do
    [ -f "$B0_ROOT/$scene/results.json" ] || B0_MISSING+=("$scene")
  done
  if [ "${#B0_MISSING[@]}" -eq 0 ]; then
    CHECK_ARGS+=(--b0-root "$B0_ROOT")
  else
    echo "NOTE: part of the benchmark's scenes only, without an own Inria B0 run for ${B0_MISSING[*]}: no B0 check"
  fi
fi
if [ "$QUICK" = "1" ]; then
  step "4. Report of the quick runs (checks the whole report path; Inria checks are N/A at $QUICK_STEPS steps)"
  python -m src.evaluation.compare_modes --root "$OUT_ROOT" --baseline "$BASELINE_VARIANT" --modes "${MODE_LIST[@]}" \
    --include-deviating "${CHECK_ARGS[@]}" --out "$REPORT.json" --markdown "$REPORT.md" || FAILED=1
else
  step "4. Compare modes and check the baseline against Inria (published numbers and own B0 run)"
  # --strict: exit 1 neu default (gsplat) khong khop Inria trong nguong - phai xem truoc khi dung lam baseline
  if ! python -m src.evaluation.compare_modes --root "$OUT_ROOT" --baseline "$BASELINE_VARIANT" --modes "${MODE_LIST[@]}" \
       "${CHECK_ARGS[@]}" --out "$REPORT.json" --markdown "$REPORT.md" --strict; then
    echo "WARN: the gsplat default runs do not match Inria (published or own B0) within tolerance, or a check is incomplete"
    FAILED=1
  fi
fi

echo
echo "DONE. Log: $LOG"
echo "Send back: $REPORT.md and .json (and $LOG if something failed)"
exit "$FAILED"
