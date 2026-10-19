#!/usr/bin/env bash
# train_gsplat_modes.sh - Hai che do huan luyen gsplat tren mot benchmark (deep_blending | mipnerf360):
#   1. default (baseline B1) cho tung canh;
#   2. mcmc voi cung so Gaussian cuoi cua lan chay default (giao thuc cua paper 3DGS-MCMC);
#   3. bang so sanh default/mcmc; doi chieu default voi so lieu Inria cong bo (muc benchmark cua configs/<benchmark>.yaml)
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
#   SCENES="..."   chi chay cac canh nay (mac dinh: moi canh trong configs/<benchmark>.yaml)
#   MCMC_SET=...   tham so them cho mcmc, mac dinh benchmark.gsplat.mcmc_set (Deep Blending: opacity_reg=0.001)
#   WS=/workspace  ENV_NAME=gs-gsplat  CONFIG=<file JSON>   (du lieu phai co san, khong tai gi)
#   OUT_ROOT, B0_ROOT   mac dinh lay tu configs/<benchmark>.yaml (results.gsplat | gsplat_quick, results.inria_b0):
#                       $WS/outputs/benchmarks/<benchmark>/{gsplat,gsplat_quick,inria_b0}
# Ket qua: $OUT_ROOT/<canh>/<che_do>_s<seed>/; log va bao cao: $WS/outputs/logs, $WS/outputs/reports/modes_<tag>[_quick]_<thoi gian>.*

set -euo pipefail
export MPLBACKEND=Agg  # xem setup_gsplat.sh: MPLBACKEND cua Jupyter lam matplotlib bao loi trong env gs-gsplat

usage() { sed -n '2,22p' "$0"; exit 2; }
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

read -r -a SCENES <<< "${SCENES:-$(read_cfg scenes)}"
read -r -a SEED_LIST <<< "$SEEDS"
MCMC_EXTRA=()
if [ -n "$MCMC_SET" ]; then
  MCMC_EXTRA=(--set "$MCMC_SET")
fi
echo "benchmark: $BENCH | scenes: ${SCENES[*]} | seeds: ${SEED_LIST[*]} | mcmc extra: ${MCMC_EXTRA[*]-none}"
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
      echo "default failed for $scene (seed $seed); skipping its mcmc run"
      FAILED=1
      continue
    fi
    step "3. mcmc (same Gaussian count) | $scene | seed $seed"
    # --replace-stale: lan mcmc OK cu lay ngan sach tu mot lan default truoc (da train lai) thi chay lai
    python -m src.training.run --mode mcmc --scene "$src" --scene-name "$scene" \
      --out "$OUT_ROOT/$scene/mcmc_s$seed" --seed "$seed" --gsplat-dir "$WS/code/gsplat" --rerun-incomplete \
      --replace-stale --cap-max-from "$base" "${MCMC_EXTRA[@]}" "${SCENE_ARGS[@]}" "${RUN_ARGS[@]}" || FAILED=1
  done
done

# ---------------------------------------------------------------------------
REPORT="$REPORT_DIR/${NAME_TAG}_$STAMP"
[ -d "$B0_ROOT" ] || echo "WARN: no own Inria run in $B0_ROOT (run bash scripts/run_train.sh $BENCH b0): the B0 check will be INCOMPLETE"
if [ "$QUICK" = "1" ]; then
  step "4. Report of the quick runs (checks the whole report path; Inria checks are N/A at $QUICK_STEPS steps)"
  python -m src.evaluation.compare_modes --root "$OUT_ROOT" --baseline "$BASELINE_VARIANT" --modes default mcmc \
    --include-deviating --reference-config "$CONFIG" --b0-root "$B0_ROOT" \
    --out "$REPORT.json" --markdown "$REPORT.md" || FAILED=1
else
  step "4. Compare modes and check the baseline against Inria (published numbers and own B0 run)"
  # --strict: exit 1 neu default (gsplat) khong khop Inria trong nguong - phai xem truoc khi dung lam baseline
  if ! python -m src.evaluation.compare_modes --root "$OUT_ROOT" --baseline "$BASELINE_VARIANT" --modes default mcmc \
       --reference-config "$CONFIG" --b0-root "$B0_ROOT" --out "$REPORT.json" --markdown "$REPORT.md" --strict; then
    echo "WARN: the gsplat default runs do not match Inria (published or own B0) within tolerance, or a check is incomplete"
    FAILED=1
  fi
fi

echo
echo "DONE. Log: $LOG"
echo "Send back: $REPORT.md and .json (and $LOG if something failed)"
exit "$FAILED"
