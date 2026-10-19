#!/usr/bin/env bash
# train_inria_b0.sh - B0: 3DGS goc cua Inria tren mot benchmark (deep_blending | mipnerf360).
# Tham so huan luyen / render / metrics giong full_eval.py cua Inria; doc tu muc benchmark cua configs/<benchmark>.yaml.
# Chay lai an toan: canh nao da co results.json se duoc bo qua. FORCE=1: chay lai, luot cu doi ten <canh>.prev_<thoi gian>
# (khong ghi de).
#
# Cach chay (tren pod, sau scripts/setup/setup_3dgs_inria.sh; thuong goi qua scripts/run_train.sh <benchmark> b0):
#   bash scripts/pipeline/train_inria_b0.sh deep_blending
#   bash scripts/pipeline/train_inria_b0.sh mipnerf360   # du lieu phai co san (xem experiments/mipnerf360/README.md)
#
# Bien tuy chon:
#   WS=/workspace  ENV_NAME=gs-inria  FORCE=1
#   OUT_ROOT=<thu muc output>   mac dinh results.inria_b0 trong configs/<benchmark>.yaml
#                               ($WS/outputs/benchmarks/<benchmark>/inria_b0); moi canh mot thu muc <canh>/
#   CONFIG=<file JSON>          mac dinh: muc benchmark cua configs/<benchmark>.yaml, chup vao $WS/outputs/reports/configs/
#   SCENES="..."                chi chay cac canh nay (mac dinh: moi canh cua benchmark)
# Du lieu phai co san o cho configs/<benchmark>.yaml ghi (khong tai gi): thieu thi dung truoc khi train.

set -euo pipefail

usage() { sed -n '2,17p' "$0"; exit 2; }
[ "$#" -eq 1 ] || usage
BENCH="$1"
[[ "$BENCH" =~ ^[a-z0-9_]+$ ]] || usage
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${WS:-/workspace}"
ENV_NAME="${ENV_NAME:-gs-inria}"
CODE_DIR="$WS/code/gaussian-splatting"
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
OUT_ROOT="${OUT_ROOT:-$(registry_results inria_b0)}"

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/gpu_lock.sh"
gpu_lock_acquire "run_inria_b0 $BENCH"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUT_ROOT"
LOG="$LOG_DIR/baseline_${TAG}_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

MON_PID=""
# shellcheck disable=SC2329  # goi qua trap EXIT
cleanup() { if [ -n "$MON_PID" ]; then kill "$MON_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
step "0. Environment"
[ -d "$CODE_DIR" ] || die "missing $CODE_DIR - run scripts/setup/setup_3dgs_inria.sh first"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found (no GPU visible)"
# shellcheck source=/dev/null
. "$WS/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate "$ENV_NAME"
set -u
python -c "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; assert torch.cuda.is_available()" \
  || die "conda env $ENV_NAME is not ready - rerun scripts/setup/setup_3dgs_inria.sh"

read -r -a SCENES <<< "${SCENES:-$(read_cfg scenes)}"
read -r -a TRAIN_ARGS <<< "$(read_cfg train_args)"
read -r -a RENDER_ITERS <<< "$(read_cfg render_iterations)"
read -r -a RENDER_ARGS <<< "$(read_cfg render_args)"
METHOD="$(read_cfg method)"
has_results() {  # results.json co du chi so cua lan danh gia can dung (vd. ours_30000)
  python - "$1/results.json" "$METHOD" <<'PY'
import json
import sys
try:
    values = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], {})
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if all(k in values for k in ("PSNR", "SSIM", "LPIPS")) else 1)
PY
}
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
echo "benchmark: $BENCH | config: $CONFIG"
echo "scenes: ${SCENES[*]} | GPU: ${GPU_NAME:-?} | output: $OUT_ROOT"

# ---------------------------------------------------------------------------
step "1. Dataset"
locate_dataset
echo "dataset root: $DATA_ROOT"
for scene in "${SCENES[@]}"; do
  check_scene_layout "$scene"
  echo "$scene: $(find "$DATA_ROOT/$scene/images" -maxdepth 1 -type f | wc -l) images | image factor $(scene_factor "$scene")"
done

# ---------------------------------------------------------------------------
REPO_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
# repo khong phai git (vd. giai nen tu ZIP khong co .git): pipefail khong duoc lam dung script
REPO_DIRTY="$( (git -C "$REPO_DIR" status --porcelain 2>/dev/null || true) | head -c 1 | wc -c)"
INRIA_COMMIT_NOW="$(git -C "$CODE_DIR" rev-parse HEAD)"

for scene in "${SCENES[@]}"; do
  src="$DATA_ROOT/$scene"
  model="$OUT_ROOT/$scene"
  if has_results "$model"; then
    if [ "${FORCE:-0}" != "1" ]; then
      step "$scene: results.json has $METHOD, skipping (FORCE=1 to rerun)"
      continue
    fi
    mv "$model" "$model.prev_$STAMP"  # FORCE=1: giu lai luot cu, khong ghi de (nhu src.training.run)
    echo "previous run kept as $model.prev_$STAMP"
  fi
  mkdir -p "$model"
  factor="$(scene_factor "$scene")"
  SCENE_ARGS=()
  if [ "$factor" != 1 ]; then
    SCENE_ARGS=(-i "images_$factor")  # nhu full_eval.py: Mip-NeRF 360 dung anh thu nho san images_4 / images_2
  fi

  step "2. Train $scene"
  # tien trinh ghi log GPU khong mang theo khoa GPU (xem lib/gpu_lock.sh)
  if [[ "${SR3D_GPU_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits -l 2 > "$model/gpu_log.csv" 2>/dev/null {SR3D_GPU_LOCK_FD}>&- &
  else
    nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu \
      --format=csv,noheader,nounits -l 2 > "$model/gpu_log.csv" 2>/dev/null &
  fi
  MON_PID=$!
  start_iso="$(date -Is)"
  t0="$(date +%s)"
  (cd "$CODE_DIR" && python train.py -s "$src" "${SCENE_ARGS[@]}" -m "$model" "${TRAIN_ARGS[@]}")
  t1="$(date +%s)"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  python - "$model/run_meta.json" "$scene" "$CONFIG" "${TRAIN_ARGS[*]}" "$start_iso" \
    "$((t1 - t0))" "${GPU_NAME:-unknown}" "$INRIA_COMMIT_NOW" "$REPO_COMMIT" "$REPO_DIRTY" "${SCENE_ARGS[*]}" \
    "$BENCH" <<'PY'
import json
import sys

import torch

out, scene, config, train_args, start, seconds, gpu, inria, repo, dirty, scene_args, bench = sys.argv[1:13]
meta = {
    "scene": scene,
    "benchmark": bench,
    "config": config,
    "train_args": train_args,
    "scene_args": scene_args,
    "train_start": start,
    "train_seconds": int(seconds),
    "gpu": gpu,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "inria_commit": inria,
    "repo_commit": repo,
    "repo_dirty": dirty.strip() != "0",
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2)
print("train time: %.1f min" % (meta["train_seconds"] / 60))
PY

  step "3. Render $scene (iterations ${RENDER_ITERS[*]})"
  for it in "${RENDER_ITERS[@]}"; do
    # render.py doc lai thu muc anh (-i) tu cfg_args cua model, nhu full_eval.py
    (cd "$CODE_DIR" && python render.py --iteration "$it" -s "$src" -m "$model" "${RENDER_ARGS[@]}")
  done

  step "4. Metrics $scene"
  (cd "$CODE_DIR" && python metrics.py -m "$model")
  has_results "$model" || die "metrics.py did not write $METHOD into $model/results.json"
done

# ---------------------------------------------------------------------------
step "5. Compare with Inria reference numbers"
SUMMARY="$REPORT_DIR/baseline_${TAG}_summary_$STAMP.json"
VERDICT=0
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python -m src.evaluation.compare_reference --config "$CONFIG" --model-root "$OUT_ROOT" --out "$SUMMARY" \
  --strict || VERDICT=$?

echo
if [ "$VERDICT" != 0 ]; then
  echo "B0 does NOT match the Inria reference within tolerance (or a scene is missing): see $SUMMARY"
fi
echo "DONE. Log: $LOG"
echo "Send back: $SUMMARY (and $LOG if something failed)"
exit "$VERDICT"
