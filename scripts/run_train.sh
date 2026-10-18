#!/usr/bin/env bash
# run_train.sh - Huan luyen 3DGS (buoc "Train 3DGS" cua luong chinh). Hai che do trong configs/base.yaml (train_modes):
# default (DefaultStrategy, baseline) va mcmc (MCMCStrategy, cung so Gaussian cuoi voi default tren cung canh).
#
#   bash scripts/run_train.sh custom <ten_canh> [che_do ...] [-- tham so them]
#       canh tu chup da chuan bi (scripts/prepare_data.sh custom); mac dinh "default mcmc", lan luot
#       (scripts/pipeline/train_scene.sh cho tung che do; tham so sau -- di tiep toi src.training.run)
#   bash scripts/run_train.sh <benchmark> [gsplat | b0]         deep_blending | mipnerf360:
#       gsplat (mac dinh): default roi mcmc cho moi canh, bang so sanh   (scripts/pipeline/train_gsplat_modes.sh)
#       b0: 3DGS goc cua Inria, doi chieu so lieu cong bo               (scripts/pipeline/train_inria_b0.sh)
#
# Bien tuy chon: SEEDS="42 43 44" (benchmark), SEED=42 (canh tu chup), QUICK=1 (benchmark: 3000 vong), SCENES="..."
# Can GPU va env da cai (bash scripts/pod/run_all.sh --setup-only). Luot da co cung cau hinh tu bo qua.
# Chay nen, xep hang GPU: bash scripts/pod/job.sh start <ten> -- bash scripts/run_train.sh ...

set -uo pipefail

usage() { sed -n '2,14p' "$0"; exit 2; }
[ "$#" -ge 1 ] || usage
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="$1"
shift
case "$WHAT" in
  custom|custom_scene)
    [ "$#" -ge 1 ] || usage
    NAME="$1"
    shift
    MODES=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
      [[ "$1" =~ ^[A-Za-z0-9_]+$ ]] || { echo "ERROR: invalid mode name: $1 (extra options go after --)"; exit 2; }
      MODES+=("$1")
      shift
    done
    [ "$#" -eq 0 ] || shift  # bo dau --
    [ "${#MODES[@]}" -gt 0 ] || MODES=(default mcmc)
    for mode in "${MODES[@]}"; do
      bash "$REPO_DIR/scripts/pipeline/train_scene.sh" "$NAME" "$mode" "$@" || exit $?
    done
    ;;
  -h|--help) usage ;;
  *)
    [ "$#" -le 1 ] || usage
    case "${1:-gsplat}" in
      gsplat) exec bash "$REPO_DIR/scripts/pipeline/train_gsplat_modes.sh" "$WHAT" ;;
      b0) exec bash "$REPO_DIR/scripts/pipeline/train_inria_b0.sh" "$WHAT" ;;
      *) usage ;;
    esac
    ;;
esac
