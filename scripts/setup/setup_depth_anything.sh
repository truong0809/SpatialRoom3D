#!/usr/bin/env bash
# setup_depth_anything.sh - Code va trong so Depth Anything V2 Small cho bien the mono (depth prior,
# src/training/mono_depth.py) vao $WS/code/depth_anything_v2 (network volume, con nguyen sau khi Stop pod).
# Chi tai code (git, dung commit ghim) va trong so (kiem SHA-256); KHONG cai goi Python nao: code chay trong env
# gs-gsplat (torch, torchvision, opencv da co). Chay lai an toan: da du thi chi kiem tra lai.
#
# Cach chay (tu thu muc goc cua repo tren pod):
#   bash scripts/setup/setup_depth_anything.sh
# scripts/pod/run_experiment.sh tu goi script nay khi ke hoach co luot dung sr3d.mono.
#
# Bien tuy chon: WS=/workspace
# Phien ban ghim doc tu configs/environment/gs-gsplat.conf (DA2_*).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONF="${CONF:-$REPO_DIR/configs/environment/gs-gsplat.conf}"
[ -r "$CONF" ] || { echo "ERROR: missing $CONF" >&2; exit 1; }
# shellcheck source=/dev/null
. "$CONF"
for v in DA2_REPO DA2_COMMIT DA2_WEIGHTS_FILE DA2_WEIGHTS_URL DA2_WEIGHTS_SHA256; do
  [ -n "${!v:-}" ] || { echo "ERROR: $v is missing in $CONF" >&2; exit 1; }
done

WS="${WS:-/workspace}"
DIR="$WS/code/depth_anything_v2"
WEIGHTS="$DIR/$DA2_WEIGHTS_FILE"

die() { echo "ERROR: $*" >&2; exit 1; }
sha_ok() { [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d ' ' -f 1)" = "$DA2_WEIGHTS_SHA256" ]; }
# viec nen khong co ai tra loi: git khong hoi mat khau; duoi 1 KB/s suot 60 giay thi git ngat, duoi 100 KB/s suot
# 5 phut thi curl ngat roi thu lai (nhu src/datasets/fetch.py), khong treo ca viec truoc khi train
git_net() { GIT_TERMINAL_PROMPT=0 git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 "$@"; }

command -v git > /dev/null 2>&1 || die "git not found"
command -v curl > /dev/null 2>&1 || die "curl not found"
mkdir -p "$WS/code"

echo "== Depth Anything V2 code @ ${DA2_COMMIT:0:7} in $DIR"
if [ -d "$DIR" ] && [ ! -d "$DIR/.git" ] && [ -z "$(ls -A "$DIR")" ]; then
  rmdir "$DIR"  # thu muc rong: khong co gi de giu
fi
if [ -e "$DIR" ] && [ ! -d "$DIR/.git" ]; then
  die "$DIR exists but is not a git checkout (e.g. weights copied there by hand): move it away, then rerun"
fi
if [ ! -d "$DIR/.git" ]; then
  rm -rf "$DIR.partial"
  # clone bi ngat giua chung khong de lai thu muc trong nhu da xong
  git_net clone "$DA2_REPO" "$DIR.partial" || { rm -rf "$DIR.partial"; die "cannot clone $DA2_REPO"; }
  mv "$DIR.partial" "$DIR"
fi
if [ "$(git -C "$DIR" rev-parse HEAD)" != "$DA2_COMMIT" ]; then
  git_net -C "$DIR" fetch origin || die "cannot fetch $DA2_REPO"
  git -C "$DIR" checkout --detach "$DA2_COMMIT" || die "commit $DA2_COMMIT not found in $DA2_REPO"
fi
[ "$(git -C "$DIR" rev-parse HEAD)" = "$DA2_COMMIT" ] || die "$DIR is not at the pinned commit $DA2_COMMIT"
if [ -n "$(git -C "$DIR" status --porcelain --untracked-files=no)" ]; then
  die "$DIR has local modifications; the depth prior expects the unmodified code"
fi

echo "== weights $DA2_WEIGHTS_FILE (SHA-256 ${DA2_WEIGHTS_SHA256:0:12}...)"
if sha_ok "$WEIGHTS"; then
  echo "already downloaded"
else
  mkdir -p "$(dirname "$WEIGHTS")"
  rm -f "$WEIGHTS.part"
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 --speed-limit 102400 --speed-time 300 \
    -o "$WEIGHTS.part" "$DA2_WEIGHTS_URL" || { rm -f "$WEIGHTS.part"; die "cannot download $DA2_WEIGHTS_URL"; }
  if ! sha_ok "$WEIGHTS.part"; then
    rm -f "$WEIGHTS.part"
    die "the downloaded weights do not have the pinned SHA-256 $DA2_WEIGHTS_SHA256"
  fi
  mv "$WEIGHTS.part" "$WEIGHTS"
fi
echo "Depth Anything V2 Small ready: $DIR (commit ${DA2_COMMIT:0:7}, $DA2_WEIGHTS_FILE)"
