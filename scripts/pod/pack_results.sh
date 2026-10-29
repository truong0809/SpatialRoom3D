#!/usr/bin/env bash
# pack_results.sh - Dong goi ket qua nhe de tai ve tu pod (vai MB):
#   - $WS/outputs/reports, $WS/outputs/logs, $WS/outputs/jobs (bao cao kem anh bieu do *_charts/, log, trang thai viec chay nen);
#   - phan con lai cua $WS/outputs: JSON (ca *.jsonl, vd. nhat ky hook sr3d_hooks.jsonl), CSV, log, cau hinh
#     (cfg_args, *.yml, *.txt) va mot anh render moi lan chay, ke ca bao cao frame / COLMAP va log COLMAP cua canh
#     tu chup (outputs/custom_scenes/<canh>/colmap/).
# KHONG gom: checkpoint, file PLY (ke ca outputs/checkpoints/), video, anh / video goc, dataset (qua lon; PLY tai
# rieng khi can xem 3D).
# Bo qua thu muc *.prev_* (lan chay cu da duoc thay bang --force / --rerun-incomplete).
#
# Cach chay: bash scripts/pod/pack_results.sh [--name <ten ban>] [file_goi.tar.gz]
#   mac dinh: $WS/ket_qua_<thang><ngay>_<gio><phut><giay>.tar.gz (khong de goi truoc); dong cuoi in duong dan goi.
#   --name <ten ban>: $WS/ket_qua_<ten ban>_<nam><thang><ngay>_<gio><phut><giay>.tar.gz (vd. --name gsplat_mcmc, o goi
#   ket qua cua notebook cung ten); noi dung goi khong doi (moi ket qua nhe tren volume).

set -uo pipefail

WS="${WS:-/workspace}"
LABEL=""
if [ "${1:-}" = --name ]; then
  [[ "${2:-}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: --name needs a name (letters, digits, '.', '_', '-')"; exit 2; }
  LABEL="$2"
  shift 2
fi
if [ -n "$LABEL" ]; then
  DEFAULT_PKG="$WS/ket_qua_${LABEL}_$(date +%Y%m%d_%H%M%S).tar.gz"
else
  DEFAULT_PKG="$WS/ket_qua_$(date +%m%d_%H%M%S).tar.gz"
fi
PKG="${1:-$DEFAULT_PKG}"
case "$PKG" in
  /*) ;;
  *) PKG="$PWD/$PKG" ;;
esac
cd "$WS" || { echo "ERROR: cannot enter $WS"; exit 1; }

LIST="$(mktemp)"
trap 'rm -f "$LIST" "$PKG.part"' EXIT
{
  for d in outputs/reports outputs/logs outputs/jobs; do
    [ -d "$d" ] && find "$d" -type f ! -name '*.tar.gz' -print0
  done
  if [ -d outputs ]; then
    # khong theo symlink (find mac dinh): mot thu muc khong bi gom hai lan qua lien ket
    find outputs \( -path outputs/reports -o -path outputs/logs -o -path outputs/jobs -o -path outputs/checkpoints \
        -o -name '*.prev_*' -o -name '.images_*_png*' \) -prune -o -type f \( \
        -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.log' -o -name '*.yml' -o -name '*.txt' -o -name cfg_args \
        -o -path '*/renders/val_step*_0000.png' -o -path '*/ours_30000/*/00000.png' \) -print0
  fi
} > "$LIST" 2>/dev/null

COUNT="$(tr -cd '\0' < "$LIST" | wc -c)"
if [ "$COUNT" -eq 0 ]; then
  echo "WARN: nothing to pack in $WS (no reports, logs or outputs yet)"
fi
mkdir -p "$(dirname "$PKG")"
# tar exit 1 = co file dang duoc ghi trong luc dong goi (vd. log cua viec dang chay): goi van dung duoc.
# --ignore-failed-read: file bien mat giua chung (thu muc doi ten .prev_*, COLMAP lam lai) chi bi bo qua.
tar czf "$PKG.part" --null -T "$LIST" --warning=no-file-changed --ignore-failed-read
rc=$?
if [ "$rc" -gt 1 ]; then
  echo "ERROR: tar failed (exit $rc)"
  exit 1
fi
mv "$PKG.part" "$PKG"
echo "packed $COUNT files ($(du -h "$PKG" | cut -f1)); no checkpoints or PLY files"
echo "$PKG"
