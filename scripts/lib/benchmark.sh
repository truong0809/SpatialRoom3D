# shellcheck shell=bash
# benchmark.sh - ham dung chung cho scripts/pipeline/train_inria_b0.sh, train_gsplat_modes.sh va prepare_benchmark.sh
# (source, khong chay). Can san: REPO_DIR, CONFIG (ban JSON cua muc benchmark trong configs/<benchmark>.yaml, do
# scripts/lib/config.sh ghi ra), WS, BENCH, REPORT_DIR; va da source scripts/lib/config.sh.
# Duong dan du lieu va thu muc ket qua lay tu configs/<benchmark>.yaml (src.datasets.registry): mot noi duy nhat ghi
# du lieu nam dau, ghi de duoc bang configs/local.yaml hoac bien SR3D_DATA_<TEN> (docs/design/architecture.md).

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { echo "ERROR: $*"; exit 1; }

# doc ban JSON: python3 cua may la du (chi thu vien chuan); khong co thi dung python cua env dang dung
BENCH_PY="$(command -v python3 || command -v python || true)"
[ -n "$BENCH_PY" ] || { echo "ERROR: python3 not found"; exit 1; }

read_cfg() {  # read_cfg <dotted.key> [mac_dinh] -> gia tri trong ban JSON cua benchmark (danh sach: noi bang dau cach)
  "$BENCH_PY" - "$CONFIG" "$@" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    for key in sys.argv[2].split("."):
        value = value[key]
except (KeyError, TypeError):
    if len(sys.argv) > 3:
        print(sys.argv[3])
        sys.exit(0)
    sys.exit(f"missing key {sys.argv[2]!r} in {sys.argv[1]}")
if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, list):
    print(" ".join(map(str, value)))
else:
    print(value)
PY
}

registry_get() {  # registry_get <dotted.key>: truong cua luong trong configs/ (da thay ${WS} va ap ghi de)
  registry_value "$(read_cfg dataset)" "$1"
}

registry_sha256() {  # registry_sha256 <ten file>: SHA-256 mong doi cua file tai ve (rong neu registry khong ghi)
  local table
  table="$(registry_get source.sha256 2>/dev/null || echo '{}')"
  "$BENCH_PY" -c 'import json, sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "$table" "$1"
}

scene_factor() {  # scene_factor <canh>: he so thu nho anh cua canh (images_<N>); 1 = anh goc trong images/
  read_cfg "image_factor.$1" 1
}

has_scenes() {  # has_scenes <thu muc>: thu muc chua canh dau tien cua benchmark
  [ -d "$1/${SCENES[0]}" ]
}

registry_results() {  # registry_results <khoa>: thu muc ket qua cua benchmark (vd. gsplat, gsplat_quick, inria_b0)
  # goi trong $(...): loi in ra stderr de khong bi nuot vao bien
  registry_get "results.$1" \
    || { echo "ERROR: the registry has no results.$1 for $(read_cfg dataset) (see the error above)" >&2; exit 1; }
}

find_scene_parent() {  # find_scene_parent <thu muc>: thu muc chua canh dau tien cua benchmark (sau toi 3 cap), rong neu khong co
  # || true: find bao loi (thu muc khong doc duoc) hoac head dong ong som khong duoc lam dung script (set -e, pipefail)
  find "$1" -maxdepth 3 -type d -name "${SCENES[0]}" -print 2>/dev/null | head -n 1 | xargs -r -d '\n' dirname || true
}

locate_dataset() {  # dat DATA_ROOT; tai va giai nen chi khi registry cho phep (auto_download) hoac DOWNLOAD=1
  local root download_dir extract_to urls url zip expected actual found
  root="$(registry_get root)" || die "cannot get the data folder of $(read_cfg dataset) from the registry (see the error above)"
  if has_scenes "$root"; then
    DATA_ROOT="$root"
    return 0
  fi
  extract_to="$(registry_get source.extract_to 2>/dev/null || true)"
  # lan giai nen truoc da xong (co dau .extracted) nhung chua doi toi root (vd. bi ngat luc doi): dung o do, kem canh
  # bao. Khong co dau thi lan giai nen do co the do dang (bi ngat giua chung): khong dung, giai nen lai o duoi.
  if [ -n "$extract_to" ] && [ -f "$extract_to/.extracted" ]; then
    found="$(find_scene_parent "$extract_to")"
    if [ -n "$found" ]; then
      DATA_ROOT="$found"
      echo "WARN: dataset found at $DATA_ROOT, not at the registry root $root (move it there or fix the registry)"
      return 0
    fi
  fi
  if [ "$(registry_get source.auto_download 2>/dev/null || echo false)" != "true" ] && [ "${DOWNLOAD:-0}" != "1" ]; then
    die "dataset not found: $root (no ${SCENES[0]}/ inside). Put the data there, point the registry to it" \
        "(configs/local.yaml or SR3D_DATA_*), or set DOWNLOAD=1 to fetch: $(registry_get source.urls 2>/dev/null || true)"
  fi
  [ -n "$extract_to" ] || die "registry has no source.extract_to for $(read_cfg dataset)"
  if [ -e "$root" ] && [ -n "$(ls -A "$root" 2>/dev/null)" ]; then
    die "$root exists but has no ${SCENES[0]}/: not extracting over it (check what is there first)"
  fi
  download_dir="$(registry_get source.download_dir 2>/dev/null || echo "$WS/data/downloads")"
  read -r -a urls <<< "$(registry_get source.urls)"
  mkdir -p "$download_dir" "$extract_to" "$REPORT_DIR"
  for url in "${urls[@]}"; do
    zip="$download_dir/$(basename "$url")"
    if [ ! -s "$zip" ]; then
      curl -fL --retry 3 -o "$zip.part" "$url"
      mv "$zip.part" "$zip"
    fi
    actual="$(sha256sum "$zip" | cut -d' ' -f1)"
    echo "$actual  $(basename "$zip")" | tee "$REPORT_DIR/$(basename "$zip").sha256"
    expected="$(registry_sha256 "$(basename "$zip")")"
    if [ -n "$expected" ] && [ "$expected" != "$actual" ]; then
      die "SHA-256 of $zip is $actual, registry expects $expected (delete the file to download again)"
    fi
    "$BENCH_PY" -I -m zipfile -e "$zip" "$extract_to"   # thu vien chuan, -I: khong nap module tu thu muc hien tai
  done
  touch "$extract_to/.extracted"  # moi file nen da giai nen het (lan giai nen do dang thi khong co dau nay)
  # file nen co the co thu muc tren cung (tandt_db.zip: db/, tandt/): doi thu muc chua cac canh toi root
  found="$(find_scene_parent "$extract_to")"
  [ -n "$found" ] || die "cannot find ${SCENES[0]}/ inside $extract_to after extracting"
  mkdir -p "$(dirname "$root")"
  [ -d "$root" ] && rmdir "$root"  # thu muc rong (kiem tra o tren)
  mv "$found" "$root"
  echo "dataset extracted from $found to $root"
  has_scenes "$root" || die "cannot find ${SCENES[0]}/ in $root after extracting"
  DATA_ROOT="$root"
}

check_scene_layout() {  # check_scene_layout <canh>: images/, sparse/0/ va images_<N>/ khi he so N > 1
  local src="$DATA_ROOT/$1" factor
  factor="$(scene_factor "$1")"
  [ -d "$src/sparse/0" ] && [ -d "$src/images" ] \
    || die "unexpected dataset layout: $src/sparse/0 or $src/images not found"
  if [ "$factor" != 1 ]; then
    [ -d "$src/images_$factor" ] || die "unexpected dataset layout: $src/images_$factor not found"
  fi
}
