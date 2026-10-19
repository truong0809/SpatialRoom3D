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

locate_dataset() {  # dat DATA_ROOT = thu muc goc cua benchmark trong configs/<benchmark>.yaml
  # du lieu do nguoi dung tu dua len (moi canh mot thu muc co images/ va sparse/0/)
  local root
  root="$(registry_get root)" || die "cannot get the data folder of $(read_cfg dataset) from the registry (see the error above)"
  has_scenes "$root" || die "dataset not found: $root (no ${SCENES[0]}/ inside). Upload the scenes there yourself" \
    "(${SCENES[*]}; each with images/ and sparse/0/), or point configs/local.yaml / SR3D_DATA_* to where they are"
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
