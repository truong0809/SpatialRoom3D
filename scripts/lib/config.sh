# shellcheck shell=bash
# config.sh - Doc cau hinh configs/*.yaml tu script bash (source file nay). Can san REPO_DIR va WS.
# configs/ la noi duy nhat ghi du lieu va ket qua nam dau; ghi de tren mot pod bang configs/local.yaml hoac bien
# SR3D_DATA_<TEN> (docs/design/architecture.md). Can Python co PyYAML.
#
#   config_python                                 python co PyYAML: python3 cua may luc source file nay (template
#                                                 Runpod thuong co, vi JupyterLab can PyYAML; o 1 cua notebook kiem
#                                                 tra), neu khong thi python cua env gs-gsplat
#   registry_value <luong> <khoa>                 vd. registry_value custom_scene results.gsplat
#                                                 in gia tri (da thay ${WS}); loi in ra stderr, tra ve 1
#   config_export_benchmark <luong> <file.json>   ghi phan benchmark cua luong ra JSON: code chay trong env khong co
#                                                 PyYAML (gs-inria) doc ban nay; ban nay o lai trong outputs/reports/configs/
#   b0_missing_scenes <thu muc B0> <method> <canh>...
#                                                 in (noi bang dau cach) cac canh chua co ket qua B0 du chi so, cung
#                                                 tieu chi voi bang so sanh (src.evaluation.compare_reference.load_scene)
#   make_charts <ten> <bao cao .json>...          bieu do tong (src.evaluation.charts: doi chieu Inria, chenh lech tong so
#                                                 voi moc; nhieu bao cao cung moc thi gop) vao <ten>_charts/ (anh PNG,
#                                                 index.md co cach doc va bang so), them muc "Bieu do" vao cuoi <ten>.md
#                                                 khi file do co. Can matplotlib: python cua env gs-gsplat, khong co thi
#                                                 python3 cua may; khong co thi chi in NOTE. CHARTS=0: bo qua. Luon tra ve
#                                                 0: bieu do loi chi in WARN, bang trong bao cao khong doi. Chay voi
#                                                 MPLBACKEND=Agg (MPLBACKEND cua Jupyter lam matplotlib bao loi khi import,
#                                                 nhu setup_gsplat.sh)

# python3 cua may, ghi lai luc source file nay: cac script source truoc khi kich hoat env conda, ma sau
# `conda activate gs-inria` thi python3 la cua env do (khong co PyYAML)
SR3D_SYSTEM_PYTHON="${SR3D_SYSTEM_PYTHON:-$(command -v python3 || true)}"
export SR3D_SYSTEM_PYTHON  # cac script con (chay sau khi kich hoat env) dung lai lua chon nay

config_python() {
  local py
  for py in "$SR3D_SYSTEM_PYTHON" "$(command -v python3 || true)" "$WS/miniforge3/envs/gs-gsplat/bin/python" \
            /usr/bin/python3 /usr/local/bin/python3; do
    if [ -n "$py" ] && [ -x "$py" ] && "$py" -c 'import yaml' 2> /dev/null; then
      echo "$py"
      return 0
    fi
  done
  echo "ERROR: no Python with PyYAML to read configs/*.yaml (fix: python3 -m pip install pyyaml)" >&2
  return 1
}

registry_value() {
  local py
  py="$(config_python)" || return 1
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$py" -m src.datasets.registry \
    --configs "$REPO_DIR/configs" --ws "$WS" --get "$1" "$2"
}

config_export_benchmark() {
  local py
  py="$(config_python)" || return 1
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$py" -m src.datasets.registry \
    --configs "$REPO_DIR/configs" --ws "$WS" --export-benchmark "$1" --out "$2" > /dev/null
}

b0_missing_scenes() {
  local py
  py="$(config_python)" || return 1
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$@" <<'PY'
import sys
from pathlib import Path

from src.evaluation.compare_reference import load_scene

root, method = Path(sys.argv[1]), sys.argv[2]


def done(scene):
    try:
        return load_scene(scene, root / scene, method, 0).status == "OK"
    except (TypeError, ValueError):  # vd. chỉ số null trong results.json: như chưa có kết quả
        return False


print(" ".join(scene for scene in sys.argv[3:] if not done(scene)))
PY
}

make_charts() {
  local name="$1" py="" candidate json reports=() args
  shift
  [ "${CHARTS:-1}" != 0 ] || return 0
  for json in "$@"; do
    [ ! -f "$json" ] || reports+=("$json")
  done
  [ "${#reports[@]}" -gt 0 ] || return 0
  for candidate in "$WS/miniforge3/envs/gs-gsplat/bin/python" "$SR3D_SYSTEM_PYTHON" "$(command -v python3 || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ] && MPLBACKEND=Agg "$candidate" -c 'import matplotlib' 2> /dev/null; then
      py="$candidate"
      break
    fi
  done
  if [ -z "$py" ]; then
    echo "NOTE: no Python with matplotlib: no charts for $(basename "$name") (the tables are in the report)"
    return 0
  fi
  args=(--report "${reports[@]}" --out "${name}_charts")
  [ ! -f "$name.md" ] || args+=(--append-to "$name.md")
  MPLBACKEND=Agg PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$py" -m src.evaluation.charts "${args[@]}" \
    || echo "WARN: the charts of $(basename "$name") could not be drawn (the tables in the report are not affected)"
  return 0
}
