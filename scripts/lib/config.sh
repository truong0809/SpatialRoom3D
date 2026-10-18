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
