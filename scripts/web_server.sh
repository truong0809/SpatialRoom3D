#!/usr/bin/env bash
# web_server.sh - SAN PHAM: server cua trang web tren pod (tai video / anh len, dung phong, theo doi, xem 3D).
#
# Cach chay (tu thu muc goc cua repo tren pod, sau scripts/pod/run_all.sh):
#   bash scripts/web_server.sh start      chay nen; in dia chi va mat khau
#   bash scripts/web_server.sh status     dang chay hay khong, dia chi, mat khau
#   bash scripts/web_server.sh stop       dung server (viec dung phong dang chay van chay tiep)
#   bash scripts/web_server.sh log [n]    cuoi log cua server (mac dinh 40 dong)
#   bash scripts/web_server.sh run        chay o tien canh (Ctrl+C de dung)
# Tren trang Runpod (mot lan): Edit Pod -> Expose HTTP Ports: them cong 8000. Roi mo
#   https://<POD_ID>-8000.proxy.runpod.net  (lenh start in san dia chi nay khi chay tren Runpod)
#
# - Trang web goi dung luong san pham: scripts/build_room.sh <nguon> <phong> --background (viec nen room_<phong>,
#   xep hang cho GPU nhu cu). Server khong giu khoa GPU va khong chay qua job.sh.
# - Lan dau: tai thu vien xem 3D (python3 -m app.web.vendor, can Internet) va tao mat khau ngau nhien
#   ($WS/outputs/product/server/token, quyen 600). Doi mat khau: xoa file do roi stop / start lai.
# - Video / anh tai len nam o $WS/data/uploads/<phong>/; phong dung xong vao thu vien nhu khi chay build_room.sh tay.
# - PID, cong: $WS/outputs/product/server/server.{pid,port}; log: $WS/outputs/logs/web_server.log
# Bien tuy chon: WS=/workspace  PORT=8000  MAX_UPLOAD_GB=8  PRODUCT_CONFIG=<file>

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WS:-/workspace}"
PORT="${PORT:-8000}"
MAX_UPLOAD_GB="${MAX_UPLOAD_GB:-8}"
STATE_DIR="$WS/outputs/product/server"
PID_FILE="$STATE_DIR/server.pid"
PORT_FILE="$STATE_DIR/server.port"  # cong server dang dung: status / start lan sau in dung dia chi du PORT khac
TOKEN_FILE="$STATE_DIR/token"
LOG="$WS/outputs/logs/web_server.log"

usage() { sed -n '2,19p' "$0"; exit 2; }
die() { echo "ERROR: $*"; exit 1; }

# shellcheck source=/dev/null
. "$REPO_DIR/scripts/lib/config.sh"

server_pid() {  # pid cua server dang chay (doi chieu lenh, vi pid co the lap lai sau khi pod khoi dong lai)
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null)" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "app.web.server" || return 1
  echo "$pid"
}

running_port() {  # cong cua server dang chay (file server.port), khong co thi PORT
  local port
  port="$(cat "$PORT_FILE" 2>/dev/null)"
  [[ "$port" =~ ^[0-9]+$ ]] && echo "$port" || echo "$PORT"
}

address() {  # address <cong>
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "https://${RUNPOD_POD_ID}-$1.proxy.runpod.net/  (cần mở cổng HTTP $1 của pod trên trang Runpod)"
  else
    echo "http://localhost:$1/"
  fi
}

# Server khong mang theo token cua mot viec nen hay khoa GPU cua tien trinh goi (vd. lenh nay chay ben trong mot
# viec cua job.sh): neu khong, job.sh coi server la tien trinh cua viec do, va server giu khoa GPU toi khi tat.
CLEAN_ENV=(env -u SR3D_JOB_TOKEN -u SR3D_JOB_NAME -u SR3D_GPU_LOCK_FD -u SR3D_GPU_LOCK_HELD -u SR3D_GPU_LOCK_FILE)
close_gpu_lock() {
  if [[ "${SR3D_GPU_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
    eval "exec ${SR3D_GPU_LOCK_FD}>&-"
  fi
}

server_args() {
  SERVER_ARGS=(-m app.web.server --ws "$WS" --port "$PORT" --max-upload-gb "$MAX_UPLOAD_GB" --pid-file "$PID_FILE")
  [ -z "${PRODUCT_CONFIG:-}" ] || SERVER_ARGS+=(--config "$PRODUCT_CONFIG")
}

prepare() {
  [[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be a number: $PORT"
  PY="$(config_python)" || die "cần python có PyYAML (đọc configs/product.yaml): python3 -m pip install pyyaml"
  mkdir -p "$STATE_DIR" "$(dirname "$LOG")" || die "cannot create $STATE_DIR"
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR" "$PY" -m app.web.vendor) \
    || die "thiếu thư viện xem 3D (app/web/static/vendor/): cần Internet một lần, hoặc python3 -m app.web.vendor --from <thư mục .tgz>"
  server_args
}

cmd_start() {
  local pid
  if pid="$(server_pid)"; then
    echo "Server đang chạy (pid $pid)."
    cmd_status
    return 0
  fi
  prepare
  close_gpu_lock
  printf '\n########## %s | web server | port %s\n' "$(date '+%F %T')" "$PORT" >> "$LOG"
  rm -f "$PID_FILE"  # server tu ghi pid khi da mo cong
  echo "$PORT" > "$PORT_FILE"
  cd "$REPO_DIR" || die "cannot cd $REPO_DIR"
  # Chi lenh server chay nen (khong boc trong subshell): subshell giu stdout cua lenh goi (vd. o Jupyter) toi khi
  # server tat, lam lenh start khong bao gio tra ve.
  "${CLEAN_ENV[@]}" PYTHONPATH="$REPO_DIR" PYTHONUNBUFFERED=1 WS="$WS" \
    setsid nohup "$PY" "${SERVER_ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 100); do
    if server_pid > /dev/null; then
      echo "Đã chạy server trang web (pid $(server_pid))."
      cmd_status
      return 0
    fi
    sleep 0.2
  done
  echo "ERROR: server không chạy được; cuối log ($LOG):"
  tail -n 20 "$LOG"
  return 1
}

cmd_status() {
  local pid port
  if pid="$(server_pid)"; then
    port="$(running_port)"
    echo "Server: đang chạy (pid $pid, cổng $port)"
    echo "Địa chỉ: $(address "$port")"
    [ -f "$TOKEN_FILE" ] && echo "Mật khẩu: $(cat "$TOKEN_FILE")"
    echo "Log: $LOG"
  else
    echo "Server: không chạy (bash scripts/web_server.sh start)"
    return 1
  fi
}

cmd_stop() {
  local pid
  if ! pid="$(server_pid)"; then
    echo "Server không chạy."
    rm -f "$PID_FILE"
    return 0
  fi
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  rm -f "$PID_FILE"
  echo "Đã dừng server (pid $pid). Việc dựng phòng đang chạy (nếu có) vẫn chạy tiếp: bash scripts/pod/job.sh status"
}

cmd_log() {
  local n="${1:-40}"
  [[ "$n" =~ ^[0-9]+$ ]] || die "number of lines must be an integer"
  [ -f "$LOG" ] && tail -n "$n" "$LOG" || echo "(chưa có log)"
}

cmd_run() {
  local pid
  if pid="$(server_pid)"; then
    die "server đang chạy (pid $pid): bash scripts/web_server.sh stop trước"
  fi
  prepare
  close_gpu_lock
  echo "$PORT" > "$PORT_FILE"
  cd "$REPO_DIR" || die "cannot cd $REPO_DIR"
  exec "${CLEAN_ENV[@]}" PYTHONPATH="$REPO_DIR" PYTHONUNBUFFERED=1 WS="$WS" "$PY" "${SERVER_ARGS[@]}"
}

case "${1:-}" in
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  log) shift; cmd_log "$@" ;;
  run) cmd_run ;;
  *) usage ;;
esac
