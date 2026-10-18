#!/usr/bin/env bash
# job.sh - Chay viec dai (cai dat, huan luyen, COLMAP) o che do nen; moi luc chi mot viec duoc chay.
#
# Cach dung (tu thu muc goc cua repo; trong terminal hoac o %%bash / ! cua Jupyter):
#   bash scripts/pod/job.sh start <ten> -- <lenh ...>   chay nen; dang co viec khac thi xep hang cho
#   bash scripts/pod/job.sh status [ten]                trang thai cac viec (co ten: chi tiet + cuoi log)
#   bash scripts/pod/job.sh log <ten> [so_dong]         cuoi log cua mot viec (mac dinh 40 dong)
#   bash scripts/pod/job.sh stop <ten> | --all          dung viec, ke ca moi tien trinh con
#   bash scripts/pod/job.sh busy                        exit 0 neu dang co viec chay hoac xep hang
#
# - Khoa: flock tren $JOB_LOCK (o dia cua pod, /run/lock hoac /tmp; xem lib/gpu_lock.sh). Viec dang chay
#   giu khoa; viec moi xep hang toi khi khoa nha. Tien trinh con ke thua khoa (tien trinh con cua Python nhan
#   fd khoa qua src.utils.gpu_lock), nen khoa chi nha khi ca cay tien trinh ket thuc. Pod tat thi khoa mat
#   theo. Khong xoa file khoa khi dang co viec.
# - Moi tien trinh cua mot viec (ke ca tien trinh con, ke ca tien trinh tu goi setsid) mang token rieng cua
#   viec trong bien moi truong SR3D_JOB_TOKEN; trang thai va lenh dung tim tien trinh theo token,
#   khong dung pgrep hay so pid (pid lap lai sau khi pod khoi dong lai).
# - Trang thai (con lai sau khi tat pod): $WS/outputs/jobs/<ten>.state; log: $WS/outputs/logs/jobs/<ten>.log.
# - setsid + nohup: dong trinh duyet hay tat kernel Jupyter khong lam dung viec. Pod van tinh tien.
# - Moi viec chay voi MPLBACKEND=Agg: Jupyter dat backend inline ma cac env conda khong co.
#
# Bien tuy chon: WS=/workspace  JOB_LOCK=<file khoa>  STOP_TIMEOUT=20 (giay)
#   AUTO_STOP=1 (khi start)   dung pod bang runpodctl khi viec CUOI CUNG ket thuc binh thuong, ke ca viec xep
#                             hang sau do (yeu cau ghi o $JOB_LOCK.auto_stop kem phien cua pod; pod khoi dong
#                             lai thi yeu cau het hieu luc; chua thu tren pod)
#   AUTO_STOP=0 (khi start)   huy yeu cau; khong dat bien: giu nguyen. "stop" khi khong con viec nao cung huy.

set -uo pipefail

# Goi tu ben trong mot viec (vd. script cua viec goi "job.sh status"): bo token cua viec khoi lenh nay, de
# chinh no va cac tien trinh con cua no (grep, subshell...) khong bi dem la tien trinh cua viec do.
if [ -n "${SR3D_JOB_TOKEN:-}" ] && [ "${1:-}" != "_run" ]; then
  exec env -u SR3D_JOB_TOKEN -u SR3D_JOB_NAME bash "${BASH_SOURCE[0]}" "$@"
fi

WS="${WS:-/workspace}"
JOB_DIR="$WS/outputs/jobs"
JOB_LOG_DIR="$WS/outputs/logs/jobs"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=/dev/null
. "$(dirname "$SELF")/../lib/gpu_lock.sh"
JOB_LOCK="$GPU_LOCK_FILE"
AUTO_STOP_FILE="$JOB_LOCK.auto_stop"  # cung cho voi file khoa (o dia cua pod): pod tat thi yeu cau mat theo

die() { echo "ERROR: $*" >&2; exit 2; }
now() { date '+%F %T'; }
usage() { sed -n '2,26p' "$SELF"; exit 2; }
state_file() { echo "$JOB_DIR/$1.state"; }

# File trang thai chi duoc ghi them; voi moi khoa, dong sau cung la gia tri hien tai.
get() { [ -f "$1" ] || return 0; sed -n "s/^$2=//p" "$1" | tail -n 1; }
put() { local sf="$1" kv; shift; for kv in "$@"; do printf '%s\n' "$kv"; done >> "$sf"; }

# Pid cua moi tien trinh con song mang token cua viec (tru chinh lenh job.sh dang chay).
job_pids() {
  local token file pid
  token="$(get "$(state_file "$1")" token)"
  [ -n "$token" ] || return 0
  while IFS= read -r file; do
    pid="${file#/proc/}"
    pid="${pid%/environ}"
    [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ] || echo "$pid"
  done < <(grep -lxzF "SR3D_JOB_TOKEN=$token" /proc/[0-9]*/environ 2>/dev/null)
}

job_alive() { [ -n "$(job_pids "$1")" ]; }

lock_busy() {  # dung khi khoa GPU dang bi giu
  local fd
  exec {fd}>>"$JOB_LOCK" || return 1
  if flock -n "$fd"; then
    exec {fd}>&-
    return 1
  fi
  exec {fd}>&-
  return 0
}

job_names() {  # moi viec da tung chay, moi nhat truoc (ten viec khong co khoang trang)
  local sf
  # shellcheck disable=SC2045
  for sf in $(ls -1t "$JOB_DIR"/*.state 2>/dev/null); do
    basename "$sf" .state
  done
}

state_label() {
  local name="$1" sf st
  sf="$(state_file "$name")"
  st="$(get "$sf" state)"
  case "$st" in
    SUBMITTED|QUEUED|RUNNING)
      if ! job_alive "$name"; then
        echo "MẤT (pod tắt hoặc tiến trình bị giết giữa chừng)"
      elif [ "$st" = "RUNNING" ]; then
        echo "ĐANG CHẠY"
      else
        echo "ĐANG XẾP HÀNG"
      fi ;;
    DONE) echo "XONG" ;;
    FAILED) echo "LỖI (exit $(get "$sf" exit))" ;;
    STOPPED) echo "ĐÃ DỪNG" ;;
    *) echo "${st:-?}" ;;
  esac
}

elapsed() {  # elapsed <tu> [den] -> h:mm:ss
  local a b s
  a="$(date -d "$1" +%s 2>/dev/null)" || return 0
  if [ -n "${2:-}" ]; then b="$(date -d "$2" +%s 2>/dev/null)" || return 0; else b="$(date +%s)"; fi
  s=$((b - a))
  printf '%d:%02d:%02d' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}

last_line() {  # dong cuoi co noi dung cua log (bo phan da bi \r ghi de cua thanh tien do)
  [ -f "$1" ] || return 0
  tail -c 4000 "$1" | sed 's/.*\r//' | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c1-160
}

any_job_alive() {  # any_job_alive [ten bo qua] -> in ten mot viec con song; exit 1 neu khong co
  local n
  for n in $(job_names); do
    [ "$n" = "${1:-}" ] && continue
    if job_alive "$n"; then
      echo "$n"
      return 0
    fi
  done
  return 1
}

# Yeu cau tu dung pod. Phien cua pod = thoi diem tien trinh 1 cua container khoi dong (truong 22 cua
# /proc/1/stat): container khoi dong lai thi phien doi, yeu cau cu het hieu luc.
pod_session() { sed 's/^.*) //' /proc/1/stat 2>/dev/null | cut -d' ' -f20; }
auto_stop_set() {
  printf 'session=%s\nrequested=%s\nby=%s\n' "$(pod_session)" "$(now)" "$1" > "$AUTO_STOP_FILE"
  echo "Tự dừng pod khi việc cuối cùng xong: BẬT"
}
auto_stop_clear() {  # auto_stop_clear <ly do>
  if [ -f "$AUTO_STOP_FILE" ]; then
    rm -f "$AUTO_STOP_FILE"
    echo "Tự dừng pod: TẮT ($1)"
  fi
}
auto_stop_active() {
  [ -f "$AUTO_STOP_FILE" ] || return 1
  if [ "$(get "$AUTO_STOP_FILE" session)" != "$(pod_session)" ]; then
    rm -f "$AUTO_STOP_FILE"  # ghi tu phien truoc cua pod
    return 1
  fi
}

# ---------------------------------------------------------------------------------------------
cmd_start() {
  [ "$#" -ge 1 ] || usage
  local name="$1"
  shift
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "job name may only contain letters, digits, '.', '_' and '-': $name"
  [ "${1:-}" = "--" ] && shift
  [ "$#" -ge 1 ] || die "missing command for job $name"
  mkdir -p "$JOB_DIR" "$JOB_LOG_DIR" || die "cannot create $JOB_DIR"
  # Khong mang theo khoa GPU cua tien trinh goi (neu co), de viec moi xep hang dung cach.
  if [[ "${SR3D_GPU_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
    eval "exec ${SR3D_GPU_LOCK_FD}>&-"
  fi

  local sf log token waiting=0 sub
  sf="$(state_file "$name")"
  log="$JOB_LOG_DIR/$name.log"
  exec {sub}>>"$JOB_LOCK.submit" || die "cannot open $JOB_LOCK.submit"
  flock "$sub"  # hai lenh start cung luc khong the cung vuot qua buoc kiem tra duoi day
  case "${AUTO_STOP-}" in  # ap dung ca khi viec da dang chay (lenh start lap lai chi de bat / tat)
    1) auto_stop_set "$name" ;;
    0) auto_stop_clear "AUTO_STOP=0" ;;
  esac
  if job_alive "$name"; then
    echo "Việc '$name' ($(state_label "$name")): không chạy thêm bản thứ hai."
    echo "Xem: bash scripts/pod/job.sh status $name"
    exec {sub}>&-
    return 0
  fi
  lock_busy && waiting=1
  token="$(date +%s%N)-$$-$RANDOM"
  {
    printf 'name=%s\n' "$name"
    printf 'token=%s\n' "$token"
    printf 'log=%s\n' "$log"
    printf 'cmd=%s\n' "$(printf '%q ' "$@")"
    printf 'cwd=%s\n' "$PWD"
    printf 'submitted=%s\n' "$(now)"
    printf 'state=SUBMITTED\n'
  } > "$sf"
  printf '\n########## %s | job %s | %s\n' "$(now)" "$name" "$(printf '%q ' "$@")" >> "$log"
  env -u SR3D_GPU_LOCK_HELD -u SR3D_GPU_LOCK_FD -u SR3D_GPU_LOCK_FILE \
    SR3D_JOB_TOKEN="$token" SR3D_JOB_NAME="$name" MPLBACKEND=Agg \
    setsid nohup bash "$SELF" _run "$name" -- "$@" >> "$log" 2>&1 < /dev/null {sub}>&- &

  for _ in $(seq 1 100); do
    [ -n "$(get "$sf" pid)" ] && break
    sleep 0.1
  done
  exec {sub}>&-
  if [ -z "$(get "$sf" pid)" ]; then
    echo "ERROR: job $name did not start; see $log"
    any_job_alive > /dev/null || auto_stop_clear "không còn việc nào"
    return 1
  fi
  if [ "$waiting" = 1 ]; then
    echo "Đã xếp hàng '$name': đang có việc khác dùng GPU, '$name' sẽ tự chạy khi việc đó xong."
  else
    echo "Đã chạy nền '$name'."
  fi
  echo "Log: $log"
  echo "Trạng thái: bash scripts/pod/job.sh status | Dừng: bash scripts/pod/job.sh stop $name"
}

# Noi bo: chay trong phien rieng do cmd_start tao ra.
JOB_NAME=""
JOB_SF=""
JOB_CHILD=""
on_signal() {
  trap - TERM INT
  if [ -n "$JOB_CHILD" ]; then kill -TERM "$JOB_CHILD" 2>/dev/null; fi
  put "$JOB_SF" "finished=$(now)" "exit=143" "state=STOPPED"
  echo "[$(now)] Đã dừng '$JOB_NAME'"
  exit 143
}

cmd_run() {
  JOB_NAME="$1"
  shift
  [ "${1:-}" = "--" ] && shift
  JOB_SF="$(state_file "$JOB_NAME")"
  put "$JOB_SF" "pid=$$" "state=QUEUED"
  trap on_signal TERM INT
  local lfd rc st cwd
  exec {lfd}>>"$JOB_LOCK" || { put "$JOB_SF" "finished=$(now)" "exit=97" "state=FAILED"; exit 97; }
  if ! flock -n "$lfd"; then
    echo "[$(now)] Đang có việc khác dùng GPU: '$JOB_NAME' xếp hàng chờ..."
    flock "$lfd"
  fi
  export SR3D_GPU_LOCK_HELD=1 SR3D_GPU_LOCK_FD="$lfd" SR3D_GPU_LOCK_FILE="$JOB_LOCK"
  put "$JOB_SF" "started=$(now)" "state=RUNNING"
  echo "[$(now)] Bắt đầu '$JOB_NAME' (pid $$)"
  cwd="$(get "$JOB_SF" cwd)"
  cd "$cwd" 2>/dev/null || cd "$WS" || true
  "$@" < /dev/null &
  JOB_CHILD=$!
  wait "$JOB_CHILD"
  rc=$?
  JOB_CHILD=""
  if [ "$rc" -eq 0 ]; then st=DONE; else st=FAILED; fi
  put "$JOB_SF" "finished=$(now)" "exit=$rc" "state=$st"
  echo "[$(now)] Kết thúc '$JOB_NAME': $st (exit $rc)"
  maybe_auto_stop
  exit "$rc"
}

maybe_auto_stop() {
  auto_stop_active || return 0
  local sub other
  exec {sub}>>"$JOB_LOCK.submit" || return 0
  flock "$sub"  # khong quyet dinh dung luc mot lenh start dang gui viec moi
  if other="$(any_job_alive "$JOB_NAME")"; then
    exec {sub}>&-
    echo "AUTO_STOP: còn việc '$other' đang chờ/chạy; pod sẽ dừng khi việc cuối cùng xong"
    return 0
  fi
  rm -f "$AUTO_STOP_FILE"  # chi thu dung pod mot lan
  exec {sub}>&-
  if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null 2>&1; then
    echo "AUTO_STOP: không có runpodctl hoặc RUNPOD_POD_ID, không dừng được pod (hãy dừng trên trang Runpod)"
    return 0
  fi
  echo "AUTO_STOP: dừng pod $RUNPOD_POD_ID"
  runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" \
    || echo "AUTO_STOP: runpodctl báo lỗi, hãy dừng pod trên trang Runpod"
}

cmd_status() {
  mkdir -p "$JOB_DIR"
  if [ "$#" -ge 1 ]; then
    status_one "$1"
    return
  fi
  local n sf any=0 started finished submitted
  for n in $(job_names); do
    any=1
    sf="$(state_file "$n")"
    started="$(get "$sf" started)"
    finished="$(get "$sf" finished)"
    submitted="$(get "$sf" submitted)"
    printf '%-20s %s' "$n" "$(state_label "$n")"
    if [ -n "$finished" ]; then
      printf ' | %s → %s (%s)' "${started:-$submitted}" "$finished" "$(elapsed "${started:-$submitted}" "$finished")"
    elif [ -n "$started" ]; then
      printf ' | từ %s (%s)' "$started" "$(elapsed "$started")"
    else
      printf ' | gửi lúc %s' "$submitted"
    fi
    printf '\n'
    if job_alive "$n"; then
      printf '    %s\n' "$(last_line "$(get "$sf" log)")"
    fi
  done
  [ "$any" = 1 ] || echo "Chưa có việc nào."
  if lock_busy; then echo "Khóa GPU: đang có việc giữ"; else echo "Khóa GPU: rảnh"; fi
  auto_stop_active && echo "Tự dừng pod khi việc cuối cùng xong: BẬT (hủy: AUTO_STOP=0 khi start)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"
  fi
}

status_one() {
  local name="$1" sf
  sf="$(state_file "$name")"
  [ -f "$sf" ] || die "no job named $name (see: bash scripts/pod/job.sh status)"
  echo "Việc:       $name"
  echo "Trạng thái: $(state_label "$name")"
  echo "Lệnh:       $(get "$sf" cmd)"
  echo "Gửi lúc:    $(get "$sf" submitted)"
  echo "Bắt đầu:    $(get "$sf" started)"
  echo "Kết thúc:   $(get "$sf" finished)"
  echo "Log:        $(get "$sf" log)"
  echo "--- cuối log ---"
  cmd_log "$name" 15
}

cmd_log() {
  [ "$#" -ge 1 ] || usage
  local sf log n="${2:-40}"
  sf="$(state_file "$1")"
  [ -f "$sf" ] || die "no job named $1"
  [[ "$n" =~ ^[0-9]+$ ]] || die "number of lines must be an integer"
  log="$(get "$sf" log)"
  [ -f "$log" ] || { echo "(chưa có log)"; return 0; }
  tail -n "$n" "$log" | sed 's/.*\r//' | cut -c1-300
}

stop_one() {
  local name="$1" sf pids
  sf="$(state_file "$name")"
  [ -f "$sf" ] || { echo "Không có việc '$name'"; return 0; }
  pids="$(job_pids "$name")"
  if [ -z "$pids" ]; then
    echo "'$name' không chạy ($(state_label "$name"))"
    return 0
  fi
  # Gui SIGTERM cho moi tien trinh mang token (gom ca tien trinh moi sinh ra trong luc dung)
  for _ in $(seq 1 "$STOP_TIMEOUT"); do
    pids="$(job_pids "$name")"
    [ -n "$pids" ] || break
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null
    sleep 1
  done
  pids="$(job_pids "$name")"
  if [ -n "$pids" ]; then
    echo "'$name' chưa dừng sau ${STOP_TIMEOUT}s: buộc dừng (SIGKILL)"
    # shellcheck disable=SC2086
    kill -KILL $pids 2>/dev/null
    sleep 1
  fi
  case "$(get "$sf" state)" in
    DONE|FAILED|STOPPED) ;;
    *) put "$sf" "finished=$(now)" "exit=137" "state=STOPPED" ;;
  esac
  echo "Đã dừng '$name'."
}

cmd_stop() {
  [ "$#" -ge 1 ] || usage
  local names=() n
  if [ "$1" = "--all" ]; then
    for n in $(job_names); do
      job_alive "$n" && names+=("$n")
    done
  else
    names=("$@")
  fi
  if [ "${#names[@]}" -eq 0 ]; then
    echo "Không có việc nào đang chạy hoặc xếp hàng."
    auto_stop_clear "không còn việc nào"
    return 0
  fi
  # dung viec dang xep hang truoc, de no khong chiem GPU ngay khi viec dang chay vua dung
  for n in "${names[@]}"; do
    [ "$(get "$(state_file "$n")" state)" = "RUNNING" ] || stop_one "$n"
  done
  for n in "${names[@]}"; do
    [ "$(get "$(state_file "$n")" state)" = "RUNNING" ] && stop_one "$n"
  done
  # nguoi dung da dung viec bang tay: yeu cau tu dung pod khong con y nghia khi khong con viec nao
  any_job_alive > /dev/null || auto_stop_clear "đã dừng mọi việc"
  return 0
}

cmd_busy() {
  local n
  lock_busy && return 0
  for n in $(job_names); do
    job_alive "$n" && return 0
  done
  return 1
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  log) shift; cmd_log "$@" ;;
  stop) shift; cmd_stop "$@" ;;
  busy) shift; cmd_busy ;;
  _run) shift; cmd_run "$@" ;;
  *) usage ;;
esac
