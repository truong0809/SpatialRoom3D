# shellcheck shell=bash
# gpu_lock.sh - Khoa "moi luc chi mot viec dung GPU", dung chung cho cac script (source file nay).
#
#   gpu_lock_acquire <ten_viec>
#     - Script cha hoac scripts/pod/job.sh da giu khoa (SR3D_GPU_LOCK_HELD=1): khong lam gi.
#     - Khoa dang ranh: giu khoa cho toi khi script va moi tien trinh con cua no ket thuc (tien trinh con
#       chay tu Python nhan fd khoa qua src.utils.gpu_lock, vi subprocess dong moi fd ke thua).
#     - Khoa dang bi viec khac giu: dung ngay, khong chay chen len GPU.
#   So fd giu khoa nam trong SR3D_GPU_LOCK_FD: tien trinh phu chay lau (vd. nvidia-smi -l ghi log GPU)
#   nen dong fd nay ({SR3D_GPU_LOCK_FD}>&-) de khong giu khoa neu script bi giet giua chung.
#
# Khoa la flock tren mot file o dia cua pod (/run/lock/spatialroom3d-gpu.lock, hoac /tmp neu khong ghi duoc),
# khong nam tren network volume: pod tat thi khoa mat theo, khong bao gio de lai khoa "treo".
# Khong xoa file khoa khi dang co viec: viec moi se mo file moi va chay chen.

if [ -z "${JOB_LOCK:-}" ]; then
  if [ -d /run/lock ] && [ -w /run/lock ]; then
    JOB_LOCK=/run/lock/spatialroom3d-gpu.lock
  else
    JOB_LOCK=/tmp/spatialroom3d-gpu.lock
  fi
fi
GPU_LOCK_FILE="$JOB_LOCK"

gpu_lock_acquire() {
  if [ "${SR3D_GPU_LOCK_HELD:-0}" = "1" ]; then
    return 0
  fi
  local fd
  exec {fd}>>"$GPU_LOCK_FILE" || { echo "ERROR: cannot open the lock file $GPU_LOCK_FILE"; exit 1; }
  if ! flock -n "$fd"; then
    echo "ERROR: another job is using the GPU; not starting ${1:-this job} next to it."
    echo "       See: bash scripts/pod/job.sh status   (or queue it: bash scripts/pod/job.sh start <name> -- <command>)"
    exit 1
  fi
  export SR3D_GPU_LOCK_HELD=1 SR3D_GPU_LOCK_FD="$fd" SR3D_GPU_LOCK_FILE="$GPU_LOCK_FILE"
}
