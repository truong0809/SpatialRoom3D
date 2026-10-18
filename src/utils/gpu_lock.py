"""Khóa GPU của các script bash (scripts/lib/gpu_lock.sh, scripts/pod/job.sh) phía Python.

Script giữ khóa bằng một fd mở trên file khóa (flock), ghi số fd vào SR3D_GPU_LOCK_FD và đường dẫn file
khóa vào SR3D_GPU_LOCK_FILE (để kiểm tra fd đúng là file khóa). Tiến trình con của bash kế thừa fd này,
nhưng `subprocess` của Python đóng mọi fd kế thừa khi chạy tiến trình con. Truyền riêng fd khóa
(`pass_fds=inherited_lock_fds()`) cho tiến trình chạy lâu (huấn luyện, COLMAP), để khóa chỉ nhả khi chính
tiến trình đó kết thúc, kể cả khi tiến trình Python cha bị giết trước.
"""

from __future__ import annotations

import os


def inherited_lock_fds() -> tuple[int, ...]:
    value = os.environ.get("SR3D_GPU_LOCK_FD", "")
    if not value.isdigit():
        return ()
    fd = int(value)
    try:
        opened = os.fstat(fd)
    except OSError:
        return ()  # biến còn nhưng fd đã đóng (vd. tiến trình trung gian đã đóng nó)
    path = os.environ.get("SR3D_GPU_LOCK_FILE")
    if path:
        try:
            if not os.path.samestat(opened, os.stat(path)):
                return ()  # số fd đã được dùng lại cho file khác
        except OSError:
            return ()
    return (fd,)
