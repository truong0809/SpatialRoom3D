"""Test bước 0-1 của scripts/setup/setup_3dgs_inria.sh và setup_gsplat.sh bằng nvidia-smi / df / curl giả.

Không cài gì: curl giả từ chối tải bộ cài Miniforge, nên script dừng ở bước 1 sau khi qua (hoặc trượt) bước 0.
"""

import os
import stat
import subprocess

import pytest

from shell_env import REPO, HoldLock

SCRIPTS = ["setup_3dgs_inria.sh", "setup_gsplat.sh"]


def fake_bin(tmp_path, compute_cap="8.9", driver_cuda="12.8", size_gb="2300000", free_gb="2000000"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tools = {
        "nvidia-smi": f'''
case "$*" in
  *query-gpu=name*) echo "NVIDIA GeForce RTX 4090" ;;
  *query-gpu=compute_cap*) echo "{compute_cap}" ;;
  *query-gpu=memory.total*) echo "24564" ;;
  "") echo "| NVIDIA-SMI 570.195   Driver Version: 570.195   CUDA Version: {driver_cuda} |" ;;
esac''',
        "df": f'''
case "$*" in
  *output=size*) printf ' 1G-blocks\\n {size_gb}G\\n' ;;
  *output=avail*) printf ' Avail\\n {free_gb}G\\n' ;;
esac''',
        "curl": '''
echo "curl $*" >> "$CALLS"
case "$*" in
  *"%{http_code}"*) printf 200 ;;
  *) exit 22 ;;
esac''',
    }
    for name, body in tools.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def run_setup(tmp_path, script, bin_dir, **extra):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "WS": str(ws),
           "JOB_LOCK": str(tmp_path / "gpu.lock"), "CALLS": str(tmp_path / "calls.txt"), "ALLOW_PROVISIONAL": "1",
           **extra}
    return subprocess.run(["bash", str(REPO / "scripts" / "setup" / script)], env=env, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("script", SCRIPTS)
def test_preflight_passes_on_network_volume_and_uses_pinned_miniforge(tmp_path, script):
    out = run_setup(tmp_path, script, fake_bin(tmp_path))
    text = out.stdout + out.stderr
    assert out.returncode != 0  # dừng ở bước tải Miniforge (curl giả)
    assert "1. Miniforge" in text
    assert "is a network volume: free space cannot be read with df" in text
    downloads = [line for line in (tmp_path / "calls.txt").read_text().splitlines() if "miniforge.sh" in line]
    assert len(downloads) == 1
    assert "/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-x86_64.sh" in downloads[0]
    assert "latest" not in downloads[0]


@pytest.mark.parametrize("script", SCRIPTS)
def test_gpu_without_kernels_in_arch_list_is_refused(tmp_path, script):
    out = run_setup(tmp_path, script, fake_bin(tmp_path, compute_cap="7.5"))
    assert out.returncode == 1
    assert "compute capability 7.5 is not in TORCH_CUDA_ARCH_LIST" in out.stdout
    assert "1. Miniforge" not in out.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_old_driver_is_refused(tmp_path, script):
    out = run_setup(tmp_path, script, fake_bin(tmp_path, driver_cuda="12.4"))
    assert out.returncode == 1
    assert "driver supports CUDA 12.4 < 12.8" in out.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_small_local_disk_is_refused(tmp_path, script):
    out = run_setup(tmp_path, script, fake_bin(tmp_path, size_gb="100", free_gb="5"))
    assert out.returncode == 1
    assert "GB free in" in out.stdout and "have 5 GB" in out.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_provisional_versions_need_allow_flag(tmp_path, script):
    out = run_setup(tmp_path, script, fake_bin(tmp_path), ALLOW_PROVISIONAL="0")
    assert out.returncode == 1
    assert "provisional" in out.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_setup_refuses_while_gpu_lock_is_held(tmp_path, script):
    with HoldLock(tmp_path / "gpu.lock"):
        out = run_setup(tmp_path, script, fake_bin(tmp_path))
    assert out.returncode == 1
    assert "another job is using the GPU" in out.stdout
    assert not (tmp_path / "calls.txt").exists()  # chưa kiểm tra mạng, chưa tải gì


def test_data_tools_setup_refuses_while_gpu_lock_is_held(tmp_path):
    bin_dir = fake_bin(tmp_path)
    with HoldLock(tmp_path / "gpu.lock"):
        out = run_setup(tmp_path, "setup_data_tools.sh", bin_dir)
    assert out.returncode == 1
    assert "another job is using the GPU" in out.stdout and "Miniforge not found" not in out.stdout
    out = run_setup(tmp_path, "setup_data_tools.sh", bin_dir)
    assert out.returncode == 1 and "Miniforge not found" in out.stdout  # khóa rảnh: qua bước khóa


def run_probe(tmp_path, bin_dir):
    out_dir = tmp_path / "reports"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "OUT_DIR": str(out_dir)}
    return subprocess.run(["bash", str(REPO / "scripts" / "setup" / "env_probe.sh")], env=env, capture_output=True,
                          text=True, timeout=180)


def test_env_probe_exit_code_follows_fail_checks(tmp_path):
    bin_dir = fake_bin(tmp_path)
    # python3 giả không có torch: chỉ là [INFO], không phải [FAIL]
    (bin_dir / "python3").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    (bin_dir / "python3").chmod(0o755)
    out = run_probe(tmp_path, bin_dir)
    assert "[PASS] VRAM 24564 MiB" in out.stdout and "[INFO] no PyTorch in the image python" in out.stdout
    assert "[FAIL]" not in out.stdout and out.returncode == 0, out.stdout[-2000:]
    # không thấy GPU (thay bằng bản lỗi, không xóa: trên pod /usr/bin có nvidia-smi thật)
    (bin_dir / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    out = run_probe(tmp_path, bin_dir)
    assert "[FAIL] GPU not visible" in out.stdout and out.returncode == 1
