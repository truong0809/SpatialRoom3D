"""Test luồng scripts/pod/run_all.sh với các script cài đặt / huấn luyện được thay bằng stub.

Kiểm tra thứ tự bước, điều kiện phụ thuộc, bảng tóm tắt, mã thoát, khóa GPU và chạy nền qua job.sh.
"""

import os
import tarfile

import pytest

from shell_env import HoldLock, ShellEnv, calls, make_fake_repo, write_stub

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

RECORD = ('echo "$(basename "$0") ARGS=$* QUICK=${QUICK:-} SEEDS=${SEEDS:-} ALLOW=${ALLOW_PROVISIONAL:-} '
          'MPL=${MPLBACKEND:-} HELD=${SR3D_GPU_LOCK_HELD:-}" >> "$CALLS"\n')
STUBS = {
    "scripts/setup/env_probe.sh": "PROBE",
    "scripts/setup/setup_3dgs_inria.sh": "INRIA",
    "scripts/setup/setup_data_tools.sh": "TOOLS",
    "scripts/setup/setup_gsplat.sh": "GSPLAT",
}
B0, MODES = "run_inria_b0.sh", "run_gsplat_modes.sh"  # tên ghi lại cho run_train.sh <benchmark> b0 | gsplat
# scripts/run_train.sh <benchmark> b0 | gsplat: ghi lại như hai bước riêng (B0, gsplat)
RUN_TRAIN = r'''step="run_gsplat_modes.sh"
[ "${2:-}" != b0 ] || step="run_inria_b0.sh"
echo "$step ARGS=$1 QUICK=${QUICK:-} SEEDS=${SEEDS:-} ALLOW=${ALLOW_PROVISIONAL:-} MPL=${MPLBACKEND:-} HELD=${SR3D_GPU_LOCK_HELD:-}" >> "$CALLS"
if [ "$step" = run_inria_b0.sh ]; then
  mkdir -p "$WS/outputs/reports"; echo B0 > "$WS/outputs/reports/B0.txt"; exit "${STUB_RC_B0:-0}"
fi
if [ "${QUICK:-0}" = 1 ]; then exit "${STUB_RC_QUICK:-0}"; fi
exit "${STUB_RC_MODES:-0}"
'''


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["pod/run_all.sh", "pod/pack_results.sh"])
    for name, key in STUBS.items():
        write_stub(repo, name, RECORD + f'mkdir -p "$WS/outputs/reports"; echo {key} > "$WS/outputs/reports/{key}.txt"\n'
                                        f'exit "${{STUB_RC_{key}:-0}}"\n')
    write_stub(repo, "scripts/run_train.sh", RUN_TRAIN)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    yield env
    env.job("stop", "--all")
    env.cleanup()


def run_all(sh, **env):
    return sh.run(str(sh.repo / "scripts" / "pod" / "run_all.sh"), env={k: str(v) for k, v in env.items()},
                  timeout=120)


def steps_called(sh):
    return [line.split()[0] for line in calls(sh.root)]


def summary(sh):
    tsv = sorted((sh.ws / "outputs" / "reports").glob("run_all_*.tsv"))
    assert len(tsv) == 1
    rows = [line.split("\t") for line in tsv[0].read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return {row[0]: row for row in rows}


def test_all_steps_run_in_order_and_succeed(sh):
    out = run_all(sh)
    assert out.returncode == 0, out.stdout + out.stderr
    assert steps_called(sh) == ["env_probe.sh", "setup_3dgs_inria.sh", "setup_data_tools.sh", "setup_gsplat.sh",
                                MODES, B0, MODES]
    lines = calls(sh.root)
    assert "QUICK=1 SEEDS=42" in lines[4] and "QUICK= " in lines[6]
    assert all("ARGS=deep_blending " in lines[index] for index in (4, 5, 6))  # benchmark chạy trong run_all
    assert all("ALLOW=1" in line and "MPL=Agg" in line and "HELD=1" in line for line in lines)
    rows = summary(sh)
    assert [rows[name][1] for name in ("env_probe", "setup_inria", "data_tools", "setup_gsplat", "quick_test",
                                       "b0_inria", "modes_full")] == ["OK"] * 7
    assert "TÓM TẮT run_all" in out.stdout and "Gói kết quả:" in out.stdout
    package = sorted(sh.ws.glob("ket_qua_*.tar.gz"))
    assert len(package) == 1
    with tarfile.open(package[0]) as tar:
        names = tar.getnames()
    assert "outputs/reports/INRIA.txt" in names and any(name.startswith("outputs/reports/run_all_") for name in names)
    assert sorted((sh.ws / "outputs" / "logs").glob("run_all_*.log"))  # chạy trực tiếp: có log riêng


def test_failed_gsplat_setup_skips_only_dependent_steps(sh):
    out = run_all(sh, STUB_RC_GSPLAT=1)
    assert out.returncode == 1
    assert steps_called(sh) == ["env_probe.sh", "setup_3dgs_inria.sh", "setup_data_tools.sh", "setup_gsplat.sh", B0]
    rows = summary(sh)
    assert rows["setup_gsplat"][1:3] == ["FAIL", "1"]
    assert rows["quick_test"][1] == "SKIP" and "setup_gsplat" in rows["quick_test"][4]
    assert rows["modes_full"][1] == "SKIP"
    assert rows["b0_inria"][1] == "OK"
    assert "LỖI (exit 1)" in out.stdout and "BỎ QUA (cần setup_gsplat thành công)" in out.stdout


def test_failed_quick_test_skips_full_modes(sh):
    out = run_all(sh, STUB_RC_QUICK=1)
    assert out.returncode == 1
    rows = summary(sh)
    assert rows["quick_test"][1] == "FAIL" and rows["modes_full"][1] == "SKIP" and rows["b0_inria"][1] == "OK"


def test_env_probe_failure_stops_before_installing(sh):
    out = run_all(sh, STUB_RC_PROBE=1)
    assert out.returncode == 1
    assert steps_called(sh) == ["env_probe.sh"]
    rows = summary(sh)
    assert rows["env_probe"][1] == "FAIL"
    assert all(rows[name][1] == "SKIP" for name in ("setup_inria", "setup_gsplat", "modes_full"))
    assert sorted(sh.ws.glob("ket_qua_*.tar.gz"))  # vẫn đóng gói log để gửi về


def test_user_skip_counts_as_satisfied(sh):
    out = run_all(sh, SKIP="quick_test setup_gsplat")
    assert out.returncode == 0, out.stdout
    called = steps_called(sh)
    assert "setup_gsplat.sh" not in called
    assert called.count(MODES) == 1  # chỉ lượt đầy đủ
    rows = summary(sh)
    assert rows["quick_test"][1] == "SKIP" and rows["quick_test"][4] == "SKIP"
    assert rows["modes_full"][1] == "OK"


def test_setup_only_runs_no_benchmark(sh):
    """Pod mới gắn volume cũ: chỉ cài / kiểm tra môi trường, không chạy Deep Blending."""
    out = sh.run(str(sh.repo / "scripts" / "pod" / "run_all.sh"), "--setup-only", timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert steps_called(sh) == ["env_probe.sh", "setup_3dgs_inria.sh", "setup_data_tools.sh", "setup_gsplat.sh"]
    rows = summary(sh)
    assert all(rows[name][1] == "SKIP" and rows[name][4] == "SKIP" for name in ("quick_test", "b0_inria", "modes_full"))


def test_setup_only_in_background(sh):
    out = sh.run(str(sh.repo / "scripts" / "pod" / "run_all.sh"), "--background", "--setup-only")
    assert out.returncode == 0, out.stdout + out.stderr
    state = sh.wait_state("run_all", "DONE", "FAILED", timeout=60)
    assert state["state"] == "DONE", sh.log("run_all")
    assert "--setup-only" in state["cmd"]
    assert MODES not in steps_called(sh) and B0 not in steps_called(sh)


def test_quick_test_uses_first_seed_only(sh):
    run_all(sh, SEEDS="7 8 9")
    lines = calls(sh.root)
    assert "QUICK=1 SEEDS=7 " in lines[4]
    assert "SEEDS=7 8 9" in lines[6]


def test_refuses_to_start_while_gpu_lock_is_held(sh):
    with HoldLock(sh.lock):
        out = run_all(sh)
    assert out.returncode == 1
    assert "another job is using the GPU" in out.stdout
    assert calls(sh.root) == []
    assert not (sh.ws / "outputs" / "reports").exists()


def test_background_runs_as_job(sh):
    out = sh.run(str(sh.repo / "scripts" / "pod" / "run_all.sh"), "--background")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Đã chạy nền 'run_all'" in out.stdout
    state = sh.wait_state("run_all", "DONE", "FAILED", timeout=60)
    assert state["state"] == "DONE", sh.log("run_all")
    assert "TÓM TẮT run_all" in sh.log("run_all")
    assert not sorted((sh.ws / "outputs" / "logs").glob("run_all_*.log"))  # log nằm trong logs/jobs/run_all.log
    again = sh.run(str(sh.repo / "scripts" / "pod" / "run_all.sh"), "--background", "extra")
    assert again.returncode == 2
