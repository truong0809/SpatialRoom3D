"""Test luồng scripts/pod/run_all.sh (chỉ cài / kiểm tra môi trường) với các script cài đặt được thay bằng stub.

Kiểm tra thứ tự bước, điều kiện phụ thuộc, bảng tóm tắt, mã thoát, khóa GPU, chạy nền qua job.sh, và việc run_all
không train hay chạy benchmark nào (chạy với dữ liệu nào do người dùng tự chọn).
"""

import os
import tarfile

import pytest

from shell_env import HoldLock, ShellEnv, calls, make_fake_repo, write_stub

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

RECORD = ('echo "$(basename "$0") ARGS=$* ALLOW=${ALLOW_PROVISIONAL:-} '
          'MPL=${MPLBACKEND:-} HELD=${SR3D_GPU_LOCK_HELD:-}" >> "$CALLS"\n')
STUBS = {
    "scripts/setup/env_probe.sh": "PROBE",
    "scripts/setup/setup_3dgs_inria.sh": "INRIA",
    "scripts/setup/setup_data_tools.sh": "TOOLS",
    "scripts/setup/setup_gsplat.sh": "GSPLAT",
}
SETUP = ["env_probe.sh", "setup_3dgs_inria.sh", "setup_data_tools.sh", "setup_gsplat.sh"]


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["pod/run_all.sh", "pod/pack_results.sh"])
    for name, key in STUBS.items():
        write_stub(repo, name, RECORD + f'mkdir -p "$WS/outputs/reports"; echo {key} > "$WS/outputs/reports/{key}.txt"\n'
                                        f'exit "${{STUB_RC_{key}:-0}}"\n')
    # run_all không được gọi bước huấn luyện nào: nếu gọi thì stub này ghi lại
    for name in ("scripts/run_train.sh", "scripts/pod/run_scene.sh"):
        write_stub(repo, name, RECORD + "exit 0\n")
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    yield env
    env.job("stop", "--all")
    env.cleanup()


def run_all(sh, *args, **env):
    return sh.run(str(sh.repo / "scripts" / "pod" / "run_all.sh"), *args,
                  env={k: str(v) for k, v in env.items()}, timeout=120)


def steps_called(sh):
    return [line.split()[0] for line in calls(sh.root)]


def summary(sh):
    tsv = sorted((sh.ws / "outputs" / "reports").glob("run_all_*.tsv"))
    assert len(tsv) == 1
    rows = [line.split("\t") for line in tsv[0].read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return {row[0]: row for row in rows}


def test_setup_steps_run_in_order_and_nothing_is_trained(sh):
    out = run_all(sh)
    assert out.returncode == 0, out.stdout + out.stderr
    assert steps_called(sh) == SETUP  # không có bước train / benchmark nào
    lines = calls(sh.root)
    assert all("ALLOW=1" in line and "MPL=Agg" in line and "HELD=1" in line for line in lines)
    rows = summary(sh)
    assert list(rows) == ["env_probe", "setup_inria", "data_tools", "setup_gsplat"]
    assert all(row[1] == "OK" for row in rows.values())
    assert "TÓM TẮT run_all" in out.stdout and "Gói log / báo cáo:" in out.stdout
    assert "Môi trường sẵn sàng" in out.stdout
    package = sorted(sh.ws.glob("ket_qua_*.tar.gz"))
    assert len(package) == 1
    with tarfile.open(package[0]) as tar:
        names = tar.getnames()
    assert "outputs/reports/INRIA.txt" in names and any(name.startswith("outputs/reports/run_all_") for name in names)
    assert sorted((sh.ws / "outputs" / "logs").glob("run_all_*.log"))  # chạy trực tiếp: có log riêng
    assert not (sh.ws / "outputs" / "benchmarks").exists() and not (sh.ws / "data").exists()


def test_failed_step_is_reported_and_the_others_still_run(sh):
    out = run_all(sh, STUB_RC_GSPLAT=1)
    assert out.returncode == 1
    assert steps_called(sh) == SETUP
    rows = summary(sh)
    assert rows["setup_gsplat"][1:3] == ["FAIL", "1"] and rows["setup_inria"][1] == "OK"
    assert "LỖI (exit 1)" in out.stdout and "Môi trường sẵn sàng" not in out.stdout


def test_env_probe_failure_stops_before_installing(sh):
    out = run_all(sh, STUB_RC_PROBE=1)
    assert out.returncode == 1
    assert steps_called(sh) == ["env_probe.sh"]
    rows = summary(sh)
    assert rows["env_probe"][1] == "FAIL"
    assert all(rows[name][1] == "SKIP" for name in ("setup_inria", "data_tools", "setup_gsplat"))
    assert sorted(sh.ws.glob("ket_qua_*.tar.gz"))  # vẫn đóng gói log để gửi về


def test_user_skip_counts_as_satisfied(sh):
    out = run_all(sh, SKIP="setup_inria")
    assert out.returncode == 0, out.stdout
    assert "setup_3dgs_inria.sh" not in steps_called(sh)
    rows = summary(sh)
    assert rows["setup_inria"][1] == "SKIP" and rows["setup_inria"][4] == "SKIP"


def test_old_setup_only_flag_is_accepted(sh):
    """Notebook cũ gọi run_all.sh --background --setup-only: nay run_all luôn chỉ cài đặt, tham số này không đổi gì."""
    out = run_all(sh, "--setup-only")
    assert out.returncode == 0, out.stdout + out.stderr
    assert steps_called(sh) == SETUP
    out = run_all(sh, "--background", "--setup-only")
    assert out.returncode == 0 and "Đã chạy nền 'run_all'" in out.stdout, out.stdout + out.stderr
    state = sh.wait_state("run_all", "DONE", "FAILED", timeout=60)
    assert state["state"] == "DONE", sh.log("run_all")
    assert steps_called(sh) == SETUP * 2


def test_refuses_to_start_while_gpu_lock_is_held(sh):
    with HoldLock(sh.lock):
        out = run_all(sh)
    assert out.returncode == 1
    assert "another job is using the GPU" in out.stdout
    assert calls(sh.root) == []
    assert not (sh.ws / "outputs" / "reports").exists()


def test_background_runs_as_job(sh):
    out = run_all(sh, "--background")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Đã chạy nền 'run_all'" in out.stdout
    state = sh.wait_state("run_all", "DONE", "FAILED", timeout=60)
    assert state["state"] == "DONE", sh.log("run_all")
    assert "TÓM TẮT run_all" in sh.log("run_all") and steps_called(sh) == SETUP
    assert not sorted((sh.ws / "outputs" / "logs").glob("run_all_*.log"))  # log nằm trong logs/jobs/run_all.log
    again = run_all(sh, "--background", "extra")
    assert again.returncode == 2
