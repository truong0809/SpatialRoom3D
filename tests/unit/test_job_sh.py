"""Test scripts/pod/job.sh và scripts/lib/gpu_lock.sh: chạy nền, xếp hàng, dừng cả cây tiến trình, trạng thái cũ.

Chạy thật bash/flock/setsid trên /proc (Linux), mỗi test có WS và file khóa riêng trong tmp_path.
"""

import os
import signal
import time

import pytest

from shell_env import REPO, HoldLock, ShellEnv

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")


@pytest.fixture
def sh(tmp_path):
    env = ShellEnv(tmp_path)
    yield env
    env.job("stop", "--all", timeout=60)
    env.cleanup()


def test_start_runs_in_background_and_records_done(sh):
    out = sh.job("start", "a", "--", "bash", "-c", "echo hello-from-job")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Đã chạy nền 'a'" in out.stdout
    state = sh.wait_state("a", "DONE")
    assert state["exit"] == "0" and state["started"] and state["finished"]
    assert "hello-from-job" in sh.log("a")
    assert sh.job("busy").returncode == 1
    assert "XONG" in sh.job("status").stdout


def test_failed_command_records_exit_code(sh):
    sh.job("start", "bad", "--", "bash", "-c", "exit 3")
    state = sh.wait_state("bad", "FAILED")
    assert state["exit"] == "3"
    assert "LỖI (exit 3)" in sh.job("status").stdout


def test_jobs_run_one_at_a_time_in_order(sh):
    marks = sh.root / "marks.txt"
    first = f"echo a-start $(date +%s.%N) >> {marks}; sleep 1.5; echo a-end $(date +%s.%N) >> {marks}"
    second = f"echo b-start $(date +%s.%N) >> {marks}"
    sh.job("start", "a", "--", "bash", "-c", first)
    sh.wait_state("a", "RUNNING")
    out = sh.job("start", "b", "--", "bash", "-c", second)
    assert "Đã xếp hàng 'b'" in out.stdout
    assert sh.wait_state("b", "QUEUED")["state"] == "QUEUED"
    assert "ĐANG XẾP HÀNG" in sh.job("status").stdout
    sh.wait_state("a", "DONE")
    sh.wait_state("b", "DONE")
    times = dict(line.split() for line in marks.read_text().splitlines())
    assert float(times["b-start"]) >= float(times["a-end"])


def test_second_start_of_same_job_is_refused(sh):
    sh.job("start", "a", "--", "sleep", "30")
    sh.wait_state("a", "RUNNING")
    token = sh.state("a")["token"]
    out = sh.job("start", "a", "--", "sleep", "30")
    assert out.returncode == 0
    assert "không chạy thêm bản thứ hai" in out.stdout
    assert sh.state("a")["token"] == token  # trạng thái của việc đang chạy không bị ghi đè


def test_stop_all_stops_running_and_queued_jobs(sh):
    sh.job("start", "a", "--", "sleep", "60")
    sh.wait_state("a", "RUNNING")
    sh.job("start", "b", "--", "sleep", "60")
    sh.wait_state("b", "QUEUED")
    out = sh.job("stop", "--all")
    assert out.returncode == 0, out.stdout + out.stderr
    for name in ("a", "b"):
        assert sh.state(name)["state"] == "STOPPED"
        sh.wait_no_pids(name)
    assert sh.lock_free()
    assert sh.job("busy").returncode == 1
    # việc đang xếp hàng không được chạy sau khi việc trước bị dừng
    time.sleep(0.5)
    assert sh.state("b").get("started") is None


def test_stop_kills_term_ignoring_children_and_setsid_grandchildren(sh):
    script = ('trap "" TERM; setsid bash -c \'trap "" TERM; sleep 120\' & '
              'bash -c \'trap "" TERM; sleep 120\' & wait')
    sh.job("start", "stubborn", "--", "bash", "-c", script)
    sh.wait_state("stubborn", "RUNNING")
    deadline = time.monotonic() + 10
    while len(sh.token_pids("stubborn")) < 4 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert len(sh.token_pids("stubborn")) >= 4  # leader, bash -c, hai nhánh con (một nhánh đã setsid)
    out = sh.job("stop", "stubborn")
    assert "SIGKILL" in out.stdout
    sh.wait_no_pids("stubborn")
    assert sh.state("stubborn")["state"] == "STOPPED"
    assert sh.lock_free()


def test_orphan_after_leader_is_killed_still_counts_and_can_be_stopped(sh):
    sh.job("start", "orphan", "--", "sleep", "120")
    state = sh.wait_state("orphan", "RUNNING")
    os.kill(int(state["pid"]), signal.SIGKILL)  # tiến trình dẫn bị giết, "sleep" mồ côi vẫn chạy
    time.sleep(0.3)
    assert sh.token_pids("orphan"), "the orphaned child should still carry the job token"
    assert "ĐANG CHẠY" in sh.job("status").stdout
    assert sh.job("busy").returncode == 0
    assert not sh.lock_free()  # tiến trình con kế thừa khóa: không có việc nào chen vào GPU
    sh.job("stop", "orphan")
    sh.wait_no_pids("orphan")
    assert sh.state("orphan")["state"] == "STOPPED"
    assert sh.lock_free()


def test_stale_state_after_pod_restart_is_reported_lost_and_restartable(sh):
    jobs = sh.ws / "outputs" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "old.state").write_text("name=old\ntoken=gone-123\nlog=/nonexistent\ncmd=sleep 1\ncwd=/\n"
                                    "submitted=2026-10-17 01:00:00\nstate=SUBMITTED\npid=999999\nstate=QUEUED\n"
                                    "started=2026-10-17 01:00:01\nstate=RUNNING\n", encoding="utf-8")
    status = sh.job("status")
    assert "MẤT" in status.stdout
    assert sh.job("busy").returncode == 1
    assert "không chạy" in sh.job("stop", "old").stdout
    sh.job("start", "old", "--", "bash", "-c", "exit 0")
    assert sh.wait_state("old", "DONE")["token"] != "gone-123"


def test_job_started_inside_a_job_waits_for_the_outer_job(sh):
    marks = sh.root / "marks.txt"
    job_sh = REPO / "scripts" / "pod" / "job.sh"
    outer = (f"bash {job_sh} start inner -- bash -c 'echo inner $(date +%s.%N) >> {marks}'; "
             f"sleep 1; echo outer-end $(date +%s.%N) >> {marks}")
    sh.job("start", "outer", "--", "bash", "-c", outer)
    sh.wait_state("outer", "DONE")
    inner = sh.wait_state("inner", "DONE")
    assert inner["token"] != sh.state("outer")["token"]
    times = dict(line.split() for line in marks.read_text().splitlines())
    assert float(times["inner"]) >= float(times["outer-end"])


def test_job_sh_works_from_inside_a_job(sh):
    job_sh = REPO / "scripts" / "pod" / "job.sh"
    sh.job("start", "lone", "--", "bash", "-c", "exit 0")
    sh.wait_state("lone", "DONE")
    # việc "probe" xem trạng thái của chính nó và của "lone"; "lone" đã xong nên không có tiến trình nào
    probe = f"bash {job_sh} busy; echo busy=$?; bash {job_sh} status lone | head -n 2"
    sh.job("start", "probe", "--", "bash", "-c", probe)
    sh.wait_state("probe", "DONE")
    log = sh.log("probe")
    assert "busy=0" in log  # chính "probe" đang giữ khóa GPU
    assert "XONG" in log


def test_job_can_stop_itself(sh):
    job_sh = REPO / "scripts" / "pod" / "job.sh"
    marks = sh.root / "marks.txt"
    # lệnh stop chạy bên trong việc không mang token của việc, nên không tự giết mình giữa chừng
    script = f"bash {job_sh} stop self > {marks} 2>&1; sleep 60"
    sh.job("start", "self", "--", "bash", "-c", script)
    sh.wait_state("self", "STOPPED")
    sh.wait_no_pids("self")
    deadline = time.monotonic() + 10
    while "Đã dừng" not in (marks.read_text() if marks.exists() else "") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert "Đã dừng 'self'." in marks.read_text()
    # tiến trình `job.sh stop` (không mang token của việc) có thể còn giữ khóa một chút sau khi in thông báo
    deadline = time.monotonic() + 10
    while not sh.lock_free() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert sh.lock_free()


def test_gpu_lock_acquire_refuses_when_lock_is_held(sh):
    script = f'. {REPO}/scripts/lib/gpu_lock.sh; gpu_lock_acquire "thu"; echo got-lock'
    with HoldLock(sh.lock):
        out = sh.run("-c", script)
        assert out.returncode == 1
        assert "another job is using the GPU" in out.stdout and "got-lock" not in out.stdout
        # script cha đã giữ khóa (SR3D_GPU_LOCK_HELD=1): không xin lại
        out = sh.run("-c", script, env={"SR3D_GPU_LOCK_HELD": "1"})
        assert out.returncode == 0 and "got-lock" in out.stdout
    out = sh.run("-c", script)  # khóa đã nhả
    assert out.returncode == 0 and "got-lock" in out.stdout


@pytest.mark.parametrize("close_fd", [True, False])
def test_helper_that_closes_the_lock_fd_does_not_keep_the_lock(sh, close_fd):
    pid_file = sh.root / "helper.pid"
    redirect = "{SR3D_GPU_LOCK_FD}>&-" if close_fd else ""
    script = (f'. {REPO}/scripts/lib/gpu_lock.sh; gpu_lock_acquire "thu"; '
              'echo fd=$SR3D_GPU_LOCK_FD held=$SR3D_GPU_LOCK_HELD; '
              f'sleep 5 > /dev/null 2>&1 < /dev/null {redirect} & echo $! > {pid_file}; '
              'bash -c "flock -n $JOB_LOCK true" && echo free-inside || echo busy-inside')
    out = sh.run("-c", script)  # script kết thúc ngay, tiến trình phụ "sleep 5" vẫn chạy
    assert out.returncode == 0, out.stdout + out.stderr
    assert "held=1" in out.stdout and "busy-inside" in out.stdout
    helper = int(pid_file.read_text())
    try:
        assert os.path.exists(f"/proc/{helper}")
        # đóng fd khóa: khóa nhả cùng script; không đóng: tiến trình phụ còn giữ khóa
        assert sh.lock_free() is close_fd
    finally:
        os.kill(helper, signal.SIGKILL)


def fake_runpodctl(sh):
    """runpodctl giả ghi lại lệnh; trả về (env cho job.sh, file ghi lệnh)."""
    bin_dir = sh.root / "bin"
    bin_dir.mkdir(exist_ok=True)
    stops = sh.root / "runpodctl_calls.txt"
    (bin_dir / "runpodctl").write_text(f'#!/usr/bin/env bash\necho "$*" >> {stops}\n', encoding="utf-8")
    (bin_dir / "runpodctl").chmod(0o755)
    return {"PATH": f"{bin_dir}:{sh.env['PATH']}", "RUNPOD_POD_ID": "pod123"}, stops


def wait_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def request_file(sh):
    return sh.lock.with_name(sh.lock.name + ".auto_stop")


def test_auto_stop_waits_for_the_last_job(sh):
    env, stops = fake_runpodctl(sh)
    out = sh.job("start", "first", "--", "sleep", "1.5", env={**env, "AUTO_STOP": "1"})
    assert "Tự dừng pod khi việc cuối cùng xong: BẬT" in out.stdout
    sh.wait_state("first", "RUNNING")
    sh.job("start", "second", "--", "bash", "-c", "exit 0", env=env)  # xếp hàng sau, không đặt AUTO_STOP
    assert "Tự dừng pod khi việc cuối cùng xong: BẬT" in sh.job("status").stdout
    sh.wait_state("first", "DONE")
    sh.wait_state("second", "DONE")
    assert wait_file(stops)
    assert stops.read_text().splitlines() == ["pod stop pod123"]  # đúng một lần, sau việc cuối cùng
    assert "pod sẽ dừng khi việc cuối cùng xong" in sh.log("first")
    assert not request_file(sh).exists()


def test_auto_stop_zero_cancels_the_request(sh):
    env, stops = fake_runpodctl(sh)
    sh.job("start", "first", "--", "sleep", "1", env={**env, "AUTO_STOP": "1"})
    sh.wait_state("first", "RUNNING")
    out = sh.job("start", "second", "--", "bash", "-c", "exit 0", env={**env, "AUTO_STOP": "0"})
    assert "Tự dừng pod: TẮT" in out.stdout
    sh.wait_state("first", "DONE")
    sh.wait_state("second", "DONE")
    time.sleep(0.5)
    assert not stops.exists()


def test_stopping_every_job_cancels_auto_stop(sh):
    env, stops = fake_runpodctl(sh)
    sh.job("start", "armed", "--", "sleep", "60", env={**env, "AUTO_STOP": "1"})
    sh.wait_state("armed", "RUNNING")
    out = sh.job("stop", "--all", env=env)
    assert "Tự dừng pod: TẮT" in out.stdout and not request_file(sh).exists()
    sh.job("start", "later", "--", "bash", "-c", "exit 0", env=env)  # việc sau đó không được dừng pod
    sh.wait_state("later", "DONE")
    time.sleep(0.5)
    assert not stops.exists()


def test_repeated_start_can_arm_auto_stop_for_a_running_job(sh):
    env, stops = fake_runpodctl(sh)
    sh.job("start", "long", "--", "sleep", "1.5", env=env)
    sh.wait_state("long", "RUNNING")
    out = sh.job("start", "long", "--", "sleep", "1.5", env={**env, "AUTO_STOP": "1"})
    assert "không chạy thêm bản thứ hai" in out.stdout and "BẬT" in out.stdout
    sh.wait_state("long", "DONE")
    assert wait_file(stops) and stops.read_text().splitlines() == ["pod stop pod123"]


def test_request_from_a_previous_pod_session_is_ignored(sh):
    env, stops = fake_runpodctl(sh)
    request_file(sh).write_text("session=previous-pod\nrequested=2026-10-16 10:00:00\nby=old\n", encoding="utf-8")
    assert "BẬT" not in sh.job("status", env=env).stdout
    sh.job("start", "fresh", "--", "bash", "-c", "exit 0", env=env)
    sh.wait_state("fresh", "DONE")
    time.sleep(0.5)
    assert not stops.exists() and not request_file(sh).exists()


def test_unknown_command_prints_usage(sh):
    out = sh.job("frobnicate")
    assert out.returncode == 2
    assert "job.sh start" in out.stdout


def test_invalid_job_name_is_rejected(sh):
    out = sh.job("start", "bad name", "--", "true")
    assert out.returncode == 2
    assert "job name" in out.stderr
