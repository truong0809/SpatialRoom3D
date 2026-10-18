"""Tiện ích cho test các script bash (scripts/pod/job.sh, run_all.sh, run_scene.sh...): môi trường tách biệt trong tmp_path.

Mỗi test có WS và file khóa GPU riêng; mọi tiến trình do test sinh ra mang biến SR3D_TEST_ID riêng để
dọn sạch khi test kết thúc (kể cả khi test lỗi giữa chừng).
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class ShellEnv:
    def __init__(self, tmp_path: Path, repo: Path = REPO):
        self.root = tmp_path
        self.repo = repo
        self.ws = tmp_path / "ws"
        self.ws.mkdir(exist_ok=True)
        self.lock = tmp_path / "gpu.lock"
        self.test_id = uuid.uuid4().hex
        # python3 của các script = Python đang chạy test (có PyYAML để đọc configs/*.yaml), kể cả khi venv chưa activate
        self.env = {
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "LANG": "C.UTF-8",
            "WS": str(self.ws),
            "JOB_LOCK": str(self.lock),
            "STOP_TIMEOUT": "3",
            "SR3D_TEST_ID": self.test_id,
        }

    # ------------------------------------------------------------------------------------------ chạy
    def run(self, *args: str, env: dict | None = None, timeout: float = 60, cwd: Path | None = None):
        merged = {**self.env, **(env or {})}
        return subprocess.run(["bash", *args], env=merged, cwd=cwd or self.repo, capture_output=True,
                              text=True, timeout=timeout)

    def job(self, *args: str, env: dict | None = None, timeout: float = 60):
        return self.run(str(self.repo / "scripts" / "pod" / "job.sh"), *args, env=env, timeout=timeout)

    # -------------------------------------------------------------------------------- trạng thái việc
    def state_file(self, name: str) -> Path:
        return self.ws / "outputs" / "jobs" / f"{name}.state"

    def state(self, name: str) -> dict:
        values: dict[str, str] = {}
        path = self.state_file(name)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    values[key] = value  # dòng sau cùng của mỗi khóa là giá trị hiện tại
        return values

    def wait_state(self, name: str, *states: str, timeout: float = 30) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.state(name)
            if current.get("state") in states:
                return current
            time.sleep(0.1)
        raise AssertionError(f"job {name} did not reach {states}: {self.state(name)}\nlog:\n{self.log(name)}")

    def log(self, name: str) -> str:
        path = self.ws / "outputs" / "logs" / "jobs" / f"{name}.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    # ------------------------------------------------------------------------------------- tiến trình
    def pids_with(self, needle: str) -> list[int]:
        found = []
        for environ in Path("/proc").glob("[0-9]*/environ"):
            try:
                data = environ.read_bytes()
            except OSError:
                continue
            if needle.encode() in data.split(b"\0"):
                found.append(int(environ.parent.name))
        return [pid for pid in found if pid != os.getpid()]

    def test_pids(self) -> list[int]:
        return self.pids_with(f"SR3D_TEST_ID={self.test_id}")

    def token_pids(self, name: str) -> list[int]:
        token = self.state(name).get("token")
        return self.pids_with(f"SR3D_JOB_TOKEN={token}") if token else []

    def wait_no_pids(self, name: str, timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.token_pids(name):
                return
            time.sleep(0.1)
        raise AssertionError(f"job {name} still has processes: {self.token_pids(name)}")

    def lock_free(self) -> bool:
        return subprocess.run(["flock", "-n", str(self.lock), "true"], capture_output=True).returncode == 0

    def cleanup(self) -> None:
        for _ in range(3):
            pids = self.test_pids()
            if not pids:
                return
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            time.sleep(0.2)


class HoldLock:
    """Giữ khóa GPU như một việc khác đang chạy (flock + sleep trong session riêng); thoát thì nhả hẳn.

    `flock <file> sleep` chạy sleep như tiến trình con mang fd khóa: chỉ giết flock thì khóa vẫn bị sleep giữ,
    nên phải giết cả nhóm tiến trình.
    """

    def __init__(self, lock: Path):
        self.lock = lock

    def __enter__(self):
        self.proc = subprocess.Popen(["flock", str(self.lock), "sleep", "60"], start_new_session=True)
        deadline = time.monotonic() + 10
        while subprocess.run(["flock", "-n", str(self.lock), "true"]).returncode == 0:
            if time.monotonic() > deadline:
                raise AssertionError("could not take the lock")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        self.proc.wait()
        deadline = time.monotonic() + 10
        while subprocess.run(["flock", "-n", str(self.lock), "true"]).returncode != 0:
            if time.monotonic() > deadline:
                raise AssertionError("lock still held after the holder was killed")
            time.sleep(0.02)


# ------------------------------------------------------------------------------------- repo giả cho stub
def make_fake_repo(root: Path, real_scripts: list[str], with_src: bool = False, with_configs: bool = False) -> Path:
    """Bản sao tối thiểu của repo: các script thật được chép, các script khác thay bằng stub (write_stub).

    `real_scripts`: đường dẫn tính từ scripts/ (vd. "pod/run_scene.sh", "run_train.sh"); scripts/lib/ và
    scripts/pod/job.sh luôn được chép."""
    repo = root / "repo"
    (repo / "scripts" / "lib").mkdir(parents=True, exist_ok=True)
    for lib in sorted((REPO / "scripts" / "lib").glob("*.sh")):
        shutil.copy2(lib, repo / "scripts" / "lib" / lib.name)
    for name in ["pod/job.sh", *real_scripts]:
        (repo / "scripts" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "scripts" / name, repo / "scripts" / name)
    if with_src:
        shutil.copytree(REPO / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    if with_configs:
        shutil.copytree(REPO / "configs", repo / "configs")
    return repo


def write_stub(repo: Path, name: str, body: str) -> Path:
    """Stub bash ghi lời gọi vào $CALLS (một dòng: tên + tham số + biến môi trường quan trọng).

    `name` tính từ gốc repo (vd. scripts/pipeline/prepare_scene.sh).
    """
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -u\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def calls(root: Path) -> list[str]:
    path = root / "calls.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
