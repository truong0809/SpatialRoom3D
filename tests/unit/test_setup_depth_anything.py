"""Test scripts/setup/setup_depth_anything.sh không cần mạng: một kho git cục bộ thay cho GitHub, `file://` thay cho
Hugging Face (file cấu hình riêng qua CONF, cùng các khóa DA2_* của configs/environment/gs-gsplat.conf)."""

import hashlib
import os
import shutil
import subprocess

import pytest

from shell_env import REPO

pytestmark = pytest.mark.skipif(not (shutil.which("git") and shutil.which("curl")), reason="needs git and curl")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def da2(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    git("init", "-q", "-b", "main", cwd=source)
    (source / "README.md").write_text("fake Depth Anything V2\n", encoding="utf-8")
    git("add", "-A", cwd=source)
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", "fake", cwd=source)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fake weights")
    state = {"source": source, "commit": git("rev-parse", "HEAD", cwd=source), "weights": weights,
             "ws": tmp_path / "ws", "conf": tmp_path / "da2.conf"}
    state["ws"].mkdir()

    def write_conf(sha256=hashlib.sha256(weights.read_bytes()).hexdigest()):
        state["conf"].write_text(f'DA2_REPO="{source}"\nDA2_COMMIT="{state["commit"]}"\n'
                                 'DA2_WEIGHTS_FILE="checkpoints/depth_anything_v2_vits.pth"\n'
                                 f'DA2_WEIGHTS_URL="file://{weights}"\nDA2_WEIGHTS_SHA256="{sha256}"\n', encoding="utf-8")

    def run():
        env = dict(os.environ, WS=str(state["ws"]), CONF=str(state["conf"]))
        return subprocess.run(["bash", str(REPO / "scripts" / "setup" / "setup_depth_anything.sh")], env=env,
                              capture_output=True, text=True, timeout=120)

    write_conf()
    state.update(write_conf=write_conf, run=run, dir=state["ws"] / "code" / "depth_anything_v2")
    return state


def test_installs_the_pinned_code_and_weights_and_reruns_safely(da2):
    out = da2["run"]()
    assert out.returncode == 0, out.stdout + out.stderr
    weights = da2["dir"] / "checkpoints" / "depth_anything_v2_vits.pth"
    assert weights.read_bytes() == b"fake weights"
    assert git("rev-parse", "HEAD", cwd=da2["dir"]) == da2["commit"]
    assert "Depth Anything V2 Small ready" in out.stdout
    again = da2["run"]()
    assert again.returncode == 0 and "already downloaded" in again.stdout


def test_weights_with_another_sha256_are_removed(da2):
    da2["write_conf"]("0" * 64)
    out = da2["run"]()
    assert out.returncode != 0 and "do not have the pinned SHA-256" in out.stderr
    assert not list((da2["dir"] / "checkpoints").glob("*"))  # không còn file tải về, kể cả bản dở


def test_a_folder_that_is_not_a_checkout_is_left_alone(da2):
    (da2["dir"] / "checkpoints").mkdir(parents=True)
    (da2["dir"] / "checkpoints" / "mine.pth").write_bytes(b"copied by hand")
    out = da2["run"]()
    assert out.returncode != 0 and "is not a git checkout" in out.stderr
    assert (da2["dir"] / "checkpoints" / "mine.pth").read_bytes() == b"copied by hand"


def test_an_empty_folder_is_replaced_by_the_checkout(da2):
    da2["dir"].mkdir(parents=True)
    out = da2["run"]()
    assert out.returncode == 0, out.stdout + out.stderr
    assert git("rev-parse", "HEAD", cwd=da2["dir"]) == da2["commit"]


def test_a_failed_clone_is_reported_and_leaves_nothing_behind(da2):
    shutil.rmtree(da2["source"])  # kho nguồn không còn (như mạng lỗi)
    out = da2["run"]()
    assert out.returncode != 0 and f"ERROR: cannot clone {da2['source']}" in out.stderr
    assert not da2["dir"].exists() and not da2["dir"].with_name(da2["dir"].name + ".partial").exists()


def test_the_checkout_follows_the_pinned_commit(da2):
    def upstream_commit(name):
        (da2["source"] / f"{name}.md").write_text(f"{name}\n", encoding="utf-8")
        git("add", "-A", cwd=da2["source"])
        git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
            "commit", "-q", "-m", name, cwd=da2["source"])
        return git("rev-parse", "HEAD", cwd=da2["source"])

    assert da2["run"]().returncode == 0
    da2["commit"] = upstream_commit("newer")  # commit ghim mới, sau lần clone: phải fetch
    da2["write_conf"]()
    out = da2["run"]()
    assert out.returncode == 0, out.stdout + out.stderr
    assert git("rev-parse", "HEAD", cwd=da2["dir"]) == da2["commit"]
    upstream_commit("newest")  # nguồn đi trước commit ghim: clone mới vẫn về đúng commit ghim
    shutil.rmtree(da2["dir"])
    out = da2["run"]()
    assert out.returncode == 0, out.stdout + out.stderr
    assert git("rev-parse", "HEAD", cwd=da2["dir"]) == da2["commit"]


def test_a_modified_checkout_is_refused(da2):
    assert da2["run"]().returncode == 0
    (da2["dir"] / "README.md").write_text("changed\n", encoding="utf-8")
    out = da2["run"]()
    assert out.returncode != 0 and "local modifications" in out.stderr
