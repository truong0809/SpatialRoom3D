"""Test scripts/pipeline/train_scene.sh thật (huấn luyện một cảnh tự chụp), gọi qua scripts/run_train.sh custom.

conda, GPU và src.training.run được thay bằng stub (src.training.run giả chỉ ghi summary.json vào --out). Kiểm tra:
- mỗi chế độ gọi src.training.run với đúng tham số như repo trước (8037a66), chỉ khác tên module và đường dẫn;
- mcmc lấy ngân sách Gaussian từ lượt default cùng seed (chính sách match_baseline trong configs/base.yaml);
- mcmc chưa có lượt default thì dừng trước khi train; mcmc_demo (cap_max cố định) không cần lượt default.
"""

import os

import pytest

from shell_env import ShellEnv, calls, make_fake_repo

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

FAKE_PYTHON = r'''#!/usr/bin/env bash
echo "python $*" >> "$CALLS"
case "${1:-}" in
  -c) exit 0 ;;  # kiểm tra fused-ssim / gsplat trên GPU
  -) exec python3 "$@" ;;  # đoạn script viết thẳng trong file .sh (đọc configs/base.yaml)
  -m)
    if [ "${2:-}" = src.training.run ]; then
      out="" prev=""
      for arg in "$@"; do [ "$prev" = --out ] && out="$arg"; prev="$arg"; done
      mkdir -p "$out" && echo '{"status": "OK"}' > "$out/summary.json"
      exit 0
    fi
    exec python3 "$@" ;;
  *) echo "fake python: unexpected $*" >&2; exit 3 ;;
esac
'''


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["run_train.sh", "pipeline/train_scene.sh"], with_src=True, with_configs=True)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(FAKE_PYTHON, encoding="utf-8")
    (bin_dir / "python").chmod(0o755)
    env.env["PATH"] = f"{bin_dir}:{env.env['PATH']}"
    conda = env.ws / "miniforge3" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
    scene = env.ws / "outputs" / "custom_scenes" / "phong1" / "colmap"
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "images").mkdir()
    yield env
    env.cleanup()


def training_calls(sh):
    prefix = "python -m src.training.run "
    return [line[len(prefix):] for line in calls(sh.root) if line.startswith(prefix)]


def test_default_then_mcmc_with_the_baseline_budget(sh):
    out = sh.run(str(sh.repo / "scripts" / "run_train.sh"), "custom", "phong1")
    assert out.returncode == 0, out.stdout + out.stderr
    root = sh.ws / "outputs" / "custom_scenes" / "phong1"
    common = f"--scene {root / 'colmap'} --scene-name phong1"
    tail = f"--seed 42 --gsplat-dir {sh.ws / 'code' / 'gsplat'} --rerun-incomplete"
    # cùng dạng lệnh với scripts/train_scene.sh của repo trước: chỉ khác tên module và đường dẫn
    assert training_calls(sh) == [
        f"--mode default {common} --out {root / 'default_s42'} {tail}",
        f"--mode mcmc {common} --out {root / 'mcmc_s42'} {tail} --cap-max-from {root / 'default_s42'}",
    ]
    assert sorted((sh.ws / "outputs" / "logs").glob("train_phong1_default_s42_*.log"))


def test_mcmc_needs_a_finished_default_run(sh):
    out = sh.run(str(sh.repo / "scripts" / "run_train.sh"), "custom", "phong1", "mcmc")
    assert out.returncode == 1 and "needs a finished default run first" in out.stdout
    assert training_calls(sh) == []


def test_demo_mode_has_a_fixed_budget_and_extra_options_pass_through(sh):
    out = sh.run(str(sh.repo / "scripts" / "run_train.sh"), "custom", "phong1", "mcmc_demo", "--", "--dry-run",
                 env={"SEED": "7"})
    assert out.returncode == 0, out.stdout + out.stderr
    (line,) = training_calls(sh)
    assert line.startswith("--mode mcmc_demo ") and "--cap-max-from" not in line
    assert " --seed 7 " in line and line.endswith("--rerun-incomplete --dry-run")
