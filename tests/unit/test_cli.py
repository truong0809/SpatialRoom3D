"""Test giao diện dòng lệnh app/cli: mỗi lệnh gọi đúng script trong scripts/, tham số đi nguyên."""

import subprocess
import sys

from app.cli.main import COMMANDS, REPO_DIR, command_for, main
from shell_env import REPO


def test_every_command_points_to_an_existing_script():
    assert REPO_DIR == REPO
    for name, (script, _, _) in COMMANDS.items():
        assert (REPO / script).is_file(), (name, script)
    assert {"prepare", "train", "render", "evaluate"} <= set(COMMANDS)  # bốn bước chính của scripts/
    assert COMMANDS["fetch"][0] == "scripts/pod/fetch_data.sh"
    assert COMMANDS["experiment"][0] == "scripts/pod/run_experiment.sh"
    assert COMMANDS["filter"][0] == "scripts/run_filter.sh"


def test_arguments_pass_through_unchanged():
    assert command_for("train", ["custom", "phong1", "mcmc", "--", "--set", "opacity_reg=0.001"]) == [
        "bash", str(REPO / "scripts" / "run_train.sh"), "custom", "phong1", "mcmc", "--", "--set", "opacity_reg=0.001"]
    assert command_for("setup", []) == ["bash", str(REPO / "scripts" / "pod" / "run_all.sh"), "--background"]
    assert "baseline" not in COMMANDS  # base chỉ cài môi trường; chạy với dữ liệu nào do người dùng chọn
    assert command_for("stop", ["--all"])[-2:] == ["stop", "--all"]


def test_dry_run_help_and_unknown_command(capsys):
    assert main(["--dry-run", "scene", "/x/phong1", "phong1", "--background"]) == 0
    assert capsys.readouterr().out.strip().endswith("scripts/pod/run_scene.sh /x/phong1 phong1 --background")
    assert main(["--help"]) == 0 and "Các lệnh:" in capsys.readouterr().out
    assert main([]) == 2
    assert main(["nope"]) == 2 and "lệnh không có: nope" in capsys.readouterr().err


def test_module_entry_point_runs_the_script():
    out = subprocess.run([sys.executable, "-m", "app.cli", "evaluate"], cwd=REPO, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 2 and "run_evaluate.sh" in out.stdout  # script in hướng dẫn khi thiếu tham số
