"""Test scripts/pod/run_segments.sh (đánh giá cảnh tự quay với tập test là các đoạn liền của video).

Chỉ chạy phần không cần GPU: kiểm tra đầu vào và DRY_RUN=1 (in đúng các lệnh: chia ảnh, train default rồi mcmc cho
từng seed vào thư mục kết quả riêng, lập bảng). Phần chia ảnh có test riêng (tests/unit/test_train_split.py); train
và lập bảng là các script đã có test.
"""

from colmap_writer import make_model
from shell_env import REPO, ShellEnv

SCRIPT = str(REPO / "scripts" / "pod" / "run_segments.sh")


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


def test_dry_run_prints_the_split_the_runs_and_the_report(tmp_path):
    sh = ShellEnv(tmp_path)
    try:
        out = sh.run(SCRIPT, "phong_nhom", env={"DRY_RUN": "1"})
        assert out.returncode == 1 and "no COLMAP model for phong_nhom" in out.stdout
        scene = sh.ws / "outputs" / "custom_scenes" / "phong_nhom" / "colmap"
        make_model(scene / "sparse" / "0", names(40))
        out = sh.run(SCRIPT, "phong_nhom", env={"DRY_RUN": "1", "SEEDS": "42 43"})
        assert out.returncode == 0, out.stdout + out.stderr
        lines = [line for line in out.stdout.splitlines() if not line.startswith("#")]
        seg = sh.ws / "outputs" / "custom_segments"
        assert lines[0].endswith(f"-m src.training.split segments --scene {scene} --out {seg}/phong_nhom/split "
                                 "--blocks 6 --test-fraction 0.125 --gap 3")
        runs = lines[1:-1]
        assert [(line.split()[1], line.split()[6]) for line in runs] == [
            ("SEED=42", "default"), ("SEED=42", "mcmc"), ("SEED=43", "default"), ("SEED=43", "mcmc")]
        assert all(line.startswith(f"OUT_ROOT={seg} ") and f"--test-list {seg}/phong_nhom/split/test.txt "
                   f"--exclude-list {seg}/phong_nhom/split/exclude.txt" in line for line in runs)
        assert lines[-1].startswith(f"CUSTOM_ROOT={seg} REPORT={sh.ws}/outputs/reports/segments_phong_nhom_")
        assert lines[-1].endswith("scripts/run_evaluate.sh custom phong_nhom")
        assert not seg.exists()  # DRY_RUN: không ghi gì
    finally:
        sh.cleanup()


def test_bad_settings_are_refused(tmp_path):
    sh = ShellEnv(tmp_path)
    try:
        make_model(sh.ws / "outputs" / "custom_scenes" / "p1" / "colmap" / "sparse" / "0", names(40))
        cases = [({}, "../p1", "invalid scene name"), ({"MODES": "mcmc"}, "p1", "must start with default"),
                 ({"SEEDS": "4x"}, "p1", "invalid seed"), ({"GAP": "-1"}, "p1", "whole numbers"),
                 ({"FRACTION": "2"}, "p1", "between 0 and 0.5")]
        for env, scene, message in cases:
            out = sh.run(SCRIPT, scene, env={"DRY_RUN": "1", **env})
            assert out.returncode == 2 and message in out.stdout, (env, out.stdout)
        assert sh.run(SCRIPT, env={"DRY_RUN": "1"}).returncode == 2  # thiếu tên cảnh
    finally:
        sh.cleanup()
