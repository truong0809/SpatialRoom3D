"""Test scripts/pod/run_segments.sh (đánh giá cảnh tự quay với tập test là các đoạn liền của video).

Chỉ chạy phần không cần GPU. DRY_RUN=1 trên repo thật: kiểm tra đầu vào và các lệnh in ra (chia ảnh, train default
rồi mcmc rồi các biến thể cho từng seed vào thư mục kết quả riêng, lập bảng). Chạy thật trên repo thu nhỏ
(make_fake_repo): chia ảnh thật, train_scene.sh và run_evaluate.sh là stub ghi lại lời gọi; kiểm tra thứ tự, biến môi
trường, cách xử lý khi một lượt lỗi và việc không ghi gì vào thư mục COLMAP nguồn. Phần chia ảnh có test riêng
(tests/unit/test_train_split.py); train và lập bảng là các script đã có test.
"""

import json
import re

import pytest

from colmap_writer import make_model
from shell_env import REPO, ShellEnv, calls, make_fake_repo, write_stub

SCRIPT = str(REPO / "scripts" / "pod" / "run_segments.sh")
VARIANTS = "mcmc:mono:sr3d.mono=true mcmc:kr:sr3d.kr=true mcmc:rbg:random_bkgd=true"
WEIGHTS = ("code", "depth_anything_v2", "checkpoints", "depth_anything_v2_vits.pth")  # DA2_WEIGHTS_FILE trong $WS/code

# stub: ghi lời gọi; FAIL_TAG làm lỗi biến thể có tag đó, FAIL_MODE làm lỗi lượt chuẩn (không tag) của chế độ đó
TRAIN_STUB = ('echo "train $2 SEED=$SEED TAG=$TAG BASELINE_TAG=$BASELINE_TAG OUT_ROOT=$OUT_ROOT SCENE=$1 '
              'ARGS=${*:3}" >> "$CALLS"\n'
              '[ -z "$TAG" ] || [ "$TAG" != "${FAIL_TAG:-}" ] || exit 1\n'
              '[ -n "$TAG" ] || [ "$2" != "${FAIL_MODE:-}" ] || exit 1\n')
EVAL_STUB = ('echo "evaluate $* CUSTOM_ROOT=$CUSTOM_ROOT REPORT=${REPORT##*/} BASELINE=${BASELINE:-} '
             'MODES=${MODES:-} TAGS=${TAGS:-}" >> "$CALLS"\n')


def names(n):
    return [f"{i:05d}.jpg" for i in range(1, n + 1)]


def run_line(line):
    """(seed, tag, chế độ, các --set) của một dòng train do DRY_RUN in ra."""
    tokens = line.split()
    return tokens[1], tokens[2], tokens[7], [tokens[i + 1] for i, token in enumerate(tokens) if token == "--set"]


def snapshot(root):
    return sorted((str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in root.rglob("*"))


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
        assert [run_line(line) for line in runs] == [
            ("SEED=42", "TAG=", "default", []), ("SEED=42", "TAG=", "mcmc", []),
            ("SEED=43", "TAG=", "default", []), ("SEED=43", "TAG=", "mcmc", [])]
        assert all(line.startswith(f"OUT_ROOT={seg} ") and line.split()[3] == "BASELINE_TAG="
                   and f"--test-list {seg}/phong_nhom/split/test.txt "
                   f"--exclude-list {seg}/phong_nhom/split/exclude.txt" in line for line in runs)
        assert lines[-1].startswith(f"CUSTOM_ROOT={seg} REPORT={sh.ws}/outputs/reports/segments_phong_nhom_")
        assert lines[-1].endswith("scripts/run_evaluate.sh custom phong_nhom") and "TAGS=" not in lines[-1]
        assert not seg.exists()  # DRY_RUN: không ghi gì
    finally:
        sh.cleanup()


def test_dry_run_with_variants_on_a_room_built_for_the_product(tmp_path):
    sh = ShellEnv(tmp_path)
    try:
        product = sh.ws / "outputs" / "product"
        make_model(product / "video10" / "colmap" / "sparse" / "0", names(40))
        out = sh.run(SCRIPT, "video10", env={"DRY_RUN": "1"})  # mặc định đọc outputs/custom_scenes: chỉ cách dùng
        assert out.returncode == 1 and f"SRC_ROOT={sh.ws}/outputs/product" in out.stdout
        env = {"DRY_RUN": "1", "SEEDS": "42 43", "SRC_ROOT": str(product), "VARIANTS": VARIANTS}
        out = sh.run(SCRIPT, "video10", env=env)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "# WARN: Depth Anything V2 Small not found" in out.stdout  # DRY_RUN chỉ cảnh báo
        lines = [line for line in out.stdout.splitlines() if not line.startswith("#")]
        seg = sh.ws / "outputs" / "custom_segments"
        assert f"--scene {product}/video10/colmap --out {seg}/video10/split " in lines[0]
        expected = []
        for seed in ("SEED=42", "SEED=43"):  # mỗi seed: các chế độ rồi các biến thể
            expected += [(seed, "TAG=", "default", []), (seed, "TAG=", "mcmc", []),
                         (seed, "TAG=mono", "mcmc", ["sr3d.mono=true"]), (seed, "TAG=kr", "mcmc", ["sr3d.kr=true"]),
                         (seed, "TAG=rbg", "mcmc", ["random_bkgd=true"])]
        assert [run_line(line) for line in lines[1:-2]] == expected
        # ngân sách Gaussian của biến thể: lượt default không tag của cùng seed trên cùng tập chia
        assert all(line.split()[3] == "BASELINE_TAG=" and f" {product}/video10/colmap " in line for line in lines[1:-2])
        report = f"REPORT={sh.ws}/outputs/reports/segments_video10_"
        assert lines[-2].startswith(f"CUSTOM_ROOT={seg} {report}") and ' TAGS="mono kr rbg" bash ' in lines[-2]
        assert lines[-1].startswith(f"CUSTOM_ROOT={seg} {report}")
        assert '_vs_mcmc BASELINE=mcmc MODES=mcmc TAGS="mono kr rbg" bash ' in lines[-1]
        assert not seg.exists()
        weights = sh.ws.joinpath(*WEIGHTS)
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"")
        out = sh.run(SCRIPT, "video10", env=env)
        assert out.returncode == 0 and "WARN" not in out.stdout, out.stdout
    finally:
        sh.cleanup()


def test_bad_settings_are_refused(tmp_path):
    sh = ShellEnv(tmp_path)
    try:
        make_model(sh.ws / "outputs" / "custom_scenes" / "p1" / "colmap" / "sparse" / "0", names(40))
        cases = [({}, "../p1", "invalid scene name"), ({"MODES": "mcmc"}, "p1", "must start with default"),
                 ({"SEEDS": "4x"}, "p1", "invalid seed"), ({"GAP": "-1"}, "p1", "whole numbers"),
                 ({"FRACTION": "2"}, "p1", "between 0 and 0.5"),
                 ({"SEG_ROOT": f"{sh.ws}/outputs/custom_scenes/"}, "p1", "SEG_ROOT must differ from SRC_ROOT"),
                 ({"VARIANTS": "mcmc:mono"}, "p1", "invalid entry in VARIANTS"),
                 ({"VARIANTS": "mcmc:Mono:sr3d.mono=true"}, "p1", "invalid entry in VARIANTS"),
                 ({"VARIANTS": "default:x:random_bkgd=true"}, "p1", "is a variant of default"),
                 ({"VARIANTS": "foo:x:random_bkgd=true"}, "p1", "is not in MODES"),
                 ({"MODES": "default mcmc mcmc_x", "VARIANTS": "mcmc:x:random_bkgd=true"}, "p1",
                  "mcmc_x is also a mode in MODES"),
                 ({"VARIANTS": "mcmc:x:random_bkgd=true mcmc:x:sr3d.kr=true"}, "p1", "mcmc_x is listed twice"),
                 ({"VARIANTS": "mcmc:x:random_bkgd=true;id"}, "p1", "invalid parameter")]
        for env, scene, message in cases:
            out = sh.run(SCRIPT, scene, env={"DRY_RUN": "1", **env})
            assert out.returncode == 2 and message in out.stdout, (env, out.stdout)
        assert sh.run(SCRIPT, env={"DRY_RUN": "1"}).returncode == 2  # thiếu tên cảnh
    finally:
        sh.cleanup()


@pytest.fixture
def fake(tmp_path):
    pytest.importorskip("numpy")  # bước chia ảnh đọc mô hình COLMAP bằng Python có numpy
    repo = make_fake_repo(tmp_path, ["pod/run_segments.sh"], with_src=True, with_configs=True)
    write_stub(repo, "scripts/pipeline/train_scene.sh", TRAIN_STUB)
    write_stub(repo, "scripts/run_evaluate.sh", EVAL_STUB)
    sh = ShellEnv(tmp_path, repo=repo)
    sh.env["CALLS"] = str(tmp_path / "calls.txt")
    yield sh
    sh.cleanup()


def run_segments(sh, scene, **env):
    return sh.run(str(sh.repo / "scripts" / "pod" / "run_segments.sh"), scene, env=env, timeout=120)


def test_runs_each_seed_then_its_variants_and_keeps_going_after_a_failed_variant(fake):
    product = fake.ws / "outputs" / "product"
    scene = product / "video10" / "colmap"
    make_model(scene / "sparse" / "0", names(120))
    before = snapshot(product)
    seg = fake.ws / "outputs" / "custom_segments"
    env = {"SRC_ROOT": str(product), "SEEDS": "42 43", "FAIL_TAG": "kr",
           "VARIANTS": "mcmc:mono:sr3d.mono=true mcmc:kr:sr3d.kr=true"}
    out = run_segments(fake, "video10", **env)  # chưa có Depth Anything V2: dừng trước khi chia ảnh hay train
    assert out.returncode == 1 and "Depth Anything V2 Small not found" in out.stdout, out.stdout
    assert calls(fake.root) == [] and not seg.exists()

    weights = fake.ws.joinpath(*WEIGHTS)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"")
    out = run_segments(fake, "video10", **env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "failed variant runs: mcmc_kr_s42 mcmc_kr_s43" in out.stdout  # lỗi biến thể: chạy tiếp, cuối cùng exit 1
    assert f"scene: {scene} -> {seg}/video10 | seeds: 42 43 | modes: default mcmc | variants: mcmc:mono:" in out.stdout
    split = seg / "video10" / "split"
    common = f"BASELINE_TAG= OUT_ROOT={seg} SCENE={scene} ARGS=--test-list {split}/test.txt " \
             f"--exclude-list {split}/exclude.txt"
    expected = []
    for seed in (42, 43):
        expected += [f"train default SEED={seed} TAG= {common}", f"train mcmc SEED={seed} TAG= {common}",
                     f"train mcmc SEED={seed} TAG=mono {common} --set sr3d.mono=true",
                     f"train mcmc SEED={seed} TAG=kr {common} --set sr3d.kr=true"]
    expected += [f"evaluate custom video10 CUSTOM_ROOT={seg} REPORT=segments_video10_<t> BASELINE= MODES= "
                 "TAGS=mono kr",
                 f"evaluate custom video10 CUSTOM_ROOT={seg} REPORT=segments_video10_<t>_vs_mcmc BASELINE=mcmc "
                 "MODES=mcmc TAGS=mono kr"]
    assert [re.sub(r"_\d{8}_\d{6}", "_<t>", line) for line in calls(fake.root)] == expected
    assert (split / "test.txt").is_file() and (split / "exclude.txt").is_file()
    assert json.loads((split / "segments.json").read_text(encoding="utf-8"))["scene"] == str(scene.resolve())
    assert snapshot(product) == before  # không ghi gì vào thư mục COLMAP nguồn


def test_a_failed_mode_stops_before_the_next_runs_and_the_report(fake):
    make_model(fake.ws / "outputs" / "custom_scenes" / "phong_nhom" / "colmap" / "sparse" / "0", names(120))
    out = run_segments(fake, "phong_nhom", SEEDS="42 43", FAIL_MODE="mcmc", VARIANTS="mcmc:rbg:random_bkgd=true")
    assert out.returncode == 1 and "mcmc seed 42 failed" in out.stdout, out.stdout
    assert [line.split()[:3] for line in calls(fake.root)] == [["train", "default", "SEED=42"],
                                                               ["train", "mcmc", "SEED=42"]]
