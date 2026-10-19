"""Test kế hoạch thực nghiệm (src/training/experiment.py) và script điều phối scripts/pod/run_experiment.sh.

Script điều phối chạy thật trong một repo giả; các bước nó gọi (tải dữ liệu, B0, gsplat, báo cáo, đóng gói) là stub ghi
lại lời gọi kèm các biến môi trường. Bước gsplat giả ghi mỗi lượt vào đúng thư mục của nó
(`<kết quả>/<cảnh>/<chế độ>_s<seed>/` có summary.json OK, bỏ qua lượt đã xong như pipeline thật); dung lượng của lượt là
file thưa (sparse): `du -b` đếm kích thước khai báo nhưng file không chiếm chỗ thật.
"""

import json
import os
import shutil

import pytest
import yaml

from shell_env import REPO, ShellEnv, calls, make_fake_repo, write_stub
from src.training.experiment import load_experiment, main, plan_lines, space_needs, tree_bytes

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

STUBS = {
    "scripts/pod/fetch_data.sh":
        'echo "fetch $* | QUICK=${QUICK-unset}" >> "$CALLS"\n'
        'case " ${FAKE_FETCH_FAIL:-} " in *" $1 "*) echo "ERROR: fake download failure"; exit 1 ;; esac\n',
    "scripts/run_train.sh":
        'echo "train $* | MODES=${MODES-} SEEDS=${SEEDS-} SCENES=${SCENES-} QUICK=${QUICK-unset}'
        ' MCMC_SET=${MCMC_SET-unset} OUT_ROOT=${OUT_ROOT-unset} FORCE=${FORCE-unset}" >> "$CALLS"\n'
        'if [ "$2" = gsplat ] && [ -n "${FAKE_RUN_SIZE:-}" ]; then\n'
        '  for scene in $SCENES; do for mode in $MODES; do\n'
        '    out="$WS/outputs/benchmarks/$1/gsplat/$scene/${mode}_s$SEEDS"\n'
        '    [ -f "$out/summary.json" ] && continue\n'
        '    mkdir -p "$out"; truncate -s "$FAKE_RUN_SIZE" "$out/blob"; echo \'{"status": "OK"}\' > "$out/summary.json"\n'
        '  done; done\n'
        'fi\n'
        'exit "${FAKE_TRAIN_RC:-0}"\n',
    "scripts/run_evaluate.sh":
        'echo "evaluate $* | BASELINE=${BASELINE-unset} MODES=${MODES-} SCENES=${SCENES-} QUICK=${QUICK-unset}"'
        ' >> "$CALLS"\n'
        'printf "| bảng so sánh %s |\\n" "$1" > "$REPORT.md"\n',
    "scripts/pod/pack_results.sh": 'echo "pack $*" >> "$CALLS"\n',
}
FAKE_PYTHON = r'''#!/usr/bin/env bash
# python của env conda giả: kiểm tra import (-c) luôn qua; smoke test tham số các chế độ ghi lại lời gọi
case "${1:-}" in
  -c) exit 0 ;;
  */smoke_test_gsplat.py) shift; echo "smoke $*" >> "$CALLS"; exit "${FAKE_SMOKE_RC:-0}" ;;
esac
echo "fake python: unexpected $*" >&2
exit 3
'''
EXTRA_MODE = {"description": "mcmc, noise_lr khác (chế độ tự thêm)", "preset": "mcmc", "args": {"noise_lr": 250000},
              "cap_max": {"policy": "match_baseline"}}


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["pod/run_experiment.sh"], with_src=True, with_configs=True)
    for name, body in STUBS.items():
        write_stub(repo, name, body)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("nvidia-smi", "#!/usr/bin/env bash\necho 'Fake GPU'\n")):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    env.env["PATH"] = f"{bin_dir}:{env.env['PATH']}"
    conda = env.ws / "miniforge3" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
    (env.ws / "code" / "gsplat" / "examples").mkdir(parents=True)
    (env.ws / "code" / "gsplat" / "examples" / "simple_trainer.py").write_text("# fake\n", encoding="utf-8")
    (env.ws / "code" / "gaussian-splatting").mkdir(parents=True)
    yield env
    env.cleanup()


def experiment(sh, *args, **env):
    return sh.run(str(sh.repo / "scripts" / "pod" / "run_experiment.sh"), *args,
                  env={key: str(value) for key, value in env.items()}, timeout=120)


def steps(sh, prefix):
    return [line.split(" | ")[0] for line in calls(sh.root) if line.startswith(prefix)]


def summary_tsv(sh, name="main"):
    """Bảng các bước của lượt chạy gần nhất (bảng so sánh tổng hợp: cùng tên, đuôi .md)."""
    return sorted((sh.ws / "outputs" / "reports").glob(f"experiment_{name}_*.tsv"))[-1]


def summary_rows(sh, name="main"):
    rows = summary_tsv(sh, name).read_text(encoding="utf-8").splitlines()
    return [row.split("\t")[:2] for row in rows if not row.startswith("#")]


def add_extra_mode_plan(configs):
    """Kế hoạch `extra` = main thêm một chế độ tự định nghĩa (mcmc_x) và bảng so với mcmc."""
    base = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    base["train_modes"]["modes"]["mcmc_x"] = EXTRA_MODE
    (configs / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    plan = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    plan.update(name="extra", modes=["default", "mcmc", "mcmc_x"], also_against=["mcmc"])
    (configs / "experiments" / "extra.yaml").write_text(yaml.safe_dump(plan, allow_unicode=True), encoding="utf-8")


# ------------------------------------------------------------------------------------------------- kế hoạch
def test_the_main_plan_is_valid():
    plan = load_experiment("main", REPO / "configs", "/workspace")
    assert plan["modes"] == ["default", "mcmc"] and plan["seeds"] == [42, 43, 44] and plan["also_against"] == []
    assert [(b["dataset"], b["scenes"], b["b0"]) for b in plan["benchmarks"]] == [
        ("deep_blending", ["drjohnson", "playroom"], True),
        ("mipnerf360", ["room", "counter", "kitchen", "bonsai"], False)]
    assert plan["benchmarks"][0]["all_scenes"] and not plan["benchmarks"][1]["all_scenes"]
    assert plan_lines(plan)[-1] == "bench|mipnerf360|0|room counter kitchen bonsai"
    modes = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))["train_modes"]["modes"]
    assert set(plan["modes"]) <= set(modes)


@pytest.mark.parametrize("change, message", [
    ({"modes": ["mcmc"]}, "must include the baseline mode"),
    ({"modes": ["default", "nope"]}, "unknown mode(s) ['nope']"),
    ({"seeds": [42, 42]}, "a seed is listed twice"),
    ({"seeds": [42, "43"]}, "non-negative integers"),
    ({"benchmarks": [{"dataset": "custom_scene", "scenes": []}]}, "dataset must be one of the benchmarks"),
    ({"benchmarks": [{"dataset": "mipnerf360", "scenes": ["room", "lobby"]}]}, "scenes must be distinct scenes"),
    ({"benchmarks": [{"dataset": "deep_blending", "b0": "yes"}]}, "b0 must be true or false"),
    ({"estimate": {"gpu_hours": 18}}, "estimate.disk_gb must be a positive number"),
    ({"also_against": ["default"]}, "also_against must list distinct modes"),
    ({"also_against": ["mcmc", "nope"]}, "also_against must list distinct modes"),
])
def test_bad_plans_are_rejected(tmp_path, change, message):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw.update(change)
    (configs / "experiments" / "bad.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")
                       .replace("]", r"\]")):
        load_experiment("bad", configs, "/workspace")
    assert main(["--configs", str(configs), "--plan", "bad"]) == 2
    assert main(["--configs", str(configs), "--plan", "nope"]) == 2


# ------------------------------------------------------------------------------------------------- điều phối
def test_plan_mode_prints_the_steps_and_runs_nothing(sh):
    out = experiment(sh, "main", "--plan", VOLUME_GB=150)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "chế độ: default mcmc | seed: 42 43 44" in out.stdout and "VOLUME_GB=150" in out.stdout
    assert "gsplat_mipnerf360_s44: MODES=\"default mcmc\" SEEDS=44" in out.stdout
    assert "report_deep_blending: MODES=\"default mcmc\"" in out.stdout and "_vs_" not in out.stdout
    assert steps(sh, "fetch") == ["fetch deep_blending drjohnson playroom --dry-run",
                                  "fetch mipnerf360 room counter kitchen bonsai --dry-run"]
    assert not steps(sh, "train") and not (sh.ws / "outputs").exists() and sh.lock_free()
    assert experiment(sh, "nope", "--plan").returncode == 2
    assert experiment(sh, "main", "--quick").returncode == 2
    assert experiment(sh, "main", "--plan", VOLUME_GB="lots").returncode == 2


def test_runs_every_step_in_order_and_user_variables_cannot_change_the_plan(sh):
    out = experiment(sh, "main", MCMC_SET="opacity_reg=0.5", OUT_ROOT="/elsewhere", QUICK=1, SEEDS=7, FORCE=1,
                     MODES="default", BASELINE="mcmc")
    assert out.returncode == 0, out.stdout + out.stderr
    order = [line.split(" | ")[0] for line in calls(sh.root)]
    db, m360 = "SCENES=drjohnson playroom", "SCENES=room counter kitchen bonsai"
    # trước khi tải / train: simple_trainer.py thật nhận đúng tham số của mọi chế độ trong kế hoạch
    assert order[0] == f"smoke --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --modes default mcmc"
    assert order[1:-1] == ["fetch deep_blending drjohnson playroom", "fetch mipnerf360 room counter kitchen bonsai",
                          "train deep_blending b0",
                          *[f"train {bench} gsplat" for _ in range(3) for bench in ("deep_blending", "mipnerf360")],
                          "evaluate deep_blending", "evaluate mipnerf360"]
    assert order[-1].startswith(f"pack {sh.ws / 'ket_qua_'}") and order[-1].endswith(".tar.gz")
    trains = [line for line in calls(sh.root) if line.startswith("train ")]
    assert f"{db} QUICK=0 MCMC_SET=unset OUT_ROOT=unset FORCE=unset" in trains[0]  # B0
    for line, (bench_scenes, seed) in zip(trains[1:], [(db, 42), (m360, 42), (db, 43), (m360, 43), (db, 44),
                                                       (m360, 44)]):
        assert f"MODES=default mcmc SEEDS={seed} {bench_scenes} QUICK=0 MCMC_SET=unset OUT_ROOT=unset" in line
    assert [line for line in calls(sh.root) if line.startswith("evaluate")] == [
        f"evaluate deep_blending | BASELINE=unset MODES=default mcmc {db} QUICK=unset",
        f"evaluate mipnerf360 | BASELINE=unset MODES=default mcmc {m360} QUICK=unset"]
    rows = summary_rows(sh)
    assert [row[0] for row in rows][:4] == ["env_check", "space_check", "fetch_deep_blending", "fetch_mipnerf360"]
    assert all(status == "OK" for _, status in rows)
    text = summary_tsv(sh).with_suffix(".md").read_text(encoding="utf-8")
    assert "| gsplat_mipnerf360_s44 | OK |" in text and "| bảng so sánh deep_blending |" in text
    assert "## mipnerf360: room counter kitchen bonsai" in text and "so với" not in text
    assert sorted((sh.ws / "outputs" / "reports" / "configs").glob("experiment_main_*.json"))
    assert "TÓM TẮT thực nghiệm main" in out.stdout and sh.lock_free()


def test_a_plan_with_an_extra_mode_adds_the_table_against_mcmc(sh):
    add_extra_mode_plan(sh.repo / "configs")
    out = experiment(sh, "extra")
    assert out.returncode == 0, out.stdout + out.stderr
    assert steps(sh, "smoke") == [f"smoke --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --modes default"
                                  " mcmc mcmc_x"]
    trains = [line for line in calls(sh.root) if line.startswith("train ") and " gsplat " in line]
    assert len(trains) == 6 and all("MODES=default mcmc mcmc_x" in line for line in trains)
    db, m360 = "SCENES=drjohnson playroom", "SCENES=room counter kitchen bonsai"
    assert [line for line in calls(sh.root) if line.startswith("evaluate")] == [
        f"evaluate deep_blending | BASELINE=unset MODES=default mcmc mcmc_x {db} QUICK=unset",
        f"evaluate deep_blending | BASELINE=mcmc MODES=mcmc mcmc_x {db} QUICK=unset",  # chế độ mới so với mcmc
        f"evaluate mipnerf360 | BASELINE=unset MODES=default mcmc mcmc_x {m360} QUICK=unset",
        f"evaluate mipnerf360 | BASELINE=mcmc MODES=mcmc mcmc_x {m360} QUICK=unset"]
    text = summary_tsv(sh, "extra").with_suffix(".md").read_text(encoding="utf-8")
    assert "### mcmc mcmc_x so với mcmc (mipnerf360)" in text and "| report_mipnerf360_vs_mcmc | OK |" in text


def test_a_failed_download_skips_only_that_benchmark(sh):
    out = experiment(sh, "main", FAKE_FETCH_FAIL="mipnerf360")
    assert out.returncode == 1
    trains = steps(sh, "train")
    assert trains == ["train deep_blending b0"] + ["train deep_blending gsplat"] * 3
    rows = dict(summary_rows(sh))
    assert rows["fetch_mipnerf360"] == "FAIL" and rows["gsplat_mipnerf360_s42"] == "SKIP"
    assert rows["gsplat_deep_blending_s44"] == "OK" and rows["report_mipnerf360"] == "OK"


def test_environment_not_ready_stops_before_any_download(sh):
    out = experiment(sh, "main", FAKE_SMOKE_RC=1)  # trainer không nhận tham số của một chế độ
    assert out.returncode == 1 and "does not accept the arguments of a mode in the plan" in out.stdout
    assert not steps(sh, "fetch") and not steps(sh, "train") and "môi trường chưa sẵn sàng" in out.stdout
    (sh.root / "calls.txt").unlink()
    shutil.rmtree(sh.ws / "code" / "gsplat")
    out = experiment(sh, "main")
    assert out.returncode == 1 and "no gsplat checkout" in out.stdout
    assert not steps(sh, "fetch") and not steps(sh, "train") and "môi trường chưa sẵn sàng" in out.stdout
    assert steps(sh, "pack")  # vẫn đóng gói log để gửi lại


# ------------------------------------------------------------------------------------------------- chỗ trống
def fake_run(sh, bench, scene, mode, seed, size=1 << 30, status="OK"):
    """Một lượt gsplat (summary.json với `status`), dung lượng `size` byte (file thưa); trả về thư mục của lượt."""
    out = sh.ws / "outputs" / "benchmarks" / bench / "gsplat" / scene / f"{mode}_s{seed}"
    out.mkdir(parents=True)
    with open(out / "blob", "wb") as handle:
        handle.truncate(size)
    (out / "summary.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    return out


PLAN_RUNS = [(bench, scene, mode) for bench, scenes in (("deep_blending", ("drjohnson", "playroom")),
                                                       ("mipnerf360", ("room", "counter", "kitchen", "bonsai")))
             for scene in scenes for mode in ("default", "mcmc")]  # 12 lượt mỗi seed


def test_space_needed_per_seed_counts_only_the_runs_still_to_do(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    sh = ShellEnv(tmp_path)
    plan = load_experiment("main", configs, str(sh.ws))

    def needs(datasets=None):
        return {row["seed"]: row for row in space_needs(plan, configs, str(sh.ws), datasets)["seeds"]}

    estimate = round(56e9 / 36)  # chưa đo được: ước lượng của kế hoạch chia đều cho 36 lượt
    assert space_needs(plan, configs, str(sh.ws))["run_estimate_bytes"] == estimate
    assert {seed: (row["need_bytes"], row["finished"], row["runs"], row["measured"])
            for seed, row in needs().items()} == {seed: (12 * estimate, 0, 12, 0) for seed in (42, 43, 44)}

    size42 = {key: tree_bytes(fake_run(sh, *key, 42, size=2_000_000)) for key in PLAN_RUNS[:4]}  # Deep Blending xong
    partial = needs()
    assert partial[42]["need_bytes"] == 8 * estimate and partial[42]["finished"] == 4
    assert partial[43]["need_bytes"] == sum(size42.values()) + 8 * estimate and partial[43]["measured"] == 4

    size42.update({key: tree_bytes(fake_run(sh, *key, 42, size=3_000_000)) for key in PLAN_RUNS[4:]})
    room_default = ("mipnerf360", "room", "default")
    size43 = tree_bytes(fake_run(sh, *room_default, 43, size=5_000_000))  # cùng lượt ở seed khác, lớn hơn
    fake_run(sh, "mipnerf360", "room", "mcmc", 43, size=10, status="FAILED")  # chưa xong: còn phải train lại
    need = needs()
    assert need[42]["need_bytes"] == 0 and need[42]["finished"] == 12
    assert need[43]["need_bytes"] == sum(size for key, size in size42.items() if key != room_default)
    assert (need[43]["finished"], need[43]["measured"]) == (1, 11)
    assert need[44]["need_bytes"] == sum(size42.values()) - size42[room_default] + size43  # số lớn nhất của mỗi lượt
    db = needs(["deep_blending"])  # chỉ benchmark sẽ train
    assert [db[seed]["runs"] for seed in (42, 43, 44)] == [4, 4, 4] and db[42]["need_bytes"] == 0
    with pytest.raises(ValueError, match="not in the plan"):
        needs(["custom_scene"])


def test_space_is_checked_before_starting_and_before_each_seed(sh):
    # chưa có lượt nào: mỗi seed ước lượng 56 / 3 ≈ 18,7 GB; volume 20 GB thì không bắt đầu
    out = experiment(sh, "main", VOLUME_GB=20)
    assert out.returncode == 1 and "seed 42 (about 18.7 GB more) would not fit in a 20 GB volume" in out.stdout
    assert not steps(sh, "fetch") and "không đủ chỗ trống" in out.stdout

    # volume 35 GB: cảnh báo; mỗi lượt thật dùng 1 GiB (12 lượt = 12,9 GB mỗi seed); seed 44 không còn đủ chỗ thì dừng
    (sh.root / "calls.txt").unlink()
    out = experiment(sh, "main", VOLUME_GB=35, FAKE_RUN_SIZE="1G")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "WARN: the rest of the plan may not fit in 35 GB" in out.stdout
    assert "Seed 42: 0/12 lượt đã xong, cần thêm khoảng 18.7 GB" in out.stdout  # chưa đo được: theo ước lượng
    assert "Seed 43: 0/12 lượt đã xong, cần thêm khoảng 12.9 GB; đang dùng 12.9 GB / volume 35 GB" in out.stdout
    assert "Dừng train: không đủ chỗ trống cho seed 44 (VOLUME_GB=35)" in out.stdout
    assert steps(sh, "train") == ["train deep_blending b0"] + ["train deep_blending gsplat", "train mipnerf360 gsplat"] * 2
    rows = dict(summary_rows(sh))
    assert rows["gsplat_mipnerf360_s44"] == "SKIP" and rows["report_deep_blending"] == "OK"

    # chạy lại sau khi tăng VOLUME_GB một chút: seed 42, 43 đã xong không làm giảm chỗ cần cho seed 44
    (sh.root / "calls.txt").unlink()
    out = experiment(sh, "main", VOLUME_GB=38, FAKE_RUN_SIZE="1G")
    assert out.returncode == 1 and "seed 44 (about 12.9 GB more) would not fit in a 38 GB volume" in out.stdout
    assert "seed 43: 12/12 lượt gsplat đã xong, còn cần khoảng 0.0 GB" in out.stdout
    assert not steps(sh, "fetch") and not steps(sh, "train")

    # đủ chỗ: chỉ seed 44 còn lượt để train
    (sh.root / "calls.txt").unlink()
    out = experiment(sh, "main", VOLUME_GB=45, FAKE_RUN_SIZE="1G")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Seed 42: 12/12 lượt gsplat đã xong, không cần thêm chỗ" in out.stdout
    assert "Seed 44: 0/12 lượt đã xong, cần thêm khoảng 12.9 GB; đang dùng 25.8 GB / volume 45 GB" in out.stdout
    assert len(list((sh.ws / "outputs" / "benchmarks").glob("*/gsplat/*/*_s44/summary.json"))) == 12

    # SPACE_CHECK=0: không đo, chạy hết
    (sh.root / "calls.txt").unlink()
    out = experiment(sh, "main", VOLUME_GB=20, SPACE_CHECK=0)
    assert out.returncode == 0, out.stdout + out.stderr
    assert len(steps(sh, "train")) == 7 and "SPACE_CHECK=0" in out.stdout


def test_space_after_a_stop_in_the_middle_of_a_seed(sh):
    # pod dừng giữa seed 43 (Mip-NeRF 360 mới xong 2 lượt): chạy lại thì seed 43 cần 6 lượt, seed 44 đủ 12 lượt
    for bench, scene, mode in PLAN_RUNS:
        fake_run(sh, bench, scene, mode, 42)
    for bench, scene, mode in PLAN_RUNS[:6]:
        fake_run(sh, bench, scene, mode, 43)
    out = experiment(sh, "main", VOLUME_GB=36, FAKE_RUN_SIZE="1G")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "WARN: the rest of the plan may not fit in 36 GB" in out.stdout
    assert "Seed 43: 6/12 lượt đã xong, cần thêm khoảng 6.4 GB; đang dùng 19.3 GB / volume 36 GB" in out.stdout
    assert "Dừng train: không đủ chỗ trống cho seed 44 (VOLUME_GB=36)" in out.stdout
    assert len(list((sh.ws / "outputs" / "benchmarks").glob("*/gsplat/*/*_s43/summary.json"))) == 12
    assert not list((sh.ws / "outputs" / "benchmarks").glob("*/gsplat/*/*_s44"))


def test_background_runs_as_one_queued_job(sh):
    out = experiment(sh, "main", "--background")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Đã chạy nền 'exp_main'" in out.stdout
    state = sh.wait_state("exp_main", "DONE", "FAILED", timeout=120)
    assert state["state"] == "DONE", sh.log("exp_main")
    assert "TÓM TẮT thực nghiệm main" in sh.log("exp_main") and len(steps(sh, "train")) == 7
