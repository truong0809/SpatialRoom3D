"""Test kế hoạch thực nghiệm (src/training/experiment.py) và script điều phối scripts/pod/run_experiment.sh.

Script điều phối chạy thật trong một repo giả; các bước nó gọi (tải dữ liệu, B0, gsplat, báo cáo, đóng gói) là stub ghi
lại lời gọi kèm các biến môi trường. Bước gsplat giả ghi mỗi lượt vào đúng thư mục của nó
(`<kết quả>/<cảnh>/<chế độ>_s<seed>/` có summary.json OK, bỏ qua lượt đã xong như pipeline thật); dung lượng của lượt là
file thưa (sparse): `du -b` đếm kích thước khai báo nhưng file không chiếm chỗ thật.
"""

import json
import os
import re
import shutil

import pytest
import yaml

from shell_env import REPO, ShellEnv, calls, make_fake_repo, write_stub
from src.datasets.registry import load_registry
from src.training.experiment import load_experiment, main, plan_lines, run_args, space_needs, tree_bytes

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

STUBS = {
    "scripts/pod/fetch_data.sh":
        'echo "fetch $* | QUICK=${QUICK-unset}" >> "$CALLS"\n'
        'case " ${FAKE_FETCH_FAIL:-} " in *" $1 "*) echo "ERROR: fake download failure"; exit 1 ;; esac\n',
    "scripts/run_train.sh":
        'echo "train $* | MODES=${MODES-} SEEDS=${SEEDS-} SCENES=${SCENES-} QUICK=${QUICK-unset}'
        ' MCMC_SET=${MCMC_SET-unset} OUT_ROOT=${OUT_ROOT-unset} FORCE=${FORCE-unset} B0_CHECK=${B0_CHECK-unset}'
        ' VARIANTS=${VARIANTS-unset}" >> "$CALLS"\n'
        'if [ "$2" = gsplat ] && [ -n "${FAKE_RUN_SIZE:-}" ]; then\n'
        '  runs="$MODES"\n'
        '  for entry in ${VARIANTS:-}; do runs="$runs $(echo "$entry" | cut -d: -f1)_$(echo "$entry" | cut -d: -f2)"; done\n'
        '  for scene in $SCENES; do for run in $runs; do\n'
        '    out="$WS/outputs/benchmarks/$1/gsplat/$scene/${run}_s$SEEDS"\n'
        '    [ -f "$out/summary.json" ] && continue\n'
        '    mkdir -p "$out"; truncate -s "$FAKE_RUN_SIZE" "$out/blob"; echo \'{"status": "OK"}\' > "$out/summary.json"\n'
        '  done; done\n'
        'fi\n'
        'exit "${FAKE_TRAIN_RC:-0}"\n',
    "scripts/run_evaluate.sh":
        'echo "evaluate $* | BASELINE=${BASELINE-unset} MODES=${MODES-} SCENES=${SCENES-} QUICK=${QUICK-unset}'
        ' TAGS=${TAGS-unset}"'
        ' >> "$CALLS"\n'
        'printf "| bảng so sánh %s |\\n" "$1" > "$REPORT.md"\n',
    "scripts/pod/pack_results.sh": 'echo "pack $*" >> "$CALLS"\n',
    "scripts/setup/setup_depth_anything.sh": 'echo "da2setup $* | WS=${WS-unset}" >> "$CALLS"\nexit "${FAKE_DA2_RC:-0}"\n',
}
FAKE_PYTHON = r'''#!/usr/bin/env bash
# python của env conda giả: kiểm tra import (-c) luôn qua; smoke test, kiểm tra điểm SfM và lọc Gaussian ghi lại lời gọi
case "${1:-}" in
  -c) exit 0 ;;
  */smoke_test_gsplat.py) shift; echo "smoke $*" >> "$CALLS"; exit "${FAKE_SMOKE_RC:-0}" ;;
  -m)
    case "${2:-}" in
      src.training.depth_check) shift 2; echo "depth $*" >> "$CALLS"
        case " $* " in *"${FAKE_DEPTH_FAIL:-@@}"*) exit 1 ;; esac; exit 0 ;;
      src.training.prune) shift 2; echo "prune $*" >> "$CALLS"; exit "${FAKE_PRUNE_RC:-0}" ;;
    esac ;;
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
    """Bảng các bước của lượt chạy gần nhất của `name` (`<kế hoạch>` hoặc `<kế hoạch>_<phương pháp>`; bảng so sánh tổng
    hợp: cùng tên, đuôi .md). `name` là tên bản (main, gsplat_mcmc...); không lẫn bảng từng benchmark hay bản khác."""
    pattern = re.compile(rf"{name}_\d{{8}}_\d{{6}}\.tsv")
    return sorted(path for path in (sh.ws / "outputs" / "reports").glob(f"{name}_*.tsv")
                  if pattern.fullmatch(path.name))[-1]


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
                     MODES="default", BASELINE="mcmc", B0_CHECK="if_present")
    assert out.returncode == 0, out.stdout + out.stderr
    order = [line.split(" | ")[0] for line in calls(sh.root)]
    db, m360 = "SCENES=drjohnson playroom", "SCENES=room counter kitchen bonsai"
    # trước khi tải / train: simple_trainer.py thật nhận đúng tham số của mọi chế độ trong kế hoạch
    assert order[0] == f"smoke --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --modes default mcmc"
    assert order[1:-1] == ["fetch deep_blending drjohnson playroom", "fetch mipnerf360 room counter kitchen bonsai",
                          "train deep_blending b0",
                          *[f"train {bench} gsplat" for _ in range(3) for bench in ("deep_blending", "mipnerf360")],
                          "evaluate deep_blending", "evaluate mipnerf360"]
    assert order[-1] == f"pack {sh.ws / ('ket_qua_' + summary_tsv(sh).stem + '.tar.gz')}"  # ket_qua_main_<thời gian>
    trains = [line for line in calls(sh.root) if line.startswith("train ")]
    assert f"{db} QUICK=0 MCMC_SET=unset OUT_ROOT=unset FORCE=unset" in trains[0]  # B0
    for line, (bench_scenes, seed) in zip(trains[1:], [(db, 42), (m360, 42), (db, 43), (m360, 43), (db, 44),
                                                       (m360, 44)]):
        assert f"MODES=default mcmc SEEDS={seed} {bench_scenes} QUICK=0 MCMC_SET=unset OUT_ROOT=unset" in line
        assert " B0_CHECK=unset " in line  # cả kế hoạch: B0 chạy trước trong cùng việc, đối chiếu bắt buộc
    assert [line for line in calls(sh.root) if line.startswith("evaluate")] == [
        f"evaluate deep_blending | BASELINE=unset MODES=default mcmc {db} QUICK=unset TAGS=",
        f"evaluate mipnerf360 | BASELINE=unset MODES=default mcmc {m360} QUICK=unset TAGS="]  # lượt có tag: để ngoài
    rows = summary_rows(sh)
    assert [row[0] for row in rows][:4] == ["env_check", "space_check", "fetch_deep_blending", "fetch_mipnerf360"]
    assert all(status == "OK" for _, status in rows)
    text = summary_tsv(sh).with_suffix(".md").read_text(encoding="utf-8")
    assert "| gsplat_mipnerf360_s44 | OK |" in text and "| bảng so sánh deep_blending |" in text
    assert "## mipnerf360: room counter kitchen bonsai" in text and "so với" not in text
    assert sorted((sh.ws / "outputs" / "reports" / "configs").glob("main_*.json"))
    assert "TÓM TẮT main: thực nghiệm main (" in out.stdout and sh.lock_free()
    assert "  tên bản: main (việc nền, báo cáo, log và gói kết quả mang tên này)" in out.stdout


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
        f"evaluate deep_blending | BASELINE=unset MODES=default mcmc mcmc_x {db} QUICK=unset TAGS=",
        f"evaluate deep_blending | BASELINE=mcmc MODES=mcmc mcmc_x {db} QUICK=unset TAGS=",  # chế độ mới so với mcmc
        f"evaluate mipnerf360 | BASELINE=unset MODES=default mcmc mcmc_x {m360} QUICK=unset TAGS=",
        f"evaluate mipnerf360 | BASELINE=mcmc MODES=mcmc mcmc_x {m360} QUICK=unset TAGS="]
    text = summary_tsv(sh, "extra").with_suffix(".md").read_text(encoding="utf-8")
    assert "### So với mcmc (mipnerf360)" in text and "| report_mipnerf360_vs_mcmc | OK |" in text


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
    assert out.returncode == 1 and "the gsplat smoke test of the plan failed" in out.stdout
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
    assert "Đã chạy nền 'main'" in out.stdout
    state = sh.wait_state("main", "DONE", "FAILED", timeout=120)
    assert state["state"] == "DONE", sh.log("main")
    assert "TÓM TẮT main: thực nghiệm main (" in sh.log("main") and len(steps(sh, "train")) == 7


# ----------------------------------------------------------------------------------- chạy riêng một phương pháp
def gsplat_runs(plan):
    return {(bench["dataset"], scene, run, seed) for bench in plan["benchmarks"] for scene in bench["scenes"]
            for run in plan["runs"] for seed in plan["seeds"]}


def b0_scenes(plan):
    return {(bench["dataset"], scene) for bench in plan["benchmarks"] if bench["b0"] for scene in bench["scenes"]}


def test_only_selects_the_part_of_one_method():
    whole = load_experiment("main", REPO / "configs", "/workspace")
    b0, default, mcmc = (load_experiment("main", REPO / "configs", "/workspace", only)
                         for only in ("b0", "default", "mcmc"))
    assert (b0["modes"], b0["runs"], b0["post_filter"]) == ([], [], None)
    assert [(b["dataset"], b["b0"]) for b in b0["benchmarks"]] == [("deep_blending", True)]  # chỉ benchmark có B0
    assert default["modes"] == ["default"] and mcmc["modes"] == ["default", "mcmc"]  # mcmc cần ngân sách của default
    assert not any(bench["b0"] for bench in default["benchmarks"] + mcmc["benchmarks"])
    # chạy riêng đủ ba phương pháp = đúng các lượt của cả kế hoạch (cùng thư mục, cùng seed)
    assert gsplat_runs(default) | gsplat_runs(mcmc) == gsplat_runs(whole) and gsplat_runs(default) < gsplat_runs(mcmc)
    assert b0_scenes(b0) == b0_scenes(whole) and not b0_scenes(default) | b0_scenes(mcmc)
    assert [m["estimate"] for m in (b0, default, mcmc)] == [{"gpu_hours": 1, "disk_gb": 5}, {"gpu_hours": 6, "disk_gb": 30},
                                                            {"gpu_hours": 12, "disk_gb": 54}]
    assert {m["estimate_scope"] for m in (b0, default, mcmc)} == {"method"} and whole["estimate_scope"] == "plan"
    assert {m["run_estimate_bytes"] for m in (whole, b0, default, mcmc)} == {round(56e9 / 36)}  # phần của mỗi lượt
    assert plan_lines(mcmc)[-6:] == ["only|mcmc", f"run_estimate_bytes|{round(56e9 / 36)}",
                                     "b0_reference|deep_blending",  # B0 của cả kế hoạch: bảng đối chiếu khi đã có
                                     "run_name|gsplat_mcmc",  # tên bản = tên notebook gsplat_mcmc.ipynb
                                     "bench|deep_blending|0|drjohnson playroom",
                                     "bench|mipnerf360|0|room counter kitchen bonsai"]
    assert [m["run_name"] for m in (whole, b0, default, mcmc)] == ["main", "b0_inria", "gsplat_default", "gsplat_mcmc"]
    assert load_experiment("tune_mcmc", REPO / "configs", "/workspace", "mcmc")["run_name"] == "tune_mcmc_gsplat_mcmc"
    assert {tuple(m["b0_reference"]) for m in (whole, b0, default, mcmc)} == {("deep_blending",)}
    with pytest.raises(ValueError, match="expected b0 or one of the modes"):
        load_experiment("main", REPO / "configs", "/workspace", "mcmc_x")
    with pytest.raises(ValueError, match="no benchmark of the plan tune_mcmc has b0: true"):
        load_experiment("tune_mcmc", REPO / "configs", "/workspace", "b0")
    assert main(["--configs", str(REPO / "configs"), "--plan", "main", "--only", "nope"]) == 2


def test_only_drops_variants_extra_tables_and_unrelated_warnings(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    tune = load_experiment("tune_mcmc", configs, "/workspace", "mcmc")
    assert (tune["runs"], tune["variants"], tune["also_against"], tune["mono_runs"]) == (["default", "mcmc"], [], [], [])
    assert tune["post_filter"] == {"min_opacity": 0.02, "max_scale": 0.1} and tune["estimate_scope"] == "plan"
    base = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    base["train_modes"]["modes"]["default2"] = {"description": "như default", "preset": "default", "args": {}}
    (configs / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw.update(name="twin", modes=["default", "mcmc", "default2"])
    raw["estimate"].pop("only")
    (configs / "experiments" / "twin.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert len(load_experiment("twin", configs, "/workspace")["warnings"]) == 2  # default2 = default trên 2 benchmark
    assert len(load_experiment("twin", configs, "/workspace", "default2")["warnings"]) == 2
    assert load_experiment("twin", configs, "/workspace", "mcmc")["warnings"] == []


@pytest.mark.parametrize("only, message", [
    ({"b0": {"gpu_hours": 1}}, "estimate.only.b0 must be a mapping with gpu_hours and disk_gb"),
    ({"mcmc_x": {"gpu_hours": 1, "disk_gb": 2}}, "estimate.only.mcmc_x: expected b0 or one of the plan's modes"),
    ({"default": {"gpu_hours": 0, "disk_gb": 2}}, "estimate.only.default.gpu_hours must be a positive number"),
    ([1], "estimate.only must map b0 or a mode of the plan"),
])
def test_bad_method_estimates_are_rejected(tmp_path, only, message):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw["estimate"]["only"] = only
    (configs / "experiments" / "bad.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment("bad", configs, "/workspace")


def test_a_mode_named_b0_is_rejected(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    base = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    base["train_modes"]["modes"]["b0"] = {"description": "x", "preset": "mcmc", "args": {"noise_lr": 1},
                                         "cap_max": {"policy": "match_baseline"}}
    (configs / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw.update(name="bad", modes=["default", "b0"])
    raw["estimate"].pop("only")
    (configs / "experiments" / "bad.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="the mode name 'b0' is reserved"):
        load_experiment("bad", configs, "/workspace")


def test_two_runs_cannot_share_a_run_name(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw.update(name="gsplat_mcmc")  # kế hoạch trùng tên bản của main --only mcmc (notebook gsplat_mcmc.ipynb)
    (configs / "experiments" / "gsplat_mcmc.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    message = ("`main --only mcmc` and `gsplat_mcmc` would both be named gsplat_mcmc (job, reports, results package): "
               "rename experiments/main.yaml or experiments/gsplat_mcmc.yaml")
    with pytest.raises(ValueError, match=re.escape(message)):
        load_experiment("main", configs, "/workspace", "mcmc")
    with pytest.raises(ValueError, match=re.escape("`gsplat_mcmc` and `main --only mcmc` would both be named")):
        load_experiment("gsplat_mcmc", configs, "/workspace")
    assert load_experiment("main", configs, "/workspace", "default")["run_name"] == "gsplat_default"
    (configs / "experiments" / "gsplat_mcmc.yaml").unlink()
    raw.update(name="tune")  # kế hoạch tune và tune_mcmc: tên bản khác nhau (tune_gsplat_mcmc / tune_mcmc)
    (configs / "experiments" / "tune.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_experiment("tune", configs, "/workspace", "mcmc")["run_name"] == "tune_gsplat_mcmc"
    assert load_experiment("tune_mcmc", configs, "/workspace")["run_name"] == "tune_mcmc"
    # tên của việc nền / báo cáo loại khác: luồng dữ liệu, việc của ô 7 (b0_ / gsplat_<benchmark>), run_all, scene_...
    for other in ("mipnerf360", "gsplat_deep_blending", "b0_deep_blending", "run_all", "scene_phong1", "modes_db"):
        raw.update(name=other)
        (configs / "experiments" / f"{other}.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape(f"`{other}` would be named {other}, which is the name of another")):
            load_experiment(other, configs, "/workspace")
        (configs / "experiments" / f"{other}.yaml").unlink()


def test_only_mcmc_trains_default_then_mcmc_without_b0(sh):
    out = experiment(sh, "main", "--only", "mcmc")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Thực nghiệm main (chỉ mcmc), một phần của kế hoạch: default / mcmc" in out.stdout
    assert "chưa có thì train trước" in out.stdout
    assert ("deep_blending: drjohnson playroom | B0 Inria: không chạy trong lần này (chạy bằng --only b0); chưa có cho"
            " drjohnson playroom: default chưa được đối chiếu với B0") in out.stdout
    assert "mipnerf360: room counter kitchen bonsai | B0 Inria: không\n" in out.stdout  # B0 không thuộc kế hoạch
    assert "khoảng 1.6 GB mỗi lượt: ước lượng dung lượng của cả kế hoạch chia đều cho mọi lượt" in out.stdout
    assert "TÓM TẮT gsplat_mcmc: thực nghiệm main --only mcmc (" in out.stdout
    assert "  tên bản: gsplat_mcmc (việc nền, báo cáo, log và gói kết quả mang tên này)" in out.stdout
    assert steps(sh, "smoke") == [f"smoke --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --modes default mcmc"]
    trains = [line for line in calls(sh.root) if line.startswith("train ")]
    assert [line.split(" | ")[0] for line in trains] == ["train deep_blending gsplat", "train mipnerf360 gsplat"] * 3
    assert all("MODES=default mcmc " in line and " B0_CHECK=if_present " in line for line in trains)
    assert [line.split(" | ")[1].split(" SCENES")[0] for line in calls(sh.root) if line.startswith("evaluate")] == [
        "BASELINE=unset MODES=default mcmc"] * 2
    rows = summary_rows(sh, "gsplat_mcmc")
    assert [name for name, _ in rows][:4] == ["env_check", "space_check", "fetch_deep_blending", "fetch_mipnerf360"]
    assert "b0_deep_blending" not in dict(rows) and all(status == "OK" for _, status in rows)
    text = summary_tsv(sh, "gsplat_mcmc").with_suffix(".md").read_text(encoding="utf-8")
    stamp = summary_tsv(sh, "gsplat_mcmc").stem
    assert f"- Tên bản: `gsplat_mcmc` (việc nền, báo cáo, log và gói kết quả `ket_qua_{stamp}.tar.gz` mang tên này)" in text
    assert text.startswith("# Thực nghiệm main (chỉ mcmc)") and "(`--only mcmc`)" in text
    assert "- Một phần của kế hoạch: default / mcmc" in text
    assert ("- deep_blending: chưa có B0 tự chạy đủ cho drjohnson playroom: `default` chưa được đối chiếu đủ với B0 từng"
            " cảnh") in text and "- mipnerf360:" not in text
    assert sorted((sh.ws / "outputs" / "reports" / "configs").glob("gsplat_mcmc_*.json"))
    assert steps(sh, "pack")[-1] == f"pack {sh.ws / f'ket_qua_{stamp}.tar.gz'}"  # ket_qua_gsplat_mcmc_<thời gian>


def test_only_default_trains_only_the_baseline(sh):
    out = experiment(sh, "main", "--only", "default", "--plan")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "khoảng 6 giờ GPU, thêm khoảng 30 GB" in out.stdout and "--only default --background" in out.stdout
    out = experiment(sh, "main", "--only", "default")
    assert out.returncode == 0, out.stdout + out.stderr
    trains = [line for line in calls(sh.root) if line.startswith("train ")]
    assert len(trains) == 6 and all("MODES=default SEEDS=" in line for line in trains)
    assert steps(sh, "smoke") == [f"smoke --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --modes default"]
    b0 = sh.ws / "outputs" / "benchmarks" / "deep_blending" / "inria_b0"
    for scene in ("drjohnson", "playroom"):  # B0 đã chạy xong (vd. bằng --only b0)
        (b0 / scene).mkdir(parents=True)
        (b0 / scene / "results.json").write_text('{"ours_30000": {"SSIM": 0.9, "PSNR": 29.4, "LPIPS": 0.24}}')
    out = experiment(sh, "main", "--only", "default")
    assert out.returncode == 0, out.stdout + out.stderr
    assert ("B0 Inria: không chạy trong lần này (chạy bằng --only b0); đã có cho mọi cảnh: default được đối chiếu"
            " với B0") in out.stdout
    text = summary_tsv(sh, "gsplat_default").with_suffix(".md").read_text(encoding="utf-8")
    assert "- deep_blending: B0 tự chạy đã có cho mọi cảnh: `default` được đối chiếu với B0 từng cảnh" in text


def test_only_b0_runs_the_inria_runs_and_their_table(sh):
    out = experiment(sh, "main", "--only", "b0", "--plan")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "chỉ B0" in out.stdout and "gsplat_" not in out.stdout and "khoảng 1 giờ GPU, thêm khoảng 5 GB" in out.stdout
    assert steps(sh, "fetch") == ["fetch deep_blending drjohnson playroom --dry-run"]  # không cần Mip-NeRF 360
    (sh.root / "calls.txt").unlink()
    out = experiment(sh, "main", "--only", "b0")
    assert out.returncode == 0, out.stdout + out.stderr
    assert not steps(sh, "smoke")  # chỉ env gs-inria
    assert [line.split(" | ")[0] for line in calls(sh.root)][:-1] == ["fetch deep_blending drjohnson playroom",
                                                                      "train deep_blending b0"]
    assert "B0 còn cảnh chưa có kết quả: cần thêm khoảng 5.0 GB" in out.stdout
    rows = summary_rows(sh, "b0_inria")
    assert rows == [["env_check", "OK"], ["space_check", "OK"], ["fetch_deep_blending", "OK"],
                    ["b0_deep_blending", "OK"], ["report_deep_blending", "OK"]]
    text = summary_tsv(sh, "b0_inria").with_suffix(".md").read_text(encoding="utf-8")
    assert "Chỉ B0 (3DGS gốc của Inria)" in text and "| drjohnson | THIẾU KẾT QUẢ |" in text  # B0 giả không ghi gì
    assert "Kết luận: **INCOMPLETE**" in text and "Đã ghi tóm tắt" not in text  # bảng, không có dòng ghi file JSON
    assert "TÓM TẮT b0_inria: thực nghiệm main --only b0 (" in out.stdout
    out = experiment(sh, "main", "--only", "b0", VOLUME_GB=6)  # 5 GB ước lượng + 2 GB chừa lại > 6 GB
    assert out.returncode == 1 and "B0 (about 5.0 GB more) would not fit in a 6 GB volume" in out.stdout
    done = sh.ws / "outputs" / "benchmarks" / "deep_blending" / "inria_b0" / "drjohnson"
    done.mkdir(parents=True)
    done.joinpath("results.json").write_text('{"ours_30000": {"PSNR": 29', encoding="utf-8")  # hỏng: chưa có kết quả
    out = experiment(sh, "main", "--only", "b0", VOLUME_GB=6)
    assert out.returncode == 1 and "B0 (about 5.0 GB more) would not fit" in out.stdout
    raw = yaml.safe_load((sh.repo / "configs" / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw.update(name="solo")
    raw["estimate"].pop("only")  # không có ước lượng riêng cho B0: ước lượng 56 GB của cả kế hoạch không dùng cho B0
    (sh.repo / "configs" / "experiments" / "solo.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True),
                                                                    encoding="utf-8")
    out = experiment(sh, "solo", "--only", "b0", VOLUME_GB=6)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "không có ước lượng riêng cho B0 (estimate.only.b0): không kiểm tra chỗ trống cho B0" in out.stdout


def test_separate_methods_then_the_whole_plan_train_each_run_once(sh):
    for only in ("b0", "default", "mcmc"):
        out = experiment(sh, "main", "--only", only, FAKE_RUN_SIZE="1M")
        assert out.returncode == 0, out.stdout + out.stderr
    b0_calls = [line for line in calls(sh.root) if line.startswith("train deep_blending b0")]
    done = {path.parent for path in (sh.ws / "outputs" / "benchmarks").glob("*/gsplat/*/*/summary.json")}
    assert len(done) == 36  # 6 cảnh × 2 chế độ × 3 seed: đủ các lượt gsplat của main
    stamps = {path: path.joinpath("summary.json").stat().st_mtime_ns for path in done}
    out = experiment(sh, "main", FAKE_RUN_SIZE="1M")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Seed 42: 12/12 lượt gsplat đã xong, không cần thêm chỗ" in out.stdout
    assert {path: path.joinpath("summary.json").stat().st_mtime_ns for path in done} == stamps  # không train lại
    # bước B0 của cả kế hoạch gọi đúng như --only b0 (cùng cảnh, cùng biến): cảnh đã có kết quả thì B0 bỏ qua
    # (scripts/pipeline/train_inria_b0.sh, test_benchmarks.py::test_b0_skips_finished_scenes_and_uses_images_dir_for_mipnerf)
    assert [line for line in calls(sh.root) if line.startswith("train deep_blending b0")] == b0_calls * 2


def test_background_with_only_names_the_job_after_the_method(sh):
    out = experiment(sh, "main", "--only", "b0", "--background")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Đã chạy nền 'b0_inria'" in out.stdout  # tên bản = tên notebook b0_inria.ipynb
    state = sh.wait_state("b0_inria", "DONE", "FAILED", timeout=120)
    assert state["state"] == "DONE", sh.log("b0_inria")
    assert "TÓM TẮT b0_inria: thực nghiệm main --only b0" in sh.log("b0_inria")
    assert steps(sh, "train") == ["train deep_blending b0"]
    assert experiment(sh, "main", "--only").returncode == 2 and experiment(sh, "main", "--only", "Bad!").returncode == 2


# ------------------------------------------------------------------------- biến thể, loss độ sâu, lọc Gaussian
TUNE_VARIANTS = [{"mode": "mcmc", "tag": "or0005", "set": {"opacity_reg": 0.005}},
                 {"mode": "mcmc", "tag": "depth", "set": {"depth_loss": True}, "description": "loss độ sâu"}]


def add_tune_plan(configs, **change):
    """Kế hoạch `tune`: hai benchmark (một cảnh mỗi bên), 2 seed, 2 biến thể của mcmc, lọc Gaussian."""
    plan = {"name": "tune", "description": "dò tham số", "modes": ["default", "mcmc"], "also_against": ["mcmc"],
            "variants": TUNE_VARIANTS, "seeds": [42, 43],
            "benchmarks": [{"dataset": "deep_blending", "scenes": ["playroom"], "b0": False},
                           {"dataset": "mipnerf360", "scenes": ["room"], "b0": False}],
            "post_filter": {"min_opacity": 0.02, "max_scale": 0.1}, "estimate": {"gpu_hours": 3, "disk_gb": 8}}
    plan.update(change)
    (configs / "experiments" / "tune.yaml").write_text(yaml.safe_dump(plan, allow_unicode=True), encoding="utf-8")


def test_the_tune_mcmc_plan_is_valid():
    plan = load_experiment("tune_mcmc", REPO / "configs", "/workspace")
    assert plan["modes"] == ["default", "mcmc"] and plan["seeds"] == [42] and plan["also_against"] == ["mcmc"]
    assert [(b["dataset"], b["scenes"], b["b0"]) for b in plan["benchmarks"]] == [("mipnerf360", ["room"], False)]
    # biến thể có đoạn code mới chạy trước (lỗi thì thấy sớm), rồi 12 biến thể tham số
    assert plan["runs"] == ["default", "mcmc", "mcmc_mono", "mcmc_kr", "mcmc_bilagrid", "mcmc_v2", "mcmc_or0001",
                            "mcmc_or0005", "mcmc_or005", "mcmc_sr0001", "mcmc_sr005", "mcmc_nlr250k", "mcmc_nlr1m",
                            "mcmc_mo002", "mcmc_rbg", "mcmc_depth", "mcmc_rs30k", "mcmc_sh2"]
    # mọi biến thể đổi đúng một thứ so với mcmc, trừ gói v2 (mọi thứ notebook v2 thêm vào, cùng lúc)
    assert all(variant["mode"] == "mcmc" for variant in plan["variants"])
    assert all(len(variant["set"]) == 1 for variant in plan["variants"] if variant["tag"] != "v2")
    bundle = next(variant["set"] for variant in plan["variants"] if variant["tag"] == "v2")
    assert bundle == {"depth_loss": True, "random_bkgd": True, "use_bilateral_grid": True,
                      "strategy.refine_stop_iter": 24000, "sr3d.kr": True, "sr3d.mono": True}
    assert plan["depth_runs"] == ["mcmc_v2", "mcmc_depth"] and plan["depth_test_every"] == 8
    assert plan["mono_runs"] == ["mcmc_mono", "mcmc_v2"]
    assert plan["post_filter"] == {"min_opacity": 0.02, "max_scale": 0.1}
    lines = plan_lines(plan)
    assert "variants|mcmc:mono:sr3d.mono=true mcmc:kr:sr3d.kr=true mcmc:bilagrid:use_bilateral_grid=true" \
           " mcmc:v2:depth_loss=true,random_bkgd=true,use_bilateral_grid=true,strategy.refine_stop_iter=24000," \
           "sr3d.kr=true,sr3d.mono=true" \
           " mcmc:or0001:opacity_reg=0.001 mcmc:or0005:opacity_reg=0.005 mcmc:or005:opacity_reg=0.05" \
           " mcmc:sr0001:scale_reg=0.001 mcmc:sr005:scale_reg=0.05 mcmc:nlr250k:strategy.noise_lr=250000" \
           " mcmc:nlr1m:strategy.noise_lr=1000000 mcmc:mo002:strategy.min_opacity=0.02 mcmc:rbg:random_bkgd=true" \
           " mcmc:depth:depth_loss=true mcmc:rs30k:strategy.refine_stop_iter=30000 mcmc:sh2:sh_degree=2" in lines
    assert "gpu_hours|8" in lines and "disk_gb|35" in lines
    assert "post_filter|0.02 0.1" in lines and "depth_runs|mcmc_v2 mcmc_depth" in lines
    assert "mono_runs|mcmc_mono mcmc_v2" in lines
    main_plan = plan_lines(load_experiment("main", REPO / "configs", "/workspace"))
    assert {"variants|", "runs|default mcmc", "depth_runs|", "depth_test_every|", "post_filter|",
            "mono_runs|"} <= set(main_plan)


def test_variant_tags_are_not_reused_across_plans():
    """Mỗi tag biến thể chỉ thuộc một kế hoạch trong configs/experiments/: bảng lấy mọi seed có trên đĩa của tag được yêu
    cầu, nên một tag dùng lại ở vòng ghép sẽ đưa lượt seed 42 của lượt dò vào phần quyết định
    (docs/design/experiments.md mục 3.6)."""
    owner = {}
    for path in sorted((REPO / "configs" / "experiments").glob("*.yaml")):
        plan = load_experiment(path.stem, REPO / "configs", "/workspace")
        for variant in plan["variants"]:
            key = (variant["mode"], variant["tag"])
            assert key not in owner, f"variant {key} is in both {owner[key]} and {path.stem}"
            owner[key] = path.stem
    assert ("mcmc", "sh2") in owner and ("mcmc", "rs30k") in owner


def test_a_combination_round_plan_has_several_parameters_per_variant(tmp_path):
    """Kế hoạch của vòng ghép (docs/design/experiments.md mục 3.6), kiểm ở mức kế hoạch: seed mới 43, 44 trên room, mỗi
    cấu hình là một biến thể có thể nhiều tham số, mọi cấu hình dùng tag mới (kể cả ứng viên đơn), cùng post_filter với
    tune_mcmc; hai cấu hình cùng tham số (khác thứ tự) là một cấu hình."""
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    tune = load_experiment("tune_mcmc", configs, "/workspace")
    single = next(variant for variant in tune["variants"] if variant["tag"] == "or005")
    combo = {"name": "combo", "description": "vòng ghép", "modes": ["default", "mcmc"], "also_against": ["mcmc"],
             "variants": [{"mode": "mcmc", "tag": "c1", "set": dict(single["set"])},
                          {"mode": "mcmc", "tag": "c2", "set": {"opacity_reg": 0.05, "strategy.min_opacity": 0.02}}],
             "seeds": [43, 44], "benchmarks": [{"dataset": "mipnerf360", "scenes": ["room"], "b0": False}],
             "post_filter": tune["post_filter"], "estimate": {"gpu_hours": 2, "disk_gb": 15}}
    (configs / "experiments" / "combo.yaml").write_text(yaml.safe_dump(combo, allow_unicode=True), encoding="utf-8")
    plan = load_experiment("combo", configs, "/workspace")
    assert plan["runs"] == ["default", "mcmc", "mcmc_c1", "mcmc_c2"] and plan["warnings"] == []
    assert not {"c1", "c2"} & {variant["tag"] for variant in tune["variants"]}  # tag mới: lượt seed 42 không vào bảng
    lines = plan_lines(plan)
    assert "variants|mcmc:c1:opacity_reg=0.05 mcmc:c2:opacity_reg=0.05,strategy.min_opacity=0.02" in lines
    assert "seeds|43 44" in lines and "post_filter|0.02 0.1" in lines
    combo["variants"].append({"mode": "mcmc", "tag": "c3", "set": {"strategy.min_opacity": 0.02, "opacity_reg": 0.05}})
    (configs / "experiments" / "combo.yaml").write_text(yaml.safe_dump(combo, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="mcmc_c3 and mcmc_c2 would train mcmc with the same configuration"):
        load_experiment("combo", configs, "/workspace")


@pytest.mark.parametrize("change, message", [
    ({"variants": [{"mode": "default", "tag": "x", "set": {"opacity_reg": 0.1}}]}, "other than 'default'"),
    ({"variants": [{"mode": "nope", "tag": "x", "set": {"opacity_reg": 0.1}}]}, "mode must be one of"),
    ({"variants": [{"mode": "mcmc", "tag": "Big", "set": {"opacity_reg": 0.1}}]}, "tag must be lowercase"),
    ({"variants": [{"mode": "mcmc", "tag": "a__b", "set": {"opacity_reg": 0.1}}]}, "tag must be lowercase"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {}}]}, "set must be a non-empty mapping"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"max_steps": 3000}}]}, "controlled by the protocol"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"strategy.cap_max": 5}}]}, "controlled by the protocol"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"opacity_reg": "0.1"}}]}, "cannot be passed"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"opacity_reg": [1, 2]}}]}, "must be a number"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"render_traj_path": "a b"}}]}, "cannot be passed"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"opacity_reg": 0.1}, "seed": 1}]}, "must be a mapping"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"opacity_reg": 0.1}},
                   {"mode": "mcmc", "tag": "x", "set": {"scale_reg": 0.1}}]}, "another variant"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"opacity_reg": 0.1}},
                   {"mode": "mcmc", "tag": "y", "set": {"opacity_reg": 0.1}}]}, "would train mcmc with the same configuration"),
    ({"variants": [{"mode": "mcmc", "tag": "x", "set": {"strategy.noise_lr": 250000}},
                   {"mode": "mcmc", "tag": "y", "set": {"strategy.noise_lr": 250000.0}}]},
     "y and mcmc_x would train mcmc with the same configuration"),  # 250000 và 250000.0 là một
    ({"variants": {"mode": "mcmc"}}, "variants must be a list"),
    ({"post_filter": {"min_opacity": 0.02}}, "post_filter must be a mapping"),
    ({"post_filter": True}, "post_filter must be a mapping"),
    ({"post_filter": {"min_opacity": 1.0, "max_scale": 0.1}}, "must be below 1"),
    ({"post_filter": {"min_opacity": -1, "max_scale": 0.1}}, "must be a number >= 0"),
    ({"post_filter": {"min_opacity": 0, "max_scale": 0}}, "removes nothing"),
])
def test_bad_variants_and_filters_are_rejected(tmp_path, change, message):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw.update(change)
    (configs / "experiments" / "bad.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_experiment("bad", configs, "/workspace")


def test_a_variant_must_differ_from_the_benchmark_mcmc_convention(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw["variants"] = [{"mode": "mcmc", "tag": "or0001", "set": {"opacity_reg": 0.001}}]
    (configs / "experiments" / "dup.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    # Deep Blending: mọi lượt mcmc đã chạy với opacity_reg 0.001 (benchmark.gsplat.mcmc_set)
    with pytest.raises(ValueError, match="mcmc_or0001 and mcmc would train mcmc with the same configuration on deep_blending"):
        load_experiment("dup", configs, "/workspace")
    raw["benchmarks"] = [{"dataset": "mipnerf360", "scenes": ["room"], "b0": False}]  # Mip-NeRF 360: không có
    (configs / "experiments" / "dup.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_experiment("dup", configs, "/workspace")["runs"] == ["default", "mcmc", "mcmc_or0001"]


def add_mode(configs, name, entry):
    path = configs / "base.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    base["train_modes"]["modes"][name] = entry
    path.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_a_mode_keeps_its_own_value_of_the_mcmc_convention_key_and_twins_are_rejected(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    add_mode(configs, "mcmc_indoor", {"description": "x", "preset": "mcmc", "args": {"opacity_reg": 0.005},
                                      "cap_max": {"policy": "match_baseline"}})
    add_mode(configs, "mcmc_twin", {"description": "x", "preset": "mcmc", "args": {},
                                    "cap_max": {"policy": "match_baseline"}})
    train_modes = load_registry(configs, "/workspace")["train_modes"]
    convention = {"opacity_reg": 0.001}  # benchmark.gsplat.mcmc_set của Deep Blending
    assert run_args(train_modes, "mcmc", {}, convention) == {"opacity-reg": 0.001}
    assert run_args(train_modes, "mcmc_indoor", {}, convention) == {"opacity-reg": 0.005}  # giá trị của chế độ
    assert run_args(train_modes, "mcmc_indoor", {"opacity_reg": 0.002}, convention) == {"opacity-reg": 0.002}
    raw = yaml.safe_load((configs / "experiments" / "main.yaml").read_text(encoding="utf-8"))
    raw["modes"] = ["default", "mcmc", "mcmc_indoor"]
    (configs / "experiments" / "three.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_experiment("three", configs, "/workspace")["runs"] == ["default", "mcmc", "mcmc_indoor"]
    raw["modes"] = ["default", "mcmc", "mcmc_twin"]  # hai chế độ khác tên, cùng cấu hình: cảnh báo, cả hai vẫn train
    (configs / "experiments" / "twin.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    plan = load_experiment("twin", configs, "/workspace")
    assert plan["runs"] == ["default", "mcmc", "mcmc_twin"] and len(plan["warnings"]) == 2  # trên cả hai benchmark
    assert plan["warnings"][0].startswith("mcmc_twin and mcmc train the same configuration on deep_blending")
    assert f"warning|{plan['warnings'][1]}" in plan_lines(plan)
    raw["modes"] = ["default", "mcmc_indoor"]  # biến thể trùng chính chế độ của nó: bảng sẽ gộp hai lượt làm một
    raw["variants"] = [{"mode": "mcmc_indoor", "tag": "or0005", "set": {"opacity_reg": 0.005}}]
    (configs / "experiments" / "same.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="mcmc_indoor_or0005 and mcmc_indoor would train mcmc_indoor with the same "
                                         "configuration"):
        load_experiment("same", configs, "/workspace")
    assert load_experiment("main", configs, "/workspace")["warnings"] == []


def test_space_needed_counts_the_variant_runs(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    add_tune_plan(configs)
    sh = ShellEnv(tmp_path)
    plan = load_experiment("tune", configs, str(sh.ws))
    space = space_needs(plan, configs, str(sh.ws))
    assert space["run_estimate_bytes"] == round(8e9 / 16)  # 2 benchmark × 1 cảnh × 4 lượt × 2 seed
    first = fake_run(sh, "mipnerf360", "room", "mcmc_depth", 42, size=3_000_000)
    need = {row["seed"]: row for row in space_needs(plan, configs, str(sh.ws))["seeds"]}
    assert need[42]["finished"] == 1 and need[43]["measured"] == 1
    assert need[43]["need_bytes"] == tree_bytes(first) + 7 * round(8e9 / 16)


def test_space_counts_the_filter_output_of_finished_runs_not_yet_filtered(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", configs)
    add_tune_plan(configs)  # post_filter: opacity 0,02, trục 0,1
    sh = ShellEnv(tmp_path)
    plan = load_experiment("tune", configs, str(sh.ws))
    estimate = round(8e9 / 16)
    run = fake_run(sh, "mipnerf360", "room", "mcmc", 42, size=1000)
    for sub, size in (("ckpts", 5000), ("ply", 4000), ("renders", 300), ("videos", 200)):
        (run / sub).mkdir()
        (run / sub / "file").write_bytes(b"x" * size)
    part = sum(tree_bytes(run / sub) for sub in ("ckpts", "ply", "renders", "videos"))

    def need():
        return {row["seed"]: row["need_bytes"] for row in space_needs(plan, configs, str(sh.ws))["seeds"]}

    size = tree_bytes(run)
    assert need() == {42: 7 * estimate + part,  # lượt đã xong, chưa lọc: còn cần chỗ cho bản đã lọc
                      43: 7 * estimate + size + part}  # cùng lượt ở seed 43: đo từ seed 42, kể cả phần lọc
    (run / "filtered").mkdir()
    meta = {"status": "OK", "thresholds": {"min_opacity": 0.02, "max_scale": 0.1}}
    (run / "filtered" / "filter_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert need() == {42: 7 * estimate, 43: 7 * estimate + tree_bytes(run)}  # đã lọc với đúng ngưỡng
    meta["thresholds"]["min_opacity"] = 0.05
    (run / "filtered" / "filter_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert need()[42] == 7 * estimate + part  # ngưỡng khác: sẽ lọc lại
    # lượt không train (bị bỏ vì kiểm tra điểm SfM) không cần chỗ
    rows = space_needs(plan, configs, str(sh.ws), without=["mipnerf360:mcmc_depth"])["seeds"]
    assert [(row["seed"], row["runs"], row["need_bytes"]) for row in rows] == [
        (42, 7, 6 * estimate + part), (43, 7, 6 * estimate + tree_bytes(run) + part)]
    with pytest.raises(ValueError, match="expected <benchmark>:<run>"):
        space_needs(plan, configs, str(sh.ws), without=["mipnerf360:nope"])


def test_variants_depth_check_and_filter_steps_run_in_order(sh):
    add_tune_plan(sh.repo / "configs")
    out = experiment(sh, "tune", "--plan")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "mcmc_or0005: mcmc, opacity_reg=0.005" in out.stdout and "loss độ sâu (mcmc_depth)" in out.stdout
    assert "depth_check_mipnerf360: python -m src.training.depth_check --test-every 8" in out.stdout
    assert "filter_deep_blending_s43: python -m src.training.prune --min-opacity 0.02 --max-scale 0.1" in out.stdout
    (sh.root / "calls.txt").unlink()

    out = experiment(sh, "tune", VARIANTS="mcmc:evil:opacity_reg=9")
    assert out.returncode == 0, out.stdout + out.stderr
    order = [line.split(" | ")[0].split(" --")[0] for line in calls(sh.root)]
    assert order == ["smoke", "fetch deep_blending playroom", "fetch mipnerf360 room", "depth", "depth",
                     "train deep_blending gsplat", "prune", "train mipnerf360 gsplat", "prune",
                     "train deep_blending gsplat", "prune", "train mipnerf360 gsplat", "prune",
                     "evaluate deep_blending", "evaluate deep_blending", "evaluate mipnerf360",
                     "evaluate mipnerf360"] + [order[-1]]
    lines = calls(sh.root)
    assert lines[0].endswith("--modes default mcmc --variants mcmc:or0005:opacity_reg=0.005 mcmc:depth:depth_loss=true")
    db = sh.ws / "data" / "benchmarks"
    assert lines[3] == f"depth --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --test-every 8 " \
                       f"{db / 'deep_blending' / 'playroom'}"
    trains = [line for line in lines if line.startswith("train ")]
    assert all("VARIANTS=mcmc:or0005:opacity_reg=0.005 mcmc:depth:depth_loss=true" in line for line in trains)
    results = sh.ws / "outputs" / "benchmarks" / "mipnerf360" / "gsplat" / "room"
    prunes = [line for line in lines if line.startswith("prune ")]
    assert prunes[3] == ("prune --min-opacity 0.02 --max-scale 0.1 --skip-unfinished "
                         + " ".join(str(results / f"{run}_s43") for run in ("default", "mcmc", "mcmc_or0005",
                                                                              "mcmc_depth")))
    rows = summary_rows(sh, "tune")
    assert [name for name, _ in rows][:6] == ["env_check", "space_check", "fetch_deep_blending", "fetch_mipnerf360",
                                              "depth_check_deep_blending", "depth_check_mipnerf360"]
    assert all(status == "OK" for _, status in rows)
    text = summary_tsv(sh, "tune").with_suffix(".md").read_text(encoding="utf-8")
    assert "- Biến thể: mcmc:or0005:opacity_reg=0.005 mcmc:depth:depth_loss=true" in text
    assert "- Lọc Gaussian sau khi train: opacity < 0.02, trục lớn nhất > 0.1 × scene_scale" in text
    assert "| filter_mipnerf360_s43 | OK |" in text
    evaluations = [line.split(" | ")[1] for line in lines if line.startswith("evaluate ")]
    assert all(line.endswith(" TAGS=or0005 depth") for line in evaluations)  # bảng của kế hoạch có các biến thể của nó
    assert evaluations[1].startswith("BASELINE=mcmc MODES=mcmc ")


def test_a_failed_depth_check_drops_only_the_depth_runs_of_that_benchmark(sh):
    add_tune_plan(sh.repo / "configs")
    out = experiment(sh, "tune", FAKE_DEPTH_FAIL="mipnerf360")
    assert out.returncode == 1  # một bước không đạt
    assert "depth_check_mipnerf360 không đạt: bỏ mcmc_depth trên mipnerf360" in out.stdout
    trains = [line for line in calls(sh.root) if line.startswith("train ")]
    assert [line.split(" | ")[0] for line in trains] == ["train deep_blending gsplat", "train mipnerf360 gsplat"] * 2
    for line in trains:
        if line.startswith("train mipnerf360"):  # chỉ bỏ lượt có loss độ sâu; các lượt khác vẫn train
            assert "MODES=default mcmc " in line and line.endswith(" VARIANTS=mcmc:or0005:opacity_reg=0.005")
        else:
            assert line.endswith(" VARIANTS=mcmc:or0005:opacity_reg=0.005 mcmc:depth:depth_loss=true")
    rows = dict(summary_rows(sh, "tune"))
    assert rows["depth_check_mipnerf360"] == "FAIL" and rows["gsplat_mipnerf360_s42"] == "OK"
    assert rows["filter_mipnerf360_s42"] == "OK" and rows["filter_deep_blending_s43"] == "OK"
    text = summary_tsv(sh, "tune").with_suffix(".md").read_text(encoding="utf-8")
    assert "- mipnerf360: kiểm tra điểm SfM không đạt, không train mcmc_depth" in text


def test_reports_after_a_dropped_depth_mode_cover_the_modes_that_trained(sh):
    configs = sh.repo / "configs"
    base = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    base["train_modes"]["modes"]["mcmc_dl"] = {"description": "mcmc + loss độ sâu", "preset": "mcmc",
                                               "args": {"depth_loss": True}, "cap_max": {"policy": "match_baseline"}}
    (configs / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    add_tune_plan(configs, modes=["default", "mcmc", "mcmc_dl"], also_against=["mcmc", "mcmc_dl"],
                  variants=[TUNE_VARIANTS[0], {"mode": "mcmc_dl", "tag": "x", "set": {"scale_reg": 0.05}}])
    out = experiment(sh, "tune", FAKE_DEPTH_FAIL="mipnerf360")
    assert out.returncode == 1
    assert "depth_check_mipnerf360 không đạt: bỏ mcmc_dl mcmc_dl_x trên mipnerf360" in out.stdout
    # chỗ trống không tính 2 lượt bị bỏ: 5 lượt trên Deep Blending + 3 trên Mip-NeRF 360, mỗi lượt 8 GB / 20 lượt
    assert "Seed 42: 0/8 lượt đã xong, cần thêm khoảng 3.2 GB" in out.stdout
    m360 = [line for line in calls(sh.root) if line.startswith("train mipnerf360")]
    assert all("MODES=default mcmc " in line and line.endswith(" VARIANTS=mcmc:or0005:opacity_reg=0.005") for line in m360)
    evaluations = [line for line in calls(sh.root) if line.startswith("evaluate mipnerf360")]
    assert [line.split(" | ")[1].split(" SCENES")[0] for line in evaluations] == [
        "BASELINE=unset MODES=default mcmc", "BASELINE=mcmc MODES=mcmc"]  # không có mcmc_dl: bảng vẫn đủ cảnh
    rows = dict(summary_rows(sh, "tune"))
    assert rows["report_mipnerf360_vs_mcmc_dl"] == "SKIP" and rows["report_deep_blending_vs_mcmc_dl"] == "OK"
    db = [line for line in calls(sh.root) if line.startswith("evaluate deep_blending")]
    assert db[0].split(" | ")[1].startswith("BASELINE=unset MODES=default mcmc mcmc_dl ")  # Deep Blending không đổi


def test_b0_does_not_wait_for_the_depth_check(sh):
    add_tune_plan(sh.repo / "configs", benchmarks=[{"dataset": "deep_blending", "scenes": ["playroom"], "b0": True}],
                  seeds=[42])
    out = experiment(sh, "tune", FAKE_DEPTH_FAIL="deep_blending")
    assert out.returncode == 1
    trains = [line for line in calls(sh.root) if line.startswith("train ")]
    assert [line.split(" | ")[0] for line in trains] == ["train deep_blending b0", "train deep_blending gsplat"]
    assert trains[1].endswith(" VARIANTS=mcmc:or0005:opacity_reg=0.005")  # chỉ bỏ lượt có loss độ sâu
    rows = dict(summary_rows(sh, "tune"))
    assert rows["depth_check_deep_blending"] == "FAIL" and rows["b0_deep_blending"] == "OK"


def test_a_failed_depth_check_skips_the_benchmark_when_the_baseline_uses_depth(sh):
    configs = sh.repo / "configs"
    base = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    base["train_modes"]["modes"]["default"]["args"] = {"depth_loss": True}  # baseline có loss độ sâu
    (configs / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    add_tune_plan(configs, variants=[TUNE_VARIANTS[0]])
    out = experiment(sh, "tune", FAKE_DEPTH_FAIL="mipnerf360")
    assert out.returncode == 1
    trains = [line.split(" | ")[0] for line in calls(sh.root) if line.startswith("train ")]
    assert trains == ["train deep_blending gsplat"] * 2  # mọi lượt khác lấy ngân sách từ default: bỏ cả benchmark
    rows = dict(summary_rows(sh, "tune"))
    assert rows["gsplat_mipnerf360_s42"] == "SKIP" and rows["filter_mipnerf360_s42"] == "SKIP"
    text = summary_tsv(sh, "tune").with_suffix(".md").read_text(encoding="utf-8")
    assert "BỎ QUA (depth_check_mipnerf360 không đạt và default dùng loss độ sâu)" in text
    assert "- mipnerf360: kiểm tra điểm SfM không đạt và default dùng loss độ sâu, không train lượt gsplat nào" in text
    assert "depth_check_mipnerf360 không đạt và default dùng loss độ sâu: không train lượt gsplat nào" in out.stdout


def test_a_failed_filter_is_reported_and_training_goes_on(sh):
    add_tune_plan(sh.repo / "configs", variants=[TUNE_VARIANTS[0]])  # không còn loss độ sâu: không kiểm tra SfM
    out = experiment(sh, "tune", FAKE_PRUNE_RC=1, FAKE_TRAIN_RC=1)
    assert out.returncode == 1
    assert not [line for line in calls(sh.root) if line.startswith("depth ")]
    assert len([line for line in calls(sh.root) if line.startswith("train ")]) == 4
    rows = dict(summary_rows(sh, "tune"))
    assert rows["gsplat_mipnerf360_s42"] == "FAIL"
    assert rows["filter_mipnerf360_s42"] == "FAIL" and rows["report_mipnerf360"] == "OK"  # lọc lượt đã xong vẫn chạy


MONO_VARIANTS = TUNE_VARIANTS + [{"mode": "mcmc", "tag": "mono", "set": {"sr3d.mono": True}},
                                 {"mode": "mcmc", "tag": "kr", "set": {"sr3d.kr": True}}]


def test_a_plan_with_a_depth_prior_sets_up_depth_anything_first(sh):
    add_tune_plan(sh.repo / "configs", variants=MONO_VARIANTS, seeds=[42])
    out = experiment(sh, "tune", "--plan")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "depth prior (mcmc_mono): bước depth_anything tải Depth Anything V2 Small" in out.stdout
    assert "Các bước: depth_anything (bash scripts/setup/setup_depth_anything.sh), env_check, space_check" in out.stdout
    assert all(line.startswith("fetch ") and "--dry-run" in line for line in calls(sh.root))  # --plan chỉ kiểm dữ liệu
    (sh.root / "calls.txt").unlink()
    out = experiment(sh, "tune")
    assert out.returncode == 0, out.stdout + out.stderr
    lines = calls(sh.root)
    assert lines[0] == f"da2setup  | WS={sh.ws}"
    variants = "mcmc:or0005:opacity_reg=0.005 mcmc:depth:depth_loss=true mcmc:mono:sr3d.mono=true mcmc:kr:sr3d.kr=true"
    assert lines[1].endswith(f"--modes default mcmc --variants {variants} "
                             f"--depth-anything-dir {sh.ws / 'code' / 'depth_anything_v2'}")
    trains = [line for line in lines if line.startswith("train ")]
    assert trains and all(line.endswith(f" VARIANTS={variants}") for line in trains)
    assert [name for name, _ in summary_rows(sh, "tune")][:2] == ["depth_anything", "env_check"]


def test_a_failed_depth_anything_setup_drops_only_the_depth_prior_runs(sh):
    add_tune_plan(sh.repo / "configs", variants=MONO_VARIANTS, seeds=[42], also_against=["mcmc"])
    out = experiment(sh, "tune", FAKE_DA2_RC=1)
    assert out.returncode == 1  # một bước lỗi
    assert "depth_anything lỗi: bỏ mcmc_mono trên mọi benchmark" in out.stdout
    lines = calls(sh.root)
    kept = "mcmc:or0005:opacity_reg=0.005 mcmc:depth:depth_loss=true mcmc:kr:sr3d.kr=true"
    smoke = [line for line in lines if line.startswith("smoke ")]
    assert smoke == [f"smoke --examples-dir {sh.ws / 'code' / 'gsplat' / 'examples'} --modes default mcmc "
                     f"--variants {kept}"]  # không kiểm lượt bị bỏ, không kiểm Depth Anything V2
    trains = [line for line in lines if line.startswith("train ")]
    assert len(trains) == 2 and all(line.endswith(f" VARIANTS={kept}") for line in trains)
    # chỗ trống không tính lượt bị bỏ: 6 lượt mỗi benchmark, trừ mcmc_mono
    assert "seed 42: 0/10 lượt gsplat đã xong" in out.stdout and "Seed 42: 0/10 lượt đã xong" in out.stdout
    rows = dict(summary_rows(sh, "tune"))
    assert rows["depth_anything"] == "FAIL" and rows["env_check"] == "OK" and rows["gsplat_mipnerf360_s42"] == "OK"
    text = summary_tsv(sh, "tune").with_suffix(".md").read_text(encoding="utf-8")
    assert "- Depth Anything V2 chưa sẵn sàng (bước depth_anything lỗi): không train mcmc_mono trên mọi benchmark" in text


def test_a_failed_depth_anything_setup_stops_when_the_baseline_needs_it(sh):
    configs = sh.repo / "configs"
    base = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    base["train_modes"]["modes"]["default"]["args"] = {"sr3d.mono": True}  # baseline có depth prior
    (configs / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    add_tune_plan(configs, variants=[TUNE_VARIANTS[0]], seeds=[42])
    plan = experiment(sh, "tune", "--plan")
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert "(không tải được thì không train lượt gsplat nào: default dùng depth prior)" in plan.stdout
    (sh.root / "calls.txt").unlink(missing_ok=True)
    out = experiment(sh, "tune", FAKE_DA2_RC=1)
    assert out.returncode == 1
    assert "depth_anything lỗi và default dùng depth prior: không train lượt gsplat nào" in out.stdout
    assert not [line for line in calls(sh.root) if line.startswith(("smoke ", "fetch ", "train "))]
    rows = dict(summary_rows(sh, "tune"))
    assert rows["depth_anything"] == "FAIL" and rows["env_check"] == "SKIP"
