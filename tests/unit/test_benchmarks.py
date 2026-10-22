"""Test các lệnh benchmark (scripts/run_train.sh, prepare_data.sh, run_evaluate.sh và scripts/pipeline/) với conda,
GPU và các chương trình huấn luyện được thay bằng stub.

Kiểm tra:
- lệnh huấn luyện gsplat trên Deep Blending giữ đúng như lệnh của repo trước (8037a66) đã chạy trên pod ngày
  17/10/2026, chỉ khác đường dẫn: cùng tên cảnh, chế độ, seed và tham số;
- Mip-NeRF 360: quy ước ảnh thu nhỏ (images_2 / images_4) của Inria và gsplat, không có tham số mcmc riêng;
- B0 bỏ qua cảnh đã có kết quả, `FORCE=1` giữ lại lượt cũ;
- dữ liệu nằm ở chỗ configs/<benchmark>.yaml ghi; thiếu thì dừng trước khi train, lệnh train không tải gì;
- MODES: default luôn chạy trước, mọi chế độ khác cùng ngân sách Gaussian và cùng quy ước mcmc của benchmark;
- chạy một phần cảnh: báo cáo không đối chiếu trung bình công bố, chỉ đối chiếu B0 khi có đủ cảnh;
- cấu hình benchmark đã dùng được chụp thành JSON trong outputs/reports/configs/;
- scripts/run_evaluate.sh chọn đúng thư mục kết quả.
"""

import json
import os
import subprocess

import pytest
import yaml

from shell_env import ShellEnv, calls, make_fake_repo

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

FAKE_PYTHON = r'''#!/usr/bin/env bash
echo "python $*" >> "$CALLS"
case "${1:-}" in
  -c) exit 0 ;;  # kiểm tra import torch / gsplat / extension
  -) exec python3 "$@" ;;  # đoạn script viết thẳng trong file .sh (đọc results.json, ghi run_meta.json)
  -m)
    case "${2:-}" in
      src.training.run|src.evaluation.compare_modes) exit 0 ;;  # cần GPU / lượt chạy thật
      *) exec python3 "$@" ;;
    esac ;;
  train.py|render.py) exit 0 ;;
  metrics.py)
    printf '{"ours_30000": {"SSIM": 0.906, "PSNR": 29.7, "LPIPS": 0.238}}' > "$3/results.json" ;;
  *) echo "fake python: unexpected $*" >&2; exit 3 ;;
esac
'''
FAKE_NVIDIA_SMI = "#!/usr/bin/env bash\necho 'Fake GPU 24564'\n"


def make_scene(root, name, factors=()):
    for folder in ("images", *(f"images_{f}" for f in factors)):
        (root / name / folder).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (root / name / folder / f"IMG_{i}.JPG").write_bytes(b"x")
    (root / name / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for model in ("cameras", "images", "points3D"):
        (root / name / "sparse" / "0" / f"{model}.bin").write_bytes(b"x")


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["pod/check_datasets.sh", "prepare_data.sh", "run_train.sh", "run_evaluate.sh",
                                     "pipeline/prepare_benchmark.sh", "pipeline/train_gsplat_modes.sh",
                                     "pipeline/train_inria_b0.sh"], with_src=True, with_configs=True)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("nvidia-smi", FAKE_NVIDIA_SMI)):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    torch_dir = tmp_path / "fake_modules" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "__init__.py").write_text("__version__ = '0+fake'\nclass version:\n    cuda = None\n",
                                           encoding="utf-8")
    env.env["PATH"] = f"{bin_dir}:{env.env['PATH']}"
    env.env["PYTHONPATH"] = str(tmp_path / "fake_modules")
    conda = env.ws / "miniforge3" / "etc" / "profile.d"
    conda.mkdir(parents=True)
    (conda / "conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
    code = env.ws / "code" / "gaussian-splatting"
    code.mkdir(parents=True)
    git = ["git", "-C", str(code), "-c", "user.name=t", "-c", "user.email=t@localhost"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "fake inria"], check=True)
    yield env
    env.cleanup()


DB = ("data", "benchmarks", "deep_blending")
M360 = ("data", "benchmarks", "mipnerf360")


def out_dir(sh, bench, key):
    return sh.ws.joinpath("outputs", "benchmarks", bench, key)


ENTRY = {"run_gsplat_modes.sh": ("run_train.sh", "gsplat"), "run_inria_b0.sh": ("run_train.sh", "b0")}


def run(sh, script, bench, **env):
    """Chạy một benchmark qua lệnh chính: scripts/run_train.sh <benchmark> gsplat | b0."""
    entry, what = ENTRY[script]
    return sh.run(str(sh.repo / "scripts" / entry), bench, what, env={k: str(v) for k, v in env.items()},
                  timeout=120)


def evaluate(sh, *args, **env):
    return sh.run(str(sh.repo / "scripts" / "run_evaluate.sh"), *args, env={k: str(v) for k, v in env.items()},
                  timeout=120)


def training_calls(sh):
    prefix = "python -m src.training.run "
    return [line[len(prefix):] for line in calls(sh.root) if line.startswith(prefix)]


def test_deep_blending_gsplat_commands_are_unchanged(sh):
    db = sh.ws.joinpath(*DB)
    for scene in ("drjohnson", "playroom"):
        make_scene(db, scene)
    out = run(sh, "run_gsplat_modes.sh", "deep_blending")
    assert out.returncode == 0, out.stdout + out.stderr
    ws, gsplat = sh.ws, sh.ws / "code" / "gsplat"
    root = out_dir(sh, "deep_blending", "gsplat")
    # lệnh của scripts/run_modes_deepblending.sh trong repo trước (8037a66), đã chạy trên pod ngày 17/10/2026
    old_db, old_root = ws / "data" / "tandt_db" / "db", ws / "outputs" / "modes_db"
    old = []
    for scene in ("drjohnson", "playroom"):
        base = old_root / scene / "default_s42"
        old += [
            f"--mode default --scene {old_db / scene} --scene-name {scene} --out {base} --seed 42 "
            f"--gsplat-dir {gsplat} --rerun-incomplete",
            f"--mode mcmc --scene {old_db / scene} --scene-name {scene} --out {old_root / scene}/mcmc_s42 --seed 42 "
            f"--gsplat-dir {gsplat} --rerun-incomplete --replace-stale --cap-max-from {base} --set opacity_reg=0.001",
        ]
    # chỉ khác đường dẫn: cùng tên cảnh, chế độ, seed, tham số
    assert training_calls(sh) == [line.replace(str(old_db), str(db)).replace(str(old_root), str(root))
                                  for line in old]
    compare = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line]
    assert len(compare) == 1 and f"--root {root} " in compare[0]
    assert f"--b0-root {out_dir(sh, 'deep_blending', 'inria_b0')} " in compare[0] and "--strict" in compare[0]
    assert sorted((ws / "outputs" / "logs").glob("modes_db_*.log"))


def test_quick_deep_blending_writes_to_the_quick_folder(sh):
    db = sh.ws.joinpath(*DB)
    for scene in ("drjohnson", "playroom"):
        make_scene(db, scene)
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", QUICK=1, SEEDS="7")
    assert out.returncode == 0, out.stdout + out.stderr
    first = training_calls(sh)[0]
    assert f"--out {out_dir(sh, 'deep_blending', 'gsplat_quick') / 'drjohnson' / 'default_s7'} " in first
    assert first.endswith("--protocol max_steps=3000")


def test_mipnerf360_uses_downscaled_images_and_no_mcmc_override(sh):
    root = sh.ws.joinpath(*M360)
    make_scene(root, "room", factors=(2, 4))
    make_scene(root, "bicycle", factors=(2, 4))
    out = run(sh, "run_gsplat_modes.sh", "mipnerf360", SCENES="room bicycle")
    assert out.returncode == 0, out.stdout + out.stderr
    lines = training_calls(sh)
    assert len(lines) == 4
    assert all("--dataset-protocol data_factor=2" in line for line in lines[:2])
    assert all("--dataset-protocol data_factor=4" in line for line in lines[2:])
    assert all("--set" not in line for line in lines)  # mcmc giữ opacity_reg mặc định của gsplat
    assert f"--out {out_dir(sh, 'mipnerf360', 'gsplat') / 'room' / 'default_s42'} " in lines[0]


def add_mode(sh, name, entry):
    """Thêm một chế độ vào configs/base.yaml của repo giả (như người dùng tự thêm chế độ của mình)."""
    path = sh.repo / "configs" / "base.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    base["train_modes"]["modes"][name] = entry
    path.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_extra_modes_share_the_default_budget_and_the_mcmc_convention(sh):
    db = sh.ws.joinpath(*DB)
    for scene in ("drjohnson", "playroom"):
        make_scene(db, scene)
    add_mode(sh, "mcmc_x", {"description": "mcmc, noise_lr khác", "preset": "mcmc", "args": {"noise_lr": 250000},
                            "cap_max": {"policy": "match_baseline"}})
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", MODES="mcmc_x default mcmc", SEEDS="42 43")
    assert out.returncode == 0, out.stdout + out.stderr
    root = out_dir(sh, "deep_blending", "gsplat")
    lines = training_calls(sh)
    assert [line.split()[1] for line in lines] == ["default", "mcmc_x", "mcmc"] * 4  # default luôn trước
    for line in lines:
        mode, scene, seed = line.split()[1], line.split("--scene-name ")[1].split()[0], line.split("--seed ")[1][:2]
        assert f"--out {root / scene / f'{mode}_s{seed}'} " in line
        if mode != "default":  # cùng ngân sách Gaussian với default, cùng quy ước mcmc của Deep Blending
            assert line.endswith(f"--cap-max-from {root / scene / f'default_s{seed}'} --set opacity_reg=0.001")
    compare = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line]
    assert "--modes default mcmc_x mcmc " in compare[0] and "--scenes drjohnson playroom " in compare[0]


def test_variants_run_after_the_modes_into_their_own_folders(sh):
    root = sh.ws.joinpath(*M360)
    make_scene(root, "room", factors=(2, 4))
    variants = "mcmc:or0001:opacity_reg=0.001 mcmc:rbg:random_bkgd=true,strategy.noise_lr=250000"
    out = run(sh, "run_gsplat_modes.sh", "mipnerf360", SCENES="room", VARIANTS=variants)
    assert out.returncode == 0, out.stdout + out.stderr
    results = out_dir(sh, "mipnerf360", "gsplat") / "room"
    lines = training_calls(sh)
    assert [line.split("--out ")[1].split()[0] for line in lines] == [
        str(results / name) for name in ("default_s42", "mcmc_s42", "mcmc_or0001_s42", "mcmc_rbg_s42")]
    base = f"--cap-max-from {results / 'default_s42'}"
    assert lines[2].startswith("--mode mcmc ") and (
        f"{base} --tag or0001 --set opacity_reg=0.001 --dataset-protocol data_factor=2") in lines[2]
    assert (f"{base} --tag rbg --set random_bkgd=true --set strategy.noise_lr=250000 --dataset-protocol data_factor=2"
            in lines[3])
    assert "--tag" not in lines[1] and "variants: mcmc:or0001" in out.stdout
    compare = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line]
    assert "--modes default mcmc " in compare[0]  # các biến thể là biến thể của mcmc trong bảng
    assert "--tags or0001 rbg " in compare[0]  # các biến thể vừa train vào bảng; lượt có tag khác thì không


def test_a_mode_that_sets_the_convention_key_keeps_its_own_value(sh):
    for scene in ("drjohnson", "playroom"):
        make_scene(sh.ws.joinpath(*DB), scene)
    add_mode(sh, "mcmc_indoor", {"description": "mcmc đã chọn tham số", "preset": "mcmc",
                                 "args": {"opacity_reg": 0.005, "scale_reg": 0.05}, "cap_max": {"policy": "match_baseline"}})
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", SCENES="playroom", MODES="default mcmc mcmc_indoor",
              VARIANTS="mcmc_indoor:nlr:strategy.noise_lr=250000")
    assert out.returncode == 0, out.stdout + out.stderr
    lines = training_calls(sh)
    assert [line.split()[1] for line in lines] == ["default", "mcmc", "mcmc_indoor", "mcmc_indoor"]
    assert lines[1].endswith("--set opacity_reg=0.001")  # quy ước của Deep Blending cho mcmc
    assert "opacity_reg=0.001" not in lines[2] and "opacity_reg=0.001" not in lines[3]  # chế độ tự đặt opacity_reg
    assert lines[3].endswith("--tag nlr --set strategy.noise_lr=250000")
    assert "mcmc_indoor sets opacity_reg itself (configs/base.yaml): MCMC_SET is not added to it" in out.stdout
    compare = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line]
    assert "--tags nlr " in compare[0]
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", SCENES="playroom", MCMC_SET="opacity_reg=1 scale_reg=2")
    assert out.returncode == 1 and "MCMC_SET must be one key=value pair" in out.stdout + out.stderr


def test_variants_add_their_parameters_after_the_mcmc_convention(sh):
    for scene in ("drjohnson", "playroom"):
        make_scene(sh.ws.joinpath(*DB), scene)
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", SCENES="playroom", VARIANTS="mcmc:or0005:opacity_reg=0.005")
    assert out.returncode == 0, out.stdout + out.stderr
    line = training_calls(sh)[-1]
    assert line.endswith("--tag or0005 --set opacity_reg=0.001 --set opacity_reg=0.005")  # biến thể được ưu tiên


@pytest.mark.parametrize("variants, message", [
    ("mcmc:or0001", "invalid entry in VARIANTS"),
    ("mcmc:Or1:opacity_reg=0.001", "invalid entry in VARIANTS"),
    ("mcmc:a:b:opacity_reg=0.001", "invalid entry in VARIANTS"),
    ("default:x:opacity_reg=0.001", "a variant of default"),
    ("mcmc_x:a:opacity_reg=0.001", "is not in MODES"),
    ("mcmc:a:opacity_reg=0.001 mcmc:a:scale_reg=0.05", "listed twice"),
    ("mcmc:a:opacity reg=1", "invalid parameter 'opacity'"),  # dấu cách tách mục
    ("mcmc:a:opacity_reg=0.001,scale_reg", "invalid parameter"),
])
def test_bad_variants_stop_before_training(sh, variants, message):
    for scene in ("drjohnson", "playroom"):
        make_scene(sh.ws.joinpath(*DB), scene)
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", VARIANTS=variants)
    assert out.returncode == 1 and message in out.stdout, out.stdout + out.stderr
    assert training_calls(sh) == []


def test_unknown_modes_or_no_baseline_stop_before_training(sh):
    for scene in ("drjohnson", "playroom"):
        make_scene(sh.ws.joinpath(*DB), scene)
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", MODES="mcmc")
    assert out.returncode == 1 and "MODES must include default" in out.stdout
    out = run(sh, "run_gsplat_modes.sh", "deep_blending", MODES="default mcmc_dept")
    assert out.returncode == 1 and "unknown mode(s) ['mcmc_dept']" in out.stdout + out.stderr
    assert run(sh, "run_gsplat_modes.sh", "deep_blending", SEEDS="42 x").returncode == 1
    assert training_calls(sh) == []


def test_part_of_the_scenes_is_checked_without_the_published_mean(sh):
    root = sh.ws.joinpath(*M360)
    make_scene(root, "room", factors=(2, 4))
    out = run(sh, "run_gsplat_modes.sh", "mipnerf360", SCENES="room")
    assert out.returncode == 0, out.stdout + out.stderr
    (compare,) = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line]
    assert "--scenes room " in compare and "--b0-root" not in compare and "no B0 check" in out.stdout
    b0 = out_dir(sh, "mipnerf360", "inria_b0") / "room"
    b0.mkdir(parents=True)
    (b0 / "results.json").write_text('{"ours_30000": {"SSIM": 0.9, "PSNR": 31.0, "LPIPS": 0.2}}')
    assert run(sh, "run_gsplat_modes.sh", "mipnerf360", SCENES="room").returncode == 0
    compare = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line][-1]
    assert f"--b0-root {out_dir(sh, 'mipnerf360', 'inria_b0')} " in compare  # có B0 cho mọi cảnh được chọn


def test_mipnerf360_missing_downscaled_folder_stops_before_training(sh):
    make_scene(sh.ws.joinpath(*M360), "room")  # thiếu images_2
    out = run(sh, "run_gsplat_modes.sh", "mipnerf360", SCENES="room")
    assert out.returncode == 1
    assert "images_2 not found" in out.stdout
    assert training_calls(sh) == []


def test_b0_skips_finished_scenes_and_uses_images_dir_for_mipnerf(sh):
    db = sh.ws.joinpath(*DB)
    for scene in ("drjohnson", "playroom"):
        make_scene(db, scene)
    done = out_dir(sh, "deep_blending", "inria_b0") / "drjohnson"
    done.mkdir(parents=True)
    (done / "results.json").write_text('{"ours_30000": {"SSIM": 0.9, "PSNR": 29.4, "LPIPS": 0.24}}')
    out = run(sh, "run_inria_b0.sh", "deep_blending")
    assert out.returncode == 0, out.stdout + out.stderr
    trains = [line for line in calls(sh.root) if line.startswith("python train.py")]
    assert trains == [f"python train.py -s {db / 'playroom'} -m {out_dir(sh, 'deep_blending', 'inria_b0')}"
                      "/playroom --disable_viewer --quiet --eval --test_iterations -1"]
    meta = json.loads((out_dir(sh, "deep_blending", "inria_b0") / "playroom" / "run_meta.json").read_text())
    assert meta["benchmark"] == "deep_blending" and meta["scene_args"] == ""
    assert sorted((sh.ws / "outputs" / "reports").glob("baseline_db_summary_*.json"))

    root = sh.ws.joinpath(*M360)
    make_scene(root, "bicycle", factors=(2, 4))
    out = run(sh, "run_inria_b0.sh", "mipnerf360", SCENES="bicycle")
    # một cảnh trong 9: không so được với trung bình công bố (thiếu cảnh), nhưng cảnh đó vẫn chạy xong
    assert out.returncode == 1 and "or a scene is missing" in out.stdout, out.stdout + out.stderr
    train = [line for line in calls(sh.root) if line.startswith(f"python train.py -s {root / 'bicycle'}")]
    assert train and " -i images_4 -m " in train[0]
    assert (out_dir(sh, "mipnerf360", "inria_b0") / "bicycle" / "results.json").is_file()


def test_b0_force_reruns_and_keeps_the_previous_run(sh):
    db = sh.ws.joinpath(*DB)
    for scene in ("drjohnson", "playroom"):
        make_scene(db, scene)
    b0 = out_dir(sh, "deep_blending", "inria_b0")
    (b0 / "drjohnson").mkdir(parents=True)
    (b0 / "drjohnson" / "results.json").write_text('{"ours_30000": {"SSIM": 0.9, "PSNR": 29.4, "LPIPS": 0.24}}')
    out = run(sh, "run_inria_b0.sh", "deep_blending", FORCE=1)
    assert out.returncode == 0, out.stdout + out.stderr
    (previous,) = b0.glob("drjohnson.prev_*")  # lượt cũ được giữ, không bị ghi đè
    assert json.loads((previous / "results.json").read_text())["ours_30000"]["PSNR"] == 29.4
    assert json.loads((b0 / "drjohnson" / "results.json").read_text())["ours_30000"]["PSNR"] == 29.7
    assert f"previous run kept as {previous}" in out.stdout


def test_missing_dataset_stops_before_training_and_nothing_is_downloaded(sh):
    """Lệnh train không tải dữ liệu: thiếu thì dừng trước khi train, chỉ chỗ cần đặt dữ liệu và lệnh tải riêng."""
    for script, bench, env, scenes in (("run_gsplat_modes.sh", "deep_blending", {}, "drjohnson playroom"),
                                       ("run_inria_b0.sh", "deep_blending", {}, "drjohnson playroom"),
                                       ("run_inria_b0.sh", "mipnerf360", {"SCENES": "room"}, "room")):
        out = run(sh, script, bench, **env)
        assert out.returncode == 1, (script, bench, out.stdout + out.stderr)
        assert "dataset not found" in out.stdout and "Upload the scenes there yourself" in out.stdout
        assert f"download them with bash scripts/pod/fetch_data.sh {bench} {scenes}, or" in out.stdout
        assert "DOWNLOAD" not in out.stdout
    assert training_calls(sh) == [] and not [line for line in calls(sh.root) if line.startswith("python train.py")]
    assert not (sh.ws / "data" / "downloads").exists()


def test_evaluate_picks_the_result_folder(sh):
    out = evaluate(sh, "deep_blending")
    assert out.returncode == 1 and "no results in" in out.stdout
    out_dir(sh, "deep_blending", "gsplat").mkdir(parents=True)
    out = evaluate(sh, "deep_blending")
    assert out.returncode == 0, out.stdout + out.stderr
    reports = sh.ws / "outputs" / "reports"
    assert "no own Inria B0 run" in out.stdout and sorted(reports.glob("compare_db_*.md"))
    snapshot = sorted((reports / "configs").glob("deep_blending_*.json"))  # cấu hình đã dùng cho bảng
    assert len(snapshot) == 1 and json.loads(snapshot[0].read_text())["tag"] == "db"
    out = evaluate(sh, "deep_blending", BASELINE="mcmc", MODES="mcmc mcmc_x", REPORT=reports / "vs_mcmc")
    assert out.returncode == 0 and "no check against Inria" in out.stdout and (reports / "vs_mcmc.md").is_file()
    assert json.loads((reports / "vs_mcmc.json").read_text())["modes"] == ["mcmc", "mcmc_x"]
    assert evaluate(sh, "deep_blending", BASELINE="mcmc", QUICK=1).returncode == 2
    assert evaluate(sh, "deep_blending", BASELINE="a b").returncode == 2
    compare = [line for line in calls(sh.root) if "src.evaluation.compare_modes" in line]
    assert not compare  # run_evaluate.sh dùng python3 của máy, không qua python của env (stub)
    for what in ("custom", "custom_scene", "app"):
        out = evaluate(sh, what, "phong01")
        assert out.returncode == 1 and "no results for scene phong01" in out.stdout
    (sh.ws / "outputs" / "custom_scenes" / "phong01").mkdir(parents=True)
    out = evaluate(sh, "custom", "phong01")
    assert out.returncode == 0, out.stdout + out.stderr
    assert sorted(reports.glob("compare_custom_*.md"))
    assert evaluate(sh, "nope").returncode == 2
    assert evaluate(sh, "custom_scene").returncode == 0  # mọi cảnh tự chụp


def test_prepare_data_checks_a_benchmark_without_gpu(sh):
    out = sh.run(str(sh.repo / "scripts" / "prepare_data.sh"), "mipnerf360", timeout=120)
    assert out.returncode == 1 and "dataset not found" in out.stdout and "DOWNLOAD" not in out.stdout
    for scene in ("drjohnson", "playroom"):
        make_scene(sh.ws.joinpath(*DB), scene)
    out = sh.run(str(sh.repo / "scripts" / "prepare_data.sh"), "deep_blending", timeout=120)
    assert out.returncode == 0 and "OK: deep_blending is ready" in out.stdout, out.stdout + out.stderr
    assert not training_calls(sh) and sh.lock_free()  # không train, không giữ khóa GPU
    assert sh.run(str(sh.repo / "scripts" / "prepare_data.sh"), "nope", timeout=60).returncode == 2


def test_run_train_rejects_unknown_what(sh):
    out = sh.run(str(sh.repo / "scripts" / "run_train.sh"), "deep_blending", "nerf", timeout=60)
    assert out.returncode == 2 and "run_train.sh" in out.stdout  # in hướng dẫn
    assert sh.run(str(sh.repo / "scripts" / "run_train.sh"), timeout=60).returncode == 2
