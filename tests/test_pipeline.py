"""Test luồng chính cho một cảnh tự chụp (scripts/pod/run_scene.sh): kiểm tra đầu vào -> chuẩn bị (frame + COLMAP)
-> train default -> train mcmc -> tóm tắt, với scripts/pipeline/prepare_scene.sh và train_scene.sh thay bằng stub.

Kiểm tra: khi nào chạy lại frame + COLMAP, tham số truyền cho train_scene.sh (tên cảnh, --replace-stale, TAG rỗng),
dừng khi COLMAP FAIL hoặc default lỗi, chạy nền qua scripts/pod/job.sh; với thư mục ảnh: kiểm tra số ảnh trước khi
xếp hàng (một ảnh / quá ít ảnh thì dừng ngay), từ chối một file ảnh, và không dùng lại tên cảnh của nguồn khác; thư
mục kết quả lấy từ configs/custom_scene.yaml (kể cả configs/local.yaml). Bố cục: outputs/custom_scenes/<cảnh>/colmap
(COLMAP) cạnh <cảnh>/<chế_độ>_s<seed> (các lượt train). Quyết định train lại một lượt OK cũ nằm ở
src.training.run --replace-stale (tests/unit/test_train_run.py).
"""

import json
import os
import time

import pytest

from shell_env import ShellEnv, calls, make_fake_repo, write_stub

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs Linux /proc")

PREPARE = r'''
echo "prepare $* OVERWRITE=${OVERWRITE:-}" >> "$CALLS"
scene="$OUT_ROOT/$2/colmap"
mkdir -p "$scene/input"
if [ "${STUB_PREP_MODEL:-1}" = 1 ]; then
  mkdir -p "$scene/sparse/0"
  for f in cameras images points3D; do echo x > "$scene/sparse/0/$f.bin"; done
fi
printf '{"verdict": "%s", "input_images": 10, "warnings": [], "hints": ["chụp thêm ảnh chồng lấp"]}' \
  "${STUB_VERDICT:-PASS}" > "$scene/sfm_report.json"
exit "${STUB_RC_PREP:-0}"
'''

TRAIN = r'''
echo "train $* SEED=${SEED:-} TAG=${TAG-unset} BASELINE_TAG=${BASELINE_TAG-unset}" >> "$CALLS"
name="$1"
[ -d "$OUT_ROOT/$name/colmap" ] || { echo "stub: scene dir $OUT_ROOT/$name/colmap not found"; exit 9; }
mode="$2"
out="$OUT_ROOT/$name/${mode}_s${SEED:-42}"
rc_var="STUB_RC_TRAIN_$mode"
[ "${!rc_var:-0}" = 0 ] || exit "${!rc_var}"
case " ${STUB_TRAIN_SKIP:-} " in
  *" $mode "*) exit 0 ;;  # lượt cũ cùng cấu hình: bỏ qua, không ghi gì
esac
mkdir -p "$out"
printf '{"run_dir": "%s", "scene": "%s", "mode": "%s", "seed": %s, "status": "OK", "metrics": {"PSNR": 30.5, "SSIM": 0.91, "LPIPS": 0.2}, "num_gaussians": 1234, "train_seconds": 600}' \
  "$out" "$name" "$mode" "${SEED:-42}" > "$out/summary.json"
'''


@pytest.fixture
def sh(tmp_path):
    repo = make_fake_repo(tmp_path, ["pod/run_scene.sh", "pod/results.sh", "prepare_data.sh", "run_train.sh"],
                          with_src=True, with_configs=True)
    write_stub(repo, "scripts/pipeline/prepare_scene.sh", PREPARE)
    write_stub(repo, "scripts/pipeline/train_scene.sh", TRAIN)
    env = ShellEnv(tmp_path, repo=repo)
    env.env["CALLS"] = str(tmp_path / "calls.txt")
    env.env["OUT_ROOT"] = str(env.ws / "outputs" / "custom_scenes")
    video = tmp_path / "raw" / "video8.mp4"
    video.parent.mkdir()
    video.write_bytes(b"fake video")
    env.video = video
    yield env
    env.job("stop", "--all")
    env.cleanup()


def run_scene(sh, *args, **env):
    return sh.run(str(sh.repo / "scripts" / "pod" / "run_scene.sh"), *args, env={k: str(v) for k, v in env.items()},
                  timeout=120)


def train_call(sh, mode, name="video8", seed=42):
    return f"train {name} {mode} --replace-stale SEED={seed} TAG= BASELINE_TAG="


def make_model(sh, name="video8", verdict="PASS", root=None):
    scene = (root or sh.ws / "outputs" / "custom_scenes") / name / "colmap"
    (scene / "sparse" / "0").mkdir(parents=True)
    for f in ("cameras", "images", "points3D"):
        (scene / "sparse" / "0" / f"{f}.bin").write_text("x")
    (scene / "sfm_report.json").write_text(json.dumps({"verdict": verdict, "warnings": []}))
    return scene


def make_run(sh, mode, name="video8", seed=42):
    out = sh.ws / "outputs" / "custom_scenes" / name / f"{mode}_s{seed}"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"run_dir": str(out), "mode": mode, "status": "OK", "metrics": {}}))
    return out


def test_new_scene_prepares_then_trains_both_modes(sh):
    out = run_scene(sh, str(sh.video), "video8")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == [f"prepare {sh.video} video8 OVERWRITE=0", train_call(sh, "default"),
                              train_call(sh, "mcmc")]
    assert "COLMAP verdict: PASS" in out.stdout
    assert "== Cảnh video8" in out.stdout and "default_s42" in out.stdout  # tóm tắt cuối từ overview


def test_relative_source_is_made_absolute(sh):
    out = sh.run(str(sh.repo / "scripts" / "pod" / "run_scene.sh"), "raw/video8.mp4", "video8", cwd=sh.root)
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root)[0] == f"prepare {sh.video} video8 OVERWRITE=0"


def test_existing_model_is_reused(sh):
    make_model(sh)
    make_run(sh, "default")
    make_run(sh, "mcmc")
    out = run_scene(sh, str(sh.video), "video8", STUB_TRAIN_SKIP="default mcmc")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == [train_call(sh, "default"), train_call(sh, "mcmc")]
    assert "already has a COLMAP model" in out.stdout


def test_modes_seed_and_leftover_tags(sh):
    make_model(sh)
    out = run_scene(sh, "-", "video8", MODES="default mcmc_demo", SEED=7, TAG="thu", BASELINE_TAG="x")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == [train_call(sh, "default", seed=7), train_call(sh, "mcmc_demo", seed=7)]
    assert "default_s7" in out.stdout  # tóm tắt cuối đọc đúng thư mục kết quả


def test_scene_named_like_a_repo_folder(sh):
    (sh.repo / "docs").mkdir()  # thư mục cùng tên trong repo không được nhầm với cảnh
    make_model(sh, name="docs")
    out = run_scene(sh, "-", "docs")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root)[0] == train_call(sh, "default", name="docs")


def test_custom_out_root_is_used_for_training_and_summary(sh):
    root = sh.root / "other_out"
    make_model(sh, root=root)
    out = run_scene(sh, "-", "video8", OUT_ROOT=root)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "video8" / "default_s42" / "summary.json").is_file()
    assert "default_s42" in out.stdout and "chưa có lượt nào" not in out.stdout


def test_retrain_forces_every_mode(sh):
    make_model(sh)
    make_run(sh, "default")
    out = run_scene(sh, "-", "video8", "--retrain")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == ["train video8 default --force SEED=42 TAG= BASELINE_TAG=",
                              "train video8 mcmc --force SEED=42 TAG= BASELINE_TAG="]


def test_redo_colmap_overwrites(sh):
    make_model(sh)
    make_run(sh, "default")
    out = run_scene(sh, str(sh.video), "video8", "--redo-colmap", "--", "--target", "300")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == [f"prepare {sh.video} video8 --target 300 OVERWRITE=1", train_call(sh, "default"),
                              train_call(sh, "mcmc")]


def test_unfinished_preparation_is_redone(sh):
    scene = sh.ws / "outputs" / "custom_scenes" / "video8" / "colmap"
    (scene / "input").mkdir(parents=True)
    (scene / "input" / "00001.jpg").write_bytes(b"jpg")
    out = run_scene(sh, str(sh.video), "video8")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root)[0] == f"prepare {sh.video} video8 OVERWRITE=1"
    assert "unfinished preparation" in out.stdout


def test_colmap_fail_stops_before_training(sh):
    out = run_scene(sh, str(sh.video), "video8", STUB_VERDICT="FAIL")
    assert out.returncode == 1
    assert calls(sh.root) == [f"prepare {sh.video} video8 OVERWRITE=0"]
    assert "COLMAP verdict FAIL" in out.stdout
    assert "  - chụp thêm ảnh chồng lấp" in out.stdout  # gợi ý trong sfm_report.json được in ra


def test_failed_preparation_stops(sh):
    out = run_scene(sh, str(sh.video), "video8", STUB_RC_PREP=1, STUB_VERDICT="FAIL", STUB_PREP_MODEL=0)
    assert out.returncode == 1
    assert len(calls(sh.root)) == 1
    assert "preparing the scene failed (exit 1, COLMAP verdict FAIL)" in out.stdout


def test_failed_default_skips_mcmc(sh):
    out = run_scene(sh, str(sh.video), "video8", STUB_RC_TRAIN_default=1)
    assert out.returncode == 1
    assert [line.split()[2] for line in calls(sh.root)[1:]] == ["default"]  # mcmc không chạy
    assert "default failed" in out.stdout


def test_dash_needs_an_existing_model(sh):
    out = run_scene(sh, "-", "video8")
    assert out.returncode == 1
    assert calls(sh.root) == []
    assert "has no COLMAP model yet" in out.stdout


@pytest.mark.parametrize("args", [("-", "bad name"), ("-", "video8", "--redo-colmap"), ("-", "video8", "--frobnicate"),
                                  ("/nonexistent.mp4", "video8")])
def test_bad_arguments_are_rejected_before_anything_runs(sh, args):
    out = run_scene(sh, *args)
    assert out.returncode in (1, 2)
    assert calls(sh.root) == []
    assert not (sh.ws / "outputs" / "jobs").exists()


def test_background_runs_as_named_job(sh):
    out = run_scene(sh, str(sh.video), "video8", "--background")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Đã chạy nền 'scene_video8'" in out.stdout
    state = sh.wait_state("scene_video8", "DONE", "FAILED", timeout=60)
    assert state["state"] == "DONE", sh.log("scene_video8")
    assert len(calls(sh.root)) == 3
    # chạy lại khi việc đang chạy: không chạy bản thứ hai
    sh.job("start", "scene_video8", "--", "sleep", "30")
    sh.wait_state("scene_video8", "RUNNING")
    again = run_scene(sh, "-", "video8", "--background")
    assert "không chạy thêm bản thứ hai" in again.stdout
    time.sleep(0.2)


# ------------------------------------------------------------------------------------- nguồn là thư mục ảnh

def make_photos(sh, count, name="phong_anh"):
    folder = sh.root / "raw" / name
    folder.mkdir(parents=True)
    for i in range(count):
        (folder / f"IMG_{i}.JPG").write_bytes(b"jpg")  # run_scene chỉ đếm file; frames.py mới đọc ảnh
    return folder


def test_photo_folder_is_prepared_then_trained(sh):
    photos = make_photos(sh, 12)
    out = run_scene(sh, str(photos), "phong_anh")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == [f"prepare {photos} phong_anh OVERWRITE=0", train_call(sh, "default", name="phong_anh"),
                              train_call(sh, "mcmc", name="phong_anh")]


@pytest.mark.parametrize("count,expected", [(1, "Chỉ có 1 ảnh"), (4, "Chỉ có 4 ảnh dùng được"),
                                            (0, "Không có ảnh JPG/JPEG/PNG")])
def test_unusable_photo_folder_is_refused_before_queueing(sh, count, expected):
    photos = make_photos(sh, count)
    out = run_scene(sh, str(photos), "phong_anh", "--background")
    assert out.returncode == 1 and expected in out.stdout, out.stdout
    assert calls(sh.root) == [] and not (sh.ws / "outputs" / "jobs").exists()
    assert not (sh.ws / "outputs" / "custom_scenes" / "phong_anh").exists()


def test_single_image_file_is_refused(sh):
    photo = make_photos(sh, 12) / "IMG_0.JPG"
    out = run_scene(sh, str(photo), "phong_anh")
    assert out.returncode == 1 and "là một ảnh" in out.stdout and "không có chế độ dựng từ một ảnh" in out.stdout
    assert calls(sh.root) == []


def test_min_images_option_reaches_the_early_check(sh):
    photos = make_photos(sh, 4)
    out = run_scene(sh, str(photos), "phong_anh", "--", "--min-images", "3")
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root)[0] == f"prepare {photos} phong_anh --min-images 3 OVERWRITE=0"


def test_scene_name_already_used_by_another_source_is_refused(sh):
    scene = make_model(sh)
    (scene / "frames_report.json").write_text(json.dumps({"source": str(sh.video)}), encoding="utf-8")
    photos = make_photos(sh, 12)
    out = run_scene(sh, str(photos), "video8", "--background")
    assert out.returncode == 1 and "đã được dựng từ" in out.stdout and "đặt tên cảnh mới" in out.stdout
    assert calls(sh.root) == [] and not (sh.ws / "outputs" / "jobs").exists()
    # cùng nguồn: dùng lại mô hình COLMAP như trước
    out = run_scene(sh, str(sh.video), "video8", STUB_TRAIN_SKIP="default mcmc")
    assert out.returncode == 0 and "already has a COLMAP model" in out.stdout
    # nguồn mới + --redo-colmap: dựng lại cảnh từ nguồn mới
    out = run_scene(sh, str(photos), "video8", "--redo-colmap")
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"prepare {photos} video8 OVERWRITE=1" in calls(sh.root)


def test_results_folder_comes_from_the_registry_override(sh):
    """Không đặt OUT_ROOT: thư mục kết quả lấy từ configs/custom_scene.yaml, kể cả configs/local.yaml của pod."""
    local = sh.repo / "configs" / "local.yaml"
    local.write_text(json.dumps({"datasets": {"custom_scene": {  # JSON cũng là YAML hợp lệ
        "results": {"gsplat": "${WS}/elsewhere"}, "prepared": "${WS}/elsewhere/<cảnh>/colmap"}}}), encoding="utf-8")
    root = sh.ws / "elsewhere"
    out = run_scene(sh, str(sh.video), "video8", OUT_ROOT="")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "video8" / "colmap" / "sfm_report.json").is_file()
    assert (root / "video8" / "default_s42" / "summary.json").is_file()
    assert not (sh.ws / "outputs" / "custom_scenes").exists() and "default_s42" in out.stdout

    local.write_text("datasets:\n  custom_scene:\n    results:\n      gsplat: ${WS}/third\n",
                     encoding="utf-8")  # chỉ đổi một khóa: dừng trước khi làm gì
    out = run_scene(sh, str(sh.video), "video9", OUT_ROOT="")
    assert out.returncode != 0 and "prepared must be <results.gsplat>/<cảnh>/colmap" in out.stdout + out.stderr
    assert not (sh.ws / "third").exists() and len(calls(sh.root)) == 3  # chỉ các lời gọi của lần chạy trước


def test_entry_scripts_run_the_same_pipeline_steps(sh):
    """scripts/prepare_data.sh custom và scripts/run_train.sh custom gọi đúng các bước của scripts/pipeline/."""
    prepare = str(sh.repo / "scripts" / "prepare_data.sh")
    train = str(sh.repo / "scripts" / "run_train.sh")
    out = sh.run(prepare, "custom", str(sh.video), "video8", "--target", "300", timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root) == [f"prepare {sh.video} video8 --target 300 OVERWRITE="]
    assert sh.run(prepare, "custom", str(sh.video), timeout=60).returncode == 2  # thiếu tên cảnh
    out = sh.run(train, "custom", "video8", timeout=60)  # mặc định: default rồi mcmc
    assert out.returncode == 0, out.stdout + out.stderr
    assert calls(sh.root)[1:] == ["train video8 default SEED= TAG=unset BASELINE_TAG=unset",
                                  "train video8 mcmc SEED= TAG=unset BASELINE_TAG=unset"]
    out = sh.run(train, "custom", "video8", "mcmc", "--", "--set", "opacity_reg=0.001", timeout=60)
    assert out.returncode == 0 and calls(sh.root)[-1].startswith("train video8 mcmc --set opacity_reg=0.001 ")
    out = sh.run(train, "custom", "video8", env={"STUB_RC_TRAIN_default": "3"}, timeout=60)
    assert out.returncode == 3 and calls(sh.root)[-1].startswith("train video8 default")  # dừng, không chạy mcmc
    assert sh.run(train, "custom", "video8", "--set", timeout=60).returncode == 2  # tham số thêm phải sau --
