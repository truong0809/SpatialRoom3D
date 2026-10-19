"""Test cấu hình dữ liệu (configs/*.yaml) và lệnh kiểm tra chỉ đọc (src.datasets.registry, scripts/pod/check_datasets.sh).

Mọi dữ liệu là file giả để kiểm tra logic.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.datasets.registry import CONFIG_DIR, load_registry, main
from src.training.experiment import load_experiment
from src.training.profiles import load_config
from shell_env import REPO


def scene(root, name, folders=("images",), model=True, images=3):
    path = root / name
    for folder in folders:
        (path / folder).mkdir(parents=True, exist_ok=True)
        for i in range(images):
            (path / folder / f"{i:05d}.jpg").write_bytes(b"x")
    if model:
        (path / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        for name_ in ("cameras", "images", "points3D"):
            (path / "sparse" / "0" / f"{name_}.bin").write_bytes(b"x")
    return path


def fake_ws(tmp_path):
    """Volume: Deep Blending, 2 lượt gsplat (+ 1 lượt .prev_), một cảnh tự chụp đã chuẩn bị, một cảnh mới tải lên."""
    ws = tmp_path / "ws"
    db = ws / "data" / "benchmarks" / "deep_blending"
    scene(db, "drjohnson")
    scene(db, "playroom")
    gsplat = ws / "outputs" / "benchmarks" / "deep_blending" / "gsplat" / "drjohnson"
    for mode in ("default_s42", "mcmc_s42"):
        (gsplat / mode).mkdir(parents=True)
        (gsplat / mode / "summary.json").write_text("{}")
    (gsplat / "default_s42.prev_20261017_101010").mkdir()
    (gsplat / "default_s42.prev_20261017_101010" / "summary.json").write_text("{}")
    source = ws / "data" / "custom_scenes" / "phong1"
    source.mkdir(parents=True)
    for i in range(12):
        (source / f"IMG_{i:04d}.JPG").write_bytes(b"x")
    prepared = scene(ws / "outputs" / "custom_scenes" / "phong1", "colmap", folders=("input", "images"))
    (prepared / "sfm_report.json").write_text('{"verdict": "PASS"}')
    (ws / "outputs" / "custom_scenes" / "phong1" / "default_s42").mkdir()
    (ws / "outputs" / "custom_scenes" / "phong1" / "default_s42" / "summary.json").write_text("{}")
    (ws / "data" / "custom_scenes" / "phong2").mkdir()  # mới tải ảnh lên, chưa chuẩn bị
    (ws / "data" / "custom_scenes" / "phong2" / "VID_1.mp4").write_bytes(b"v")
    return ws


def config_copy(tmp_path):
    """Bản sao của configs/ (không có local.yaml) để thử ghi đè."""
    target = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, target, ignore=shutil.ignore_patterns("local.yaml", "environment"))
    return target


def snapshot(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_check_reports_deep_blending_ok_and_missing_mipnerf_is_not_an_error(tmp_path, capsys):
    ws = fake_ws(tmp_path)
    before = snapshot(ws)
    assert main(["--ws", str(ws)]) == 0
    out = capsys.readouterr().out
    assert "== deep_blending (benchmark): OK" in out
    assert "drjohnson    OK" in out and "playroom     OK" in out
    assert "kết quả gsplat: " in out and "(2 lượt)" in out  # .prev_* không tính
    assert "== mipnerf360 (benchmark): chưa có" in out and "room         MISSING" in out
    assert "phong1       OK            nguồn: 12 ảnh | input=3 | images=3 | sparse/0=ok | COLMAP PASS | 1 lượt train" in out
    assert "phong2       CHƯA CHUẨN BỊ nguồn: video VID_1.mp4 | 0 lượt train" in out
    assert snapshot(ws) == before  # chỉ đọc


def test_missing_required_scene_fails(tmp_path, capsys):
    """Luồng có `required: true` mà thiếu thì mã thoát 1. Cấu hình của repo không có luồng bắt buộc nào (chạy với
    dữ liệu nào do người dùng chọn): thiếu dữ liệu chỉ được báo, mã thoát vẫn 0."""
    ws = fake_ws(tmp_path)
    for path in (ws / "data" / "benchmarks" / "deep_blending" / "playroom" / "sparse" / "0").iterdir():
        path.unlink()
    configs = config_copy(tmp_path)
    config = configs / "deep_blending.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("required: false", "required: true", 1),
                      encoding="utf-8")
    assert main(["--configs", str(configs), "--ws", str(ws), "--dataset", "deep_blending"]) == 1
    out = capsys.readouterr().out
    assert "playroom     INCOMPLETE" in out and "(benchmark, bắt buộc): THIẾU" in out
    assert main(["--ws", str(ws), "--dataset", "deep_blending"]) == 0
    assert "== deep_blending (benchmark): chưa có" in capsys.readouterr().out
    assert main(["--ws", str(ws), "--dataset", "nope"]) == 2


def test_complete_mipnerf_scene_and_json_output(tmp_path, capsys):
    ws = fake_ws(tmp_path)
    root = ws / "data" / "benchmarks" / "mipnerf360"
    for name in ("room", "counter", "kitchen", "bonsai", "bicycle", "garden", "stump", "flowers", "treehill"):
        scene(root, name, folders=("images", "images_2", "images_4"))
    assert main(["--ws", str(ws), "--dataset", "mipnerf360", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]
    assert result["ok"] and all(s["status"] == "OK" for s in result["scenes"])
    assert result["scenes"][0]["found"] == {"images": 3, "images_2": 3, "images_4": 3, "sparse/0": "ok"}


def test_configs_match_the_scripts():
    registry = load_registry(CONFIG_DIR, "$WS", local=Path("/nonexistent"), environ={})
    datasets = registry["datasets"]
    assert list(datasets) == ["deep_blending", "mipnerf360", "custom_scene"]
    assert registry["code_dir"] == "$WS/SpatialRoom3D"
    pipeline = {name: (REPO / "scripts" / "pipeline" / name).read_text(encoding="utf-8")
                for name in ("train_inria_b0.sh", "train_gsplat_modes.sh", "train_scene.sh", "prepare_scene.sh")}
    # thư mục kết quả lấy từ configs/, không ghép tay trong script
    assert 'OUT_ROOT="${OUT_ROOT:-$(registry_results inria_b0)}"' in pipeline["train_inria_b0.sh"]
    assert 'OUT_ROOT="${OUT_ROOT:-$(registry_results "$RESULTS_KEY")}"' in pipeline["train_gsplat_modes.sh"]
    for name in ("deep_blending", "mipnerf360"):
        entry = datasets[name]
        bench = entry["benchmark"]
        assert bench["tag"] == {"deep_blending": "db", "mipnerf360": "mipnerf360"}[name]  # tên báo cáo, log
        assert set(bench["image_factor"]) <= set(entry["scenes"])
        assert set(bench["reference"]["metrics"]) == set(bench["tolerance"]) == {"PSNR", "SSIM", "LPIPS"}
        assert entry["root"] == f"$WS/data/benchmarks/{name}"
        assert entry["results"] == {key: f"$WS/outputs/benchmarks/{name}/{key}"
                                    for key in ("inria_b0", "gsplat", "gsplat_quick")}
        for command in entry["commands"].values():
            script, *args = command.split("bash ", 1)[1].split()
            assert (REPO / script).is_file(), script
            if script == "scripts/pod/run_experiment.sh":  # lệnh của một thực nghiệm có chạy trên luồng này
                plan = load_experiment(args[0], CONFIG_DIR, "$WS")
                assert name in [b["dataset"] for b in plan["benchmarks"]], command
            else:
                assert args[0] == name, command
    # chỉ tải khi người dùng gọi scripts/pod/fetch_data.sh: nguồn tải chỉ nằm trong mục download, các lệnh train không tải
    assert all("source" not in entry for entry in datasets.values())
    assert not any(entry.get("required") for entry in datasets.values())  # không luồng nào bắt buộc phải có dữ liệu
    lib = (REPO / "scripts" / "lib" / "benchmark.sh").read_text(encoding="utf-8")
    assert "curl" not in lib and "DOWNLOAD" not in lib and "zipfile" not in lib
    m360 = datasets["mipnerf360"]
    indoor = set(m360["indoor_scenes"])
    assert all(m360["benchmark"]["image_factor"][s] == (2 if s in indoor else 4) for s in m360["scenes"])
    custom = datasets["custom_scene"]
    assert custom["root"] == "$WS/data/custom_scenes" and custom["results"]["gsplat"] == "$WS/outputs/custom_scenes"
    assert custom["prepared"] == "$WS/outputs/custom_scenes/<cảnh>/colmap"
    for command in custom["commands"].values():
        assert (REPO / command.split("bash ", 1)[1].split()[0]).is_file(), command
    run_scene = (REPO / "scripts" / "pod" / "run_scene.sh").read_text(encoding="utf-8")
    for text in (run_scene, pipeline["train_scene.sh"], pipeline["prepare_scene.sh"]):
        assert 'OUT_ROOT="$(registry_value custom_scene results.gsplat)"' in text
        assert "outputs/custom_scenes" not in text.split("registry_value custom_scene results.gsplat", 1)[1]
    assert 'SCENE_DIR="$OUT_ROOT/$NAME/colmap"' in run_scene
    assert 'SCENE="$OUT_ROOT/$NAME/colmap"' in pipeline["prepare_scene.sh"]
    # các chế độ huấn luyện: mục train_modes của configs/base.yaml
    modes = load_config(CONFIG_DIR / "base.yaml")
    assert modes["baseline_mode"] == "default" and {"default", "mcmc"} <= set(modes["modes"])


def test_local_file_and_environment_override_the_root(tmp_path, capsys):
    ws = fake_ws(tmp_path)
    moved = ws / "data" / "MipNeRF360"
    for name in ("room", "counter", "kitchen", "bonsai", "bicycle", "garden", "stump", "flowers", "treehill"):
        scene(moved, name, folders=("images", "images_2", "images_4"))
    configs = config_copy(tmp_path)
    (configs / "local.yaml").write_text("datasets:\n  mipnerf360:\n    root: ${WS}/data/MipNeRF360\n", encoding="utf-8")
    loaded = load_registry(configs, str(ws), environ={})
    assert loaded["datasets"]["mipnerf360"]["root"] == str(moved)
    assert loaded["datasets"]["mipnerf360"]["scenes"][0] == "room"  # phần không ghi đè giữ nguyên
    assert main(["--configs", str(configs), "--ws", str(ws), "--dataset", "mipnerf360"]) == 0
    assert "ghi đè:" in capsys.readouterr().out
    env_root = load_registry(configs, str(ws), environ={"SR3D_DATA_MIPNERF360": "/elsewhere"})
    assert env_root["datasets"]["mipnerf360"]["root"] == "/elsewhere"  # biến môi trường thắng file cục bộ


def test_get_prints_one_field(tmp_path, capsys):
    ws = fake_ws(tmp_path)
    assert main(["--ws", str(ws), "--get", "deep_blending", "root"]) == 0
    assert capsys.readouterr().out.strip() == str(ws / "data" / "benchmarks" / "deep_blending")
    assert main(["--ws", str(ws), "--get", "mipnerf360", "results.gsplat_quick"]) == 0
    assert capsys.readouterr().out.strip() == str(ws / "outputs" / "benchmarks" / "mipnerf360" / "gsplat_quick")
    assert main(["--ws", str(ws), "--get", "deep_blending", "benchmark.tag"]) == 0
    assert capsys.readouterr().out.strip() == "db"
    assert main(["--ws", str(ws), "--get", "mipnerf360", "scenes"]) == 0
    assert capsys.readouterr().out.strip().split() == ["room", "counter", "kitchen", "bonsai", "bicycle", "garden",
                                                       "stump", "flowers", "treehill"]
    assert main(["--ws", str(ws), "--get", "deep_blending", "nope"]) == 1
    assert main(["--ws", str(ws), "--get", "nope", "root"]) == 2


def test_export_benchmark_writes_the_json_used_by_the_conda_envs(tmp_path, capsys):
    out = tmp_path / "reports" / "configs" / "db.json"
    assert main(["--ws", "/ws", "--export-benchmark", "deep_blending", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["dataset"] == "deep_blending" and data["scenes"] == ["drjohnson", "playroom"]
    assert data["tag"] == "db" and data["gsplat"]["mcmc_set"] == "opacity_reg=0.001"
    assert data["reference"]["metrics"]["PSNR"] == 29.69 and data["method"] == "ours_30000"
    m360 = tmp_path / "m360.json"
    assert main(["--ws", "/ws", "--export-benchmark", "mipnerf360", "--out", str(m360)]) == 0
    assert json.loads(m360.read_text(encoding="utf-8"))["indoor_scenes"][0] == "room"
    capsys.readouterr()
    assert main(["--ws", "/ws", "--export-benchmark", "custom_scene", "--out", str(tmp_path / "x.json")]) == 1
    assert "no benchmark section" in capsys.readouterr().err
    assert main(["--ws", "/ws", "--export-benchmark", "deep_blending"]) == 2  # thiếu --out


def test_broken_local_override_is_named_in_the_error(tmp_path, capsys):
    configs = config_copy(tmp_path)
    (configs / "local.yaml").write_text("datasets: [unclosed\n", encoding="utf-8")
    assert main(["--configs", str(configs), "--ws", str(tmp_path), "--get", "deep_blending", "root"]) == 2
    err = capsys.readouterr().err
    assert "local.yaml" in err and "fix or delete it" in err
    (configs / "local.yaml").write_text("datasets:\n  mipnerf_360:\n    root: /x\n", encoding="utf-8")  # gõ sai tên
    assert main(["--configs", str(configs), "--ws", str(tmp_path), "--get", "deep_blending", "root"]) == 2
    assert "unknown dataset(s) ['mipnerf_360']" in capsys.readouterr().err
    # cùng dạng với base.yaml (danh sách) thay vì mapping: báo lỗi có tên file, không traceback
    for text in ("datasets: [deep_blending]\n", "datasets:\n  deep_blending: /x\n"):
        (configs / "local.yaml").write_text(text, encoding="utf-8")
        assert main(["--configs", str(configs), "--ws", str(tmp_path), "--get", "deep_blending", "root"]) == 2
        assert "must map dataset names to settings" in capsys.readouterr().err
    # protocol / chế độ huấn luyện không ghi đè được trên pod (chỉ nằm trong configs/base.yaml)
    (configs / "local.yaml").write_text("train_modes:\n  default_seed: 7\n", encoding="utf-8")
    assert main(["--configs", str(configs), "--ws", str(tmp_path), "--get", "deep_blending", "root"]) == 2
    assert "may only change 'datasets', not ['train_modes']" in capsys.readouterr().err


def test_config_python_is_chosen_before_conda_activate_shadows_python3(tmp_path):
    """Script B0 đọc cấu hình sau `conda activate gs-inria`: python3 lúc đó là của env, không có PyYAML.
    config.sh phải dùng python3 tìm thấy lúc được source (trước khi kích hoạt env), kể cả khi chưa có env gs-gsplat."""
    no_yaml = tmp_path / "gs-inria" / "bin"
    no_yaml.mkdir(parents=True)
    (no_yaml / "python3").write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" -S "$@"\n', encoding="utf-8")
    (no_yaml / "python3").chmod(0o755)
    script = (f'REPO_DIR="{REPO}"; WS="{tmp_path}"; . "$REPO_DIR/scripts/lib/config.sh"\n'
              f'PATH="{no_yaml}:$PATH"\n'  # như conda activate gs-inria
              'python3 -c "import yaml" 2> /dev/null && exit 9\n'
              'config_python && registry_value deep_blending benchmark.tag\n')
    env = {"PATH": f"{Path(sys.executable).parent}{os.pathsep}/usr/bin:/bin", "HOME": str(tmp_path)}
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.split() == [str(Path(sys.executable).parent / "python3"), "db"]


def test_custom_prepared_must_follow_the_results_folder(tmp_path, capsys):
    """Script đặt COLMAP ở <results.gsplat>/<cảnh>/colmap: ghi đè chỉ một trong hai khóa thì báo lỗi, không đoán."""
    configs = config_copy(tmp_path)
    local = configs / "local.yaml"
    local.write_text("datasets:\n  custom_scene:\n    results:\n      gsplat: ${WS}/other\n", encoding="utf-8")
    assert main(["--configs", str(configs), "--ws", str(tmp_path), "--get", "custom_scene", "root"]) == 2
    err = capsys.readouterr().err
    assert "prepared must be <results.gsplat>/<cảnh>/colmap" in err and f"{tmp_path}/other/<cảnh>/colmap" in err
    local.write_text("datasets:\n  custom_scene:\n    results:\n      gsplat: ${WS}/other\n"
                     "    prepared: ${WS}/other/<cảnh>/colmap\n", encoding="utf-8")
    loaded = load_registry(configs, str(tmp_path), environ={})["datasets"]["custom_scene"]
    assert loaded["prepared"] == f"{tmp_path}/other/<cảnh>/colmap"


def test_configs_have_no_old_layout_entries():
    """Repo mới, volume mới: cấu hình không còn chỗ cũ của bố cục trước (không có bước dời dữ liệu)."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "previous_locations" not in text and "migrate" not in text, path


def test_check_datasets_script_uses_ws(tmp_path):
    ws = fake_ws(tmp_path)
    env = {**os.environ, "WS": str(ws)}
    out = subprocess.run(["bash", str(REPO / "scripts" / "pod" / "check_datasets.sh"), "--dataset", "deep_blending"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"WS = {ws}" in out.stdout and "deep_blending" in out.stdout
