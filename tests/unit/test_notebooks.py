"""Test các notebook của app/notebooks/ (sinh bằng app/notebooks/build_notebooks.py): file trong repo đúng là bản sinh ra,
các ô chung giống hệt nhau, mỗi notebook một phương pháp gọi đúng phần của thực nghiệm chính, mã trong các ô hợp lệ."""

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess

import pytest

from shell_env import REPO
from src.training.experiment import load_experiment, run_name

NOTEBOOKS = REPO / "app" / "notebooks"
METHOD_FILES = {"b0": "b0_inria.ipynb", "default": "gsplat_default.ipynb", "mcmc": "gsplat_mcmc.ipynb"}


def builder():
    spec = importlib.util.spec_from_file_location("build_notebooks", NOTEBOOKS / "build_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells(name):
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))["cells"]


def source(cell):
    return "".join(cell["source"])


def code_after(name, heading):
    """Ô code ngay sau ô markdown có tiêu đề bắt đầu bằng `heading` (như cách bạn tìm ô theo số)."""
    nb = cells(name)
    for index, cell in enumerate(nb):
        if cell["cell_type"] == "markdown" and source(cell).lstrip("# ").startswith(heading):
            assert nb[index + 1]["cell_type"] == "code", heading
            return source(nb[index + 1])
    raise AssertionError(f"{name}: no cell {heading!r}")


def test_the_notebooks_in_the_repo_are_the_generated_ones(tmp_path):
    module = builder()
    assert module.main(["--check"]) == 0
    assert module.main(["--out", str(tmp_path)]) == 0
    built = sorted(path.name for path in tmp_path.iterdir())
    assert built == sorted(["runpod.ipynb", *METHOD_FILES.values()])
    assert sorted(path.name for path in NOTEBOOKS.glob("*.ipynb")) == built  # không có notebook ngoài bộ sinh
    (tmp_path / "runpod.ipynb").write_text("{}", encoding="utf-8")
    assert module.main(["--check", "--out", str(tmp_path)]) == 1


def test_the_setup_cells_are_the_same_in_every_notebook():
    reference = cells("runpod.ipynb")[1:11]  # ô 1–5 (tiêu đề và mã): máy, khóa deploy, code, môi trường, theo dõi
    for name in METHOD_FILES.values():
        assert [dict(cell, id=None) for cell in cells(name)[1:11]] == [dict(cell, id=None) for cell in reference], name
    stop = code_after("runpod.ipynb", "11. Dừng")
    pack = code_after("runpod.ipynb", "10. Mọi kết quả")
    assert pack.endswith("pack_results.sh --name main")  # gói của code tổng: ket_qua_main_<thời gian>.tar.gz
    assert code_after("b0_inria.ipynb", "9. Dừng") == stop
    assert code_after("b0_inria.ipynb", "8. Mọi kết quả") == pack.replace("--name main", "--name b0_inria")
    for name in ("gsplat_default.ipynb", "gsplat_mcmc.ipynb"):  # cùng ô, gói mang tên bản của notebook
        assert code_after(name, "11. Dừng") == stop
        assert code_after(name, "10. Mọi kết quả") == pack.replace("--name main", f"--name {name[:-6]}")


def test_each_method_notebook_runs_its_part_of_the_main_experiment():
    for method, name in METHOD_FILES.items():
        run = code_after(name, "6. Chạy")
        assert f'"run_experiment.sh", "main", "--only", "{method}",' in run.replace('f"{repo}/scripts/pod/', '"')
        assert '"--background" if CHAY else "--plan"' in run and "CHAY = False" in run
        label = name[:-6]  # tên bản = tên notebook: việc nền, báo cáo, gói kết quả, PLY xuất ra
        assert run_name("main", method) == label == load_experiment("main", REPO / "configs", "/workspace",
                                                                    method)["run_name"]
        assert f'/workspace/outputs/reports/{label}_*.md' in code_after(name, f"{7 if method == 'b0' else 8}. Bảng")
        header = source(cells(name)[0])
        assert f"**Tên bản: `{label}`.**" in header and f"`ket_qua_{label}_<thời gian>.tar.gz`" in header
        assert ("PLY xuất ra" in header) == (method != "b0")  # B0 không có ô xuất PLY
    assert 'MODES="default")' in code_after("gsplat_default.ipynb", "7. Cảnh tự chụp")
    assert 'MODES="default mcmc")' in code_after("gsplat_mcmc.ipynb", "7. Cảnh tự chụp")
    assert "run_scene.sh" not in "".join(source(cell) for cell in cells("b0_inria.ipynb"))  # B0: chỉ benchmark
    assert 'RUN = "mcmc_s42"' in code_after("gsplat_mcmc.ipynb", "9. Kết quả")
    assert "MODES" not in code_after("runpod.ipynb", "8. Cảnh tự chụp")  # code tổng: mặc định default, mcmc


def headings_of(name):
    """{số ô: tiêu đề} của các ô markdown `## <số>. <tiêu đề>`."""
    return {int(match.group(1)): match.group(2) for cell in cells(name) if cell["cell_type"] == "markdown"
            for match in [re.match(r"## (\d+)\. (.*)", source(cell))] if match}


RUNPOD_REFERENCE = re.compile(r"ô (\d+) của `runpod\.ipynb`|`runpod\.ipynb`,? ô (\d+)")


def test_cell_numbers_are_in_order_and_the_tables_point_at_existing_cells():
    runpod = headings_of("runpod.ipynb")
    for name in ["runpod.ipynb", *METHOD_FILES.values()]:
        headings = list(headings_of(name))
        assert headings == list(range(1, len(headings) + 1)), name
        text = "".join(source(cell) for cell in cells(name) if cell["cell_type"] == "markdown")
        # ô của runpod.ipynb mà notebook theo phương pháp nhắc tới: kiểm theo runpod.ipynb, không theo notebook này
        for match in RUNPOD_REFERENCE.finditer(text):
            number = int(match.group(1) or match.group(2))
            context = text[match.start() - 120:match.end() + 120]
            expected = {7: "Benchmark", 8: "Cảnh tự chụp", 9: "Kết quả một cảnh"}[number]
            assert runpod[number].startswith(expected), (name, number, context)
        text = RUNPOD_REFERENCE.sub("", text)
        referenced = {int(number) for number in re.findall(r"\bô (\d+)\b", text)}
        referenced |= {int(number) for row in re.findall(r"\| ([\d →]+)(?: \(| \|)", text)  # cột "Chạy các ô"
                       for number in re.findall(r"\d+", row)}
        assert referenced and referenced <= set(headings), (name, sorted(referenced - set(headings)))


def test_the_report_cell_shows_the_newest_summary_of_its_own_runs(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {  # tên: mtime (s); bảng từng benchmark và chế độ mcmc_indoor mới hơn, báo cáo mới nhất có mtime cũ hơn
        "gsplat_mcmc_20261028_100000.md": 100, "gsplat_mcmc_20261029_090000.md": 50,
        "gsplat_mcmc_20261029_090000_deep_blending.md": 300, "gsplat_mcmc_indoor_20261030_080000.md": 400,
        "main_20261031_070000.md": 500, "tune_mcmc_gsplat_mcmc_20261031_080000.md": 600,
    }
    for file, mtime in files.items():
        (reports / file).write_text(f"# {file}\n", encoding="utf-8")
        os.utime(reports / file, (mtime, mtime))
    code = code_after("gsplat_mcmc.ipynb", "8. Bảng").replace("/workspace/outputs/reports", str(reports))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "gsplat_mcmc.ipynb[report]", "exec"), {})
    assert out.getvalue().splitlines()[0] == str(reports / "gsplat_mcmc_20261029_090000.md")
    for file in files:
        (reports / file).unlink()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "gsplat_mcmc.ipynb[report]", "exec"), {})
    assert out.getvalue().startswith("Chưa có báo cáo")


def test_the_code_cells_are_valid_python_or_bash(tmp_path):
    bash = shutil.which("bash")
    for name in ["runpod.ipynb", *METHOD_FILES.values()]:
        for index, cell in enumerate(cells(name)):
            if cell["cell_type"] != "code":
                continue
            text = source(cell)
            if text.startswith("%%bash"):
                if bash:
                    script = tmp_path / f"{name}_{index}.sh"
                    script.write_text(text.split("\n", 1)[1], encoding="utf-8")
                    assert subprocess.run([bash, "-n", str(script)]).returncode == 0, (name, index)
                continue
            python = "\n".join(line for line in text.splitlines() if not line.startswith("!"))
            compile(python, f"{name}[{index}]", "exec")


@pytest.mark.parametrize("name", list(METHOD_FILES.values()))
def test_method_notebooks_name_the_estimates_of_the_plan(name):
    header = source(cells(name)[0])
    method = next(key for key, value in METHOD_FILES.items() if value == name)
    estimate = load_experiment("main", REPO / "configs", "/workspace", method)["estimate"]
    assert f"khoảng {estimate['gpu_hours']:g} giờ GPU" in source(cells(name)[11])  # ô 6 ghi đúng ước lượng của kế hoạch
    assert f"--only {method}" in header and f"app/notebooks/{name}" in header
