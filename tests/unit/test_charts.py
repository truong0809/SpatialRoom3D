"""Test biểu đồ tổng (src/evaluation/charts.py). Mọi con số là dữ liệu giả để kiểm tra logic, KHÔNG phải kết quả.

Báo cáo dùng trong test được lập bằng chính src.evaluation.compare_modes / compare_reference từ các lượt giả (thư mục
như trên pod: outputs/benchmarks/<benchmark>/...), nên biểu đồ được kiểm trên đúng cấu trúc JSON mà pipeline ghi ra.
Phần vẽ cần matplotlib (env gs-gsplat có sẵn): test vẽ bị bỏ qua khi venv không có.
"""

import importlib.util
import json
import statistics
import struct
import warnings

import pytest

from src.evaluation import charts
from src.evaluation.charts import (build, fmt, fmt_compact, fmt_signed, label_order, main, make_source, report_name,
                                   short_labels)
from src.evaluation.compare_modes import main as compare_main
from src.evaluation.compare_reference import main as reference_main

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
PROTOCOL = {"data_factor": 1, "test_every": 8, "max_steps": 30000, "lpips_net": "vgg", "antialiased": False,
            "save_ply": True, "disable_video": False}
CONFIG = {"name": "baseline_inria_deepblending", "tag": "db", "scenes": ["drjohnson", "playroom"], "method": "ours_30000",
          "iteration": 30000, "reference": {"metrics": {"PSNR": 29.690, "SSIM": 0.906, "LPIPS": 0.238},
                                            "source": "số giả của test"},
          "tolerance": {"PSNR": 0.5, "SSIM": 0.01, "LPIPS": 0.01}}
M360_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]


def write_run(root, scene, mode, seed, metrics, *, tag="", args=None, steps=30000):
    """Một lượt gsplat đã xong (summary.json như src.training.run ghi, rút gọn); `steps` khác 30000: lượt chạy thử."""
    run = root / scene / f"{mode}{'_' + tag if tag else ''}_s{seed}"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text("{}", encoding="utf-8")
    deviations = {} if steps == 30000 else {"max_steps": {"base": 30000, "used": steps, "source": "quick"}}
    summary = {"run_dir": str(run), "scene": scene, "mode": mode, "seed": seed, "status": "OK", "metrics": metrics,
               "num_gaussians": 2_000_000, "train_seconds": 1800, "peak_vram_mib": 10240, "ply_size_mb": 450.0,
               "render_ms_per_image": 9.0, "protocol": {**PROTOCOL, "max_steps": steps}, "deviations": deviations,
               "split_sha256": "s1", "split_rule": "every_n", "split_applied_ok": True, "gsplat_commit": "6e8c837",
               "gpu": "RTX A6000", "args": args or {}, "lpips_net": "vgg", "scene_fingerprint": "fp", "tag": tag or None,
               "eval_step": steps - 1}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def write_b0(root, scene, metrics):
    model = root / scene
    (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
    (model / "results.json").write_text(json.dumps({"ours_30000": metrics}), encoding="utf-8")
    (model / "run_meta.json").write_text(json.dumps({"train_seconds": 1500}), encoding="utf-8")
    (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2500000\nend_header\n")


def compare(tmp_path, name, *args):
    out = tmp_path / f"{name}.json"
    assert compare_main([*args, "--out", str(out), "--markdown", str(tmp_path / f"{name}.md")]) == 0
    return json.loads(out.read_text(encoding="utf-8")), out


def main_like(tmp_path):
    """Kiểu thực nghiệm chính trên Deep Blending: 2 cảnh × default / mcmc × 3 seed, B0 tự chạy, số công bố."""
    runs, b0 = tmp_path / "benchmarks" / "deep_blending" / "gsplat", tmp_path / "benchmarks" / "deep_blending" / "inria_b0"
    for scene, base in (("drjohnson", 29.2), ("playroom", 29.8)):
        for seed, delta in ((42, 0.10), (43, 0.05), (44, -0.02)):
            write_run(runs, scene, "default", seed, {"PSNR": base, "SSIM": 0.900, "LPIPS": 0.245})
            write_run(runs, scene, "mcmc", seed, {"PSNR": base + delta, "SSIM": 0.902, "LPIPS": 0.235},
                      args={"opacity_reg": "0.001"})
        write_b0(b0, scene, {"PSNR": base + 0.2, "SSIM": 0.905, "LPIPS": 0.236})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    return compare(tmp_path, "main", "--root", str(runs), "--baseline", "default", "--modes", "default", "mcmc",
                   "--reference-config", str(config), "--b0-root", str(b0))


def m360_like(tmp_path):
    """Kiểu Mip-NeRF 360 của thực nghiệm chính: 2 cảnh × default / mcmc × 3 seed (một phần benchmark: đối chiếu số công
    bố SKIPPED), thêm một biến thể có tag chỉ ở room, seed 42."""
    runs = tmp_path / "benchmarks" / "mipnerf360" / "gsplat"
    for scene, base in (("room", 31.6), ("counter", 29.0)):
        for seed, delta in ((42, 0.30), (43, 0.20), (44, 0.25)):
            write_run(runs, scene, "default", seed, {"PSNR": base, "SSIM": 0.920, "LPIPS": 0.210})
            write_run(runs, scene, "mcmc", seed, {"PSNR": base + delta, "SSIM": 0.921, "LPIPS": 0.200})
    write_run(runs, "room", "mcmc", 42, {"PSNR": 31.75, "SSIM": 0.919, "LPIPS": 0.205}, tag="sh2", args={"sh_degree": 2})
    config = tmp_path / "config_m360.json"
    config.write_text(json.dumps({**CONFIG, "name": "baseline_inria_mipnerf360", "scenes": M360_SCENES}), encoding="utf-8")
    return compare(tmp_path, "m360", "--root", str(runs), "--baseline", "default", "--modes", "default", "mcmc",
                   "--scenes", "room", "counter", "--reference-config", str(config), "--tags", "sh2")


def tune_like(tmp_path, against="default"):
    """Kiểu dò tham số: một cảnh, một seed, default, mcmc và hai biến thể có tag; bảng so với `against`."""
    runs = tmp_path / "benchmarks" / "mipnerf360" / "gsplat"
    if not runs.exists():
        def metrics(psnr, lpips):
            return {"PSNR": psnr, "SSIM": 0.92, "LPIPS": lpips}
        write_run(runs, "room", "default", 42, metrics(31.6, 0.220))
        write_run(runs, "room", "mcmc", 42, metrics(31.9, 0.205))
        write_run(runs, "room", "mcmc", 42, metrics(32.0, 0.204), tag="sh2", args={"sh_degree": 2})
        write_run(runs, "room", "mcmc", 42, metrics(31.7, 0.210), tag="or0001", args={"opacity_reg": 0.001})
    modes = ["default", "mcmc"] if against == "default" else ["mcmc"]
    return compare(tmp_path, f"tune_vs_{against}", "--root", str(runs), "--baseline", against, "--modes", *modes,
                   "--tags", "sh2", "or0001")


def long_names(tmp_path):
    """Hai biến thể của mcmc cùng tên thư mục (hai thư mục kết quả) với tên tham số dài: nhãn là tên biến thể đã cắt."""
    first, second = tmp_path / "runs", tmp_path / "runs2"
    write_run(first, "room", "default", 42, {"PSNR": 31.60, "SSIM": 0.918, "LPIPS": 0.221})
    for root, value, psnr in ((first, 1, 31.85), (second, 2, 31.70)):
        write_run(root, "room", "mcmc", 42, {"PSNR": psnr, "SSIM": 0.921, "LPIPS": 0.212},
                  args={"a_very_long_parameter_name": value})
    return compare(tmp_path, "long", "--root", str(first), "--root", str(second), "--baseline", "default",
                   "--modes", "default", "mcmc")


def b0_only(tmp_path, scenes=("drjohnson", "playroom")):
    b0 = tmp_path / "benchmarks" / "deep_blending" / "inria_b0"
    write_b0(b0, "drjohnson", {"PSNR": 29.40, "SSIM": 0.905, "LPIPS": 0.236})
    write_b0(b0, "playroom", {"PSNR": 30.00, "SSIM": 0.909, "LPIPS": 0.240})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**CONFIG, "scenes": list(scenes)}), encoding="utf-8")
    out = tmp_path / "b0.json"
    reference_main(["--config", str(config), "--model-root", str(b0), "--out", str(out)])
    return json.loads(out.read_text(encoding="utf-8")), out


def sources(*made):
    """Báo cáo [(báo cáo, đường dẫn)] như main() đọc."""
    return [make_source(report, path.name) for report, path in made]


def by_slug(found):
    return {chart.slug: chart for chart in found}


def rows(chart):
    return {row[0]: row[1:] for row in chart.rows}


# --------------------------------------------------------------------------------------------- số và nhãn
def test_numbers_use_a_decimal_comma_and_a_real_minus_sign():
    assert fmt(29.3691, 3) == "29,369" and fmt(-1.26, 1) == "−1,3" and fmt(2_000_000 / 1e6, 2) == "2,00"
    assert fmt(-0.00004, 3) == "0,000"  # không có "−0,000"
    assert fmt(None, 3) == fmt(float("nan"), 3) == fmt(True, 1) == "–"
    assert fmt_signed(0.07, 3) == "+0,070" and fmt_signed(-0.0118, 4) == "−0,0118"
    # chênh lệch giữ dấu như bảng của compare_modes ("+0.0000" vẫn tính là tốt hơn ở bảng số cặp)
    assert fmt_signed(-0.00001, 3) == "−0,000" and fmt_signed(0.0, 4) == "+0,0000" and fmt_signed(None, 3) == "–"
    assert fmt_compact(0.5) == "0,5" and fmt_compact(0.01) == "0,01" and fmt_compact(2.0) == "2"


def test_labels_are_the_run_folders_without_the_seed():
    def agg(*dirs):
        return {"run_dirs": list(dirs)}
    long = "mcmc[c=3, d=4, e=5, f=6, g=7, h=8, i=9]"
    report = {"variants": ["default", "mcmc[a=1]", "mcmc[b=2]", long, "mcmc[x=1]", "mcmc[y=1]", "mcmc"],
              "scenes": {"room": {"default": agg("/r/room/default_s42", "/r/room/default_s43"),
                                  "mcmc[a=1]": agg("/r/room/mcmc_a_s42"),
                                  "mcmc[b=2]": agg("/r/room/mcmc_b_s42", "/r/room/mcmc_b2_s43"),
                                  long: agg("/r/room/mcmc_c_s42", "/r/room/mcmc_c2_s42"),
                                  "mcmc[x=1]": agg("/r/room/mcmc_z_s42"), "mcmc[y=1]": agg("/q/room/mcmc_z_s42"),
                                  "mcmc": agg("/r/room/mcmc_s42")}}}
    labels = short_labels(report)
    assert labels["default"] == "default" and labels["mcmc[a=1]"] == "mcmc_a" and labels["mcmc"] == "mcmc"
    assert labels["mcmc[b=2]"] == "mcmc[b=2]"  # lượt của biến thể nằm ở hai tên thư mục: dùng tên biến thể
    assert labels[long] == long[:31] + "…" and len(labels[long]) == 32
    assert labels["mcmc[x=1]"] == "mcmc[x=1]" and labels["mcmc[y=1]"] == "mcmc[y=1]"  # cùng tên thư mục: tên biến thể
    # lượt chính của mỗi chế độ trước, theo thứ tự của báo cáo; các biến thể khác theo nhãn
    assert label_order([(labels[v], v) for v in report["variants"]]) == [
        "default", "mcmc", "mcmc[b=2]", labels[long], "mcmc[x=1]", "mcmc[y=1]", "mcmc_a"]
    # tên dài trùng nhau sau khi cắt: thêm số thứ tự, nhãn vẫn khác nhau và không quá 32 ký tự
    names = [f"mcmc[a_very_long_parameter_name={value}]" for value in (1, 2)]
    labels = short_labels({"variants": names, "scenes": {"room": {name: agg(f"/r{i}/room/mcmc_s42")
                                                                 for i, name in enumerate(names)}}})
    assert labels == {names[0]: names[0][:26] + "… (1)", names[1]: names[1][:26] + "… (2)"}
    assert max(len(label) for label in labels.values()) <= 32


def test_reports_are_named_after_the_benchmark_of_their_runs(tmp_path):
    assert report_name(main_like(tmp_path)[0], "main_x") == "deep_blending"
    assert report_name(b0_only(tmp_path / "b0")[0], "b0_x") == "deep_blending"  # thư mục mô hình B0
    # lượt ngoài bố cục outputs/benchmarks/<benchmark>/ (vd. cảnh tự chụp): tên file báo cáo
    assert report_name(long_names(tmp_path / "long")[0], "compare_custom_1") == "compare_custom_1"


# ----------------------------------------------------------------------------------------- dữ liệu biểu đồ
def test_one_report_gives_the_inria_check_and_the_total_against_its_baseline(tmp_path):
    report, path = main_like(tmp_path)
    (source,) = sources((report, path))
    found, skipped = build([source])
    # mốc có khớp Inria không (điều kiện để dùng nó làm mốc) trước, rồi chênh lệch tổng
    assert [chart.key for chart in found] == ["01_doi_chieu_inria", "02_so_voi_default"] and not skipped
    assert source.labels == {"default": "default", "mcmc[opacity_reg=0.001]": "mcmc"} and source.base == "default"
    inria, total = found
    assert total.title == "Chênh lệch tổng so với default" and total.sources == ["main.json"]
    # trung bình ± độ lệch chuẩn và số cặp tốt hơn trên 6 cặp (cảnh, seed): đúng số ở mục wins (bảng số cặp tốt hơn)
    assert total.rows == [["mcmc", "drjohnson, playroom", "42, 43, 44", "6", "+0,043 ± 0,054", "+0,0020 ± 0,0000",
                           "−0,0100 ± 0,0000", "4/6 · 6/6 · 6/6"]]
    won = report["wins"]["mcmc[opacity_reg=0.001]"]
    assert total.rows[0][4] == f"{fmt_signed(won['mean']['PSNR'], 3)} ± {fmt(won['std']['PSNR'], 3)}"
    panels = total.groups[0].panels
    assert [panel.items[0]["text"] for panel in panels] == ["+0,043 · 4/6", "+0,0020 · 6/6", "−0,0100 · 6/6"]
    assert sorted(panels[0].items[0]["points"]) == pytest.approx([-0.02, -0.02, 0.05, 0.05, 0.10, 0.10])
    assert [panel.invert for panel in panels] == [False, False, True]  # LPIPS: trục đảo, phải vẫn là tốt hơn
    assert total.groups[0].labels == ["mcmc"] and panels[0].items[0]["color"] == charts.ORANGE
    assert total.subtitle[0] == "Mốc default; 2 cảnh (drjohnson, playroom); seed 42, 43, 44"
    assert total.subtitle[2] == "Mọi mục điều kiện so với mốc default đạt (cột Điều kiện của báo cáo)."
    assert ("từng cặp (cảnh, seed)", charts.INK, "small") in total.legend
    checks = rows(inria)
    assert checks["drjohnson: default − B0"] == ["29,400 → 29,200 (−0,200 ✓; ±0,5)", "0,9050 → 0,9000 (−0,0050 ✓; ±0,01)",
                                                 "0,2360 → 0,2450 (+0,0090 ✓; ±0,01)", "OK"]
    assert checks["trung bình: default − công bố"][0] == "29,690 → 29,500 (−0,190 ✓; ±0,5)"
    assert checks["trung bình: default − công bố"][3] == "PASS"
    assert inria.subtitle[0] == "B0 tự chạy, từng cảnh: PASS; số Inria công bố, trung bình các cảnh: PASS"


def test_reports_of_several_benchmarks_are_pooled_with_one_row_per_benchmark(tmp_path):
    db, m360 = main_like(tmp_path), m360_like(tmp_path)
    found, skipped = build(sources(db, m360))
    assert [chart.key for chart in found] == ["01_doi_chieu_inria", "02_so_voi_default"] and not skipped
    inria, total = found
    pairs = [pair for report, _ in (db, m360) for pair in report["pairs"]["mcmc" if "mcmc" in report["pairs"]
                                                                          else "mcmc[opacity_reg=0.001]"]]
    assert len(pairs) == 12
    psnr = [pair["PSNR"] for pair in pairs]
    table = rows(total)
    # mcmc của Deep Blending (opacity_reg 0.001) và của Mip-NeRF 360 cùng nhãn mcmc: gộp 12 cặp
    assert table["mcmc: tổng"][:4] == ["deep_blending: drjohnson, playroom; mipnerf360: counter, room", "42, 43, 44", "12",
                                       f"+0,147 ± {fmt(statistics.stdev(psnr), 3)}"]
    assert fmt_signed(statistics.fmean(psnr), 3) == "+0,147"
    assert table["mcmc: tổng"][4:] == ["+0,0015 ± 0,0005", "−0,0100 ± 0,0000", "10/12 · 12/12 · 12/12"]
    # dòng của từng benchmark: đúng số ở bảng số cặp tốt hơn của báo cáo benchmark đó
    for name, (report, _) in (("deep_blending", db), ("mipnerf360", m360)):
        won = next(value for variant, value in report["wins"].items() if variant.startswith("mcmc") and "sh_" not in variant)
        assert table[f"mcmc: {name}"][3] == f"{fmt_signed(won['mean']['PSNR'], 3)} ± {fmt(won['std']['PSNR'], 3)}"
    assert table["mcmc: deep_blending"][:3] == ["deep_blending: drjohnson, playroom", "42, 43, 44", "6"]
    # biến thể chỉ có ở một benchmark: một dòng, ghi rõ benchmark; một cặp nên số ghi không kèm số cặp tốt hơn
    assert table["mcmc_sh2: mipnerf360"][:6] == ["mipnerf360: room", "42", "1", "+0,150", "−0,0010", "−0,0050"]
    labels = total.groups[0].labels
    assert labels == ["mcmc: tổng", "mcmc: deep_blending", "mcmc: mipnerf360", "mcmc_sh2: mipnerf360"]
    colors = [item["color"] for item in total.groups[0].panels[0].items]
    assert colors == [charts.ORANGE, charts.tint(charts.ORANGE, charts.PART_TINT), charts.tint(charts.ORANGE, charts.PART_TINT),
                      charts.AQUA]
    assert total.groups[0].panels[0].items[3]["text"] == "+0,150"
    assert total.subtitle[0] == ("Mốc default; 4 cảnh (deep_blending: drjohnson, playroom; mipnerf360: counter, room); "
                                 "seed 42, 43, 44")
    assert total.subtitle[2] == ("Điều kiện so với mốc default chưa đạt: seeds=DIFF ở 1/5 cặp (cảnh, biến thể) (cột Điều "
                                 "kiện của báo cáo): chưa được nói \"cải thiện\".")
    assert [entry[0] for entry in total.legend] == ["mcmc", "mcmc_sh2", "từng cặp (cảnh, seed)"]  # dòng nhạt: không
    assert total.sources == ["main.json", "m360.json"]
    # đối chiếu Inria: mỗi báo cáo có phần đối chiếu một dòng trạng thái; chỉ Deep Blending có số để vẽ
    assert [group.title for group in inria.groups] == ["deep_blending"]
    assert inria.subtitle[:2] == ["deep_blending: B0 tự chạy, từng cảnh: PASS; số Inria công bố, trung bình các cảnh: PASS",
                                  "mipnerf360: số Inria công bố, trung bình các cảnh: SKIPPED"]
    assert inria.subtitle[2].startswith("mipnerf360: chỉ chạy 2 cảnh (room, counter), số liệu công bố là trung bình của 9")
    assert inria.header[0] == "Báo cáo" and inria.rows[0][:2] == ["deep_blending", "drjohnson: default − B0"]


def test_the_variants_of_mcmc_get_their_own_total_against_mcmc(tmp_path):
    found, skipped = build(sources(tune_like(tmp_path), tune_like(tmp_path, against="mcmc")))
    assert [chart.key for chart in found] == ["01_so_voi_default", "02_so_voi_mcmc"]
    assert dict(skipped) == {"Đối chiếu default với Inria": "báo cáo không có phần đối chiếu (mốc không phải default, hoặc "
                                                               "chưa có B0 tự chạy và cấu hình benchmark)"}
    default, mcmc = found
    assert default.groups[0].labels == ["mcmc", "mcmc_or0001", "mcmc_sh2"]  # mcmc trước, biến thể theo nhãn
    assert [item["text"] for item in default.groups[0].panels[0].items] == ["+0,300", "+0,100", "+0,400"]
    assert [item["color"] for item in default.groups[0].panels[0].items] == [charts.ORANGE, charts.AQUA, charts.AQUA]
    assert mcmc.title == "Chênh lệch tổng so với mcmc" and mcmc.groups[0].labels == ["mcmc_or0001", "mcmc_sh2"]
    assert [item["text"] for item in mcmc.groups[0].panels[0].items] == ["−0,200", "+0,100"]
    assert [item["text"] for item in mcmc.groups[0].panels[2].items] == ["+0,0050", "−0,0010"]
    assert not any(item["points"] for item in mcmc.groups[0].panels[0].items)  # một cặp: không chấm từng cặp
    assert mcmc.subtitle[1].startswith("Mỗi phương pháp một cặp (cảnh, seed) chung với mốc")
    assert "chưa đạt: seeds=FEW ở 2/2 cặp (cảnh, biến thể)" in mcmc.subtitle[2]
    assert "chưa được nói \"cải thiện\"" in mcmc.subtitle[2]
    assert [entry[0] for entry in default.legend] == ["mcmc", "biến thể khác"]  # mỗi màu một mục
    assert [entry[0] for entry in mcmc.legend] == ["biến thể khác"]
    index = charts.index_markdown(found, skipped, sources(tune_like(tmp_path), tune_like(tmp_path, against="mcmc")))
    assert index.count("| mcmc_sh2 | `mcmc[sh_degree=2]` |\n") == 1  # hai báo cáo, cùng benchmark: một dòng


def test_b0_only_report_shows_b0_against_the_published_numbers(tmp_path):
    found, skipped = build(sources(b0_only(tmp_path)))
    assert [chart.key for chart in found] == ["01_b0_so_voi_cong_bo"] and not skipped
    table = rows(found[0])
    assert table["drjohnson"] == ["29,400", "0,9050", "0,2360"] and table["trung bình B0"] == ["29,700", "0,9070", "0,2380"]
    assert table["Inria công bố"] == ["29,690 ± 0,5", "0,9060 ± 0,01", "0,2380 ± 0,01"]
    assert table["chênh lệch (trung bình − công bố)"] == ["+0,010 ✓", "+0,0010 ✓", "+0,0000 ✓"]
    assert found[0].groups[0].panels[0].items[2]["text"] == "29,700 (Δ +0,010 ✓)"
    assert found[0].subtitle[0].startswith("baseline_inria_deepblending, ours_30000: kết luận PASS (ngưỡng PSNR ±0,5")
    found, _ = build(sources(b0_only(tmp_path / "missing", scenes=("drjohnson", "playroom", "kitchen"))))
    assert "Thiếu kết quả: kitchen" in found[0].subtitle[1]
    assert [row[0] for row in found[0].rows] == ["drjohnson", "playroom", "Inria công bố"]  # chưa có trung bình


def test_the_inria_chart_says_why_it_is_not_drawn():
    report = {"baseline": "default", "variants": ["default"], "scenes": {},
              "b0_check": {"verdict": "N/A", "reasons": ["default_s42: max_steps 3000 != 30000"], "scenes": []},
              "reference_check": {"verdict": "SKIPPED", "reasons": ["chỉ chạy 1 cảnh (room)"]}}
    title, reason = charts.inria_chart([make_source(report, "r.json")])
    assert title == "Đối chiếu default với Inria"
    assert "B0 tự chạy, từng cảnh: N/A" in reason and "max_steps 3000 != 30000" in reason and "chỉ chạy 1 cảnh" in reason


def test_reports_sharing_runs_are_not_pooled(tmp_path):
    report, path = main_like(tmp_path)
    again = path.with_name("main_lap_lai.json")  # cùng các lượt, lập lại vào file khác
    again.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    found, skipped = build(sources((report, path), (report, again)))
    assert [chart.slug for chart in found] == ["doi_chieu_inria"]
    reason = dict(skipped)["Chênh lệch tổng so với default"]
    assert reason.startswith("main.json và main_lap_lai.json có chung lượt ") and "không gộp" in reason
    found, skipped = build(sources((report, path), (report, path)))  # cùng một file đưa hai lần
    assert "so_voi_default" not in [chart.slug for chart in found]
    assert dict(skipped)["Chênh lệch tổng so với default"].startswith("main.json và main.json có chung lượt ")


def test_other_protocols_same_benchmarks_and_lone_baselines(tmp_path):
    """Mốc của lượt chạy thử (sai khác protocol) không gộp với mốc đúng protocol cùng tên thư mục; hai báo cáo cùng
    benchmark mang tên file; báo cáo chỉ có mốc thì nói đúng lý do."""
    db = main_like(tmp_path)
    quick = tmp_path / "benchmarks" / "deep_blending" / "gsplat_quick"
    for scene in ("drjohnson", "playroom"):
        write_run(quick, scene, "default", 42, {"PSNR": 26.0, "SSIM": 0.86, "LPIPS": 0.33}, steps=3000)
        write_run(quick, scene, "mcmc", 42, {"PSNR": 26.3, "SSIM": 0.87, "LPIPS": 0.31}, steps=3000)
    quick_report = compare(tmp_path, "quick", "--root", str(quick), "--baseline", "default{max_steps=3000}",
                           "--modes", "default", "mcmc", "--include-deviating")
    found, _ = build(sources(db, quick_report))
    assert [chart.slug for chart in found] == ["doi_chieu_inria", "so_voi_default", "so_voi_default_max_steps_3000"]
    assert found[1].groups[0].labels == ["mcmc"] and found[1].rows[0][3] == "6"  # chỉ các lượt 30.000 vòng
    assert found[2].title == "Chênh lệch tổng so với default{max_steps=3000}" and found[2].rows[0][3] == "2"
    assert any("sai khác protocol" in line for line in found[2].subtitle)
    runs = tmp_path / "benchmarks" / "deep_blending" / "gsplat"
    halves = [compare(tmp_path, name, "--root", str(runs), "--baseline", "default", "--modes", "default", "mcmc",
                      "--scenes", scene) for name, scene in (("db_a", "drjohnson"), ("db_b", "playroom"))]
    total = by_slug(build(sources(*halves))[0])["so_voi_default"]
    assert total.groups[0].labels == ["mcmc: tổng", "mcmc: db_a", "mcmc: db_b"]  # cùng benchmark: tên file
    only = tmp_path / "only" / "benchmarks" / "deep_blending" / "gsplat"
    write_run(only, "drjohnson", "default", 42, {"PSNR": 29.2, "SSIM": 0.900, "LPIPS": 0.245})
    found, skipped = build(sources(compare(tmp_path / "only", "only", "--root", str(only), "--modes", "default")))
    assert dict(skipped)["Chênh lệch tổng so với default"] == ("báo cáo chỉ có mốc default: không có phương pháp nào "
                                                               "khác để so")


def test_old_or_broken_reports_only_drop_their_own_chart(tmp_path):
    report, _ = main_like(tmp_path)
    for made in (tune_like(tmp_path / "tune"), b0_only(tmp_path / "b0"), (report, tmp_path / "main.json")):
        for chart in build(sources(made))[0]:  # mọi ảnh mang tên mà module nhận là của nó khi dọn thư mục
            assert charts.OWN_IMAGE.fullmatch(f"{chart.key}.png"), chart.key
    old = {key: value for key, value in report.items() if key != "pairs"}  # báo cáo lập bằng bản cũ
    found, skipped = build([make_source(old, "old.json")])
    assert [chart.slug for chart in found] == ["doi_chieu_inria"]
    assert "chưa có mục `pairs`" in dict(skipped)["Chênh lệch tổng so với default"]
    broken = {**report, "pairs": [], "b0_check": {"scenes": 1}}  # mục hỏng
    found, skipped = build([make_source(broken, "broken.json")])
    assert not found
    for title in ("Chênh lệch tổng so với default", "Đối chiếu mốc với Inria"):
        assert dict(skipped)[title].startswith("không đọc được phần của báo cáo cần cho biểu đồ này")


# ------------------------------------------------------------------------------------------------------- CLI
def test_cli_rejects_files_that_are_not_comparison_reports(tmp_path, capsys):
    other = tmp_path / "other.json"
    other.write_text('{"scenes": 1}', encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    for path in (other, broken, tmp_path / "missing.json"):
        assert main(["--report", str(path)]) == 2
    assert "is not a report of src.evaluation.compare_modes" in capsys.readouterr().err
    _, report = main_like(tmp_path / "main")
    assert main(["--report", str(report), str(other), "--out", str(tmp_path / "out")]) == 2  # một file hỏng: dừng
    with pytest.raises(SystemExit):  # nhiều báo cáo: phải chỉ thư mục ghi
        main(["--report", str(report), str(report)])
    assert not list(tmp_path.rglob("*_charts")) and not (tmp_path / "out").exists()


def test_cli_without_matplotlib_draws_nothing(tmp_path, monkeypatch, capsys):
    _, path = main_like(tmp_path)
    find_spec = charts.importlib.util.find_spec
    monkeypatch.setattr(charts.importlib.util, "find_spec",
                        lambda name, *args: None if name == "matplotlib" else find_spec(name, *args))
    assert main(["--report", str(path)]) == 3
    assert "matplotlib is not installed" in capsys.readouterr().err
    assert not path.with_name("main_charts").exists()


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", data[16:24])


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib (env gs-gsplat)")
def test_cli_writes_the_images_the_index_and_a_section_of_the_report(tmp_path, capsys):
    _, path = main_like(tmp_path)
    markdown = tmp_path / "main.md"
    tables = markdown.read_text(encoding="utf-8")
    out = tmp_path / "main_charts"
    out.mkdir()
    (out / "09_anh_rieng.png").write_bytes(b"x")  # file của người dùng: giữ nguyên
    (out / "ghi_chu.txt").write_text("của người dùng", encoding="utf-8")
    for _ in range(2):  # chạy lại: thay ảnh và mục Biểu đồ, không thêm mục thứ hai
        assert main(["--report", str(path), "--append-to", str(markdown)]) == 0
    keys = ["01_doi_chieu_inria", "02_so_voi_default"]
    assert sorted(p.name for p in out.glob("*.png")) == [f"{key}.png" for key in keys] + ["09_anh_rieng.png"]
    assert (out / "ghi_chu.txt").read_text(encoding="utf-8") == "của người dùng"
    assert (out / "09_anh_rieng.png").read_bytes() == b"x"
    for key in keys:
        width, height = png_size(out / f"{key}.png")
        assert width > 900 and height > 300
    index = (out / "index.md").read_text(encoding="utf-8")
    assert index.startswith("# Biểu đồ của `main.json`\n")
    assert "![Chênh lệch tổng so với default](02_so_voi_default.png)" in index
    assert ("| mcmc | drjohnson, playroom | 42, 43, 44 | 6 | +0,043 ± 0,054 | +0,0020 ± 0,0000 | −0,0100 ± 0,0000 "
            "| 4/6 · 6/6 · 6/6 |") in index
    assert "| mcmc | `mcmc[opacity_reg=0.001]` |" in index  # nhãn trong biểu đồ ứng với biến thể nào
    text = markdown.read_text(encoding="utf-8")
    assert text.startswith(tables.rstrip("\n")) and text.count("**Biểu đồ**") == 1  # bảng của báo cáo không đổi
    assert "**Biểu đồ** (`main_charts/`; cách đọc và bảng số của từng biểu đồ: `main_charts/index.md`):" in text
    for key in keys:
        assert f"(main_charts/{key}.png)" in text
    # vẽ lại vào thư mục khác: mục Biểu đồ cũ được thay, không thành hai mục
    assert main(["--report", str(path), "--out", str(tmp_path / "khac_charts"), "--append-to", str(markdown)]) == 0
    text = markdown.read_text(encoding="utf-8")
    assert text.count("**Biểu đồ**") == 1 and "(khac_charts/02_so_voi_default.png)" in text
    # cùng thư mục, báo cáo có ít biểu đồ hơn: ảnh cũ do module ghi được xóa, file của người dùng vẫn giữ
    _, tune = tune_like(tmp_path / "tune")
    assert main(["--report", str(tune), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["01_so_voi_default.png", "09_anh_rieng.png"]
    # lần chạy trước bị ngắt sau khi xóa index.md: ảnh mang tên biểu đồ của module vẫn là của nó, chạy lại được
    (out / "index.md").unlink()
    assert main(["--report", str(path), "--out", str(out)]) == 0 and (out / "09_anh_rieng.png").read_bytes() == b"x"
    # thư mục có index.md không do module ghi: dừng, không ghi gì
    other = tmp_path / "khac"
    other.mkdir()
    (other / "index.md").write_text("của người dùng", encoding="utf-8")
    capsys.readouterr()
    assert main(["--report", str(path), "--out", str(other)]) == 2
    assert f"{other}/index.md was not written by src.evaluation.charts" in capsys.readouterr().err
    assert [p.name for p in other.iterdir()] == ["index.md"]


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib (env gs-gsplat)")
def test_cli_pools_the_reports_of_an_experiment_into_one_folder(tmp_path, capsys):
    _, db = main_like(tmp_path)
    _, m360 = m360_like(tmp_path)
    summary = tmp_path / "main_20261029_090000.md"
    summary.write_text("# Thực nghiệm main (20261029_090000)\n", encoding="utf-8")
    out = tmp_path / "main_20261029_090000_charts"
    assert main(["--report", str(db), str(m360), "--out", str(out), "--append-to", str(summary)]) == 0
    assert f"Biểu đồ: 2 ảnh trong {out}" in capsys.readouterr().out
    assert sorted(p.name for p in out.glob("*.png")) == ["01_doi_chieu_inria.png", "02_so_voi_default.png"]
    index = (out / "index.md").read_text(encoding="utf-8")
    assert index.startswith("# Biểu đồ của `main.json`, `m360.json`\n")
    assert "| mcmc | `mcmc[opacity_reg=0.001]` | deep_blending |" in index  # nhãn gộp: biến thể của từng báo cáo
    assert "| mcmc: tổng | deep_blending: drjohnson, playroom; mipnerf360: counter, room | 42, 43, 44 | 12 |" in index
    text = summary.read_text(encoding="utf-8")
    assert "![Chênh lệch tổng so với default](main_20261029_090000_charts/02_so_voi_default.png)" in text


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib (env gs-gsplat)")
def test_a_chart_that_fails_does_not_stop_the_others(tmp_path, monkeypatch, capsys):
    _, path = main_like(tmp_path)
    save = charts.save

    def failing(chart, *args, **kwargs):
        if chart.slug == "doi_chieu_inria":
            raise RuntimeError("hỏng giả")
        return save(chart, *args, **kwargs)
    monkeypatch.setattr(charts, "save", failing)
    assert main(["--report", str(path)]) == 0
    out = tmp_path / "main_charts"
    assert not (out / "01_doi_chieu_inria.png").exists() and (out / "02_so_voi_default.png").is_file()
    assert "WARN: chart 01_doi_chieu_inria could not be drawn: RuntimeError: hỏng giả" in capsys.readouterr().err
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "- Đối chiếu default với Inria: vẽ lỗi (RuntimeError: hỏng giả)" in index
    # số mục = số của file ảnh
    assert ("## 2. Chênh lệch tổng so với default\n\n![Chênh lệch tổng so với default](02_so_voi_default.png)"
            in index)


def pooled(tmp_path):
    return sources(main_like(tmp_path), m360_like(tmp_path))


def tune_both(tmp_path):
    return sources(tune_like(tmp_path), tune_like(tmp_path, against="mcmc"))


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib (env gs-gsplat)")
@pytest.mark.parametrize("make", [lambda p: sources(main_like(p)), pooled, tune_both, lambda p: sources(b0_only(p)),
                                  lambda p: sources(long_names(p))], ids=["main", "pooled", "tune", "b0", "long"])
def test_figures_show_every_number_inside_their_frame(tmp_path, make):
    import matplotlib.axes
    import matplotlib.text
    found, _ = build(make(tmp_path))
    assert found
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for chart in found:
            fig = charts.figure(chart)
            renderer = fig.canvas.get_renderer()
            texts = {text.get_text() for text in fig.findobj(matplotlib.text.Text)}
            numbers = [item["text"] for group in chart.groups for panel in group.panels for item in panel.items if item]
            assert numbers and set(numbers) <= texts, chart.key
            assert chart.title in texts and "Nguồn: " + ", ".join(chart.sources) in texts
            for ax in fig.findobj(matplotlib.axes.Axes):
                box = ax.get_window_extent(renderer)
                for text in ax.texts:  # nhãn số: không tràn khỏi khung (không bị cắt, không đè lên khung bên cạnh)
                    extent = text.get_window_extent(renderer)
                    assert box.x0 - 1 <= extent.x0 and extent.x1 <= box.x1 + 1, (chart.key, text.get_text())
                title = ax.get_title(loc="left")
                if title.startswith(("LPIPS ↓", "ΔLPIPS ↓")):  # khung chỉ số LPIPS: trục đảo chiều, phải là tốt hơn
                    left, right = ax.get_xlim()
                    assert "trục đảo" in title and left > right, chart.key
                heading = ax.title if title else None
                if heading is not None:  # tên khung không bị cắt ở mép hình
                    extent = heading.get_window_extent(renderer)
                    assert extent.x0 >= 0 and extent.x1 <= fig.bbox.x1 + 1, (chart.key, title)
                if chart.slug.startswith("so_voi_"):  # nới trục cho nhãn không được làm thanh và chấm co lại quá mức
                    left, right = ax.get_xlim()
                    reach = [abs(patch.get_width()) for patch in ax.patches]
                    reach += [abs(x) for line in ax.lines for x in line.get_xdata()]
                    if max(reach, default=0) > 0:
                        assert max(reach) >= 0.25 * abs(right - left), (chart.key, title, ax.get_xlim(), max(reach))
    assert not [w for w in caught if "missing from" in str(w.message)]  # chữ Việt, ↑ ↓ ✓ ✗ − đều có trong font
