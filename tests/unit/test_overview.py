"""Test src.evaluation.overview (xem nhanh kết quả) và scripts/pod/pack_results.sh (đóng gói kết quả nhẹ).

Mọi số liệu trong file này là dữ liệu tổng hợp để kiểm tra logic.
"""

import json
import os
import subprocess
import tarfile

from src.evaluation.overview import build, main
from shell_env import REPO


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        content = json.dumps(content)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def make_ws(tmp_path):
    ws = tmp_path / "ws"
    reports = ws / "outputs" / "reports"
    write(reports / "run_all_20261017_100000.tsv",
          "# run_all 20261017_100000 | repo abc1234\n# step\tstatus\texit\tseconds\tnote\n"
          "env_probe\tFAIL\t1\t5\t\nsetup_inria\tSKIP\t\t\tcần env_probe thành công\n")
    write(reports / "baseline_db_summary_20261017_042824.json", {
        "scenes": [{"scene": "drjohnson", "status": "OK", "metrics": {"PSNR": 29.37, "SSIM": 0.905, "LPIPS": 0.236},
                    "train_seconds": 1405, "num_gaussians": 3105559},
                   {"scene": "playroom", "status": "MISSING", "metrics": {}}],
        "mean": {}, "reference": {"PSNR": 29.69, "SSIM": 0.906, "LPIPS": 0.238},
        "tolerance": {"PSNR": 0.5}, "verdict": "INCOMPLETE"})
    write(reports / "modes_db_quick_20261017_110000.json",
          {"scenes": {"drjohnson": {"default{max_steps=3000}": {"n": 1}, "mcmc{max_steps=3000}": {"n": 1}}},
           "incomplete": []})
    write(reports / "modes_db_20261017_120000.md", "| Cảnh | Biến thể |\n|---|---|\n| drjohnson | default |\n")
    write(ws / "data" / "custom_scenes" / "video8" / "video8.mp4", b"video")  # nguồn gốc
    scene = ws / "outputs" / "custom_scenes" / "video8" / "colmap"
    write(scene / "frames_report.json", {"source_type": "video", "selected": 300, "candidates": 1200,
                                         "rejected_blurry": 40, "fps": 4.0, "fps_auto": True,
                                         "duplicates_dropped": 3})
    write(scene / "sfm_report.json", {
        "verdict": "WARN", "input_images": 300, "registration_ratio": 0.93,
        "model": {"registered_images": 279, "points3d": 37125, "mean_reprojection_error_px": 0.81},
        "distorted_models": [{"path": "0", "registered": 279}, {"path": "1", "registered": 21}],
        "warnings": ["only 93% of input images registered"]})
    runs = ws / "outputs" / "custom_scenes" / "video8"
    write(runs / "default_s42" / "summary.json", {
        "run_dir": str(runs / "default_s42"), "scene": "video8", "mode": "default", "seed": 42, "status": "OK",
        "metrics": {"PSNR": 31.25, "SSIM": 0.93, "LPIPS": 0.15}, "num_gaussians": 1500000, "train_seconds": 900})
    write(runs / "default_s42" / "ply" / "point_cloud_29999.ply", b"x" * 2048)
    write(runs / "default_s42" / "videos" / "traj_29999.mp4", b"v" * 1024)  # video quỹ đạo của bước đánh giá
    write(runs / "default_s42" / "render_ellipse_20261020_101010" / "videos" / "traj_29999.mp4", b"v")  # render lại
    write(runs / "mcmc_s42" / "run_meta.json", {"scene": "video8", "mode": "mcmc", "status": "RUNNING",
                                                "owner": {"host": "elsewhere", "pid": 1}, "spec": {}})
    write(runs / "default_s42.prev_20261016_101010" / "summary.json",
          {"run_dir": "old", "status": "OK", "metrics": {"PSNR": 1.0}})
    write(ws / "ket_qua_1017_1200.tar.gz", b"tar")
    return ws


def runs_dir(ws):
    return ws / "outputs" / "custom_scenes" / "video8"


def test_overview_shows_every_section(tmp_path):
    ws = make_ws(tmp_path)
    text = "\n".join(build(ws, None))
    assert "run_all (run_all_20261017_100000.tsv)" in text and "repo abc1234" in text
    assert "env_probe     LỖI (exit 1)" in text and "cần env_probe thành công" in text
    assert "drjohnson" in text and "29.370" in text and "THIẾU KẾT QUẢ" in text and "INCOMPLETE" in text
    assert "Chạy thử gsplat 3000 vòng (modes_db_quick_20261017_110000.json): 2 lượt OK | OK" in text
    assert "| drjohnson | default |" in text
    assert "chọn 300 / 1200 ứng viên" in text and "fps 4.0 (tự chọn)" in text and "bỏ 3 frame trùng" in text
    assert "COLMAP: WARN | đăng ký 279/300 ảnh (93%) | 37,125 điểm | sai số chiếu lại 0.81 px" in text
    assert "COLMAP tách 2 nhóm ảnh" in text and "only 93% of input images registered" in text
    assert "default_s42" in text and "31.25" in text and "1,500,000" in text and "15.0" in text
    assert "mcmc_s42" in text and "DỞ DANG" in text  # RUNNING nhưng tiến trình ghi không còn
    assert "prev_" not in text and "1.00" not in text  # lượt cũ đã thay không được liệt kê
    assert "point_cloud_29999.ply" in text
    assert f"video: {runs_dir(ws) / 'default_s42' / 'videos' / 'traj_29999.mp4'}" in text
    assert "render_ellipse_20261020_101010/videos/traj_29999.mp4" in text
    section = text.split("== Cảnh video8")[1].split("\n== ")[0]
    assert not [line for line in section.splitlines() if line.split()[:1] == ["colmap"]]  # COLMAP không phải lượt train
    assert f"nguồn: {ws / 'data' / 'custom_scenes' / 'video8'} | 0 ảnh JPG/JPEG/PNG | file khác: video8.mp4" in text
    assert "ket_qua_1017_1200.tar.gz" in text


def test_overview_shows_the_latest_experiment(tmp_path):
    ws = make_ws(tmp_path)
    text = "\n".join(build(ws, None))
    assert "Thực nghiệm (scripts/pod/run_experiment.sh): chưa có" in text
    reports = ws / "outputs" / "reports"
    write(reports / "experiment_main_20261019_120000.tsv",
          "# experiment main 20261019_120000 | repo abc1234\n# step\tstatus\texit\tseconds\tnote\n"
          "fetch_mipnerf360\tOK\t0\t600\t\ngsplat_mipnerf360_s44\tSKIP\t\t\tkhông đủ chỗ trống cho seed 44\n")
    text = "\n".join(build(ws, None))
    assert f"== Thực nghiệm (experiment_main_20261019_120000.tsv); bảng so sánh: {reports}" in text
    assert "experiment main 20261019_120000 | repo abc1234" in text
    assert "fetch_mipnerf360             OK             0:10:00" in text
    assert "gsplat_mipnerf360_s44        BỎ QUA         không đủ chỗ trống cho seed 44" in text


def test_uploaded_scene_not_yet_prepared_is_listed(tmp_path):
    ws = make_ws(tmp_path)
    for i in range(3):
        write(ws / "data" / "custom_scenes" / "phong1" / f"IMG_{i}.JPG", b"x")
    text = "\n".join(build(ws, None))
    assert "== Cảnh phong1" in text and "| 3 ảnh JPG/JPEG/PNG" in text
    assert "chưa chuẩn bị" in text and "run_scene.sh <nguồn> phong1" in text


def test_overview_scene_filter_and_empty_workspace(tmp_path, capsys):
    ws = make_ws(tmp_path)
    text = "\n".join(build(ws, ["video8"]))
    assert "run_all" not in text and "B0" not in text and "== Cảnh video8" in text
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--ws", str(empty)]) == 0
    out = capsys.readouterr().out
    assert "chưa có tóm tắt" in out and "Cảnh tự chụp: chưa có" in out and "Gói kết quả: chưa có" in out
    assert main(["--ws", str(tmp_path / "missing")]) == 2


def test_overview_reads_runs_from_another_out_root(tmp_path):
    ws = make_ws(tmp_path)
    other = tmp_path / "other"
    write(other / "video8" / "mcmc_s42" / "summary.json", {
        "run_dir": str(other / "video8" / "mcmc_s42"), "mode": "mcmc", "status": "OK",
        "metrics": {"PSNR": 32.5, "SSIM": 0.94, "LPIPS": 0.12}})
    text = "\n".join(build(ws, ["video8"], out_root=other))
    assert "mcmc_s42" in text and "32.50" in text and "default_s42" not in text


def test_overview_shows_photo_scene_coverage_and_hints(tmp_path):
    ws = tmp_path / "ws"
    scene = ws / "outputs" / "custom_scenes" / "phong_anh" / "colmap"
    write(scene / "frames_report.json", {"source_type": "images", "selected": 40, "photos_found": 43,
                                         "duplicates_dropped": 2, "unreadable": [{"name": "x.png"}],
                                         "blurry_kept": ["IMG_7.JPG"]})
    write(scene / "sfm_report.json", {
        "verdict": "FAIL", "input_images": 40, "registration_ratio": 0.6,
        "model": {"registered_images": 24, "points3d": 9000, "mean_reprojection_error_px": 0.7},
        "distorted_models": [{"path": "0", "registered": 24}, {"path": "1", "registered": 16}],
        "warnings": ["only 60% of input images registered"],
        "coverage": {"status": "LOW", "azimuth_coverage_deg": 150, "largest_gap_deg": 210,
                     "camera_extent_over_depth": 0.31, "warnings": ["có khoảng hướng ngang khoảng 210 độ ..."]},
        "hints": ["COLMAP tách ảnh thành 2 nhóm ..."]})
    text = "\n".join(build(ws, ["phong_anh"]))
    assert "ảnh: dùng 40 / 43 ảnh JPG/JPEG/PNG, bỏ 2 ảnh trùng, bỏ 1 file hỏng | 1 ảnh có độ nét thấp (vẫn giữ)" in text
    assert "COLMAP: FAIL" in text
    assert "góc nhìn: THIẾU | phủ 150/360 độ theo phương ngang, khoảng trống lớn nhất 210 độ" in text
    assert "0.31 x khoảng cách tới cảnh" in text and "- có khoảng hướng ngang khoảng 210 độ" in text
    assert "- gợi ý: COLMAP tách ảnh thành 2 nhóm" in text


B0 = "outputs/benchmarks/deep_blending/inria_b0"


def test_pack_results_keeps_small_files_and_skips_heavy_ones(tmp_path):
    ws = tmp_path / "ws"
    keep = [
        "outputs/reports/b0.json", "outputs/reports/compare_db_1.md", "outputs/logs/jobs/run_all.log",
        "outputs/jobs/run_all.state",
        "outputs/custom_scenes/v/default_s42/summary.json", "outputs/custom_scenes/v/default_s42/train.log",
        "outputs/custom_scenes/v/default_s42/cfg.yml",
        "outputs/custom_scenes/v/default_s42/renders/val_step29999_0000.png",
        f"{B0}/drjohnson/results.json", f"{B0}/drjohnson/cfg_args",
        f"{B0}/drjohnson/test/ours_30000/renders/00000.png", f"{B0}/drjohnson/test/ours_30000/gt/00000.png",
        "outputs/custom_scenes/v/colmap/frames_report.json", "outputs/custom_scenes/v/colmap/sfm_report.json",
        "outputs/custom_scenes/v/colmap/colmap_logs/mapper.log",
    ]
    skip = [
        "outputs/custom_scenes/v/default_s42/ply/point_cloud_29999.ply",
        "outputs/custom_scenes/v/default_s42/ckpts/ckpt_29999_rank0.pt",
        "outputs/custom_scenes/v/default_s42/renders/val_step29999_0001.png",
        "outputs/custom_scenes/v/default_s42.prev_20261016_101010/summary.json",
        f"{B0}/drjohnson/point_cloud/iteration_30000/point_cloud.ply",
        f"{B0}/drjohnson/test/ours_30000/renders/00001.png",
        "outputs/custom_scenes/v/colmap/input/00001.jpg", "outputs/custom_scenes/v/colmap/sparse/0/images.bin",
        "outputs/benchmarks/mipnerf360/gsplat/room/.images_2_png_0123456789ab/frame_00001.png",
        "outputs/custom_scenes/v/default_s42/videos/traj_29999.mp4",
        "outputs/checkpoints/custom_scene/v/default_s42.ply",
        "data/custom_scenes/v/v.mp4", "ket_qua_1016_0900.tar.gz",
    ]
    for name in keep + skip:
        write(ws / name, "x")
    # liên kết tới một thư mục đã gom: không gom lại lần nữa (find không theo liên kết)
    (ws / "outputs" / "links").mkdir(parents=True)
    (ws / "outputs" / "links" / "v").symlink_to(ws / "outputs" / "custom_scenes" / "v")
    package = tmp_path / "out" / "pkg.tar.gz"
    out = subprocess.run(["bash", str(REPO / "scripts" / "pod" / "pack_results.sh"), str(package)],
                         env={**os.environ, "WS": str(ws)}, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().splitlines()[-1] == str(package)
    with tarfile.open(package) as tar:
        names = set(tar.getnames())
    assert set(keep) <= names
    assert not names & set(skip)
    assert not [name for name in names if name.startswith("outputs/links")]
    assert not package.with_name("pkg.tar.gz.part").exists()
