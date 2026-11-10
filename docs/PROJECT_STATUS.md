# PROJECT STATUS

_Cập nhật: 10/11/2026. Bản rút gọn (D21): lịch sử chi tiết (nhật ký kiểm thử, rà soát độc lập, ghi chú từng lần sửa) nằm trong lịch sử Git của file này và trong các bản bàn giao của Project trên claude.ai._

## Mục tiêu hiện tại

1. **Đo nốt sản phẩm trên pod** (D20, D22): ngày 08/11/2026 đã tải video lên trang web, dựng phòng `video10` và xem 3D trên trình duyệt; còn thiếu thời gian từng bước ngoài train, thời gian tải trang, FPS, và kiểm tra số Gaussian của `model.splat` (mục "Còn mở").
2. **Hoàn thiện trang web theo bản thiết kế** (giai đoạn 5): trang Kết quả thực nghiệm; chấm điểm phòng tự dựng bằng lượt quay kiểm tra (`docs/design/product.md` mục 5).
3. **Báo cáo đồ án** (giai đoạn 6) từ kết quả thực nghiệm ngày 31/10/2026 (mục "Kết quả").

## Quy ước trạng thái

`Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`. Chỉ mức cuối cùng mới được tính là hoàn thành.

## Quyết định đã chốt

| # | Ngày | Quyết định |
|---|---|---|
| D1 | 09/10/2026 | Phương pháp trọng tâm là 3DGS; baseline là 3DGS gốc (Kerbl et al., 2023) |
| D2 | 09/10/2026 | Hạ tầng tính toán: Runpod |
| D3 | 09/10/2026 | Chạy kiểm chứng trên GPU gộp vào cuối; mọi script có kiểm thử trên CPU trước |
| D4 | 09/10/2026 | `src/reconstruction/colmap_runner.py` thay `convert.py` của Inria (COLMAP 4.x đổi tên tùy chọn GPU) |
| D5 | 09/10/2026 | Hai chế độ trên gsplat: `default` (DefaultStrategy, baseline) và `mcmc` (MCMCStrategy, theo notebook demo của nhóm); B0 (3DGS gốc của Inria) làm mốc đối chiếu cho `default` |
| D6 | 18/10/2026 | Đầu vào: một video, hoặc một thư mục ≥ 10 ảnh JPG/JPEG/PNG chụp cùng một phòng; một ảnh luôn bị từ chối |
| D7 | 19/10/2026 | Cấu trúc repo: `app/`, package `src/`, `configs/*.yaml`, bốn lệnh chính trong `scripts/`, `tests/`, `docs/`, `experiments/` |
| D8 | 19/10/2026 | Bố cục `/workspace`: `data/` chỉ chứa dữ liệu gốc, `outputs/` chứa mọi thứ sinh ra (`docs/design/architecture.md` mục 2) |
| D9 | 19/10/2026 | Repo mới `truong0809/SpatialRoom3D`, cài lại từ đầu trên pod mới; repo cũ giữ làm lưu trữ |
| D10 | 19/10/2026 | `scripts/pod/run_all.sh` chỉ cài và kiểm tra môi trường; chạy với dữ liệu nào do người dùng chọn |
| D11 | 19/10/2026 | Dữ liệu benchmark tải về pod bằng bước riêng (`scripts/pod/fetch_data.sh`), kiểm SHA-256 khi có |
| D12 | 19/10/2026 | Thực nghiệm chính: `default` / `mcmc` cùng số Gaussian; Deep Blending (2 cảnh, kèm B0) và 4 cảnh trong nhà của Mip-NeRF 360; seed 42, 43, 44; 30.000 vòng |
| D14 | 21/10/2026 | Dò tham số `mcmc` trên `room` (`tune_mcmc`), kèm kiểm tra điểm SfM và lọc Gaussian rồi chấm lại; `room` chỉ dùng để chọn, kết luận dựa trên 5 cảnh còn lại |
| D15 | 24/10/2026 | Thêm biến thể `rs30k`, `sh2`; vòng ghép: tối đa 3 ứng viên, 2 seed mới, luật chọn ghi trước khi có số liệu (`docs/design/experiments.md` mục 3.6) |
| D16 | 28/10/2026 | Biến thể từ notebook v2 của nhóm: `bilagrid`, `kr`, `mono`, gói `v2`; `kr`, `mono` chạy bằng hook trên bản sao có vá của trainer, không sửa gsplat |
| D17 | 28/10/2026 | Ngoài code tổng, mỗi phương pháp của thực nghiệm chính có một notebook chạy riêng (`b0_inria`, `gsplat_default`, `gsplat_mcmc`); tên bản dùng cho việc nền, báo cáo, gói kết quả |
| D18 | 29/10/2026 | Hai biểu đồ tổng cho báo cáo so sánh: đối chiếu với Inria, chênh lệch so với từng mốc |
| D19 | 31/10/2026 | Repo để public; pod lấy code qua HTTPS; ô 1 của notebook kiểm tra volume cho đổi quyền file (`chmod`) |
| D20 | 04/11/2026 | Sản phẩm dùng **một mô hình chính**: `mcmc` (`configs/product.yaml`). Code train riêng `src/product/train_main.py`, không dùng code thực nghiệm; một lệnh `scripts/build_room.sh`; web chỉ đọc thư mục tĩnh `outputs/product/web/`; trang web không so sánh hai chế độ. Theo yêu cầu của bạn ("sản phẩm thì chỉ chọn 1 base tốt nhất", "code để train 3d chính sẽ là 1 file riêng", đầu vào "có thể là ảnh hoặc video") |
| D21 | 04/11/2026 | Dọn repo: bỏ `docs/handoff/` (bản bàn giao lưu trong Project), rút gọn `PROJECT_STATUS.md` và `docs/ENVIRONMENT_AUDIT.md`; lịch sử còn trong Git. Theo yêu cầu của bạn ("bớt những cái thừa trên git") |
| D22 | 04/11/2026 | Trang web của sản phẩm (`app/web/`): server thư viện chuẩn Python chạy trên pod, gọi đúng `scripts/build_room.sh`; tải lên từng phần 8 MiB vào `data/uploads/<phòng>/`; mật khẩu; trình xem Spark 2.4.0 + three.js 0.180.0 ghim phiên bản. Theo yêu cầu của bạn ("trực quan up ảnh và video lên sau đó ra sản phẩm kết quả rồi xem") |
| D23 | 05/11/2026 | Giao diện trang web làm lại theo hướng bảng điều khiển nền tối: thanh bên thu gọn được; Tổng quan (cảnh mới nhất, tạo dự án, dự án chưa xong, lịch sử xử lý) → Dự án mới → Tiến độ → Xem 3D (khung xem lớn, bảng thông tin, thanh camera nổi, 12 góc nhìn). Giữ nguyên server, API, luồng dựng và trình vẽ Spark; chỉ thêm hai trường chỉ đọc (`pipeline`, `last_build`) và phông Be Vietnam Pro ghim phiên bản. Theo yêu cầu của bạn ("Redesign my existing SpatialRoom3D website as a high-end commercial AI product") |
| D24 | 09/11/2026 | Giấy phép MIT cho mã nguồn của repo (`LICENSE`); thêm thông tin nhóm vào README; kết quả tóm tắt của lượt 31/10/2026 ở `docs/RESULTS.md`. Theo yêu cầu của bạn ("Thêm cho tôi File LICENSE", "Thêm kết quả tóm tắt trong repo") |
| D25 | 09/11/2026 | Bỏ vòng ghép (D15) và chế độ thứ ba `mcmc_indoor`: theo luật chỉ có một ứng viên (`rbg`, +0,156 dB so với ngưỡng 0,150 dB); `random_bkgd` là tùy chọn có sẵn của gsplat; sản phẩm đã chốt `mcmc` (D20). Kế hoạch `tune_mcmc_combo.yaml` chưa từng lên Git, đã xóa; thiết kế còn ở `docs/design/experiments.md` tại commit `b428d95`. Luật đọc kết quả của `tune_mcmc` giữ nguyên (mục 3.6 của file đó). Theo yêu cầu của bạn ("bỏ cái nội dung vòng ghép đi vì chưa chạy dc nên nó bị thừa") |

## Đề xuất chờ xác nhận

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Hai tham số của mô hình chính (D20) | `cap_max` cố định 1.000.000; train trên mọi ảnh trừ ảnh số 0, không chấm điểm | Không phải train `default` trước; biết trước dung lượng file web (≤ 32 MB). Ảnh hưởng trên phòng lớn chưa đo (`docs/design/product.md` mục 2) |
| Lưu trữ, GPU | Network volume (không dùng Global volume) cùng data center với GPU; ≥ 100 GB (khoảng 110 GB nếu chạy lại cả `tune_mcmc`); RTX 4090 như lượt 31/10/2026 | `docs/ENVIRONMENT_AUDIT.md` mục 1 |
| Bản gsplat | Giữ commit `6e8c837` | v1.5.3 có lỗi khiến `DefaultStrategy` không reset opacity |
| Repo cũ trên GitHub | Chuyển sang private hoặc archive | Repo cũ đang public |

Ba lựa chọn được chốt **sau khi đã có kết quả**: dùng `mcmc` làm mô hình của sản phẩm, không xét gói `v2` như một ứng viên, và bỏ vòng ghép (D25; theo luật đã ghi, một ứng viên thì vẫn chạy). Báo cáo phải ghi rõ điều này.

## Kết quả (lượt chạy pod ngày 31/10/2026)

Bảng đầy đủ, biểu đồ và ảnh so sánh: `docs/RESULTS.md` (cùng số liệu với mục này).

Code: commit `c0a7a92`. Pod và thời gian từng việc: `docs/ENVIRONMENT_AUDIT.md` mục 7. Số liệu lấy từ các báo cáo `main_20261031_114145*.md`, `tune_mcmc_20261031_155033*.md` và `compare_custom_20261031_214710.md` trên volume.

**Thực nghiệm chính** (cùng số Gaussian, 3 seed, 30.000 vòng): `mcmc` hơn `default` ở 17/18 cặp (cảnh, seed) về PSNR, 18/18 về SSIM và LPIPS.

| Cảnh | ΔPSNR (dB) | ΔSSIM | ΔLPIPS | σ PSNR `default` / `mcmc` |
|---|---|---|---|---|
| drjohnson (`mcmc` với `opacity_reg` 0,001) | +0,480 | +0,0034 | −0,0120 | 0,017 / 0,096 |
| playroom (như trên) | +0,197 | +0,0026 | −0,0150 | 0,085 / 0,294 |
| room | +0,809 | +0,0118 | −0,0246 | 0,060 / 0,137 |
| counter | +0,544 | +0,0116 | −0,0195 | 0,193 / 0,026 |
| kitchen | +0,435 | +0,0052 | −0,0058 | 0,083 / 0,123 |
| bonsai | +0,571 | +0,0067 | −0,0158 | 0,034 / 0,082 |

- Trung bình: Deep Blending +0,338 ± 0,239 dB; Mip-NeRF 360 +0,590 ± 0,181 dB (12/12 cặp tốt hơn ở cả 3 chỉ số). Ở playroom, ΔPSNR nhỏ hơn dao động giữa các seed của `mcmc`.
- Kiểm chứng baseline:
  - `default` so với số Inria công bố trên Deep Blending: PASS (−0,168 dB);
  - so với B0 tự chạy: PASS (drjohnson −0,339 dB, playroom −0,136 dB);
  - Mip-NeRF 360: không đối chiếu với số công bố (SKIPPED, mới chạy 4/9 cảnh).
- Chi phí: thời gian train ×0,98–1,10 so với `default`; VRAM đỉnh của `mcmc` thấp hơn 0,1–0,9 GiB; PLY như nhau.
- Phòng nhóm `phong_nhom` (1 seed): `default` 39,318 / 0,9769 / 0,0959; `mcmc` 39,069 / 0,9788 / 0,0873 (PSNR / SSIM / LPIPS; trái chiều). Ảnh test là các khung kề khung train của cùng video, nên số này không so với benchmark.
- Dò tham số `tune_mcmc` (`room`, seed 42; ngưỡng d = 0,150 dB / 0,0012 / 0,0018):
  - hơn `mcmc` quá d: chỉ `rbg` (+0,156 dB);
  - trong mức dao động: `sr0001`, `depth`, `nlr1m`, `nlr250k`, `kr`, `mono`, `sr005`;
  - kém quá d: `or0005`, `or0001`, `rs30k`, `sh2`, `mo002`, `or005`, `bilagrid` (−2,476 dB), `v2` (−2,750 dB);
  - lọc Gaussian (opacity < 0,02, trục > 0,1 × `scene_scale`): `default` −1,14 dB, `mcmc` và các biến thể −2,1 đến −9,1 dB. Không dùng bộ lọc này để làm nhẹ file cho MCMC.

## Đã thực hiện

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Cài 3 env conda (`scripts/pod/run_all.sh`) | Đã chạy trên pod (31/10/2026); 08/11/2026 trên pod mới, cùng volume: kiểm tra lại 5/5 bước OK trong 3 phút 34 giây | `docs/ENVIRONMENT_AUDIT.md` mục 7 |
| Ảnh / video → khung ảnh → COLMAP → kiểm tra trước khi train (`src/input/`, `src/reconstruction/`) | Đã chạy trên pod: video phòng nhóm (31/10/2026) | `sfm_report.json` của `phong_nhom`; `tests/unit/test_frames.py`, `test_sfm.py`, `tests/test_pipeline.py` |
| Tải benchmark (`scripts/pod/fetch_data.sh`) | Đã chạy trên pod: Mip-NeRF 360, 4 cảnh trong nhà (SHA-256 ghi lại) | `outputs/reports/fetch_*.json`; `tests/unit/test_fetch.py` |
| Thực nghiệm chính, B0, bảng và biểu đồ (D12, D17, D18) | Đã chạy trên pod, mọi bước OK; baseline đã kiểm chứng | Mục "Kết quả"; `tests/unit/test_experiment.py`, `test_benchmarks.py`, `test_compare_modes.py`, `test_charts.py` |
| Dò tham số, lọc Gaussian (D14–D16) | Đã chạy trên pod, mọi bước OK | Mục "Kết quả"; `tests/unit/test_trainer_hooks.py`, `test_mono_depth.py`, `test_prune.py` |
| Render lại từ checkpoint, xuất PLY (`src/rendering/render.py`) | Đã kiểm thử với runner giả; **chưa xác nhận trên pod** | `tests/unit/test_render.py` |
| Ảnh so sánh, video xếp chồng (`scripts/run_panels.sh`) | Đã chạy trên pod; `VIDEO=1` bị dừng vì hết RAM ở 2/3 cảnh | Mục "Còn mở" |
| Notebook lấy code qua HTTPS, kiểm tra `chmod` (D19) | Đã viết, đã mô phỏng; **ô 1–3 bản mới chưa chạy trên pod** (lượt 31/10 clone bằng tay) | `tests/unit/test_notebooks.py` |
| Luồng sản phẩm (D20): `configs/product.yaml`, `src/product/`, `scripts/build_room.sh` | Đã viết; đã kiểm thử bằng stub và gsplat giả; đã đối chiếu trên CPU với code thật của gsplat @6e8c837; **đã chạy trên pod 08/11/2026** (RTX 4090): phòng `video10` từ video 2:54, 249 ảnh định vị, 637.031 Gaussian, `model.splat` 20,4 MB, train 8 phút 23 giây | `tests/unit/test_product.py`, `test_product_export.py`, `test_build_room.py`; `docs/design/product.md` mục 7 |
| Trang web (D22, D23): `app/web/`, `scripts/web_server.sh` | Đã viết; đã kiểm thử trên CPU: API qua HTTP thật, tải lên → `build_room.sh` thật → thư viện; Chromium headless (trọn luồng; camera khớp mô hình theo điểm mốc của phòng tổng hợp); đã rà soát bảo mật độc lập, sửa các lỗi tìm được (mỗi lỗi có test; `docs/design/product.md` mục 8). Giao diện làm lại (D23): đã soát bằng ảnh chụp màn hình ở 1440 × 900 và 390 × 664 với dữ liệu tổng hợp. **đã chạy trên pod 08/11/2026** (sau khi thêm cổng HTTP 8000 bằng Edit Pod): tải video lên, dựng xong `video10`, xem được 3D (ảnh: `docs/results/web/tong_quan.jpg`) | `tests/unit/test_web_server.py`, `test_web_uploads.py`, `tests/browser/` (`SR3D_BROWSER=1`); `docs/design/product.md` mục 8 |
| Đánh giá phòng tự quay theo đoạn (`python -m src.training.split segments`, `--exclude-list` của `src.training.run`, `scripts/pod/run_segments.sh`): 6 đoạn liền của video làm test, bỏ 3 khung mỗi bên, `default` / `mcmc` × 3 seed; đọc được COLMAP của phòng sản phẩm (`SRC_ROOT`), thêm biến thể của `mcmc` trên cùng tập chia (`VARIANTS`, vd. `mono`, `kr`, `rbg`) | Đã viết, đã kiểm thử trên CPU (DRY_RUN; chạy với stub thay train / lập bảng); **chưa chạy trên pod** | `tests/unit/test_train_split.py`, `test_train_run.py`, `test_run_segments.py`; `docs/USAGE_GUIDE.md` mục 3.6 |
| Toàn bộ test trên CPU | 599 test đạt, 11 bỏ qua vì cần PyTorch hoặc `SR3D_BROWSER=1` (10/11/2026, Python 3.10) | `python -m pytest` |

## Còn mở

- Luồng sản phẩm đã chạy trên pod (08/11/2026, `video10`) nhưng mới có thời gian train (8 phút 23 giây); thời gian các bước còn lại và ảnh hưởng của `cap_max` 1 triệu chưa đo. `model.splat` của `video10` có 637.031 Gaussian, thấp hơn `cap_max`. Giả thuyết chưa kiểm: train đủ 1.000.000, bước xuất bỏ các Gaussian trong suốt hoàn toàn (alpha về 0 khi lượng tử hóa 8 bit, `src/product/export_web.py`). Kiểm bằng `num_gaussians_trained` trong `outputs/product/web/manifest.json` hoặc dòng `[export_web]` trong `outputs/logs/build_room_video10_*.log`.
- Trang web đã chạy trên pod, tải video lên qua proxy của Runpod được; thời gian tải trang và FPS trên máy thật chưa đo. Chưa có trang Kết quả thực nghiệm, chấm điểm bằng lượt quay kiểm tra, tự dừng pod.
- Phòng tự dựng chưa có điểm đánh giá: cần một đoạn quay kiểm tra định vị chung mô hình COLMAP với video chính (`scripts/pipeline/prepare_scene.sh`, chưa có).
- Lượt 31/10/2026: chưa xác minh `repo_dirty` trong `run_meta.json` (notebook từng được mở thẳng trong repo trên pod).
- `run_panels.sh` với `VIDEO=1` bị `Killed` (hết RAM) ở `phong_nhom` và `mipnerf360 room`; link tải `/files/workspace/...` của Jupyter chưa xác nhận chạy được.
- `bilagrid`, `v2`: PSNR giảm mạnh nhưng LPIPS gần như không đổi. Giả thuyết chưa kiểm: lệch màu vì ảnh test chấm không qua bilateral grid.
- Mip-NeRF 360 mới chạy 4/9 cảnh, nên chưa so được với trung bình công bố. Deep Blending dùng `opacity_reg` 0,001 cho cả hai cảnh (paper chỉ ghi cho `drjohnson`); báo cáo phải nêu rõ.
- Ảnh chụp: các ngưỡng (≥ 10 ảnh, nên ≥ 30) là điểm xuất phát. Ảnh không có EXIF tiêu cự cho kết quả COLMAP kém ổn định. COLMAP có thể ghép nhầm hai nơi có họa tiết giống nhau mà vẫn PASS.
- `configs/environment/*.conf` vẫn để `provisional`; `AUTO_STOP=1` chưa thử trên pod.
- Câu trả lời của GVHD cho Q1–Q3, Q5–Q12 (`docs/PROJECT_PLAN.md` mục 2); chưa công bố việc dùng AI theo quy định của trường.

## Môi trường

Phiên bản ghim, cách cài, sự cố đã biết, cấu hình pod đã chạy: `docs/ENVIRONMENT_AUDIT.md`. Lệnh chính: `README.md` mục 2; từng bước và notebook: `docs/USAGE_GUIDE.md`.

## Dữ liệu, checkpoint, kết quả

Tất cả nằm trên network volume tại `/workspace`, không đưa lên Git; bố cục: `docs/design/architecture.md` mục 2.
- Dữ liệu gốc: `data/benchmarks/{deep_blending,mipnerf360}/`, `data/custom_scenes/<cảnh>/`.
- Kết quả thực nghiệm: `outputs/benchmarks/<benchmark>/{inria_b0,gsplat}/`, `outputs/custom_scenes/phong_nhom/`.
- Báo cáo và biểu đồ: `outputs/reports/` (`main_20261031_114145.*`, `tune_mcmc_20261031_155033.*`, `compare_custom_20261031_214710.md`).
- Gói kết quả: `ket_qua_main_20261031_114145.tar.gz` (123 MB), `ket_qua_tune_mcmc_20261031_155033.tar.gz` (244 MB).
- Sản phẩm: `outputs/product/<phòng>/` và thư mục web `outputs/product/web/`; video / ảnh tải lên từ trang web: `data/uploads/<phòng>/`; mật khẩu của server: `outputs/product/server/token`. Phòng đã dựng: `video10` (08/11/2026).
- Ảnh và biểu đồ kết quả trong repo (chép từ volume, khoảng 4 MB): `docs/results/` (`docs/RESULTS.md`).

## Việc tiếp theo

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Đo nốt `video10` trên pod: thời gian từng bước (trang Tiến độ hoặc `outputs/product/video10/progress.json`), thời gian tải và FPS (nút "Đo trên máy này" ở trang Xem 3D); kiểm số Gaussian đã train (mục "Còn mở") | Ghi đủ các số vào `docs/RESULTS.md` mục 7; trên trang Xem 3D, góc mở đầu giống ảnh đầu vào tương ứng, ba chế độ chạy được |
| 2 | Đánh giá theo đoạn trên phòng thật (`docs/USAGE_GUIDE.md` mục 3.6): `phong_nhom`, rồi `video10`, `video8`, `video9` với `SRC_ROOT=/workspace/outputs/product`; thêm biến thể `mono`, `kr`, `rbg` của `mcmc` nếu còn giờ GPU | Bảng `segments_<phòng>_*.md` (và `_vs_mcmc.md` khi có biến thể) đủ 3 seed, không lượt lỗi; số chép vào `docs/RESULTS.md`; chỉ nói "cải thiện" theo `docs/design/training_modes.md` mục 4.3 |
| 3 | Trang Kết quả thực nghiệm của bản thiết kế, số lấy từ báo cáo 31/10 | Mọi số trên trang truy được về `main_20261031_114145*.md` / `.json` |
| 4 | Viết chương kết quả của báo cáo từ mục "Kết quả" (bảng, biểu đồ `main_*_charts/`, ảnh so sánh) | Mọi số trong báo cáo truy được về báo cáo `.md` / `.json` trên volume |
| 5 | Repo cũ để private hoặc archive; nếu cần, chép báo cáo `.md` gốc từ volume vào `docs/results/` (`docs/RESULTS.md` mục 11) | Repo cũ không còn public |
