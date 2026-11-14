# PROJECT STATUS

_Cập nhật: 13/11/2026. Bản rút gọn (D21): lịch sử chi tiết (nhật ký kiểm thử, rà soát độc lập, ghi chú từng lần sửa) nằm trong lịch sử Git của file này và trong các bản bàn giao của Project trên claude.ai._

## Mục tiêu hiện tại

1. **Sản phẩm trên pod** (D20, D22): đã chạy trọn luồng và đo ngày 08/11/2026 (7 phòng; `docs/RESULTS.md` mục 7); còn kiểm số Gaussian đã train so với `cap_max` (mục "Còn mở").
2. **Hoàn thiện trang web theo bản thiết kế** (giai đoạn 5): trang Kết quả thực nghiệm; chấm điểm phòng tự dựng bằng lượt quay kiểm tra (`docs/design/product.md` mục 5).
3. **Báo cáo đồ án** (giai đoạn 6) từ kết quả thực nghiệm ngày 31/10/2026 và 12–13/11/2026 (hai mục "Kết quả").

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
| D26 | 12/11/2026 | Đánh giá trên phòng thật theo đoạn: 3 phòng (`video8`, `video9`, `video10`, đọc mô hình COLMAP của luồng sản phẩm), 3 seed, `default`, `mcmc` và 3 biến thể `rbg`, `mono`, `kr`; luật đọc kết quả ghi trước khi chạy (`docs/design/experiments.md` mục 3.8). Không so với cách chia mỗi khung thứ 8. Theo lựa chọn của bạn ("giữ nguyên ban đầu chạy thêm 3 biến thể thôi ko cần thêm nữa", "tai video len dau 3 video 8910", "t chỉ cần so sánh cho hiện tại thôi cái chia cũ bỏ đi") |

## Đề xuất chờ xác nhận

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Hai tham số của mô hình chính (D20) | `cap_max` cố định 1.000.000; train trên mọi ảnh trừ ảnh số 0, không chấm điểm | Không phải train `default` trước; biết trước dung lượng file web (≤ 32 MB). Ảnh hưởng trên phòng lớn chưa đo (`docs/design/product.md` mục 2) |
| Lưu trữ, GPU | Network volume (không dùng Global volume) cùng data center với GPU; ≥ 100 GB (khoảng 110 GB nếu chạy lại cả `tune_mcmc`); RTX 4090 như lượt 31/10/2026 | `docs/ENVIRONMENT_AUDIT.md` mục 1 |
| Bản gsplat | Giữ commit `6e8c837` | v1.5.3 có lỗi khiến `DefaultStrategy` không reset opacity |
| Repo cũ trên GitHub | Chuyển sang private hoặc archive | Repo cũ đang public |
| Thêm một lượt ở `cap_max` 1 triệu (tùy chọn) | Chạy `mcmc` trên ba phòng tự quay, cùng tập chia, 3 seed, với `cap_max` 1.000.000 như sản phẩm; ghi luật đọc kết quả trước khi chạy | Đánh giá theo đoạn dùng ngân sách của `default` (tới 3,6 triệu ở `video10`); chưa biết sản phẩm mất bao nhiêu chất lượng ở 1 triệu |

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

## Kết quả: ba phòng tự quay, test theo đoạn (lượt chạy pod ngày 12–13/11/2026)

Bảng đầy đủ và biểu đồ: `docs/RESULTS.md` mục 6.2 (ảnh so sánh chỉ có trong gói kết quả, không đưa vào repo). Thiết kế và luật ghi trước khi chạy: `docs/design/experiments.md` mục 3.8 (D26).

Code: commit `49c208b` (28 lượt) và `52afb1d` (17 lượt chạy sau khi sửa bộ kiểm NaN ở `4251ff2`); gsplat `6e8c837`; RTX 4090. Số liệu lấy từ `outputs/reports/danh_gia_theo_doan/` trên volume (bản chép: `docs/results/segment_eval/`).

- 3 phòng (`video8` phòng ngủ, `video9` phòng học, `video10` phòng ký túc xá) × 5 cấu hình × 3 seed: **45/45 lượt OK**, mọi điều kiện so sánh đạt; ảnh test là 6 đoạn liền của chính video (30 ảnh mỗi phòng).
- `mcmc` so với `default`: tốt hơn ở **9/9** cặp (phòng, seed) ở cả ba chỉ số; trung bình PSNR +2,182 dB, SSIM +0,0262, LPIPS −0,0262. Theo phòng: `video8` +0,339 dB (trong mức dao động, d_r 0,765 dB), `video9` +3,923 dB, `video10` +2,285 dB.
- Ba biến thể so với `mcmc` theo luật mục 3.8: **không biến thể nào đạt**, sản phẩm giữ `mcmc`.
  - `rbg`: trong mức dao động ở cả ba phòng;
  - `mono`: tốt hơn quá d_r chỉ ở `video10` (cả ba chỉ số; +0,549 dB), luật cần 2 phòng;
  - `kr`: kém quá d_r ở `video10` (cả ba chỉ số; −0,460 dB).
- Chi phí: `mono` train lâu hơn `mcmc` ×1,20–1,77, các biến thể khác ×0,99–1,05; 45 lượt mất 9,1 giờ train, PLY cộng checkpoint 34 GB.
- Giới hạn: ảnh test vẫn nằm trên đường đã quay; ngân sách Gaussian bằng `default` (0,76 / 0,83 / 3,63 triệu), khác `cap_max` 1 triệu của sản phẩm.

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
| Luồng sản phẩm (D20): `configs/product.yaml`, `src/product/`, `scripts/build_room.sh` | Đã viết; đã kiểm thử bằng stub và gsplat giả; đã đối chiếu trên CPU với code thật của gsplat @6e8c837; **đã chạy trên pod 08/11/2026** (RTX 4090, 7 phòng): video 2:54 → 250 khung, 249 ảnh định vị, khoảng 640 nghìn Gaussian, `model.splat` 20,5 MB; tổng 18 phút 45 giây (COLMAP 9 phút 40 giây, train 8 phút 32 giây) | `tests/unit/test_product.py`, `test_product_export.py`, `test_build_room.py`; `docs/design/product.md` mục 7 |
| Trang web (D22, D23): `app/web/`, `scripts/web_server.sh` | Đã viết; đã kiểm thử trên CPU: API qua HTTP thật, tải lên → `build_room.sh` thật → thư viện; Chromium headless (trọn luồng; camera khớp mô hình theo điểm mốc của phòng tổng hợp); đã rà soát bảo mật độc lập, sửa các lỗi tìm được (mỗi lỗi có test; `docs/design/product.md` mục 8). Giao diện làm lại (D23): đã soát bằng ảnh chụp màn hình ở 1440 × 900 và 390 × 664 với dữ liệu tổng hợp. **đã chạy trên pod 08/11/2026** (sau khi thêm cổng HTTP 8000 bằng Edit Pod): tải video lên, dựng phòng, xem 3D ở cả ba chế độ; trên laptop GPU tích hợp Intel UHD: tải 956 ms, khung đầu 1,15 giây, 10,5 FPS khi đang quay màn hình (`docs/RESULTS.md` mục 7; video quay màn hình demo: [Google Drive](https://drive.google.com/file/d/1FG4LS7gpBTIIer8L2jTSQQHPk12BszV4/view)) | `tests/unit/test_web_server.py`, `test_web_uploads.py`, `tests/browser/` (`SR3D_BROWSER=1`); `docs/design/product.md` mục 8 |
| Đánh giá phòng tự quay theo đoạn (`python -m src.training.split segments`, `--exclude-list` của `src.training.run`, `scripts/pod/run_segments.sh`): 6 đoạn liền của video làm test, bỏ 3 khung mỗi bên, `default` / `mcmc` × 3 seed; đọc được COLMAP của phòng sản phẩm (`SRC_ROOT`), thêm biến thể của `mcmc` trên cùng tập chia (`VARIANTS`, vd. `mono`, `kr`, `rbg`) | Đã kiểm thử trên CPU (DRY_RUN; chạy với stub thay train / lập bảng). **Đã chạy trên pod 12–13/11/2026: 45/45 lượt OK**; bảng từng phòng, bảng gộp (so với `default`, so với `mcmc`), luật mục 3.8 (`python -m src.evaluation.segment_rule`) và ảnh so sánh đã có; đạt tiêu chí nghiệm thu đã đặt (đủ 3 seed, không lượt lỗi, kết luận theo luật, số chép vào `docs/RESULTS.md`). Ở `video9`, 6 lượt `mono`, `kr` đầu tiên dừng ở vòng 0 vì bộ kiểm NaN coi Gaussian kích thước 0 lúc khởi tạo là lỗi; sửa ở `4251ff2` rồi chạy lại (`docs/design/experiments.md` cuối mục 3.8) | Mục "Kết quả: ba phòng tự quay"; `tests/unit/test_train_split.py`, `test_train_run.py`, `test_run_segments.py`, `test_segment_rule.py`; `docs/USAGE_GUIDE.md` mục 3.6 |
| Toàn bộ test trên CPU | 599 test đạt, 19 bỏ qua vì máy kiểm thử không có PyTorch, matplotlib hoặc `SR3D_BROWSER=1` (12/11/2026, Python 3.13, commit `52afb1d`) | `python -m pytest` |

## Còn mở

- Luồng sản phẩm: `model.splat` của mọi phòng có 430–820 nghìn Gaussian, thấp hơn `cap_max` 1.000.000. Giả thuyết chưa kiểm: train đủ 1.000.000, bước xuất bỏ các Gaussian trong suốt hoàn toàn (alpha về 0 khi lượng tử hóa 8 bit, `src/product/export_web.py`). Kiểm bằng `num_gaussians_trained` trong `outputs/product/web/manifest.json` hoặc dòng `[export_web]` trong `outputs/logs/build_room_<phòng>_*.log`. Ảnh hưởng của `cap_max` 1 triệu chưa đo: đánh giá theo đoạn chạy với ngân sách của `default` (tới 3,6 triệu ở `video10`), nên chưa trả lời câu này.
- Trang web: FPS mới đo một lần trên GPU tích hợp, trong lúc đang quay màn hình (10,5 FPS); chưa đo khi không quay màn hình, ở các mức chất lượng, trên GPU rời, điện thoại. Đi bộ ra xa đường đã quay thấy vệt nhòe, floater. Chưa có trang Kết quả thực nghiệm, chấm điểm bằng lượt quay kiểm tra, tự dừng pod.
- `video9`: 157 điểm SfM nằm trong cụm ít nhất 4 điểm trùng vị trí, nên thành Gaussian kích thước 0 lúc khởi tạo (`zero_size` trong `sr3d_hooks.jsonl`; về 0 từ vòng 1.000–1.500); `video8`, `video10` không có. Chưa kiểm `points3D` của mô hình COLMAP phòng này (`outputs/product/video9/colmap` trên volume) để biết vì sao có điểm trùng.
- `scripts/pod/job.sh status`: `last_line` cắt 4.000 byte cuối của log bằng `tail -c`; khi chỗ cắt rơi vào giữa một ký tự UTF-8 nhiều byte (chữ tiếng Việt, thanh tiến độ), `grep` (bản 3.11 trong locale UTF-8) coi là file nhị phân và in `binary file matches` thay cho dòng cuối. Chỉ ảnh hưởng phần hiển thị; sửa bằng `grep -a` (chưa sửa).
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
- Kết quả thực nghiệm: `outputs/benchmarks/<benchmark>/{inria_b0,gsplat}/`, `outputs/custom_scenes/phong_nhom/`, đánh giá theo đoạn `outputs/custom_segments/<phòng>/<cấu hình>_s<seed>/` (lượt dừng lỗi trước khi sửa giữ ở `*.prev_<thời điểm>`).
- Báo cáo và biểu đồ: `outputs/reports/` (`main_20261031_114145.*`, `tune_mcmc_20261031_155033.*`, `compare_custom_20261031_214710.md`, `danh_gia_theo_doan/`).
- Gói kết quả: `ket_qua_main_20261031_114145.tar.gz` (123 MB), `ket_qua_tune_mcmc_20261031_155033.tar.gz` (244 MB), `ket_qua_danh_gia_theo_doan_day_du_20261112_180406.tar.gz` (337 MB; gói `ket_qua_danh_gia_theo_doan_20261112_180311.tar.gz` đóng trước đó một phút giống hệt, chỉ thiếu ảnh so sánh và vài dòng log cuối).
- Sản phẩm: `outputs/product/<phòng>/` và thư mục web `outputs/product/web/`; video / ảnh tải lên từ trang web: `data/uploads/<phòng>/`; mật khẩu của server: `outputs/product/server/token`. Phòng đã dựng ngày 08/11/2026: `t`, `video2`, `video8`, `video9`, `1789028710123_…`, `video10`, `1`. Video quay màn hình demo (11 phút): [Google Drive](https://drive.google.com/file/d/1FG4LS7gpBTIIer8L2jTSQQHPk12BszV4/view), file không đưa lên Git.
- Ảnh, biểu đồ và báo cáo kết quả trong repo (chép từ volume, khoảng 4 MB): `docs/results/` (`docs/RESULTS.md`).

## Việc tiếp theo

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Kiểm số Gaussian đã train (`num_gaussians_trained` trong `outputs/product/web/manifest.json`) của `video10`, `1`; nếu có máy GPU rời, đo FPS bằng nút "Đo trên máy này" | Ghi vào `docs/RESULTS.md` mục 7; giả thuyết ở mục "Còn mở" được xác nhận hoặc bác bỏ |
| 2 | Trang Kết quả thực nghiệm của bản thiết kế, số lấy từ báo cáo 31/10 | Mọi số trên trang truy được về `main_20261031_114145*.md` / `.json` |
| 3 | Viết chương kết quả của báo cáo từ hai mục "Kết quả" (bảng, biểu đồ `main_*_charts/`, ảnh so sánh; ba phòng tự quay: `docs/RESULTS.md` mục 6.2) | Mọi số trong báo cáo truy được về báo cáo `.md` / `.json` trên volume hoặc bản chép trong `docs/results/` |
| 4 | Repo cũ để private hoặc archive; nếu cần, chép báo cáo `.md` gốc từ volume vào `docs/results/` (`docs/RESULTS.md` mục 11) | Repo cũ không còn public |
