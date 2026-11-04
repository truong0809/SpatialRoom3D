# PROJECT STATUS

_Cập nhật: 04/11/2026. Bản rút gọn (D21): lịch sử chi tiết (nhật ký kiểm thử, rà soát độc lập, ghi chú từng lần sửa) nằm trong lịch sử Git của file này và trong các bản bàn giao của Project trên claude.ai._

## Mục tiêu hiện tại

1. **Chạy luồng sản phẩm lần đầu trên pod** (D20, `docs/design/product.md`): dựng phòng nhóm bằng mô hình chính `mcmc`, mở file web trên một trình xem `.splat`.
2. **Trang web xem 3D** (giai đoạn 5): đọc `manifest.json`, `model.splat`, `cameras.json` theo hợp đồng ở `docs/design/product.md` mục 4; ba chế độ xoay quanh, đi bộ, tham quan; sau đó là tải video / ảnh lên và dựng phòng từ web.
3. **Báo cáo đồ án** (giai đoạn 6) từ kết quả thực nghiệm ngày 31/10/2026 (mục "Kết quả").
4. Tùy chọn: vòng ghép của lượt dò tham số (D15), chỉ khi được duyệt chi phí GPU.

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

## Đề xuất chờ xác nhận

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Hai tham số của mô hình chính (D20) | `cap_max` cố định 1.000.000; train trên mọi ảnh trừ ảnh số 0, không chấm điểm | Không phải train `default` trước; biết trước dung lượng file web (≤ 32 MB). Ảnh hưởng trên phòng lớn chưa đo (`docs/design/product.md` mục 2) |
| Vòng ghép (D15) | Chỉ một ứng viên đạt luật (`rbg`, +0,156 dB, sát ngưỡng): chạy cấu hình {A} với seed 43, 44 trên `room`, hoặc bỏ vòng ghép | Dự báo khó đạt bước 4 của luật. Kế hoạch đã viết (`configs/experiments/tune_mcmc_combo.yaml`, chưa đưa lên Git). Khoảng 1,5 giờ GPU |
| Lưu trữ, GPU | Network volume (không dùng Global volume) cùng data center với GPU; 170 GB nếu chạy tiếp thực nghiệm; RTX 4090 như lượt 31/10/2026 | `docs/ENVIRONMENT_AUDIT.md` mục 1 |
| Bản gsplat | Giữ commit `6e8c837` | v1.5.3 có lỗi khiến `DefaultStrategy` không reset opacity |
| Repo cũ trên GitHub | Chuyển sang private hoặc archive | Repo cũ đang public |

Hai lựa chọn được chốt **sau khi đã có kết quả**: dùng `mcmc` làm mô hình của sản phẩm, và không đưa gói `v2` vào vòng ghép. Báo cáo phải ghi rõ điều này.

## Kết quả (lượt chạy pod ngày 31/10/2026)

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
| Cài 3 env conda (`scripts/pod/run_all.sh`) | Đã chạy trên pod (31/10/2026) | `docs/ENVIRONMENT_AUDIT.md` mục 7 |
| Ảnh / video → khung ảnh → COLMAP → kiểm tra trước khi train (`src/input/`, `src/reconstruction/`) | Đã chạy trên pod: video phòng nhóm (31/10/2026) | `sfm_report.json` của `phong_nhom`; `tests/unit/test_frames.py`, `test_sfm.py`, `tests/test_pipeline.py` |
| Tải benchmark (`scripts/pod/fetch_data.sh`) | Đã chạy trên pod: Mip-NeRF 360, 4 cảnh trong nhà (SHA-256 ghi lại) | `outputs/reports/fetch_*.json`; `tests/unit/test_fetch.py` |
| Thực nghiệm chính, B0, bảng và biểu đồ (D12, D17, D18) | Đã chạy trên pod, mọi bước OK; baseline đã kiểm chứng | Mục "Kết quả"; `tests/unit/test_experiment.py`, `test_benchmarks.py`, `test_compare_modes.py`, `test_charts.py` |
| Dò tham số, lọc Gaussian (D14–D16) | Đã chạy trên pod, mọi bước OK | Mục "Kết quả"; `tests/unit/test_trainer_hooks.py`, `test_mono_depth.py`, `test_prune.py` |
| Vòng ghép (D15) | Đã viết kế hoạch; **chưa chạy**, chưa đưa lên Git | `configs/experiments/tune_mcmc_combo.yaml` (trong sandbox) |
| Render lại từ checkpoint, xuất PLY (`src/rendering/render.py`) | Đã kiểm thử với runner giả; **chưa xác nhận trên pod** | `tests/unit/test_render.py` |
| Ảnh so sánh, video xếp chồng (`scripts/run_panels.sh`) | Đã chạy trên pod; `VIDEO=1` bị dừng vì hết RAM ở 2/3 cảnh | Mục "Còn mở" |
| Notebook lấy code qua HTTPS, kiểm tra `chmod` (D19) | Đã viết, đã mô phỏng; **ô 1–3 bản mới chưa chạy trên pod** (lượt 31/10 clone bằng tay) | `tests/unit/test_notebooks.py` |
| Luồng sản phẩm (D20): `configs/product.yaml`, `src/product/`, `scripts/build_room.sh` | Đã viết; đã kiểm thử bằng stub và gsplat giả; đã đối chiếu trên CPU với code thật của gsplat @6e8c837; **chưa chạy trên GPU, chưa có trang web** | `tests/unit/test_product.py`, `test_product_export.py`, `test_build_room.py`; `docs/design/product.md` mục 7 |
| Toàn bộ test trên CPU | 518 test đạt, 9 bỏ qua vì cần PyTorch (04/11/2026, Python 3.10, bản clone sạch) | `python -m pytest` |

## Còn mở

- Luồng sản phẩm chưa chạy trên GPU; thời gian của cả luồng và ảnh hưởng của `cap_max` 1 triệu chưa đo; chưa có trang web.
- Phòng tự dựng chưa có điểm đánh giá: cần một đoạn quay kiểm tra định vị chung mô hình COLMAP với video chính (`scripts/pipeline/prepare_scene.sh`, chưa có).
- Lượt 31/10/2026: chưa xác minh `repo_dirty` trong `run_meta.json` (notebook từng được mở thẳng trong repo trên pod).
- `run_panels.sh` với `VIDEO=1` bị `Killed` (hết RAM) ở `phong_nhom` và `mipnerf360 room`; link tải `/files/workspace/...` của Jupyter chưa xác nhận chạy được.
- `bilagrid`, `v2`: PSNR giảm mạnh nhưng LPIPS gần như không đổi. Giả thuyết chưa kiểm: lệch màu vì ảnh test chấm không qua bilateral grid.
- Mip-NeRF 360 mới chạy 4/9 cảnh, nên chưa so được với trung bình công bố. Deep Blending dùng `opacity_reg` 0,001 cho cả hai cảnh (paper chỉ ghi cho `drjohnson`); báo cáo phải nêu rõ.
- Ảnh chụp: các ngưỡng (≥ 10 ảnh, nên ≥ 30) là điểm xuất phát. Ảnh không có EXIF tiêu cự cho kết quả COLMAP kém ổn định. COLMAP có thể ghép nhầm hai nơi có họa tiết giống nhau mà vẫn PASS.
- `configs/environment/*.conf` vẫn để `provisional`; `AUTO_STOP=1` chưa thử trên pod.
- Câu trả lời của GVHD cho Q1–Q3, Q5–Q12 (`docs/PROJECT_PLAN.md` mục 2); chưa chọn LICENSE; chưa công bố việc dùng AI theo quy định của trường.

## Môi trường

Phiên bản ghim, cách cài, sự cố đã biết, cấu hình pod đã chạy: `docs/ENVIRONMENT_AUDIT.md`. Lệnh chính: `README.md` mục "Chạy trên Runpod"; từng bước và notebook: `docs/USAGE_GUIDE.md`.

## Dữ liệu, checkpoint, kết quả

Tất cả nằm trên network volume tại `/workspace`, không đưa lên Git; bố cục: `docs/design/architecture.md` mục 2.
- Dữ liệu gốc: `data/benchmarks/{deep_blending,mipnerf360}/`, `data/custom_scenes/<cảnh>/`.
- Kết quả thực nghiệm: `outputs/benchmarks/<benchmark>/{inria_b0,gsplat}/`, `outputs/custom_scenes/phong_nhom/`.
- Báo cáo và biểu đồ: `outputs/reports/` (`main_20261031_114145.*`, `tune_mcmc_20261031_155033.*`, `compare_custom_20261031_214710.md`).
- Gói kết quả: `ket_qua_main_20261031_114145.tar.gz` (123 MB), `ket_qua_tune_mcmc_20261031_155033.tar.gz` (244 MB).
- Sản phẩm: `outputs/product/<phòng>/` và thư mục web `outputs/product/web/`. Chưa có phòng nào.

## Việc tiếp theo

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Trên pod: lấy code mới (ô 3 hoặc `git pull`), rồi `bash scripts/build_room.sh /workspace/data/custom_scenes/room/video2.mp4 phong_nhom --background` | `manifest.json` có `phong_nhom`; `model.splat` khoảng 32 MB; `cameras.json` có đủ khung; mở `model.splat` trên một trình xem `.splat`, góc `default_view` giống ảnh đầu vào tương ứng; ghi thời gian từng bước |
| 2 | Trang web xem 3D theo `docs/design/product.md` mục 4 | Tải được `manifest.json` và mô hình phòng nhóm; ba chế độ xem chạy được; đo thời gian tải |
| 3 | Quyết định vòng ghép: chạy hoặc bỏ (mục "Đề xuất chờ xác nhận") | Ghi lại quyết định và lý do. Nếu chạy: `tune_mcmc_combo` exit 0, kết luận theo bước 4 của `docs/design/experiments.md` mục 3.6 |
| 4 | Viết chương kết quả của báo cáo từ mục "Kết quả" (bảng, biểu đồ `main_*_charts/`, ảnh so sánh) | Mọi số trong báo cáo truy được về báo cáo `.md` / `.json` trên volume |
| 5 | Chọn LICENSE; repo cũ để private hoặc archive | Có file LICENSE |
