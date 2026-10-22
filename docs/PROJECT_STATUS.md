# PROJECT STATUS

_Cập nhật: 21/10/2026_

## Mục tiêu hiện tại

1. Dựng **pod mới gắn network volume mới (trống)** bằng repo này (`docs/USAGE_GUIDE.md` mục 1–2):
   - khóa deploy chỉ đọc, clone code vào `/workspace/SpatialRoom3D`;
   - cài 3 env conda (`scripts/pod/run_all.sh`: chỉ cài môi trường, không train).
2. **Dò tham số `mcmc`** (D14, `docs/design/experiments.md` mục 3): ô 7 với `WHAT = "experiment"`, `EXP = "tune_mcmc"`.
   - Một việc nền trên `room` (Mip-NeRF 360), seed 42, 30.000 vòng: `default`, `mcmc` và 10 biến thể của `mcmc` (mỗi biến thể đổi một tham số); kiểm tra điểm SfM trước khi train; lọc Gaussian rồi chấm lại sau khi train.
   - Ước lượng: khoảng 5 giờ GPU trên RTX 4090, thêm khoảng 24 GB trên volume. **Chỉ chạy khi đã duyệt chi phí.**
   - Từ kết quả: định nghĩa chế độ `mcmc_indoor` trong `configs/base.yaml` (**chưa có**), chạy xác nhận một lượt trên `room`.
3. **Thực nghiệm chính** (D12, `docs/design/experiments.md` mục 2): ô 7 với `EXP = "main"`; thêm `mcmc_indoor` vào kế hoạch sau bước 2.
   - Một việc nền: tự tải Deep Blending và 4 cảnh trong nhà của Mip-NeRF 360, B0 trên Deep Blending, rồi `default` / `mcmc` × seed 42, 43, 44 trên 6 cảnh, rồi bảng so sánh.
   - Ước lượng với hai chế độ: khoảng 12 giờ GPU trên RTX 4090 và thêm khoảng 56 GB trên volume (volume 100 GB đủ theo ước lượng). Chạy cả bước 2 trên cùng volume rồi thêm `mcmc_indoor`: nên 150 GB. **Chỉ chạy khi đã duyệt chi phí.**
4. Cảnh tự chụp, video của nhóm (thư mục ảnh hoặc video): bạn tự đưa lên (`docs/USAGE_GUIDE.md` mục 2.1), chạy qua ô 8 của notebook, sau thực nghiệm chính.

**Repo này:** bắt đầu ngày 19/10/2026 (D7–D12, D14). **Chưa chạy trên pod nào.** Mọi số liệu GPU dưới đây là của repo trước (`3D-Indoor-Scene-Reconstruction-from-Images`), chạy ngày 17/10/2026 trên volume cũ.

## Quy ước trạng thái

`Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`.
Chỉ mức cuối cùng mới được tính là hoàn thành.

## Quyết định đã chốt

| # | Ngày | Quyết định | Ghi chú |
|---|---|---|---|
| D1 | 09/10/2026 | Phương pháp trọng tâm là 3DGS; baseline là 3DGS gốc (Kerbl et al., 2023) | Bản cài đặt (Inria chính chủ hay gsplat) chọn sau khi so sánh số liệu |
| D2 | 09/10/2026 | Hạ tầng tính toán: Runpod | Cấu hình cụ thể xem mục dưới |
| D3 | 09/10/2026 | Chạy kiểm chứng trên Runpod gộp vào cuối, sau khi các bước không cần GPU đã xong | Mọi script phải có kiểm thử CPU trước đó |
| D4 | 09/10/2026 | Dùng `src/reconstruction/colmap_runner.py` thay cho `convert.py` của Inria | `convert.py` dùng tên tùy chọn GPU cũ, không chạy với COLMAP 4.x (`docs/ENVIRONMENT_AUDIT.md` mục 4.1) |
| D5 | 09/10/2026 | Pipeline hỗ trợ hai chế độ huấn luyện trên gsplat: 3DGS với Default Strategy làm baseline; 3DGS với MCMC Strategy dựa trên notebook demo của nhóm | Theo yêu cầu. B0 (Inria) giữ làm mốc đối chiếu cho Default. Thiết kế: `docs/design/training_modes.md` |
| D6 | 18/10/2026 | Đầu vào: video, hoặc thư mục nhiều ảnh JPG/JPEG/PNG chụp cùng một căn phòng. Ảnh dùng thẳng, không chọn frame; chung bước COLMAP và huấn luyện với video. Một ảnh luôn bị từ chối, không có chế độ dựng từ một ảnh | Theo yêu cầu. Không đổi thuật toán huấn luyện, chế độ, baseline hay phiên bản thư viện |
| D7 | 19/10/2026 | Cấu trúc repo theo sơ đồ "SpatialRoom3D_v2": `app/{cli,web,notebooks}`, package `src/{input,reconstruction,training,rendering,evaluation,datasets,utils}`, `configs/{base,custom_scene,mipnerf360,deep_blending}.yaml`, bốn lệnh chính `scripts/{prepare_data,run_train,run_render,run_evaluate}.sh` (phần thực thi trong `scripts/{pipeline,pod,setup,lib}/`), `tests/{test_dataset,test_pipeline}.py` + `tests/unit/`, `README.md` (giới thiệu) và `docs/{README,ENVIRONMENT_AUDIT,USAGE_GUIDE,PROJECT_PLAN}.md`, `experiments/{deep_blending,mipnerf360}/` (chỉ README). Code dùng chung, mỗi luồng dữ liệu một file cấu hình và thư mục riêng | Theo sơ đồ và các lựa chọn của bạn ngày 19/10/2026 ("đủ như ảnh, giữ phần cần chạy", "src/ là package", "YAML như ảnh"). Phần đã chạy trên pod ngày 17/10/2026 được dời chỗ, không viết lại; thuật toán, chế độ, protocol, phiên bản không đổi. Phần viết sau ngày đó chưa chạy trên pod (`docs/design/architecture.md` mục 7) |
| D8 | 19/10/2026 | Bố cục `/workspace`: code ở `SpatialRoom3D/`; `data/` chỉ chứa dữ liệu gốc (`custom_scenes/<cảnh>/`, `benchmarks/{deep_blending,mipnerf360}/`); `outputs/` chứa mọi thứ sinh ra (`custom_scenes/<cảnh>/{colmap,<chế_độ>_s<seed>}/`, `benchmarks/<benchmark>/{inria_b0,gsplat,gsplat_quick}/`, `checkpoints/`, `logs/`, `reports/`, `jobs/`) | Theo sơ đồ của bạn. `docs/design/architecture.md` mục 2 |
| D9 | 19/10/2026 | Repo GitHub mới `truong0809/SpatialRoom3D` (private), lịch sử bắt đầu bằng một commit; repo cũ giữ làm lưu trữ, không sửa. Pod mới + volume mới: cài lại từ đầu, dữ liệu benchmark tải về pod (D11), chạy lại B0 + gsplat bằng code mới khi được duyệt GPU. Bỏ phần dời dữ liệu của volume cũ và mọi phần tương thích bố cục cũ | Theo ba lựa chọn của bạn ngày 19/10/2026 |
| D10 | 19/10/2026 | `scripts/pod/run_all.sh` (ô 4) chỉ cài / kiểm tra môi trường: không train, không tải dữ liệu. Chạy với dữ liệu nào do người dùng chọn (ô 7, ô 8). Lệnh train không tự tải: thiếu dữ liệu thì dừng trước khi train và báo chỗ cần đặt. Không luồng dữ liệu nào bắt buộc | Theo yêu cầu của bạn ngày 19/10/2026 ("base cài set up môi trường thôi, còn chạy với data nào để t tự quyết"). Ô 6 chỉ kiểm tra bố cục và số ảnh |
| D11 | 19/10/2026 | Dữ liệu benchmark tải thẳng về pod bằng một bước riêng, chỉ khi người dùng gọi: `scripts/pod/fetch_data.sh` (ô 7 `WHAT = "fetch"`, hoặc thực nghiệm chính). Nguồn trong mục `download` của `configs/<benchmark>.yaml`; kiểm SHA-256 khi có; chỉ tải cảnh còn thiếu, chỉ giải nén thư mục cần dùng, xóa file zip sau đó; không sửa dữ liệu đã có. Ảnh / video tự chụp vẫn do người dùng tự đưa lên | Theo yêu cầu của bạn ngày 19/10/2026 ("chuẩn bị để setup runpod tải data về vì ổ trên runpod t có 100gb") |
| D12 | 19/10/2026 | Thực nghiệm chính: `default` / `mcmc` (cùng số Gaussian) trên Deep Blending (2 cảnh, kèm B0) và 4 cảnh trong nhà của Mip-NeRF 360; seed 42, 43, 44; 30.000 vòng, không chạy thử; một việc nền (`configs/experiments/main.yaml`, `scripts/pod/run_experiment.sh`); giữ mọi file kết quả của mọi seed | Theo các lựa chọn của bạn ngày 19/10/2026: "Deep Blending + 4 cảnh M360", "3 seed", "chạy đầy đủ ko chạy test", "giữ lại hết" (seed 43, 44) |
| D14 | 21/10/2026 | Chế độ cải tiến dựa trên `mcmc`, chọn trên dữ liệu mở trước, thử trên video của nhóm sau: (1) dò tham số `mcmc` trên `room`, một seed, mỗi biến thể đổi một tham số (`opacity_reg`, `scale_reg`, `strategy.noise_lr`, `random_bkgd`, `depth_loss`; `configs/experiments/tune_mcmc.yaml`); (2) ghép các tham số đáng giữ thành `mcmc_indoor`, xác nhận một lượt trên `room`; (3) so `default` / `mcmc` / `mcmc_indoor` trong thực nghiệm chính (D12). Kèm: kiểm tra điểm SfM trước khi train với `depth_loss`; lọc Gaussian sau khi train rồi chấm lại, báo cáo cạnh chỉ số gốc. `room` là cảnh dùng để chọn tham số: kết luận về `mcmc_indoor` dựa trên 5 cảnh còn lại và video của nhóm | Theo lựa chọn của bạn ("Duyệt", phương án "mô hình hoàn chỉnh": chạy bằng dữ liệu mở trước, rồi chạy thử trên video của nhóm). Tinh chỉnh tham số và hậu xử lý trên phương pháp có sẵn, không phải thuật toán mới (`docs/design/experiments.md` mục 3.1, 4) |

## Đề xuất chờ xác nhận

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Loại cloud | Secure Cloud | Network volume chỉ dùng được với pod trên Secure Cloud |
| Lưu trữ | Network volume ≥ 100 GB, cùng data center với GPU; thực nghiệm chính (D12) giữ mọi file: khoảng 56 GB cộng môi trường; chạy cả D14 (dò tham số, rồi `mcmc_indoor` trong thực nghiệm chính): nên 150 GB | Dữ liệu tồn tại độc lập với pod. Volume mới: chọn data center có sẵn loại GPU ở dòng dưới. Volume hiện có: 100 GB, EU-RO-1; tăng được, không giảm được |
| GPU | RTX A6000 48 GB (thay thế: A40 48 GB, RTX 4090 24 GB) | Xem tiêu chí bên dưới. RTX 4090 đã chạy được cả luồng hai lần ngày 17/10/2026 (repo trước) |
| Bản gsplat | Commit `6e8c837` (v1.5.3 + 4 commit), không dùng v1.3.0 của demo hay bản phát hành v1.5.3 | v1.5.3 có lỗi khiến `DefaultStrategy` không reset opacity; v1.3.0 reset cả ở bước 0 (`docs/research/demo_review.md`, mục 3.2) |
| So sánh MCMC với Default | Cùng protocol; `cap_max` của MCMC bằng số Gaussian cuối của Default trên cùng cảnh; ≥ 3 seed trước khi kết luận | Giao thức của paper 3DGS-MCMC; chất lượng MCMC phụ thuộc trực tiếp vào `cap_max` |
| Protocol Mip-NeRF 360 | Theo benchmark chính chủ của mỗi bên: Inria `-i images_2` (4 cảnh trong nhà) / `-i images_4` (5 cảnh ngoài trời); gsplat `data_factor` 2 / 4 | Đã đối chiếu `full_eval.py` @54c035f và `examples/benchmarks/basic.sh` @6e8c837 (`experiments/mipnerf360/README.md`) |
| Repo cũ trên GitHub | Chuyển sang private hoặc archive | Kiểm kê ngày 18/10/2026 thấy repo cũ đang **public** |

**Tiêu chí chọn GPU**

- Bắt buộc:
  - VRAM ≥ 24 GB (mức Inria khuyến nghị để huấn luyện đạt chất lượng như paper).
  - Compute capability 8.0 / 8.6 / 8.9 / 9.0 (Ampere, Ada, Hopper): extension được build sẵn cho đúng các kiến trúc này. Không dùng Blackwell (RTX 50xx...).
  - Có trên Secure Cloud, cùng data center với network volume.
- Ưu tiên:
  - Giá/giờ thấp nhất trong số GPU đạt các tiêu chí bắt buộc.
  - Cùng dòng với máy tham chiếu của Inria (A6000) để so sánh được thời gian huấn luyện.

## Đã thực hiện

### Repo này (19–21/10/2026, trong sandbox, không GPU)

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Cấu trúc repo (D7), cấu hình YAML, bốn lệnh chính, bố cục `/workspace` (D8) | Đã viết; đã kiểm thử trên CPU | Mục "Kiểm thử trong sandbox" bên dưới |
| Lệnh huấn luyện sinh ra cho Deep Blending | Đã kiểm thử bằng stub | Giống lệnh repo trước đã chạy trên pod ngày 17/10/2026, chỉ khác đường dẫn (`tests/unit/test_benchmarks.py::test_deep_blending_gsplat_commands_are_unchanged`) |
| Cấu hình YAML so với các file JSON cũ | Đã kiểm tra | So từng khóa `configs/*.yaml` với `configs/{datasets,train_modes}.json` và `experiments/*/benchmark.json` của repo trước: `train_modes` giống hệt; dữ liệu, kết quả, protocol, số liệu công bố, ngưỡng giống hệt. Chỉ khác mô tả, lệnh, trạng thái; bỏ `previous_locations` (chỗ cũ, dùng cho việc dời) và khóa đường dẫn `benchmark`; `dataset`, `scenes` của phần benchmark nay lấy từ chính file cấu hình khi xuất JSON |
| Render lại từ checkpoint, xuất PLY: `src/rendering/render.py`, `scripts/run_render.sh` | Đã viết; đã kiểm thử với runner giả và stub; **chưa chạy với gsplat thật** | `tests/unit/test_render.py`: dùng lại lệnh đánh giá đã ghi trong `run_meta.json` (chế độ `--ckpt` của `simple_trainer.py`), cùng seed và tập test, ghi vào thư mục mới; từ chối lượt chưa xong; PLY của bước cuối, liên kết cứng, không ghi đè |
| Giao diện dòng lệnh `python -m app.cli` | Đã viết; đã kiểm thử | `tests/unit/test_cli.py` |
| Notebook `app/notebooks/runpod.ipynb` cho pod mới, volume trống | Đã viết; đã mô phỏng; **chưa chạy trên pod** | Mục "Kiểm thử trong sandbox" |
| Luồng cảnh tự chụp với COLMAP thật trên cây này | Đã chạy với COLMAP 3.9.1 bản CPU trên dữ liệu tổng hợp, huấn luyện thay bằng stub | Mục "Kiểm thử trong sandbox" |
| Rà soát độc lập (19/10/2026 và 21/10/2026, mỗi lần hai lượt, người rà soát không tham gia viết code) | Đã làm; lỗi chặn và lỗi vừa đã sửa, kèm test | Mục "Rà soát độc lập ngày 21/10/2026" và "... ngày 19/10/2026" bên dưới |
| Tải dữ liệu benchmark (D11): `src/datasets/fetch.py`, `scripts/pod/fetch_data.sh` | Đã viết; đã kiểm thử với curl giả và file zip tổng hợp; **chưa tải file thật** (sandbox không ra được Internet) | `tests/unit/test_fetch.py`: chỉ cảnh còn thiếu, chỉ thư mục `keep`; SHA-256 sai hoặc zip hỏng thì xóa file và dừng; đường dẫn không an toàn, thiếu thư mục thì dừng trước khi giải nén; bỏ file ẩn / symlink; không đụng thư mục cảnh đã có; cảnh đã vào chỗ có bản ghi ngay cả khi cảnh sau lỗi, báo cáo được ghi cả khi lỗi; curl ngắt khi kết nối treo |
| Thực nghiệm chính (D12): `configs/experiments/main.yaml`, `src/training/experiment.py`, `scripts/pod/run_experiment.sh`; nhiều chế độ trong `train_gsplat_modes.sh` (`MODES`); bảng theo tập cảnh con và theo mốc khác `default` | Đã viết; đã kiểm thử bằng stub; đã mô phỏng qua notebook; **chưa chạy trên pod** | `tests/unit/test_experiment.py` (thứ tự các bước, xen kẽ seed, biến môi trường của người dùng không đổi được kế hoạch, chỗ trống trước khi chạy và trước mỗi seed kể cả khi chạy lại sau khi dừng giữa chừng, tải lỗi chỉ bỏ benchmark đó, chạy nền một việc, kế hoạch có thêm chế độ tự định nghĩa), `tests/unit/test_benchmarks.py`, `tests/unit/test_compare_modes.py`; rà soát độc lập lần 2 (mục bên dưới) |
| Dò tham số và lọc Gaussian (D14): biến thể tham số trong kế hoạch (`variants`; `VARIANTS` của `train_gsplat_modes.sh`; `src.training.run --tag`), kiểm tra điểm SfM (`src/training/depth_check.py`, bước `depth_check_<benchmark>`), lọc Gaussian rồi chấm lại (`src/training/prune.py`, `scripts/run_filter.sh`, khóa `post_filter`, bảng trước / sau trong `compare_modes`, dòng `+ đã lọc` của `results.sh`), `configs/experiments/tune_mcmc.yaml`; notebook ô 7 `EXP`, ô 9 `LOC` | Đã viết; đã kiểm thử bằng stub và runner giả; đọc / ghi checkpoint bằng PyTorch 2.7.1 (CPU) và ghi PLY bằng `export_splats` thật của gsplat @6e8c837 đã chạy trên CPU; **chưa chạy trên GPU, chưa có số liệu** | `tests/unit/test_experiment.py` (kế hoạch `tune_mcmc`, biến thể và ngưỡng lọc sai bị từ chối, thứ tự các bước, kiểm tra điểm SfM lỗi chỉ bỏ benchmark đó, lọc lỗi không dừng train), `test_prune.py`, `test_depth_check.py`, `test_benchmarks.py`, `test_compare_modes.py`, `test_gsplat_results.py`, `test_overview.py`, `test_train_run.py`, `test_cli.py` |

### Lịch sử: repo trước (đã chạy trên pod ngày 17/10/2026)

Các lượt dưới đây chạy bằng repo trước, trên volume cũ, với bố cục và tên file cũ (đối chiếu tên: `docs/design/architecture.md` mục 7). Code tương ứng đã được dời vào repo này, không viết lại.

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Kế hoạch Giai đoạn 0 | Đã viết, đã được duyệt | `docs/PROJECT_PLAN.md` |
| Danh sách tài liệu khảo sát ban đầu | Đã viết | `docs/research/reading_list.md` (venue/link chưa kiểm chứng hết) |
| Kiểm tra máy (`scripts/setup/env_probe.sh`) | Đã chạy trên Runpod (17/10/2026) | SUMMARY 6/6 PASS: RTX 4090 24 GB, driver CUDA 12.8, `/workspace` là network volume, mọi host truy cập được (`docs/ENVIRONMENT_AUDIT.md` mục 7) |
| Cài môi trường 3DGS gốc (`scripts/setup/setup_3dgs_inria.sh`, `configs/environment/gs-inria.conf`) | Đã chạy trên Runpod (17/10/2026) | 9 bước xong trong 29 phút; smoke test PASSED (simple-knn, fused-ssim, rasterizer forward/backward) với torch 2.7.1+cu128, nvcc 12.4, gcc 11.4. Phiên bản trong file conf vẫn để `provisional` |
| Env `gs-tools` (COLMAP 4.2.1, ffmpeg) | Đã chạy trên Runpod (17/10/2026, lượt thứ hai) | COLMAP 4.2.1 bản CUDA và ffmpeg 9.0.2 với driver CUDA 13.0; lượt đầu (driver 12.8) không cài được bản CUDA, nay tự chọn bản CPU |
| Env `gs-gsplat` (`scripts/setup/setup_gsplat.sh`) | Đã chạy trên Runpod (17/10/2026), smoke test PASSED | Lượt đầu FAIL vì Jupyter đặt `MPLBACKEND=module://matplotlib_inline...`; đã sửa (`MPLBACKEND=Agg`). Lượt thứ hai: `SMOKE TEST PASSED` |
| Quy trình quay video và chụp ảnh | Đã viết | `docs/data/capture_protocol.md` (mục 8: chụp ảnh), sẽ hiệu chỉnh sau lần thu thử |
| Bước 4: tái lập Deep Blending (B0) | **Đã xác nhận đạt** (17/10/2026, RTX 4090) | Trung bình 2 cảnh ở step 30.000: PSNR 29,723 / SSIM 0,9066 / LPIPS 0,2384, so với Inria công bố 29,690 / 0,906 / 0,238 (lệch +0,033 dB). Chi tiết: `docs/ENVIRONMENT_AUDIT.md` mục 7 |
| Bước 4a, 4b: gsplat `default` / `mcmc` trên Deep Blending (seed 42) | Đã chạy trên Runpod (17/10/2026); `default` đạt điều kiện làm baseline; **chưa đủ seed để kết luận về `mcmc`** | `default` so với số liệu Inria công bố: PASS (trung bình ΔPSNR −0,192 dB); so với B0 tự chạy, từng cảnh: PASS (ΔPSNR −0,209 / −0,241 dB). `mcmc` (`opacity_reg` 0,001, cùng số Gaussian với `default`): ΔPSNR +0,070 / +0,003 dB, ΔLPIPS −0,010 / −0,015 so với `default` ở drjohnson / playroom; một seed nên bảng ghi `seeds=FEW`, không gọi là cải thiện |
| Chạy nền và khóa GPU (`run_all.sh`, `job.sh`) | Đã chạy trên Runpod (17/10/2026) | `run_all` báo OK cả 7 bước trong khoảng 2 giờ; trạng thái việc đi đúng SUBMITTED → QUEUED → RUNNING → DONE, exit 0. `AUTO_STOP=1` chưa thử |
| Chuẩn bị cảnh tự chụp (ảnh / video → COLMAP) trên pod | **Chưa có lượt chạy thành công trên pod** | Các lượt `video8` của gói kết quả 17/10/2026 dừng trước khi cài `gs-tools` |
| Kiểm tra notebook demo 3DGS-MCMC và repo SpatialScene3D | Đã viết | `docs/research/demo_review.md`; notebook gốc lưu ở `docs/research/references/demo_3dgs_mcmc/` |
| Hai chế độ huấn luyện (D5), đầu vào ảnh và kiểm tra trước khi train (D6), cấu hình Mip-NeRF 360 | Đã viết; đã kiểm thử trên CPU; đã qua các lượt rà soát độc lập ngày 17–18/10/2026 | `docs/design/training_modes.md` mục 8, `docs/USAGE_GUIDE.md` mục 3, `experiments/mipnerf360/README.md` |

### Kiểm thử trong sandbox (không GPU)

**Repo này, 21/10/2026 (D14: biến thể tham số, kiểm tra điểm SfM, lọc Gaussian):**
- **pytest:** 351 test; 350 qua, 1 bỏ qua (cần PyTorch) trên bốn môi trường như lượt 19/10/2026 (Python 3.13.16 / 3.10.20; numpy 2.5.4 / 2.2.6 / 1.26.4; opencv 4.11 / 5.0). Test bị bỏ qua (đọc / ghi checkpoint bằng PyTorch thật) qua trên Python 3.10.20 + PyTorch 2.7.1 bản CPU, cùng các test đọc kết quả đã lọc (`test_prune.py`, `test_gsplat_results.py`, `test_compare_modes.py`, `test_overview.py`: 50 test).
- **Ghi PLY bằng `export_splats` thật** của gsplat @6e8c837 (kiểm tra tay trong sandbox, không nằm trong bộ test: CPU, 1.000 Gaussian tổng hợp, qua `src.training.prune.TorchIO`): đọc lại đúng vị trí, opacity, hệ số SH theo bố cục PLY của Inria, đúng số Gaussian còn lại. Người rà soát độc lập dựng lại riêng: PLY giống từng byte với PLY do `export_splats` ghi cho cùng tập Gaussian.
- **shellcheck**, **pyflakes**, liên kết tương đối trong mọi file `.md`: sạch.
- **Notebook**, cùng cách mô phỏng như dưới: **64/64 kiểm tra đạt**. Thêm so với lượt 19/10/2026: ô 7 với `EXP = "tune_mcmc"` và `CHAY = False` in đúng kế hoạch (10 biến thể, `room` còn thiếu, bước `depth_check_mipnerf360` và `filter_mipnerf360_s42`, ước lượng), không tạo việc, không tải; `CHAY = True` thành đúng một việc nền `exp_tune_mcmc`, dừng ở `env_check`, không tải, không train; ô 9 với `LOC = True` xếp hàng việc `filter_<cảnh>_<lượt>`, dừng sớm vì không có GPU, lượt gốc không đổi, không tạo `filtered/`; lượt không có thì báo lỗi rõ; khi lượt có `filtered/`, ô 9 in dòng `+ đã lọc` ngay dưới lượt đó kèm PLY đã lọc.

**Repo này, 19/10/2026:**
- **pytest:** 282 test đều qua trên bốn môi trường: Python 3.13.16 với numpy 2.5.4 / opencv 4.11.0.86; Python 3.13.16 và 3.10.20 cài đúng `requirements.txt` (numpy 2.2.6, opencv 5.0.0.93 như env `gs-inria`); Python 3.10.20 với numpy 1.26.4 / opencv 4.11.0.86 như env `gs-gsplat`. Không test nào bị bỏ qua (máy có `colmap`, `ffmpeg`).
- **shellcheck** 0.11.0 (`scripts/*.sh scripts/*/*.sh`) và **pyflakes** (`src app tests scripts/setup/*.py`): sạch.
- **Notebook** (`app/notebooks/runpod.ipynb`) trên `/workspace` giả, pod mới, volume trống. GitHub được thay bằng một bare repo cục bộ, `ssh` và `curl` là stub (không kết nối mạng; mọi lần gọi `curl` được ghi lại; khi tải thì báo lỗi mạng, hoặc trả về một file zip giả), không GPU, không conda. **59/59 kiểm tra đạt**:
  - không ô code nào tự tải (`curl`, `wget`, `unzip`); chỉ ô 7 gọi bước tải riêng (`fetch_data.sh`, `run_experiment.sh`). Cả lượt, `curl` chỉ được `env_probe` gọi để thử kết nối tới GitHub, gitlab.inria.fr, PyPI, kho PyTorch và conda (`-o /dev/null`), và cho hai lần tải `tandt_db.zip` được bật ở ô 7; không tải Mip-NeRF 360; không còn `data/downloads`;
  - ô 1 báo ĐẠT với GPU giả (RTX 4090, driver CUDA 13.0) và cảnh báo khi `python3` không có PyYAML;
  - ô 2 tạo khóa ed25519 (`600`, thư mục `700`), chỉ in khóa công khai, chạy lại không tạo khóa mới; `ssh` được gọi với đúng khóa, `IdentitiesOnly`, `known_hosts` trên volume;
  - ô 3 clone đúng commit, cây file giống hệt repo, giữ quyền thực thi; chạy lại thì `pull --ff-only`; code trên pod có thay đổi chưa commit thì từ chối (exit 1), không stash, thay đổi còn nguyên; đang có việc nền hoặc thư mục không phải repo Git thì từ chối;
  - ô 4 chạy nền `run_all.sh` (không tham số); tóm tắt chỉ có `env_probe` (FAIL vì không có GPU) và ba bước cài đặt (SKIP); không cài gì, không tạo `outputs/benchmarks` hay `data/benchmarks`; log và báo cáo nằm trong `outputs/`;
  - ô 6 trên volume trống: Deep Blending và Mip-NeRF 360 chỉ báo "chưa có" (không luồng nào bắt buộc), không tạo thư mục nào; thiếu PyYAML thì lỗi hiện rõ trong ô; sau khi đưa Deep Blending giả (đúng bố cục) lên thì báo OK;
  - ô 7, thực nghiệm chính: `WHAT = "experiment"` (mặc định) với `CHAY = False` chỉ in kế hoạch (2 chế độ, 3 seed, 6 cảnh còn thiếu, ước lượng), không tạo việc, không tải, không tạo thư mục dữ liệu; `QUICK = True` bị từ chối; `CHAY = True` thành đúng một việc nền `exp_main`, dừng ở `env_check` (không GPU, không env), không tải, không train, vẫn có bảng các bước và gói kết quả;
  - ô 7, tải dữ liệu: `fetch` với `CHAY = False` không chạy; `CHAY = True` thành việc nền `fetch_deep_blending`, gọi đúng `tandt_db.zip` của Inria; lỗi mạng thì dừng, không có cảnh, không còn file tải dở; file sai SHA-256 thì dừng trước khi giải nén và xóa file zip;
  - ô 7, các lệnh khác: `WHAT = "check"` báo `dataset not found` kèm chỗ cần đưa lên, cho cả Deep Blending và Mip-NeRF 360; `gsplat` / `b0` với `CHAY = False` không chạy; `CHAY = True` khi chưa có dữ liệu thì báo ngay, không xếp hàng; `evaluate` khi chưa có kết quả báo rõ; `WHAT` không hợp lệ chỉ in hướng dẫn. Khi đã có dữ liệu: `check` báo OK; `gsplat` với `QUICK = True` thành việc nền `gsplat_quick_deep_blending` (thư mục `gsplat_quick`), `b0` thành `b0_deep_blending`, cả hai dừng sớm vì không có GPU / env và không tạo lượt nào; dữ liệu đã đưa lên không bị sửa;
  - ô 8 từ chối 1 ảnh, 5 ảnh, một file ảnh, `SOURCE = "-"` không có `SCENE`; tên cảnh lấy đúng từ thư mục ảnh, thư mục chứa video, hoặc tên file video; ảnh / video gốc không bị sửa;
  - ô 8 với `CHAY = False` (mặc định) không chạy gì; ô 9 xuất PLY bằng liên kết cứng; render lại chạy nền và dừng sớm khi không có GPU, không đụng lượt gốc;
  - ô 10 tạo gói kết quả không có PLY, checkpoint, video, ảnh gốc, `data/` hay `.ssh`; ô 11 với `DUNG=0` (mặc định) không dừng gì, `DUNG=1` dừng mọi việc; sau mọi ô `git status` của repo trên pod vẫn sạch.
- **COLMAP 3.9.1 bản CPU thật** trên ảnh và video tổng hợp (cảnh dựng bằng code, không phải dữ liệu thực nghiệm), qua `scripts/pod/run_scene.sh` và `scripts/prepare_data.sh custom`, huấn luyện thay bằng stub:
  - thư mục 1 ảnh: từ chối trước khi xếp hàng GPU;
  - roomA, 36 ảnh có tiêu cự EXIF 27 mm (`run_scene.sh`): 36/36 ảnh, 31.396 điểm, 0,146 px, phủ 360°, PASS (188 giây);
  - video tổng hợp 5 giây (150 frame, 30 fps) đi vòng quanh phòng: chọn 68/75 frame ứng viên, 68/68 đăng ký, 42.208 điểm, 0,251 px, PASS (646 giây);
  - `- roomA`: dùng lại COLMAP, không đọc lại nguồn;
  - roomB, 24 ảnh không EXIF, qua `scripts/prepare_data.sh custom` rồi `scripts/run_train.sh custom`: 24/24, 15.576 điểm, 0,134 px, PASS kèm cảnh báo dưới 30 ảnh và thiếu tiêu cự EXIF (73 giây);
  - dấu SHA-256 của mọi file trong `data/` không đổi; `scripts/pod/results.sh` và `scripts/pod/check_datasets.sh --dataset custom_scene` đọc đúng cả bốn cảnh;
  - chạy lại roomB qua `scripts/prepare_data.sh custom` sau khi sửa lỗi (147 giây): 24/24, 13.894 điểm, 0,600 px, PASS. Cùng 24 ảnh không EXIF nhưng lượt này COLMAP tự hiệu chỉnh ra tiêu cự lệch hai trục (fx 689 / fy 500 px) và ảnh sau khử méo bị cắt còn 548×393 (lượt trước: fx ≈ fy ≈ 500, 639×479). Kiểm tra trước khi train chưa bắt được trường hợp này (mục "Chưa xác minh / còn mở").

### Rà soát độc lập ngày 21/10/2026 (D14)

Hai lượt, người rà soát không tham gia viết code: làm trên bản sao, chạy lại test, dựng lại lỗi bằng stub và bằng đoạn code của gsplat @6e8c837 chạy trên CPU. Lượt 1: 1 lỗi chặn, 2 lỗi vừa, 7 lỗi nhỏ. Lượt 2 kiểm lại từng bản sửa (đều đạt; dựng 84 tổ hợp quy ước mcmc × lượt, bash và Python cho cùng tham số) và tìm thêm 8 lỗi nhỏ. Đã sửa (đã kiểm thử trên CPU, chưa chạy trên pod):

| Mức | Lỗi | Đã sửa | Test |
|---|---|---|---|
| Chặn | Chạy `main` sau `tune_mcmc`: bảng so sánh lẫn các biến thể có tag của `room`, nên chỉ `room` "đủ mọi biến thể" và trung bình trên các cảnh chỉ còn `room`, seed 42 | Lượt có tag chỉ vào bảng khi được yêu cầu (`compare_modes --tags`, `TAGS` của `run_evaluate.sh`); bảng của một kế hoạch có các biến thể của chính nó | `test_compare_modes.py::test_runs_with_a_tag_stay_out_unless_their_tag_is_asked_for`, `test_experiment.py`, `test_benchmarks.py` |
| Vừa | Quy ước `opacity_reg=0.001` của Deep Blending đè tham số mà một chế độ mới tự đặt (vd. `mcmc_indoor`) | Ưu tiên tăng dần: preset → quy ước của benchmark → tham số của chế độ → tham số của biến thể, như nhau ở bash và Python. Lượt cùng chế độ, cùng cấu hình là lỗi; hai chế độ khác tên cùng cấu hình chỉ là cảnh báo | `test_benchmarks.py::test_a_mode_that_sets_the_convention_key_keeps_its_own_value`, `test_experiment.py` |
| Vừa | Tài liệu nói dòng của `room` "được ghi riêng", công cụ không làm vậy | Tài liệu ghi đúng: bảng tự động vẫn tính `room`; thêm lệnh lập bảng không có `room` | — |
| Nhỏ | Kiểm tra điểm SfM coi ảnh mà mọi điểm nằm ngoài ảnh là lỗi (thực ra chỉ làm giá trị loss ghi ra là NaN, gradient hữu hạn), và bỏ cả benchmark, kể cả B0 | Trường hợp đó chỉ cảnh báo; không đạt thì chỉ bỏ các lượt có loss độ sâu, B0 vẫn chạy; bảng so sánh và chỗ trống không tính lượt bị bỏ | `test_depth_check.py`, `test_experiment.py` |
| Nhỏ | Lọc Gaussian: lỗi lạ ở một lượt dừng cả loạt; không kiểm được ảnh test, hoặc chỉ số NaN, vẫn ghi OK; bản đã lọc mất file vẫn được bỏ qua | Bắt lỗi theo từng lượt; hai trường hợp đó là lỗi; chỉ bỏ qua khi chỉ số đọc được và PLY còn | `test_prune.py` |
| Nhỏ | Kiểm tra chỗ trống không tính bản đã lọc của lượt đã xong mà chưa lọc | Tính thêm checkpoint, PLY, ảnh, video của lượt đó | `test_experiment.py` |
| Nhỏ | Ngưỡng lọc bị làm tròn 6 chữ số khi chuyển cho bash; ngưỡng khác nhau giữa các dòng không ghi theo dòng; ước lượng thời gian của `tune_mcmc` khác giả định của `main`; vài câu tài liệu cũ; ô 7 không đưa được biến thể vào bảng | Đã sửa (ô 7: biến `TAGS`) | `test_experiment.py`, `test_compare_modes.py` |

### Rà soát độc lập ngày 19/10/2026

Người rà soát làm trên một bản sao, chạy lại test và lint, dựng lại từng lỗi bằng stub. Kết quả: 1 lỗi chặn, 8 lỗi vừa, khoảng 15 lỗi nhỏ / tài liệu. Đã sửa (đã kiểm thử trên CPU, chưa chạy trên pod):

| Mức | Lỗi | Đã sửa | Test |
|---|---|---|---|
| Chặn | `scripts/pipeline/train_scene.sh` đọc `configs/base.yaml (train_modes)` như một file JSON (lỗi khi đổi tên hàng loạt): **mọi lượt train cảnh tự chụp lỗi trước khi bắt đầu**. Test cũ không bắt được vì thay `train_scene.sh` bằng stub | Đọc `train_modes` bằng `src.training.profiles.load_config` | Mới: `tests/unit/test_train_scene.py` chạy `train_scene.sh` thật (3 test; cả 3 lỗi với bản cũ) |
| Vừa | B0 đọc cấu hình sau `conda activate gs-inria` (python3 của env không có PyYAML): chưa có env `gs-gsplat` thì B0 dừng | `scripts/lib/config.sh` giữ python3 tìm thấy lúc được source (trước khi kích hoạt env) | `tests/test_dataset.py` (lỗi với bản cũ) |
| Vừa | Xuất PLY trùng tên: lượt chạy thử và lượt chuẩn, hoặc lượt cũ trước khi train lại, làm lệnh xuất lỗi sau khi đã render | Lượt chạy thử xuất vào `checkpoints/<benchmark>_quick/`; tên đã là PLY khác thì xuất `<lượt>_<thời điểm train xong>.ply`; thiếu PLY thì dừng trước khi render | `tests/unit/test_render.py` |
| Vừa | Render lại một lượt sau khi cảnh đã làm lại COLMAP: checkpoint cũ với camera mới, ảnh và chỉ số sai mà không báo | So dấu SHA-256 của mô hình COLMAP với dấu ghi trong `run_meta.json`; khác thì dừng | `tests/unit/test_render.py` |
| Vừa | Ô 8 của notebook chạy GPU ngay khi chạy ô (hoặc "Run All"); ô 11 dừng mọi việc vừa được xếp hàng | Thêm `CHAY = False` cho ô 8, `DUNG=0` cho ô 11 | Mô phỏng notebook |
| Vừa | B0 với `FORCE=1` ghi đè kết quả cũ | Lượt cũ đổi tên `<cảnh>.prev_<thời gian>` trước khi chạy lại | `tests/unit/test_benchmarks.py` |
| Vừa | Tài liệu nói quá: test chạy `train_scene.sh` thật (sai), luồng cảnh tự chụp đã chạy với COLMAP thật (bước train chỉ là stub), mọi code đã chạy trên pod (phần viết sau 17/10 chưa chạy) | Sửa lại cho đúng (`docs/design/architecture.md` mục 3 và 7, `docs/design/training_modes.md` mục 8, `docs/USAGE_GUIDE.md` mục 9) | — |
| Nhỏ | Bảng đối chiếu tên cũ → mới ghi sai tên của lượt 17/10; `configs/local.yaml` sai dạng gây traceback, khóa `train_modes` trong đó bị bỏ qua lặng lẽ; phần hướng dẫn của vài script in cả dòng code; chú thích cũ (`PYTHONPATH=src`, `env_probe.sh`, `run_scene.sh`); `ssh-keygen` trong tài liệu hỏi lại khi khóa đã có; hai gói kết quả trong cùng một phút trùng tên; CLI chạy script trong thư mục repo thay vì thư mục hiện tại; `runs/` trong `.gitignore` không neo gốc | Đã sửa (`local.yaml` chỉ nhận mục `datasets` dạng mapping; tên gói có giây) | `tests/test_dataset.py` |

Giữ nguyên, có lý do:
- Render lại ghi một thư mục mới `render_<quỹ_đạo>_<thời gian>/` **bên trong** lượt gốc: không sửa file nào của lượt, giữ ảnh render cạnh lượt của nó; tài liệu đã ghi rõ.
- File PLY xuất là liên kết cứng tới PLY của lượt (không tốn thêm chỗ): tài liệu ghi chỉ đọc / tải về, không sửa tại chỗ.
- Gói `ket_qua_*.tar.gz` ở gốc `/workspace` (như sơ đồ).

**Lượt 2: thực nghiệm chính, tải dữ liệu** (bản chưa commit sau `5d83cd7`). Người rà soát chạy lại 279 test và shellcheck trên một bản sao và dựng lại lỗi bằng stub. Kiểm tra thêm và thấy đúng: chạy lại bỏ qua lượt đã xong, lượt dở dang được đổi tên `.prev_*` rồi train lại; biến môi trường của người dùng không vào được các bước; tải lỗi chỉ bỏ benchmark đó; đường dẫn không an toàn trong zip bị chặn; bảng so với một mốc khác `default` ghép đúng seed.

Đã sửa (đã kiểm thử trên CPU, chưa chạy trên pod):

| Mức | Lỗi | Đã sửa | Test |
|---|---|---|---|
| Vừa | Chạy lại sau khi dừng (đúng cách tài liệu hướng dẫn) thì kiểm tra chỗ trống gần như không còn tác dụng: chỗ cần cho seed sau lấy theo phần ghi thêm trong lần chạy này (lượt đã xong không ghi gì), và kiểm tra lúc đầu không tính phần thực nghiệm đã dùng. Người rà soát dựng lại được: volume 52 GB, đã xong 2 seed, seed 44 vẫn được train và lên tới 64 GB | Chỗ cần cho một seed = tổng các lượt chưa xong của seed đó, mỗi lượt theo dung lượng thật của lượt cùng cảnh, cùng chế độ đã xong ở seed khác (chưa có thì theo ước lượng); kiểm tra trước mọi seed, kể cả seed đầu; lúc đầu so với tổng dung lượng đang dùng | `tests/unit/test_experiment.py`: hai ca của người rà soát (chạy lại sau khi dừng vì thiếu chỗ; pod dừng giữa seed 43) |
| Nhỏ | Bản ghi nguồn tải chỉ được ghi khi mọi cảnh của file zip xong: cảnh sau lỗi thì cảnh trước đã vào chỗ mà không có nguồn, SHA-256 | Ghi sau mỗi cảnh; báo cáo được ghi cả khi lỗi, kèm lỗi | `tests/unit/test_fetch.py` |
| Nhỏ | curl không ngắt khi kết nối treo: việc treo ở giờ đầu mà vẫn giữ khóa GPU | `--speed-limit 102400 --speed-time 300` (dưới 100 KB/s suốt 5 phút thì ngắt, `--retry` thử lại) | `tests/unit/test_fetch.py` |

Tìm thêm khi chạy lại test (trước lượt rà soát): khi `run_experiment.sh` chạy trực tiếp (không qua việc nền), tiến trình `tee` ghi log giữ fd khóa GPU, nên khóa có thể được nhả chậm một chút sau khi script kết thúc (4/30 lượt test hỏng khi máy bận). Sửa: `tee` đóng fd khóa, như `scripts/lib/gpu_lock.sh` hướng dẫn cho tiến trình phụ; 60/60 lượt qua.

**Repo trước, 18–19/10/2026** (cùng code chuẩn bị cảnh, bố cục trước khi dời sang repo này):

| Ca | Kết quả COLMAP | Kết luận | Đúng mong đợi? |
|---|---|---|---|
| Video 5 giây, đi vòng quanh phòng | 68/68 frame, khoảng 42.200 điểm, sai số 0,251 px, phủ 360° | PASS | Có |
| roomB: 24 ảnh, không EXIF | 24/24, 16.425 điểm, 0,134 px, phủ 360° | PASS (cảnh báo dưới 30 ảnh) | Có |
| roomA: 36 ảnh, **không** tiêu cự EXIF | 23/36 ảnh, tách 2 nhóm | FAIL, không train | Có (thiếu tiêu cự ban đầu) |
| roomA: cùng 36 ảnh, **có** tiêu cự EXIF 27 mm | 36/36, khoảng 31.400 điểm, 0,145 px, phủ 360° | PASS | Có |
| Hai phòng lẫn nhau (24 + 16 ảnh), có EXIF | 24/40: đúng 24 ảnh phòng A; 16 ảnh phòng B được liệt kê kèm tên file gốc | FAIL, không train | Có |
| Đứng một chỗ chỉ xoay máy, có EXIF | Không dựng được: không có cặp ảnh khởi tạo đủ thị sai | FAIL, gợi ý chụp lại có di chuyển | Có |
| 1 ảnh; 1 file ảnh; 5 ảnh | — | Từ chối trước khi xếp hàng GPU | Có |

Số điểm 3D lệch nhẹ giữa các lượt vì COLMAP chạy nhiều luồng. Log: sandbox của phiên làm việc (không đưa vào repo). Các ngưỡng còn phải hiệu chỉnh với ảnh thật.

## Chưa xác minh / còn mở

- **Repo này chưa chạy trên pod.** Chưa chạy thật: khóa deploy với GitHub thật, clone qua SSH, cài 3 env trên volume trống, tải dữ liệu benchmark, B0 và gsplat bằng code mới, render lại bằng chế độ `--ckpt` của gsplat thật, mở PLY trên một trình xem 3DGS.
- Thực nghiệm chính (D12) chưa chạy:
  - ước lượng thời gian (khoảng 12 giờ) và dung lượng (khoảng 56 GB) chỉ có phần Deep Blending dựa trên số đo; Mip-NeRF 360 chưa đo; môi trường conda trên volume (khoảng 15–20 GB) chưa đo;
  - `AUTO_STOP=1` (dừng pod khi xong) chưa thử trên pod.
- Tải dữ liệu: URL và SHA-256 của `tandt_db.zip` đã dùng được trên pod ngày 17/10/2026 (repo trước). URL của `360_v2.zip` lấy từ script tải của gsplat @6e8c837, chưa tải thử bằng repo này; dung lượng file chưa biết (script in ra khi tải).
- Ảnh không có EXIF tiêu cự: hai lượt COLMAP trên cùng 24 ảnh tổng hợp cho hai mô hình khác nhau; một lượt có tiêu cự lệch hai trục (fx 689 / fy 500 px) và ảnh sau khử méo bị cắt còn 548×393, vẫn PASS. Cần thêm cảnh báo (vd. fx/fy lệch nhau, ảnh sau khử méo nhỏ hơn hẳn ảnh vào) và thử lại với ảnh thật trước khi dùng ảnh không EXIF.
- PyYAML trong `python3` của template Runpod: suy ra từ JupyterLab 4.2.5 có trong template (`jupyter_events` cần PyYAML), chưa kiểm tra trực tiếp. Ô 1 của notebook kiểm tra; `env_probe.sh` nay liệt kê PyYAML. Không có thì chỉ các lệnh đọc cấu hình chạy trước khi cài xong `gs-gsplat` bị ảnh hưởng, kèm thông báo cách sửa.
- Câu trả lời của GVHD cho Q1–Q3 và Q5–Q12 (`docs/PROJECT_PLAN.md`, mục 2): đang dùng giả định tạm.
- Cách chia train/test cho dữ liệu tự quay: code Inria gán cứng mỗi ảnh thứ 8 làm test; cần quyết định ở M4.
- Chưa chọn LICENSE cho repository.
- Cảnh tự chụp: chưa có lượt nào chạy xong xác minh được trên pod.
- Mip-NeRF 360:
  - chưa có dữ liệu trên volume (`bash scripts/pod/check_datasets.sh --dataset mipnerf360`);
  - `360_v2.zip` chỉ có 7 cảnh; flowers và treehill nằm trong gói riêng `360_extra_scenes.zip` (đã ghi trong mục `download`, bố cục bên trong chưa kiểm tra);
  - trung bình công bố chỉ so được khi chạy đủ 9 cảnh;
  - gsplat và Inria dùng ảnh thu nhỏ khác nhau (gsplat tự thu nhỏ từ `images/`, Inria đọc `images_N/`), cần ghi rõ khi so `default` với B0.
- Đầu vào ảnh:
  - các ngưỡng (tối thiểu 10 ảnh, khuyến nghị 30, khoảng hướng trống ≥ 90 độ, camera dịch chuyển < 5 % khoảng cách tới cảnh) là điểm xuất phát, cần hiệu chỉnh sau lần chụp thử;
  - ảnh được ghi lại bằng OpenCV và chỉ giữ thẻ tiêu cự 35 mm của EXIF (các thẻ khác, kể cả GPS, bị bỏ). Ảnh không có thẻ này thì COLMAP tự đoán tiêu cự; trên ảnh tổng hợp, điều này làm số ảnh ghép được giảm từ 36/36 xuống 23/36;
  - COLMAP có thể ghép nhầm hai nơi có họa tiết giống hệt nhau và vẫn báo PASS (`docs/USAGE_GUIDE.md` mục 3.4).
- Chưa đăng ký quyền truy cập ScanNet++.
- Đoạn quay kiểm tra riêng: `src.training.run --test-list` đã hỗ trợ cho gsplat. Còn thiếu bước đưa frame của đoạn kiểm tra vào cùng model COLMAP với video chính (`scripts/pipeline/prepare_scene.sh`, mốc M4).
- `opacity_reg` của MCMC trên Deep Blending: paper dùng 0,001, config chính chủ chỉ đặt cho `drjohnson`. Benchmark Deep Blending dùng 0,001 cho cả hai cảnh (`benchmark.gsplat.mcmc_set` trong `configs/deep_blending.yaml`); cần ghi rõ khi báo cáo.
- COLMAP: chưa đo thời gian với video hay ảnh thật trên pod. Matcher mặc định là `exhaustive` như `convert.py` của Inria (250 frame ≈ 31.000 cặp ảnh); nếu quá chậm trên CPU thì thử `COLMAP_ARGS="--matcher sequential --loop-detection"` và so sánh tỉ lệ đăng ký.
- Dò tham số và lọc Gaussian (D14) chưa chạy:
  - `tune_mcmc`: ước lượng khoảng 5 giờ GPU và 24 GB chưa đo; `mcmc_indoor` chưa được định nghĩa (chờ số liệu);
  - kiểm tra điểm SfM chưa chạy trên `room` thật; lọc rồi chấm lại bằng chế độ `--ckpt` của gsplat thật chưa chạy;
  - ngưỡng lọc (opacity 0,02, trục lớn nhất 0,1 × `scene_scale`) là điểm xuất phát, chưa hiệu chỉnh; bỏ Gaussian lớn có thể làm thủng tường hay trần ở xa;
  - tham số được chọn bằng ảnh test của `room` (cảnh dùng để chọn, không tính vào kết luận); cách chặt hơn là tách tập kiểm định từ ảnh train của `room` (chưa làm).
- `configs/environment/*.conf` vẫn để `provisional` (`run_all.sh` đặt `ALLOW_PROVISIONAL=1`). Điều kiện để chốt (B0 đạt, smoke test của `gs-gsplat` PASSED trên pod) đã có từ 17/10/2026 với repo trước; chưa sửa file.

## Môi trường

Phương án, bố cục `/workspace`, sự cố đã biết: `docs/ENVIRONMENT_AUDIT.md`. Pod và phiên bản thực tế của các lượt chạy 17/10/2026 (repo trước): `docs/ENVIRONMENT_AUDIT.md` mục 7 (RTX 4090, EU-RO-1, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; driver CUDA 12.8 ở lượt đầu, 13.0 ở lượt thứ hai).

Lệnh chính: `README.md` ở gốc repo, mục "Chạy trên Runpod"; từng bước và notebook: `docs/USAGE_GUIDE.md`; benchmark: `docs/design/experiments.md`.

## Dữ liệu, checkpoint, kết quả

Lưu trên network volume của Runpod tại `/workspace`, không đưa lên Git. Nơi lưu của từng luồng: `configs/*.yaml`; bố cục: `docs/design/architecture.md` mục 2.

- Volume mới (100 GB, EU-RO-1): **chưa có gì**. Sau ô 3–4 của notebook và khi chạy ô 7, 8:
  - code `/workspace/SpatialRoom3D`; env `/workspace/miniforge3`; mã nguồn Inria / gsplat `/workspace/code`;
  - Deep Blending (ô 7 tải về, hoặc bạn tự đưa lên) `/workspace/data/benchmarks/deep_blending/{drjohnson,playroom}`; kết quả `/workspace/outputs/benchmarks/deep_blending/{inria_b0,gsplat,gsplat_quick}/`;
  - Mip-NeRF 360, 4 cảnh trong nhà (ô 7 tải về) `/workspace/data/benchmarks/mipnerf360/{room,counter,kitchen,bonsai}`; kết quả `/workspace/outputs/benchmarks/mipnerf360/gsplat/`;
  - thực nghiệm chính: `/workspace/outputs/reports/experiment_main_<thời_gian>.{md,tsv}`; dò tham số: `experiment_tune_mcmc_<thời_gian>.{md,tsv}` cùng thư mục, các lượt ở `/workspace/outputs/benchmarks/mipnerf360/gsplat/room/<chế_độ>[_<tag>]_s42/`, bản đã lọc trong `<lượt>/filtered/`;
  - cảnh tự chụp: nguồn `/workspace/data/custom_scenes/<cảnh>/`, kết quả `/workspace/outputs/custom_scenes/<cảnh>/`;
  - PLY đã xuất `/workspace/outputs/checkpoints/`; báo cáo, log, trạng thái việc nền `/workspace/outputs/{reports,logs,jobs}`; gói kết quả nhẹ `/workspace/ket_qua_*.tar.gz`.
- Volume cũ (EU-RO-1): còn kết quả ngày 17/10/2026 của repo trước, ở bố cục cũ. Repo này không đọc và không sửa volume đó.

## Việc tiếp theo

**Trên pod mới, volume mới.** Hướng dẫn: `docs/USAGE_GUIDE.md` mục 1–2, `app/notebooks/runpod.ipynb`.

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| P1 | Tạo pod gắn network volume (150 GB để chạy P5–P7 trên cùng volume; 100 GB đủ cho thực nghiệm chính hai chế độ theo ước lượng; `docs/USAGE_GUIDE.md` mục 1); tải `app/notebooks/runpod.ipynb` lên `/workspace`; chạy ô 1 | Ô 1 báo `=> ĐẠT`; có dòng `✓ python3 có PyYAML` (không có thì cài theo hướng dẫn trong ô) |
| P2 | Ô 2; thêm khóa công khai vào Deploy keys của `SpatialRoom3D` (không tick write); chạy lại ô 2 | Dòng cuối `✓ GitHub nhận khóa` |
| P3 | Ô 3 | `Đã clone ... (nhánh main)`; commit in ra trùng commit mới nhất trên GitHub |
| P4 | Ô 4, theo dõi ở ô 5 (khoảng 80 phút) | Việc `run_all` exit 0: `env_probe`, `setup_inria`, `data_tools`, `setup_gsplat` OK, smoke test PASSED; không có bước train nào |
| P5 | Ô 7, `WHAT = "experiment"`, `EXP = "tune_mcmc"`, `CHAY = False`: xem kế hoạch; rồi `VOLUME_GB` = dung lượng thật của volume, `CHAY = True` (khoảng 5 giờ GPU, **cần duyệt**). Theo dõi ở ô 5 | Việc `exp_tune_mcmc` exit 0: `env_check` nhận tham số của 10 biến thể; `depth_check_mipnerf360` OK; 12 lượt của `room` có `summary.json` OK, cùng ngân sách Gaussian; mọi lượt có `filtered/filter_meta.json` OK; báo cáo có bảng so với `default`, với `mcmc` và sau khi lọc |
| P6 | Chọn tham số cho `mcmc_indoor` từ P5 theo `docs/design/experiments.md` mục 3.1; thêm chế độ vào `configs/base.yaml` và vào `modes` của `tune_mcmc.yaml` (sửa trong repo rồi commit), chạy lại `tune_mcmc` (chỉ lượt mới được train). Chỉ đổi một tham số thì lượt của biến thể tương ứng đã là lượt xác nhận, không cần chạy lại | Mỗi tham số được chọn có lý do ghi theo số liệu P5; không biến thể nào hơn `mcmc` rõ ràng thì ghi lại như vậy, không thêm chế độ. Lượt `mcmc_indoor_s42` của `room` OK; ghi rõ là một seed, trên cảnh dùng để chọn tham số |
| P7 | Thêm `mcmc_indoor` vào `modes` của `configs/experiments/main.yaml`, `also_against: [mcmc]`, cập nhật `estimate` (commit); ô 7 với `EXP = "main"`, `CHAY = False` xem kế hoạch, rồi `VOLUME_GB`, `CHAY = True` (**cần duyệt**). Không có `mcmc_indoor` thì chạy `main` như hiện tại | Việc `exp_main` exit 0: mọi bước OK trong `experiment_main_*.md`; `fetch_*.json` có SHA-256 (Deep Blending khớp cấu hình); trên Deep Blending, B0 đạt ngưỡng so với số Inria công bố và `default` PASS so với B0 từng cảnh; mỗi cảnh × chế độ đủ 3 seed, mọi mục điều kiện OK, ngân sách EQUAL; ghi chênh lệch `default` / `mcmc` so với lượt 17/10/2026 của repo trước |
| P8 | Đọc kết quả P7 theo `docs/design/experiments.md` mục 2.5, 3.1 và `docs/design/training_modes.md` mục 4.3; lập bảng `mcmc_indoor` / `mcmc` không có `room` (hai lệnh ở mục 3.1, không dùng GPU) | Kết luận về `mcmc` và `mcmc_indoor` ghi đúng như số liệu: cảnh nào tốt hơn, kém hơn, chênh lệch so với độ lệch chuẩn giữa các seed; kết luận về `mcmc_indoor` dựa trên bảng không có `room`; không cải thiện thì vẫn ghi đủ |
| P9 | `bash scripts/run_render.sh deep_blending playroom default_s42` (render lại từ checkpoint, cần GPU, vài phút) | `render_meta.json` exit 0; PSNR / SSIM / LPIPS của lần render bằng số của lượt gốc (cùng checkpoint, cùng tập test); có video; PLY ở `outputs/checkpoints/deep_blending/playroom/default_s42.ply` |
| P10 | Video của nhóm hoặc một cảnh tự chụp (≥ 30 ảnh theo `docs/data/capture_protocol.md` mục 8) vào `data/custom_scenes/<cảnh>/`, chạy ô 8, xem ô 9; thêm `mcmc_indoor` (`bash scripts/run_train.sh custom <cảnh> mcmc_indoor`) nếu P8 cho thấy đáng dùng | `sfm_report.json` không FAIL; có `summary.json` cho cả `default` và `mcmc` (và `mcmc_indoor` nếu chạy); mô hình xem được trên trình xem 3DGS; `coverage` khớp với cách chụp thực tế |
| P11 | Mip-NeRF 360 đủ 9 cảnh (tùy chọn, khi được duyệt chi phí GPU): ô 7 `WHAT = "fetch"` với `BENCH = "mipnerf360"`, rồi `b0`, `gsplat` | Số liệu đầy đủ đối chiếu được với trung bình công bố |

**Không cần GPU**

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| — | Chọn trình xem 3DGS cho file PLY, rồi viewer web bản thô (`app/web/`, giai đoạn 5) | Mở được mô hình mẫu trên trình duyệt; có số đo thời gian tải; có kiểm thử chuyển đổi định dạng |
| — | Thu thử 1 cảnh: quay video theo `docs/data/capture_protocol.md` (video chính + đoạn quay kiểm tra), hoặc chụp ≥ 30 ảnh theo mục 8 | Đặt tên và ghi metadata đúng quy ước; ảnh cùng chiều máy, cùng ống kính, dùng file gốc |
| — | Chọn LICENSE; quyết định chế độ hiển thị của repo cũ | Có file LICENSE; repo cũ private hoặc archive |
