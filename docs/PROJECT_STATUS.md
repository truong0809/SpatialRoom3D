# PROJECT STATUS

_Cập nhật: 19/10/2026_

## Mục tiêu hiện tại

1. Dựng **pod mới gắn network volume mới (trống)** bằng repo này (`docs/USAGE_GUIDE.md` mục 1–2):
   - khóa deploy chỉ đọc, clone code vào `/workspace/SpatialRoom3D`;
   - cài 3 env conda (`scripts/pod/run_all.sh`: chỉ cài môi trường, không train, không tải dữ liệu, D10).
2. Bạn tự đưa dữ liệu lên volume (`docs/USAGE_GUIDE.md` mục 2.1), kiểm tra bằng ô 6.
3. Chạy với dữ liệu bạn chọn:
   - cảnh tự chụp (thư mục ảnh hoặc video) qua ô 8 của notebook;
   - benchmark qua ô 7, ví dụ chạy lại Deep Blending bằng repo này: B0 Inria rồi gsplat `default` / `mcmc`, seed 42. Cả chạy thử, B0 và gsplat khoảng 2,4 giờ GPU trên RTX 4090: **chỉ chạy khi đã duyệt chi phí**.

**Repo này:** lịch sử bắt đầu ngày 19/10/2026 bằng một commit chứa toàn bộ cấu trúc mới (D7–D9); D10, D11 là các thay đổi sau commit đó. **Chưa chạy trên pod nào.** Mọi số liệu GPU dưới đây là của repo trước (`3D-Indoor-Scene-Reconstruction-from-Images`), chạy ngày 17/10/2026 trên volume cũ.

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
| D7 | 19/10/2026 | Cấu trúc repo theo sơ đồ "SpatialRoom3D_v2": `app/{cli,web,notebooks}`, package `src/{input,reconstruction,training,rendering,evaluation,datasets,utils}`, `configs/{base,custom_scene,mipnerf360,deep_blending}.yaml`, bốn lệnh chính `scripts/{prepare_data,run_train,run_render,run_evaluate}.sh` (phần thực thi trong `scripts/{pipeline,pod,setup,lib}/`), `tests/{test_dataset,test_pipeline}.py` + `tests/unit/`, `docs/{README,ENVIRONMENT_AUDIT,USAGE_GUIDE,PROJECT_PLAN}.md`, `experiments/{deep_blending,mipnerf360}/` (chỉ README). Code dùng chung, mỗi luồng dữ liệu một file cấu hình và thư mục riêng | Theo sơ đồ và các lựa chọn của bạn ngày 19/10/2026 ("đủ như ảnh, giữ phần cần chạy", "src/ là package", "YAML như ảnh"). Phần đã chạy trên pod ngày 17/10/2026 được dời chỗ, không viết lại; thuật toán, chế độ, protocol, phiên bản không đổi. Phần viết sau ngày đó chưa chạy trên pod (`docs/design/architecture.md` mục 7) |
| D8 | 19/10/2026 | Bố cục `/workspace`: code ở `SpatialRoom3D/`; `data/` chỉ chứa dữ liệu gốc (`custom_scenes/<cảnh>/`, `benchmarks/{deep_blending,mipnerf360}/`; `downloads/` bỏ theo D10); `outputs/` chứa mọi thứ sinh ra (`custom_scenes/<cảnh>/{colmap,<chế_độ>_s<seed>}/`, `benchmarks/<benchmark>/{inria_b0,gsplat,gsplat_quick}/`, `checkpoints/`, `logs/`, `reports/`, `jobs/`) | Theo sơ đồ của bạn. `docs/design/architecture.md` mục 2 |
| D9 | 19/10/2026 | Repo GitHub mới `truong0809/SpatialRoom3D` (private), lịch sử bắt đầu bằng một commit; repo cũ giữ làm lưu trữ, không sửa. Pod mới + volume mới: cài lại từ đầu, đưa lại dữ liệu (D10: bạn tự đưa lên), chạy lại B0 + gsplat bằng code mới khi được duyệt GPU. Bỏ phần dời dữ liệu của volume cũ và mọi phần tương thích bố cục cũ | Theo ba lựa chọn của bạn ngày 19/10/2026 |
| D10 | 19/10/2026 | Repo không tải dữ liệu nào: bỏ hẳn bước tải, kiểm tra SHA-256 gói tải về, giải nén và dời dữ liệu (cả mục `source` trong `configs/*.yaml` và thư mục `data/downloads/`). `scripts/pod/run_all.sh` (ô 4) chỉ cài / kiểm tra môi trường, không train. Chạy với dữ liệu nào (cảnh tự chụp, Deep Blending, Mip-NeRF 360) do người dùng chọn (ô 7, ô 8); thiếu dữ liệu thì dừng trước khi train và báo chỗ cần đặt. Không luồng nào còn `required: true` (trước đây Deep Blending là bắt buộc): ô 6 chỉ báo "chưa có". `env_probe.sh` không còn thử kết nối tới máy chủ dữ liệu của Inria | Theo yêu cầu của bạn ngày 19/10/2026 ("base cài set up môi trường thôi, còn chạy với data nào để t tự quyết"). Bỏ lệnh `baseline` của CLI và ô "Baseline Deep Blending" của notebook. Repo không còn kiểm tra tổng kiểm dữ liệu benchmark: ô 6 chỉ kiểm tra bố cục và số ảnh |
| D11 | 19/10/2026 | Chỉ còn hai chế độ huấn luyện `default` và `mcmc`: bỏ chế độ `mcmc_demo` (tham số notebook demo: 40.000 vòng, antialiased, `cap_max` cố định 3.500.000). `mcmc` không có số Gaussian cố định: `cap_max` luôn bằng số Gaussian cuối của lượt `default` tương ứng. Đổi tham số khi cần: `--set` / `--protocol` / `--cap-max` kèm `TAG`, hoặc tự thêm chế độ có tên vào `configs/base.yaml` (giữ cơ chế `cap_max` cố định cho trường hợp này) | Theo yêu cầu của bạn ngày 19/10/2026. `mcmc_demo` không luồng nào tự chạy, khác protocol nên không vào bảng so sánh công bằng, và `cap_max` 3,5 triệu chưa có cơ sở. Không đổi `default`, `mcmc`, protocol hay kết quả đã có. Cách đổi tham số: `docs/design/training_modes.md` mục 3, `docs/USAGE_GUIDE.md` mục 3.7 |

## Đề xuất chờ xác nhận

| Hạng mục | Đề xuất | Lý do |
|---|---|---|
| Loại cloud | Secure Cloud | Network volume chỉ dùng được với pod trên Secure Cloud |
| Lưu trữ | Network volume ≥ 100 GB, cùng data center với GPU | Dữ liệu tồn tại độc lập với pod. Volume mới: chọn data center có sẵn loại GPU ở dòng dưới |
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

### Repo này (19/10/2026, trong sandbox, không GPU)

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Cấu trúc repo (D7), cấu hình YAML, bốn lệnh chính, bố cục `/workspace` (D8) | Đã viết; đã kiểm thử trên CPU | Mục "Kiểm thử trong sandbox" bên dưới |
| Lệnh huấn luyện sinh ra cho Deep Blending | Đã kiểm thử bằng stub | Giống lệnh repo trước đã chạy trên pod ngày 17/10/2026, chỉ khác đường dẫn (`tests/unit/test_benchmarks.py::test_deep_blending_gsplat_commands_are_unchanged`) |
| Cấu hình YAML so với các file JSON cũ | Đã kiểm tra | So từng khóa `configs/*.yaml` với `configs/{datasets,train_modes}.json` và `experiments/*/benchmark.json` của repo trước: `train_modes` giống hệt; dữ liệu, kết quả, protocol, số liệu công bố, ngưỡng giống hệt. Chỉ khác mô tả, lệnh, trạng thái; bỏ `previous_locations` (chỗ cũ, dùng cho việc dời) và khóa đường dẫn `benchmark`; `dataset`, `scenes` của phần benchmark nay lấy từ chính file cấu hình khi xuất JSON |
| Render lại từ checkpoint, xuất PLY: `src/rendering/render.py`, `scripts/run_render.sh` | Đã viết; đã kiểm thử với runner giả và stub; **chưa chạy với gsplat thật** | `tests/unit/test_render.py`: dùng lại lệnh đánh giá đã ghi trong `run_meta.json` (chế độ `--ckpt` của `simple_trainer.py`), cùng seed và tập test, ghi vào thư mục mới; từ chối lượt chưa xong; PLY của bước cuối, liên kết cứng, không ghi đè |
| Giao diện dòng lệnh `python -m app.cli` | Đã viết; đã kiểm thử | `tests/unit/test_cli.py` |
| Notebook `app/notebooks/runpod.ipynb` cho pod mới, volume trống | Đã viết; đã mô phỏng; **chưa chạy trên pod** | Mục "Kiểm thử trong sandbox" |
| Luồng cảnh tự chụp với COLMAP thật trên cây này | Đã chạy với COLMAP 3.9.1 bản CPU trên dữ liệu tổng hợp, huấn luyện thay bằng stub | Mục "Kiểm thử trong sandbox" |
| Rà soát độc lập (một lượt, người rà soát không tham gia viết code) | Đã làm; lỗi chặn và lỗi vừa đã sửa, kèm test | Mục "Rà soát độc lập ngày 19/10/2026" bên dưới |

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

**Repo này, 19/10/2026** (pytest, shellcheck, pyflakes: lượt cuối sau D11; mô phỏng notebook: lượt sau D10, D11 không đổi notebook hay script chạy trên pod ngoài một dòng chú thích của `train_scene.sh`):
- **pytest:** 245 test đều qua trên bốn môi trường: Python 3.13.16 với numpy 2.5.4 / opencv 4.11.0.86; Python 3.13.16 và 3.10.20 cài đúng `requirements.txt` (numpy 2.2.6, opencv 5.0.0.93 như env `gs-inria`); Python 3.10.20 với numpy 1.26.4 / opencv 4.11.0.86 như env `gs-gsplat`. Không test nào bị bỏ qua (máy có `colmap`, `ffmpeg`). Commit đầu `f5b1216` có 251 test; D10 bỏ các test của phần tải / giải nén và của các bước benchmark trong `run_all.sh`, thêm test "thiếu dữ liệu thì dừng trước khi train, không tải gì".
- **shellcheck** 0.11.0 (`scripts/*.sh scripts/*/*.sh`) và **pyflakes** (`src app tests scripts/setup/*.py`): sạch.
- **Notebook** (`app/notebooks/runpod.ipynb`) trên `/workspace` giả, pod mới, volume trống. GitHub được thay bằng một bare repo cục bộ, `ssh` và `curl` là stub (không kết nối mạng; mọi lần gọi `curl` được ghi lại), không GPU, không conda. **54/54 kiểm tra đạt**:
  - không ô code nào có lệnh tải (`curl`, `wget`, `DOWNLOAD`, `zipfile`, `unzip`); cả lượt, `curl` chỉ được `env_probe` gọi để thử kết nối tới GitHub, gitlab.inria.fr, PyPI, kho PyTorch và conda (`-o /dev/null`); không có `data/downloads`;
  - ô 1 báo ĐẠT với GPU giả (RTX 4090, driver CUDA 13.0) và cảnh báo khi `python3` không có PyYAML;
  - ô 2 tạo khóa ed25519 (`600`, thư mục `700`), chỉ in khóa công khai, chạy lại không tạo khóa mới; `ssh` được gọi với đúng khóa, `IdentitiesOnly`, `known_hosts` trên volume;
  - ô 3 clone đúng commit, cây file giống hệt repo, giữ quyền thực thi; chạy lại thì `pull --ff-only`; code trên pod có thay đổi chưa commit thì từ chối (exit 1), không stash, thay đổi còn nguyên; đang có việc nền hoặc thư mục không phải repo Git thì từ chối;
  - ô 4 chạy nền `run_all.sh` (không tham số); tóm tắt chỉ có `env_probe` (FAIL vì không có GPU) và ba bước cài đặt (SKIP); không cài gì, không tạo `outputs/benchmarks` hay `data/benchmarks`; log và báo cáo nằm trong `outputs/`;
  - ô 6 trên volume trống: Deep Blending và Mip-NeRF 360 chỉ báo "chưa có" (không luồng nào bắt buộc), không tạo thư mục nào; thiếu PyYAML thì lỗi hiện rõ trong ô; sau khi đưa Deep Blending giả (đúng bố cục) lên thì báo OK;
  - ô 7: `WHAT = "check"` (mặc định) báo `dataset not found` kèm chỗ cần đưa lên, cho cả Deep Blending và Mip-NeRF 360; `gsplat` / `b0` với `CHAY = False` không chạy; `CHAY = True` khi chưa có dữ liệu thì báo ngay, không xếp hàng; `evaluate` khi chưa có kết quả báo rõ; `WHAT` không hợp lệ chỉ in hướng dẫn. Khi đã có dữ liệu: `check` báo OK; `gsplat` với `QUICK = True` thành việc nền `gsplat_quick_deep_blending` (thư mục `gsplat_quick`), `b0` thành `b0_deep_blending`, cả hai dừng sớm vì không có GPU / env và không tạo lượt nào; dữ liệu đã đưa lên không bị sửa;
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

### Rà soát độc lập ngày 19/10/2026

Người rà soát làm trên một bản sao, chạy lại test và lint, dựng lại từng lỗi bằng stub. Kết quả: 1 lỗi chặn, 8 lỗi vừa, khoảng 15 lỗi nhỏ / tài liệu. Đã sửa (đã kiểm thử trên CPU, chưa chạy trên pod):

| Mức | Lỗi | Đã sửa | Test |
|---|---|---|---|
| Chặn | `scripts/pipeline/train_scene.sh` đọc `configs/base.yaml (train_modes)` như một file JSON (lỗi khi đổi tên hàng loạt): **mọi lượt train cảnh tự chụp lỗi trước khi bắt đầu**. Test cũ không bắt được vì thay `train_scene.sh` bằng stub | Đọc `train_modes` bằng `src.training.profiles.load_config` | Mới: `tests/unit/test_train_scene.py` chạy `train_scene.sh` thật (3 test; cả 3 lỗi với bản cũ) |
| Vừa | B0 đọc cấu hình sau `conda activate gs-inria` (python3 của env không có PyYAML): chưa có env `gs-gsplat` thì B0 dừng | `scripts/lib/config.sh` giữ python3 tìm thấy lúc được source (trước khi kích hoạt env) | `tests/test_dataset.py` (lỗi với bản cũ) |
| Vừa | Xuất PLY trùng tên: lượt chạy thử và lượt chuẩn, hoặc lượt cũ trước khi train lại, làm lệnh xuất lỗi sau khi đã render | Lượt chạy thử xuất vào `checkpoints/<benchmark>_quick/`; tên đã là PLY khác thì xuất `<lượt>_<thời điểm train xong>.ply`; thiếu PLY thì dừng trước khi render | `tests/unit/test_render.py` |
| Vừa | Render lại một lượt sau khi cảnh đã làm lại COLMAP: checkpoint cũ với camera mới, ảnh và chỉ số sai mà không báo | So dấu SHA-256 của mô hình COLMAP với dấu ghi trong `run_meta.json`; khác thì dừng | `tests/unit/test_render.py` |
| Vừa | Ô 8 của notebook chạy GPU ngay khi chạy ô (hoặc "Run All"); ô dừng (lúc đó là ô 12, nay là ô 11) dừng mọi việc vừa được xếp hàng | Thêm `CHAY = False` cho ô 8, `DUNG=0` cho ô dừng | Mô phỏng notebook |
| Vừa | Lần giải nén Deep Blending bị ngắt giữa chừng (vd. dừng ô 6) được dùng lại như dữ liệu đủ | Đã sửa (dấu `.extracted`), sau đó bỏ hẳn theo D10: repo không còn tải hay giải nén gì | Test của phần giải nén bỏ cùng phần đó |
| Vừa | B0 với `FORCE=1` ghi đè kết quả cũ | Lượt cũ đổi tên `<cảnh>.prev_<thời gian>` trước khi chạy lại | `tests/unit/test_benchmarks.py` |
| Vừa | Tài liệu nói quá: test chạy `train_scene.sh` thật (sai), luồng cảnh tự chụp đã chạy với COLMAP thật (bước train chỉ là stub), mọi code đã chạy trên pod (phần viết sau 17/10 chưa chạy) | Sửa lại cho đúng (`docs/design/architecture.md` mục 3 và 7, `docs/design/training_modes.md` mục 8, `docs/USAGE_GUIDE.md` mục 9) | — |
| Nhỏ | Bảng đối chiếu tên cũ → mới ghi sai tên của lượt 17/10; `configs/local.yaml` sai dạng gây traceback, khóa `train_modes` trong đó bị bỏ qua lặng lẽ; phần hướng dẫn của vài script in cả dòng code; chú thích cũ (`PYTHONPATH=src`, `env_probe.sh`, `run_scene.sh`); `ssh-keygen` trong tài liệu hỏi lại khi khóa đã có; hai gói kết quả trong cùng một phút trùng tên; CLI chạy script trong thư mục repo thay vì thư mục hiện tại; `runs/` trong `.gitignore` không neo gốc | Đã sửa (`local.yaml` chỉ nhận mục `datasets` dạng mapping; tên gói có giây) | `tests/test_dataset.py` |

Giữ nguyên, có lý do:
- Render lại ghi một thư mục mới `render_<quỹ_đạo>_<thời gian>/` **bên trong** lượt gốc: không sửa file nào của lượt, giữ ảnh render cạnh lượt của nó; tài liệu đã ghi rõ.
- File PLY xuất là liên kết cứng tới PLY của lượt (không tốn thêm chỗ): tài liệu ghi chỉ đọc / tải về, không sửa tại chỗ.
- Gói `ket_qua_*.tar.gz` ở gốc `/workspace` (như sơ đồ).

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

- **Repo này chưa chạy trên pod.** Chưa chạy thật: khóa deploy với GitHub thật, clone qua SSH, cài 3 env trên volume trống, đọc dữ liệu benchmark do bạn đưa lên, B0 và gsplat bằng code mới, render lại bằng chế độ `--ckpt` của gsplat thật, mở PLY trên một trình xem 3DGS.
- Ảnh không có EXIF tiêu cự: hai lượt COLMAP trên cùng 24 ảnh tổng hợp cho hai mô hình khác nhau; một lượt có tiêu cự lệch hai trục (fx 689 / fy 500 px) và ảnh sau khử méo bị cắt còn 548×393, vẫn PASS. Cần thêm cảnh báo (vd. fx/fy lệch nhau, ảnh sau khử méo nhỏ hơn hẳn ảnh vào) và thử lại với ảnh thật trước khi dùng ảnh không EXIF.
- PyYAML trong `python3` của template Runpod: suy ra từ JupyterLab 4.2.5 có trong template (`jupyter_events` cần PyYAML), chưa kiểm tra trực tiếp. Ô 1 của notebook kiểm tra; `env_probe.sh` nay liệt kê PyYAML. Không có thì chỉ các lệnh đọc cấu hình chạy trước khi cài xong `gs-gsplat` bị ảnh hưởng, kèm thông báo cách sửa.
- Câu trả lời của GVHD cho Q1–Q3 và Q5–Q12 (`docs/PROJECT_PLAN.md`, mục 2): đang dùng giả định tạm.
- Cách chia train/test cho dữ liệu tự quay: code Inria gán cứng mỗi ảnh thứ 8 làm test; cần quyết định ở M4.
- Chưa chọn LICENSE cho repository.
- Cảnh tự chụp: chưa có lượt nào chạy xong xác minh được trên pod.
- Mip-NeRF 360:
  - chưa có dữ liệu trên volume (`bash scripts/pod/check_datasets.sh --dataset mipnerf360`);
  - `360_v2.zip` chỉ có 7 cảnh; flowers và treehill nằm trong gói riêng (cần đưa lên thêm nếu chạy đủ 9 cảnh);
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
- `configs/environment/*.conf` vẫn để `provisional` (`run_all.sh` đặt `ALLOW_PROVISIONAL=1`). Điều kiện để chốt (B0 đạt, smoke test của `gs-gsplat` PASSED trên pod) đã có từ 17/10/2026 với repo trước; chưa sửa file.

## Môi trường

Phương án, bố cục `/workspace`, sự cố đã biết: `docs/ENVIRONMENT_AUDIT.md`. Pod và phiên bản thực tế của các lượt chạy 17/10/2026 (repo trước): `docs/ENVIRONMENT_AUDIT.md` mục 7 (RTX 4090, EU-RO-1, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; driver CUDA 12.8 ở lượt đầu, 13.0 ở lượt thứ hai).

Lệnh chính: `docs/README.md` mục "Chạy trên pod Runpod"; từng bước và notebook: `docs/USAGE_GUIDE.md`; benchmark: `docs/design/experiments.md`.

## Dữ liệu, checkpoint, kết quả

Lưu trên network volume của Runpod tại `/workspace`, không đưa lên Git. Nơi lưu của từng luồng: `configs/*.yaml`; bố cục: `docs/design/architecture.md` mục 2.

- Volume mới: **chưa có gì**. Sau ô 3–4 của notebook, khi bạn đã đưa dữ liệu lên và chạy ô 7, 8:
  - code `/workspace/SpatialRoom3D`; env `/workspace/miniforge3`; mã nguồn Inria / gsplat `/workspace/code`;
  - Deep Blending (bạn tự đưa lên) `/workspace/data/benchmarks/deep_blending/{drjohnson,playroom}`; kết quả `/workspace/outputs/benchmarks/deep_blending/{inria_b0,gsplat,gsplat_quick}/`;
  - cảnh tự chụp: nguồn `/workspace/data/custom_scenes/<cảnh>/`, kết quả `/workspace/outputs/custom_scenes/<cảnh>/`;
  - PLY đã xuất `/workspace/outputs/checkpoints/`; báo cáo, log, trạng thái việc nền `/workspace/outputs/{reports,logs,jobs}`; gói kết quả nhẹ `/workspace/ket_qua_*.tar.gz`.
- Volume cũ (EU-RO-1): còn kết quả ngày 17/10/2026 của repo trước, ở bố cục cũ. Repo này không đọc và không sửa volume đó.

## Việc tiếp theo

**Trên pod mới, volume mới.** Hướng dẫn: `docs/USAGE_GUIDE.md` mục 1–2, `app/notebooks/runpod.ipynb`.

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| P1 | Tạo network volume ≥ 100 GB và pod (`docs/USAGE_GUIDE.md` mục 1); tải `app/notebooks/runpod.ipynb` lên `/workspace`; chạy ô 1 | Ô 1 báo `=> ĐẠT`; có dòng `✓ python3 có PyYAML` (không có thì cài theo hướng dẫn trong ô) |
| P2 | Ô 2; thêm khóa công khai vào Deploy keys của `SpatialRoom3D` (không tick write); chạy lại ô 2 | Dòng cuối `✓ GitHub nhận khóa` |
| P3 | Ô 3 | `Đã clone ... (nhánh main)`; commit in ra trùng commit mới nhất trên GitHub |
| P4 | Ô 4, theo dõi ở ô 5 (khoảng 80 phút) | Việc `run_all` exit 0: `env_probe`, `setup_inria`, `data_tools`, `setup_gsplat` OK, smoke test PASSED; không có bước train nào |
| P5 | Bạn tự đưa Deep Blending lên `/workspace/data/benchmarks/deep_blending/` (không cần GPU, làm được song song với P4), chạy ô 6 | Deep Blending OK (drjohnson 263 ảnh, playroom 225 ảnh). Ô 6 chỉ kiểm tra bố cục và số ảnh, không kiểm tra tổng kiểm |
| P6 | Ô 7, `BENCH = "deep_blending"`, `CHAY = True`: `WHAT = "gsplat"` với `QUICK = True` (chạy thử), rồi `WHAT = "b0"`, rồi `WHAT = "gsplat"` (tổng khoảng 2,4 giờ GPU, **cần duyệt**) | B0 đạt ngưỡng so với số liệu Inria công bố; `default` PASS so với B0 từng cảnh; ghi lại chênh lệch so với lượt 17/10/2026 của repo trước |
| P7 | `bash scripts/run_render.sh deep_blending playroom default_s42` (render lại từ checkpoint, cần GPU, vài phút) | `render_meta.json` exit 0; PSNR / SSIM / LPIPS của lần render bằng số của lượt gốc (cùng checkpoint, cùng tập test); có video; PLY ở `outputs/checkpoints/deep_blending/playroom/default_s42.ply` |
| P8 | Thu một cảnh (≥ 30 ảnh theo `docs/data/capture_protocol.md` mục 8, hoặc video) vào `data/custom_scenes/<cảnh>/`, chạy ô 8, xem ô 9 | `sfm_report.json` không FAIL; có `summary.json` cho cả `default` và `mcmc`; mô hình xem được trên trình xem 3DGS; `coverage` khớp với cách chụp thực tế |
| P9 | Ô 7: `WHAT = "gsplat"`, `SEEDS = "42 43 44"`, `CHAY = True` (khoảng 2–3 giờ GPU, cần duyệt) | Bảng so sánh có ≥ 3 seed chung, mọi mục điều kiện OK; chỉ khi đó mới nói về cải thiện của `mcmc` |
| P10 | Mip-NeRF 360, chỉ khi bạn đã đưa dữ liệu lên (khoảng 12 GB) và được duyệt chi phí GPU | Ô 7 với `BENCH = "mipnerf360"`, `WHAT = "check"` OK; chạy `QUICK = True` trước; số liệu đầy đủ đối chiếu được với trung bình công bố khi đủ 9 cảnh |

**Không cần GPU**

| Bước | Việc | Tiêu chí nghiệm thu |
|---|---|---|
| — | Chọn trình xem 3DGS cho file PLY, rồi viewer web bản thô (`app/web/`, giai đoạn 5) | Mở được mô hình mẫu trên trình duyệt; có số đo thời gian tải; có kiểm thử chuyển đổi định dạng |
| — | Thu thử 1 cảnh: quay video theo `docs/data/capture_protocol.md` (video chính + đoạn quay kiểm tra), hoặc chụp ≥ 30 ảnh theo mục 8 | Đặt tên và ghi metadata đúng quy ước; ảnh cùng chiều máy, cùng ống kính, dùng file gốc |
| — | Chọn LICENSE; quyết định chế độ hiển thị của repo cũ | Có file LICENSE; repo cũ private hoặc archive |
