# SpatialRoom3D — Tái tạo không gian nội thất 3D bằng ảnh chụp

**3D Indoor Scene Reconstruction from Images** — Đồ án tốt nghiệp

Hệ thống nhận ảnh hoặc video quay không gian nội thất bằng camera thông thường (không yêu cầu LiDAR), tái tạo cảnh 3D bằng **3D Gaussian Splatting (3DGS)**, xuất mô hình để xem và tương tác trên trình xem 3D, và đánh giá chất lượng bằng thực nghiệm.

> **Trạng thái (19/10/2026):** repo mới, bắt đầu từ bản cấu trúc hoàn chỉnh; **chưa chạy trên pod**, chưa có kết quả nào do repo này tạo ra.
> - Ngày 17/10/2026, repo trước (`3D-Indoor-Scene-Reconstruction-from-Images`, commit `8037a66` kèm bản vá chưa commit) đã tái lập B0 (3DGS gốc của Inria) trên Deep Blending bằng RTX 4090, khớp số liệu Inria công bố trong ngưỡng, và chạy hai chế độ gsplat (Default / MCMC, seed 42). Các lượt đó nằm trên volume cũ; repo này sẽ chạy lại trên volume mới.
> - Chi tiết: [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Luồng xử lý chính (baseline)

```
Ảnh / Video (RGB) → Validate Input → Extract Frames (video) → COLMAP (SfM) → Prepare Dataset → Train 3DGS → Export PLY / Checkpoint → Evaluate (PSNR / SSIM / LPIPS)
   data/custom_scenes/   src/input/        src/input/frames      src/reconstruction/          src/training/       src/rendering/            src/evaluation/
   scripts/prepare_data.sh ────────────────────────────────────────────────┘   scripts/run_train.sh   scripts/run_render.sh   scripts/run_evaluate.sh
```

- Ảnh chụp dùng thẳng (một camera chung); video được chọn frame nét. Trước khi train có bước kiểm tra: số ảnh, tỉ lệ ảnh COLMAP ghép được, tách nhóm, độ phủ góc nhìn.
- Hai chế độ huấn luyện trên gsplat, so sánh cùng điều kiện: `default` (Default Strategy, baseline) và `mcmc` (3DGS-MCMC, MCMC Strategy, cùng số Gaussian).
- B0: 3DGS gốc của Inria, chỉ để đối chiếu số liệu công bố trên benchmark.

Phương pháp và kế hoạch: [PROJECT_PLAN.md](PROJECT_PLAN.md). Thiết kế hai chế độ: [design/training_modes.md](design/training_modes.md). Kiến trúc, bố cục `/workspace`, ba luồng dữ liệu: [design/architecture.md](design/architecture.md).

## Cấu trúc repository

```
SpatialRoom3D/
├── app/                      ứng dụng
│   ├── cli/                  giao diện dòng lệnh: python -m app.cli <lệnh>
│   ├── web/                  giao diện web (chưa làm, giai đoạn 5)
│   ├── notebooks/            runpod.ipynb: chạy trên pod Runpod
│   └── __init__.py
├── src/                      mã nguồn chính (package `src`)
│   ├── input/                ảnh / video đầu vào: kiểm tra, chọn frame
│   ├── reconstruction/       COLMAP (SfM), kiểm tra trước khi train
│   ├── training/             huấn luyện 3DGS trên gsplat (default / mcmc)
│   ├── rendering/            render ảnh / video từ checkpoint, xuất PLY
│   ├── evaluation/           chỉ số, so sánh, xem nhanh kết quả
│   ├── datasets/             đọc configs/*.yaml, kiểm tra dữ liệu từng luồng
│   ├── utils/                khóa GPU dùng chung với script bash
│   └── __init__.py
├── configs/                  cấu hình (đổi tham số không cần sửa code)
│   ├── base.yaml             phần chung: thư mục code, hai chế độ huấn luyện và protocol
│   ├── custom_scene.yaml     cảnh tự chụp
│   ├── mipnerf360.yaml       Mip-NeRF 360
│   ├── deep_blending.yaml    Deep Blending
│   └── environment/          phiên bản ghim của các env conda
├── scripts/                  script chạy trên Runpod
│   ├── prepare_data.sh       chuẩn bị dữ liệu đầu vào
│   ├── run_train.sh          huấn luyện
│   ├── run_render.sh         render ảnh / video, xuất PLY
│   ├── run_evaluate.sh       đánh giá chất lượng
│   ├── pipeline/             phần thực thi của các bước trên
│   ├── pod/                  chạy cả luồng, chạy nền / xếp hàng GPU, xem / đóng gói kết quả
│   ├── setup/                cài và kiểm tra môi trường
│   └── lib/                  hàm bash dùng chung
├── tests/
│   ├── test_dataset.py       kiểm tra cấu hình và dữ liệu
│   ├── test_pipeline.py      kiểm tra luồng chính của cảnh tự chụp
│   ├── unit/                 kiểm thử chi tiết từng phần
│   └── helpers/              tiện ích cho test
├── docs/
│   ├── README.md             (file này)
│   ├── ENVIRONMENT_AUDIT.md  môi trường, phiên bản, kiểm tra trên pod
│   ├── USAGE_GUIDE.md        hướng dẫn sử dụng
│   ├── PROJECT_PLAN.md       kế hoạch phát triển
│   ├── PROJECT_STATUS.md     trạng thái hiện tại, bằng chứng, việc tiếp theo
│   ├── design/, data/, research/
├── experiments/
│   ├── deep_blending/        protocol, lệnh và kết quả của benchmark (dữ liệu không đưa lên Git)
│   └── mipnerf360/
├── .gitignore
└── requirements.txt
```

Dữ liệu và kết quả nằm trên network volume, ngoài repo, không đưa lên Git:

```
/workspace/
├── SpatialRoom3D/            repo này
├── data/                     custom_scenes/<cảnh>/ (ảnh, video tự chụp), benchmarks/{deep_blending,mipnerf360}/
└── outputs/                  custom_scenes/, benchmarks/, checkpoints/ (PLY đã xuất), logs/, reports/, jobs/
```

Ảnh / video tự chụp và dữ liệu benchmark (Deep Blending, Mip-NeRF 360) do bạn tự đưa lên `data/` ([USAGE_GUIDE.md](USAGE_GUIDE.md) mục 2.1); chạy với dữ liệu nào là do bạn chọn.

## Chạy trên pod Runpod

Cách dễ nhất: notebook `app/notebooks/runpod.ipynb`, các bước trong [USAGE_GUIDE.md](USAGE_GUIDE.md). Bằng lệnh, từ `/workspace/SpatialRoom3D`:

```bash
bash scripts/pod/run_all.sh --background                      # pod mới: chỉ cài / kiểm tra môi trường (không train)
bash scripts/pod/job.sh status                                # theo dõi việc chạy nền (log: bash scripts/pod/job.sh log <tên>)
bash scripts/pod/check_datasets.sh                            # dữ liệu và kết quả của mọi luồng (chỉ đọc)
bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/phong1 phong1 --background   # cả luồng cho một cảnh tự chụp
bash scripts/prepare_data.sh custom /workspace/data/custom_scenes/phong1 phong1          # từng bước: chuẩn bị
bash scripts/run_train.sh custom phong1                                                  #             train default, mcmc
bash scripts/run_render.sh custom phong1 default_s42                                     #             render, xuất PLY
bash scripts/run_evaluate.sh custom phong1                                               #             so sánh
bash scripts/prepare_data.sh deep_blending                    # benchmark: kiểm tra dữ liệu bạn đã đưa lên (không dùng GPU)
bash scripts/run_train.sh deep_blending gsplat                #            train (experiments/deep_blending/README.md)
bash scripts/pod/results.sh [cảnh]                            # xem nhanh kết quả
bash scripts/pod/job.sh stop --all                            # dừng mọi việc chạy nền
python3 -m app.cli --help                                     # cùng các lệnh trên, qua một giao diện dòng lệnh
```

- Chạy lại an toàn: phần đã xong tự bỏ qua. Mỗi lúc chỉ một việc dùng GPU; việc mới tự xếp hàng.
- Kết quả nhẹ để tải về (báo cáo, log, chỉ số; không có checkpoint / PLY / video): `/workspace/ket_qua_*.tar.gz` (`bash scripts/pod/pack_results.sh`).
- Môi trường, cấu hình pod, sự cố đã biết: [ENVIRONMENT_AUDIT.md](ENVIRONMENT_AUDIT.md).

## Kiểm thử trên máy không có GPU

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # vài test tích hợp cần ffmpeg / colmap; tự bỏ qua nếu thiếu
```

Test các script bash (chạy nền, `run_all.sh`, `run_scene.sh`, script benchmark, bước kiểm tra của script cài đặt) chạy bash thật với stub thay cho GPU / conda, cần Linux có `flock`, `setsid` và `/proc`.

## Lịch sử

Repo này bắt đầu ngày 19/10/2026 từ bản cấu trúc hoàn chỉnh của repo trước `truong0809/3D-Indoor-Scene-Reconstruction-from-Images` (commit `8037a66` cùng các thay đổi chưa commit ngày 17–19/10/2026). Lịch sử trước đó nằm ở repo cũ.

## Quy ước

- Không đưa dataset, checkpoint, file `.ply`, video, ảnh hay log thô lên Git.
- Không đưa token, mật khẩu, API key hoặc file `.env` lên Git.
- Mọi số liệu trong báo cáo phải truy được về file kết quả, cấu hình và commit tương ứng.
- Dữ liệu trong `tests/` là dữ liệu tổng hợp để kiểm tra logic, không phải kết quả thực nghiệm.
