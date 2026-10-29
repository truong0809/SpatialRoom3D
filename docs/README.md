# Tài liệu

Giới thiệu dự án, luồng xử lý và cách chạy nhanh: [README](../README.md) ở gốc repo.

## Mục lục

| Tài liệu | Nội dung |
|---|---|
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Hướng dẫn sử dụng trên Runpod: chuẩn bị volume và pod, notebook, cảnh tự chụp, xuất PLY, benchmark, giao diện dòng lệnh, xử lý lỗi |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Trạng thái hiện tại, quyết định đã chốt, việc đã làm kèm bằng chứng, việc còn mở, việc tiếp theo |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Đề bài, phạm vi, phương pháp, giao thức đánh giá, mốc nghiệm thu, rủi ro |
| [ENVIRONMENT_AUDIT.md](ENVIRONMENT_AUDIT.md) | Môi trường trên pod, phiên bản ghim, cài đặt và kiểm tra, sự cố đã biết; số đo GPU tham khảo ở mục 7 (chưa phải lượt chạy của repo này) |
| [design/architecture.md](design/architecture.md) | Cây thư mục, bố cục `/workspace`, ba luồng dữ liệu, khai báo đường dẫn, nguồn gốc mã nguồn |
| [design/training_modes.md](design/training_modes.md) | Hai chế độ huấn luyện `default` và `mcmc`: protocol, giao thức so sánh, thành phần |
| [design/experiments.md](design/experiments.md) | Benchmark (Deep Blending, Mip-NeRF 360), thực nghiệm chính, dò tham số `mcmc`, lọc Gaussian sau khi train |
| [design/comparisons.md](design/comparisons.md) | Bảng và biểu đồ so sánh các chế độ (`scripts/run_evaluate.sh`, `src/evaluation/charts.py`) |
| [data/capture_protocol.md](data/capture_protocol.md) | Quy trình quay video và chụp ảnh không gian nội thất |
| [research/reading_list.md](research/reading_list.md) | Danh sách tài liệu khảo sát |
| [research/demo_review.md](research/demo_review.md) | Kiểm tra notebook demo 3DGS-MCMC của nhóm và repo SpatialScene3D (bản gốc của notebook: [research/references/demo_3dgs_mcmc/](research/references/demo_3dgs_mcmc/README.md)) |
| [handoff/2026-10-28_d17_per_method_runs.md](handoff/2026-10-28_d17_per_method_runs.md) | Bàn giao D17: chạy riêng từng phương pháp của thực nghiệm chính (B0, `default`, `mcmc`), tên bản; thiết kế, thay đổi, kiểm thử, việc còn mở |

Protocol và lệnh của từng benchmark: [experiments/deep_blending/](../experiments/deep_blending/README.md), [experiments/mipnerf360/](../experiments/mipnerf360/README.md).

## Quy ước

- Trạng thái một hạng mục đi theo bốn mức: `Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`. Chỉ mức cuối mới gọi là hoàn thành.
- Số liệu luôn kèm nguồn (file kết quả, cấu hình, commit); số công bố và số tự chạy ghi tách bạch.
- Tài liệu mô tả hiện trạng của repo. Quyết định và lý do được ghi ở [PROJECT_STATUS.md](PROJECT_STATUS.md), mục "Quyết định đã chốt".
