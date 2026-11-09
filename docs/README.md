# Tài liệu

Giới thiệu dự án theo bốn mục của báo cáo tuần 10 (cấu trúc mã nguồn, cài đặt, phần nhóm làm, thực nghiệm và kết quả sơ bộ): [README](../README.md) ở gốc repo.

## Mục lục

| Tài liệu | Nội dung |
|---|---|
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Hướng dẫn sử dụng trên Runpod: chuẩn bị volume và pod, notebook, cảnh tự chụp, xuất PLY, benchmark, giao diện dòng lệnh, xử lý lỗi |
| [RESULTS.md](RESULTS.md) | Kết quả tóm tắt: lượt chạy 31/10/2026 (bảng số, biểu đồ, ảnh so sánh trong `results/`) và sản phẩm chạy trên pod 08/11/2026 |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Trạng thái hiện tại, quyết định đã chốt, việc đã làm kèm bằng chứng, việc còn mở, việc tiếp theo |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Đề bài, phạm vi, phương pháp, giao thức đánh giá, mốc nghiệm thu, rủi ro |
| [ENVIRONMENT_AUDIT.md](ENVIRONMENT_AUDIT.md) | Môi trường trên pod, phiên bản ghim, cài đặt và kiểm tra, sự cố đã biết; các lượt chạy trên pod (31/10/2026, 08/11/2026) và số đo của repo trước ở mục 7 |
| [design/architecture.md](design/architecture.md) | Cây thư mục, bố cục `/workspace`, ba luồng dữ liệu, khai báo đường dẫn, nguồn gốc mã nguồn |
| [design/training_modes.md](design/training_modes.md) | Hai chế độ huấn luyện `default` và `mcmc`: protocol, giao thức so sánh, thành phần |
| [design/experiments.md](design/experiments.md) | Benchmark (Deep Blending, Mip-NeRF 360), thực nghiệm chính, dò tham số `mcmc`, lọc Gaussian sau khi train |
| [design/product.md](design/product.md) | Sản phẩm: mô hình chính (`configs/product.yaml`), dựng phòng từ video / ảnh (`scripts/build_room.sh`), hợp đồng với web: `manifest.json`, `model.splat`, `cameras.json`, `progress.json` |
| [../app/web/README.md](../app/web/README.md) | Trang web của sản phẩm: chạy trên pod (`scripts/web_server.sh`), xem trên laptop không cần pod, các màn, code, kiểm thử |
| [design/comparisons.md](design/comparisons.md) | Bảng và biểu đồ so sánh các chế độ (`scripts/run_evaluate.sh`, `src/evaluation/charts.py`) |
| [data/capture_protocol.md](data/capture_protocol.md) | Quy trình quay video và chụp ảnh không gian nội thất |
| [research/reading_list.md](research/reading_list.md) | Danh sách tài liệu khảo sát |
| [research/demo_review.md](research/demo_review.md) | Kiểm tra notebook demo 3DGS-MCMC của nhóm và repo SpatialScene3D (bản gốc của notebook: [research/references/demo_3dgs_mcmc/](research/references/demo_3dgs_mcmc/README.md)) |

Protocol và lệnh của từng benchmark: [experiments/deep_blending/](../experiments/deep_blending/README.md), [experiments/mipnerf360/](../experiments/mipnerf360/README.md).

## Quy ước

- Trạng thái một hạng mục đi theo bốn mức: `Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`. Chỉ mức cuối mới gọi là hoàn thành.
- Số liệu luôn kèm nguồn (file kết quả, cấu hình, commit); số công bố và số tự chạy ghi tách bạch.
- Tài liệu mô tả hiện trạng của repo. Quyết định và lý do được ghi ở [PROJECT_STATUS.md](PROJECT_STATUS.md), mục "Quyết định đã chốt".
