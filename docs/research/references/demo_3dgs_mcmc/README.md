# Notebook demo 3DGS-MCMC của nhóm (tài liệu tham khảo)

Các notebook Colab của nhóm, giữ nguyên bản nhận được. Chúng **không** thuộc pipeline và không được chạy trong repo; được lưu để đối chiếu nguồn gốc của chế độ `mcmc` và các biến thể dò tham số.

## Demo 1: `3D_demo.ipynb`

| | |
|---|---|
| File | `3D_demo.ipynb` — giữ nguyên bản nhận được, kể cả output của lần chạy trên Colab |
| Nhận ngày | 09/10/2026 |
| SHA-256 | `618277f58b878a940500b5a49f5d51fa911d23862effbd17683a6efd87d22380` |
| Nguồn tham khảo của demo | [Ziro21/SpatialScene3D](https://github.com/Ziro21/SpatialScene3D) và paper 3DGS-MCMC ([arXiv 2404.09591](https://arxiv.org/abs/2404.09591)) |
| Môi trường đã chạy (theo output) | Google Colab, NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu130, gsplat 1.3.0, COLMAP 3.9.1 (apt) |

- Kết quả kiểm tra và phân loại tái sử dụng / sửa / bỏ: [docs/research/demo_review.md](../../demo_review.md).
- Phần huấn luyện MCMC của demo được tích hợp thành chế độ `mcmc` (cùng protocol, cùng số Gaussian với Default): [docs/design/training_modes.md](../../../design/training_modes.md).
- Chỉ số trong output của notebook (PSNR 39,97 dB, SSIM 0,982, LPIPS 0,037) **không** được dùng làm kết quả của đồ án: LPIPS tính bằng AlexNet, ảnh test xen kẽ trong cùng video và chỉ có một lần chạy, không có baseline trên cùng dữ liệu.

## Demo v2: `3D_demo_v2.ipynb`

| | |
|---|---|
| File | `3D_demo_v2.ipynb` — giữ nguyên bản nhận được |
| Nhận ngày | 24/10/2026 (gửi lại ngày 10/11/2026, trùng từng byte) |
| SHA-256 | `4be69cb9b32e5f05d75778b4c6c3f062edfe5a31d9a365b5dac93bd8176fd933` |
| Nội dung | Video phòng → chọn khung → COLMAP → 3DGS-MCMC thêm Kill-then-Relocate, bilateral grid, depth prior (Depth Anything V2), 7.000 vòng → lọc floater → `.ply`, `.splat`, trình xem HTML; một ô ablation tùy chọn |
| Output | Không có (file nhận được không kèm output, nên không có số liệu của notebook) |
| Môi trường ghi trong file | Google Colab, GPU A100 (metadata `gpuType`) |
| Bản v4 | `3D_demo_v4_improved_30k.ipynb` (nhận ngày 24/10/2026, SHA-256 `8929ea692e6f935bf6efd1beb96f57204dcd165aa10d3bc26bdb6160c6f67f06`) giống hệt v2, chỉ khác một dòng `MAX_STEPS = 30_000`; không lưu riêng |

- Các thành phần v2 thêm vào `mcmc` được đưa về giao thức chung (30.000 vòng, cùng số Gaussian) thành các biến thể `kr`, `mono`, `bilagrid` và gói `v2` của lượt dò tham số: [docs/design/experiments.md](../../../design/experiments.md) mục 3.7. Kill-then-Relocate là heuristic của notebook, không có paper.
- Kết quả trên pod ngày 31/10/2026 (`room`, 1 seed): `kr`, `mono` trong mức dao động; `bilagrid`, `v2` kém `mcmc` quá ngưỡng ([docs/RESULTS.md](../../../RESULTS.md) mục 5).
