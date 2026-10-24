# Mip-NeRF 360

**Trạng thái: đã cấu hình, chưa chạy. Chưa có dữ liệu trên volume.**

Thư mục này chỉ chứa tài liệu của benchmark. Dữ liệu và kết quả nằm trên volume, không đưa lên Git.

## Dữ liệu

- Cấu hình: `configs/mipnerf360.yaml`. Mặc định `$WS/data/benchmarks/mipnerf360/<cảnh>`, mỗi cảnh có `images/`, `images_2/`, `images_4/`, `sparse/0/`. Kết quả: `$WS/outputs/benchmarks/mipnerf360/{inria_b0,gsplat,gsplat_quick}/`.
- 9 cảnh: 4 trong nhà (room, counter, kitchen, bonsai), 5 ngoài trời (bicycle, garden, stump, flowers, treehill).
- Kiểm tra (chỉ đọc): `bash scripts/pod/check_datasets.sh --dataset mipnerf360`; kiểm tra bố cục từng cảnh: `bash scripts/prepare_data.sh mipnerf360`.
- Dữ liệu nằm chỗ khác hoặc tên khác (vd. `MipNeRF360`): tạo `configs/local.yaml`, hoặc đặt `SR3D_DATA_MIPNERF360=<đường dẫn>` (`docs/design/architecture.md` mục 5). Không cần sao chép dữ liệu.
- Tải về pod: `bash scripts/pod/fetch_data.sh mipnerf360 [cảnh ...]` (mục `download` của cấu hình; URL lấy từ `examples/datasets/download_dataset.py` của gsplat @6e8c837).
  - `360_v2.zip` có 7 cảnh; flowers và treehill nằm trong gói riêng `360_extra_scenes.zip` (bố cục bên trong chưa kiểm tra: thiếu thư mục cảnh thì script dừng trước khi giải nén).
  - Chọn vài cảnh vẫn phải tải cả file zip; script chỉ giải nén `images/`, `images_2/`, `images_4/`, `sparse/` của cảnh được chọn (bỏ `images_8/`, `poses_bounds.npy`) rồi xóa file zip.
  - Không có SHA-256 công bố: script đo và ghi lại SHA-256 của file đã tải; CRC của từng file được kiểm khi giải nén.
- Hoặc tự đưa các cảnh lên chỗ trên. Thiếu cảnh hay thiếu `images_N/` thì lệnh dừng trước khi train; chỉ có một phần cảnh thì chạy với `SCENES="..."`.

## Protocol (đã đối chiếu mã nguồn)

| | Inria `full_eval.py` @54c035f | gsplat `examples/benchmarks/basic.sh` @6e8c837 |
|---|---|---|
| Cảnh trong nhà | `-i images_2` | `data_factor 2` |
| Cảnh ngoài trời | `-i images_4` | `data_factor 4` |
| Tập test | `--eval`: cứ 8 ảnh lấy 1 | `test_every 8` |

- Hệ số theo cảnh nằm ở `benchmark.image_factor` trong `configs/mipnerf360.yaml`.
- Với gsplat, hệ số này là **quy ước của benchmark** (`--dataset-protocol data_factor=N`), không bị ghi là sai khác protocol. Vì vậy `default` và `mcmc` trên cùng cảnh so được với nhau và với số liệu công bố.
- gsplat với `data_factor` > 1 tự thu nhỏ ảnh gốc thành PNG và ghi vào thư mục dữ liệu nó được đưa, rồi ghép với `images/` theo thứ tự tên mọi file trong thư mục PNG. `src.training.run` đưa cho gsplat một bản xem dữ liệu (liên kết tới cảnh gốc), và PNG nằm ở `$WS/outputs/benchmarks/mipnerf360/gsplat/<cảnh>/.images_<N>_png_<dấu mô hình COLMAP>`, dùng chung cho mọi chế độ và seed. Dấu "cache đủ" nằm cạnh thư mục đó, không nằm trong; cache dở dang, có file lạ hoặc thiếu ảnh thì bị xóa và ghi lại. Bộ dữ liệu gốc không bị ghi thêm gì.
- Hai bản cài đặt dùng ảnh khác nhau, theo đúng benchmark chính chủ của mỗi bên:
  - Inria đọc ảnh JPG thu nhỏ sẵn trong `images_N/`;
  - gsplat thu nhỏ lại từ `images/` (bicubic, PNG).
  Khi so `default` (gsplat) với B0 (Inria), cần ghi rõ điểm khác này.
- MCMC giữ `opacity_reg` mặc định của preset mcmc trong gsplat (0,01); chỉ Deep Blending dùng 0,001.
- Chạy một phần cảnh (`SCENES`, vd. 4 cảnh trong nhà): bảng so sánh không đối chiếu với trung bình công bố (ghi SKIPPED), và chỉ đối chiếu B0 khi mọi cảnh được chọn đều có lượt B0.
- Thực nghiệm chính dùng 4 cảnh trong nhà, không chạy B0 ở đây: `docs/design/experiments.md` mục 2.
- Dò tham số `mcmc` (`configs/experiments/tune_mcmc.yaml`) chỉ dùng `room`, seed 42: `default`, `mcmc` và 12 biến thể của `mcmc` trong `gsplat/room/<chế_độ>[_<tag>]_s42/`, kèm bản đã lọc Gaussian trong `filtered/` của mỗi lượt (`docs/design/experiments.md` mục 3, 4). Các biến thể có tag nên bảng của thực nghiệm chính không lẫn chúng. `room` là cảnh dùng để chọn tham số: bảng tự động vẫn tính `room`; kết luận về chế độ chọn ra (`mcmc_indoor`) dùng bảng không có `room` (`docs/design/experiments.md` mục 3.1). Vòng ghép chọn `mcmc_indoor` cũng chạy trên `room`, với seed 43, 44 (mục 3.6).
- B0: lần đầu mở một cảnh, `train.py` của Inria tạo `sparse/0/points3D.ply` trong thư mục dữ liệu (bộ này không có sẵn file đó). Đây là hành vi của bản gốc, giữ nguyên để đúng baseline; dấu mô hình COLMAP không đổi vì chỉ tính trên file `.bin`.

**Số liệu đối chiếu:** trung bình Inria công bố cho Mip-NeRF 360: PSNR 27,572, SSIM 0,816, LPIPS 0,215 (biểu đồ `assets/charts/base_*.png` trong `results.md` của Inria @54c035f, cấu hình ORIGINAL, rasterizer mặc định; đọc ngày 18/10/2026). So với trung bình này chỉ có nghĩa khi chạy đủ 9 cảnh.

## Lệnh

```bash
cd /workspace/SpatialRoom3D
bash scripts/pod/fetch_data.sh mipnerf360 room counter kitchen bonsai   # tải 4 cảnh trong nhà nếu chưa có
bash scripts/prepare_data.sh mipnerf360                       # trước tiên: dữ liệu phải báo OK (SCENES="..." khi chỉ có một phần)
QUICK=1 bash scripts/run_train.sh mipnerf360 gsplat           # chạy thử 3.000 vòng
bash scripts/run_train.sh mipnerf360 b0                       # B0 (9 cảnh; chạy trước gsplat để bảng đối chiếu đủ)
bash scripts/run_train.sh mipnerf360 gsplat                   # default + mcmc, bảng so sánh
SCENES="room counter kitchen bonsai" bash scripts/run_train.sh mipnerf360 gsplat   # chỉ 4 cảnh trong nhà
bash scripts/run_evaluate.sh mipnerf360
bash scripts/pod/run_experiment.sh tune_mcmc --plan           # dò tham số mcmc trên room: kế hoạch, các bước
```

Chạy trên GPU khi được duyệt. Trên RTX 4090, Deep Blending mất khoảng 13–20 phút mỗi cảnh mỗi chế độ (số đo ở `docs/ENVIRONMENT_AUDIT.md` mục 7). Mip-NeRF 360 chưa chạy nên chưa có số đo thời gian (9 cảnh).

## Đã kiểm tra những gì

- Lệnh sinh ra (hệ số ảnh theo cảnh, không có tham số mcmc riêng, dừng khi thiếu `images_N/`, báo cáo khi chỉ có một phần cảnh): `tests/unit/test_benchmarks.py`, chạy với stub.
- Biến thể tham số của `tune_mcmc` (thư mục `<chế_độ>_<tag>_s<seed>`, cùng `cap_max` lấy từ `default`, tham số của biến thể đứng sau quy ước mcmc của benchmark): `tests/unit/test_benchmarks.py`, `tests/unit/test_experiment.py`, chạy với stub.
- Tải và giải nén (chỉ cảnh và thư mục cần dùng, bỏ file ẩn / symlink, dừng khi đường dẫn không an toàn hoặc thiếu thư mục): `tests/unit/test_fetch.py`, với curl giả và file zip tổng hợp; **chưa tải file thật**.
- Bản xem dữ liệu, cache PNG dùng chung, cache dở dang hoặc có file lạ bị xóa, hai cảnh không dùng chung cache, cùng cấu hình thì bỏ qua: `tests/unit/test_train_run.py`, với runner giả ghép ảnh đúng như gsplat 6e8c837 (bắt được lỗi khi dấu `.complete` nằm trong thư mục PNG).
- **Chưa chạy** với gsplat thật hay dữ liệu thật.
