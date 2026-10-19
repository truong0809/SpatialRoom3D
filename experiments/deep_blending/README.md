# Deep Blending

Thư mục này chỉ chứa tài liệu của benchmark. Dữ liệu và kết quả nằm trên volume, không đưa lên Git (`.gitignore` chặn mọi file khác trong `experiments/<benchmark>/`).

- **Cấu hình:** `configs/deep_blending.yaml` (dữ liệu, nguồn tải, nơi lưu kết quả, mục `benchmark`: protocol, số liệu công bố, ngưỡng).
- **Dữ liệu:** `$WS/data/benchmarks/deep_blending/{drjohnson,playroom}` (263 và 225 ảnh), mỗi cảnh có `images/` và `sparse/0/`: thư mục `db/` trong `tandt_db.zip` mà Inria cung cấp.
  - Tải về pod: `bash scripts/pod/fetch_data.sh deep_blending` (file 650 MB của Inria; kiểm SHA-256 với giá trị đo ngày 17/10/2026; chỉ giải nén `db/<cảnh>/images` và `sparse`, bỏ `tandt/`; xóa file zip sau khi giải nén).
  - Hoặc tự đưa hai cảnh lên chỗ trên, hoặc trỏ cấu hình tới chỗ đã có (`configs/local.yaml` hoặc `SR3D_DATA_DEEP_BLENDING`, `docs/design/architecture.md` mục 5).
  - Thiếu thì lệnh train dừng trước khi train, báo `dataset not found` kèm lệnh tải.
- **Kết quả:** `$WS/outputs/benchmarks/deep_blending/{inria_b0,gsplat,gsplat_quick}/`; báo cáo `$WS/outputs/reports/{baseline_db_summary,modes_db,compare_db}_<thời_gian>.*`.
- **Protocol:**
  - tập test cứ 8 ảnh lấy 1 (`--eval` của Inria = `test_every 8` của gsplat, cùng một tập; đã đối chiếu mã nguồn);
  - ảnh có sẵn của bộ dữ liệu, không thu nhỏ thêm (`data_factor 1`);
  - 30.000 vòng; LPIPS VGG.
- **Số liệu đối chiếu:** trung bình Inria công bố cho Deep Blending (PSNR 29,690, SSIM 0,906, LPIPS 0,238; nguồn ghi trong cấu hình); ngưỡng 0,5 dB / 0,01 / 0,01.
- **MCMC:** `opacity_reg=0.001` cho mọi chế độ preset `mcmc` (`mcmc` và chế độ tự thêm cùng preset). Paper 3DGS-MCMC dùng giá trị này cho Deep Blending; config chính chủ chỉ đặt cho drjohnson, ở đây dùng cho cả hai cảnh như lượt 17/10/2026. Cần ghi rõ khi báo cáo.
- **Thực nghiệm chính** (`default` / `mcmc`, 3 seed, cùng 4 cảnh trong nhà của Mip-NeRF 360): `docs/design/experiments.md` mục 2.

```bash
cd /workspace/SpatialRoom3D
bash scripts/pod/fetch_data.sh deep_blending                     # tải dữ liệu nếu chưa có (không dùng GPU)
bash scripts/prepare_data.sh deep_blending                       # kiểm tra dữ liệu (không dùng GPU)
QUICK=1 bash scripts/run_train.sh deep_blending gsplat           # chạy thử 3.000 vòng
bash scripts/run_train.sh deep_blending b0                       # B0 (chạy trước gsplat để bảng đối chiếu đủ)
bash scripts/run_train.sh deep_blending gsplat                   # default + mcmc, bảng so sánh
SEEDS="42 43 44" bash scripts/run_train.sh deep_blending gsplat  # 3 seed
bash scripts/run_evaluate.sh deep_blending                       # chỉ lập lại bảng (mọi chế độ đã chạy)
bash scripts/run_render.sh deep_blending playroom mcmc_s42       # render lại video quỹ đạo, xuất PLY
```

Một chiến lược trên một cảnh, đúng thư mục và tham số mà `scripts/pipeline/train_gsplat_modes.sh` dùng:

```bash
. /workspace/miniforge3/etc/profile.d/conda.sh && conda activate gs-gsplat && cd /workspace/SpatialRoom3D
DB=/workspace/data/benchmarks/deep_blending; OUT=/workspace/outputs/benchmarks/deep_blending/gsplat
python -m src.training.run --mode default --scene $DB/drjohnson \
    --scene-name drjohnson --out $OUT/drjohnson/default_s42 --seed 42 --rerun-incomplete
python -m src.training.run --mode mcmc --scene $DB/drjohnson \
    --scene-name drjohnson --out $OUT/drjohnson/mcmc_s42 --seed 42 --rerun-incomplete \
    --cap-max-from $OUT/drjohnson/default_s42 --set opacity_reg=0.001
```

Bỏ `--set opacity_reg=0.001` thì cấu hình khác lượt đã có, nên bị báo lỗi chứ không ghi đè.

**Kết quả:**
- Repo này: **chưa chạy** trên pod.
- Repo trước, ngày 17/10/2026 (RTX 4090, seed 42; cùng lệnh huấn luyện, chỉ khác đường dẫn, `tests/unit/test_benchmarks.py`): `docs/ENVIRONMENT_AUDIT.md` mục 7.
