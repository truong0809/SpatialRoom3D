# Deep Blending

Thư mục này chỉ chứa tài liệu của benchmark. Dữ liệu và kết quả nằm trên volume, không đưa lên Git (`.gitignore` chặn mọi file khác trong `experiments/<benchmark>/`).

- **Cấu hình:** `configs/deep_blending.yaml` (dữ liệu, nơi lưu kết quả, mục `benchmark`: protocol, số liệu công bố, ngưỡng).
- **Dữ liệu:** `$WS/data/benchmarks/deep_blending/{drjohnson,playroom}` (263 và 225 ảnh), từ `tandt_db.zip` của Inria (đường dẫn tải và SHA-256 trong cấu hình). Thiếu dữ liệu thì script tự tải (khoảng 650 MB), kiểm tra SHA-256, giải nén vào `$WS/data/downloads/` rồi dời thư mục các cảnh tới chỗ trên.
- **Kết quả:** `$WS/outputs/benchmarks/deep_blending/{inria_b0,gsplat,gsplat_quick}/`; báo cáo `$WS/outputs/reports/{baseline_db_summary,modes_db,compare_db}_<thời_gian>.*`.
- **Protocol:**
  - tập test cứ 8 ảnh lấy 1 (`--eval` của Inria = `test_every 8` của gsplat, cùng một tập; đã đối chiếu mã nguồn);
  - ảnh có sẵn của bộ dữ liệu, không thu nhỏ thêm (`data_factor 1`);
  - 30.000 vòng; LPIPS VGG.
- **Số liệu đối chiếu:** trung bình Inria công bố cho Deep Blending (PSNR 29,690, SSIM 0,906, LPIPS 0,238; nguồn ghi trong cấu hình); ngưỡng 0,5 dB / 0,01 / 0,01.
- **MCMC:** `opacity_reg=0.001` (paper 3DGS-MCMC dùng giá trị này cho Deep Blending; config chính chủ chỉ đặt cho drjohnson, ở đây dùng cho cả hai cảnh như lượt 17/10/2026). Cần ghi rõ khi báo cáo.

```bash
cd /workspace/SpatialRoom3D
bash scripts/prepare_data.sh deep_blending                       # tải / kiểm tra dữ liệu (không dùng GPU)
bash scripts/run_train.sh deep_blending b0                       # B0
bash scripts/run_train.sh deep_blending gsplat                   # default + mcmc, bảng so sánh
QUICK=1 bash scripts/run_train.sh deep_blending gsplat           # chạy thử 3.000 vòng
SEEDS="42 43 44" bash scripts/run_train.sh deep_blending gsplat  # 3 seed
bash scripts/run_evaluate.sh deep_blending                       # chỉ lập lại bảng
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
