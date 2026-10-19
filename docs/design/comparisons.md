# So sánh các chế độ huấn luyện (`default`, `mcmc`, chế độ tự thêm)

`scripts/run_evaluate.sh` chỉ đọc kết quả đã có, không train, không dùng GPU. Script gọi `src.evaluation.compare_modes` với đúng thư mục kết quả của từng luồng, lấy từ `configs/<luồng>.yaml`.

```bash
bash scripts/run_evaluate.sh deep_blending           # outputs/benchmarks/deep_blending/gsplat, đối chiếu Inria công bố + B0 tự chạy
bash scripts/run_evaluate.sh mipnerf360              # outputs/benchmarks/mipnerf360/gsplat
QUICK=1 bash scripts/run_evaluate.sh deep_blending   # các lượt chạy thử 3.000 vòng (gsplat_quick)
bash scripts/run_evaluate.sh custom phong1 phong2    # cảnh tự chụp (outputs/custom_scenes/<cảnh>)
bash scripts/run_evaluate.sh custom                  # mọi cảnh tự chụp
BASELINE=mcmc MODES="mcmc <chế độ X>" bash scripts/run_evaluate.sh deep_blending   # chế độ X so với mcmc
```

Biến tùy chọn:
- `MODES`: chỉ các chế độ này (mặc định: mọi chế độ có lượt chạy).
- `SCENES`: benchmark, chỉ các cảnh này. Một phần các cảnh thì không đối chiếu được với trung bình công bố: mục đó ghi SKIPPED.
- `BASELINE`: chế độ làm mốc (mặc định `default`). Khác `default` thì không đối chiếu với Inria, vì phần đó chỉ dành cho baseline `default`. Tên chế độ chỉ có một biến thể là đủ, vd. `mcmc` cho biến thể `mcmc[opacity-reg=0.001]` của Deep Blending.
- `REPORT`: tên file báo cáo (không có đuôi).

Báo cáo: `$WS/outputs/reports/compare_<tên>_<thời_gian>.{md,json}`. Với benchmark, phần cấu hình benchmark đã dùng (số liệu công bố, ngưỡng) được lưu kèm ở `$WS/outputs/reports/configs/<benchmark>_<thời_gian>.json`.

Bảng gồm:
- số liệu từng cảnh × biến thể (trung bình theo seed);
- chênh lệch so với baseline (`default`, hoặc chế độ trong `BASELINE`), chỉ tính trên các seed chung;
- các mục điều kiện: protocol, tập test, commit gsplat, GPU, ảnh gsplat thực dùng, số seed, ngân sách Gaussian;
- với benchmark và baseline `default`: đối chiếu với số liệu công bố (khi đủ mọi cảnh của benchmark), và với B0 tự chạy ở từng cảnh.

Báo cáo không tự kết luận chế độ nào tốt hơn. Chỉ gọi là "cải thiện" khi có ít nhất 3 seed, mọi điều kiện đạt (ngân sách Gaussian EQUAL) và kết quả nhất quán trên các cảnh (`docs/design/training_modes.md` mục 4.3).

Đã kiểm tra trong repo trước, trên kết quả thật của pod ngày 17/10/2026: cùng mã so sánh cho ra báo cáo giống hệt `reports/modes_db_20261017_162815.md`. Repo này chưa có kết quả nào trên pod (`docs/PROJECT_STATUS.md`).
