# So sánh DefaultStrategy (`default`) và MCMC (`mcmc`)

`scripts/run_evaluate.sh` chỉ đọc kết quả đã có, không train, không dùng GPU. Script gọi `src.evaluation.compare_modes` với đúng thư mục kết quả của từng luồng, lấy từ `configs/<luồng>.yaml`.

```bash
bash scripts/run_evaluate.sh deep_blending           # outputs/benchmarks/deep_blending/gsplat, đối chiếu Inria công bố + B0 tự chạy
bash scripts/run_evaluate.sh mipnerf360              # outputs/benchmarks/mipnerf360/gsplat
QUICK=1 bash scripts/run_evaluate.sh deep_blending   # các lượt chạy thử 3.000 vòng (gsplat_quick)
bash scripts/run_evaluate.sh custom phong1 phong2    # cảnh tự chụp (outputs/custom_scenes/<cảnh>)
bash scripts/run_evaluate.sh custom                  # mọi cảnh tự chụp
```

Báo cáo: `$WS/outputs/reports/compare_<tên>_<thời_gian>.{md,json}`. Với benchmark, phần cấu hình benchmark đã dùng (số liệu công bố, ngưỡng) được lưu kèm ở `$WS/outputs/reports/configs/<benchmark>_<thời_gian>.json`.

Bảng gồm:
- số liệu từng cảnh × biến thể (trung bình theo seed);
- chênh lệch so với `default`, chỉ tính trên các seed chung;
- các mục điều kiện: protocol, tập test, commit gsplat, GPU, ảnh gsplat thực dùng, số seed, ngân sách Gaussian;
- với benchmark: đối chiếu `default` với số liệu công bố, và với B0 tự chạy ở từng cảnh.

Báo cáo không tự kết luận chế độ nào tốt hơn. Chỉ gọi là "cải thiện" khi có ít nhất 3 seed, mọi điều kiện đạt (ngân sách Gaussian EQUAL) và kết quả nhất quán trên các cảnh (`docs/design/training_modes.md` mục 4.3).

Đã kiểm tra trong repo trước, trên kết quả thật của pod ngày 17/10/2026: cùng mã so sánh cho ra báo cáo giống hệt `reports/modes_db_20261017_162815.md`. Repo này chưa có kết quả nào trên pod (`docs/PROJECT_STATUS.md`).
