# Đánh giá theo đoạn: kiểm tra đủ lượt và luật mục 3.8

Thư mục: `/workspace/outputs/custom_segments` | phòng: video8, video9, video10 | seed: 42, 43, 44. Luật ghi trước khi chạy: `docs/design/experiments.md` mục 3.8 (D26). Các mục kiểm tra khác của bảng so sánh (`src.evaluation.compare_modes`) vẫn phải đọc riêng.

## 1. Độ đầy đủ (số seed OK)

| Phòng | default | mcmc | mcmc_rbg | mcmc_mono | mcmc_kr |
|---|---|---|---|---|---|
| video8 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| video9 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| video10 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

Tổng: 45/45 lượt OK.

## 2. Điều kiện so sánh trong từng phòng

| Phòng | Tập chia | Commit gsplat | GPU | Số Gaussian bằng default cùng seed |
|---|---|---|---|---|
| video8 | OK | OK | OK | EQUAL |
| video9 | OK | OK | OK | EQUAL |
| video10 | OK | OK | OK | EQUAL |

## 3. Chỉ số và chênh lệch

Trung bình ± độ lệch chuẩn trên các seed OK; chênh lệch là trung bình theo cặp cùng seed, nhãn so với d_r của phòng (tốt hơn / dao động / KÉM).

### video8

d_r: PSNR 0.765 | SSIM 0.0035 | LPIPS 0.0036

| Cấu hình | Seed OK | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| default | 3/3 | 25.359 ± 0.529 | 0.9371 ± 0.0028 | 0.1421 ± 0.0019 |
| mcmc | 3/3 | 25.698 ± 0.553 | 0.9443 ± 0.0021 | 0.1340 ± 0.0031 |
| mcmc_rbg | 3/3 | 25.604 ± 0.320 | 0.9431 ± 0.0030 | 0.1350 ± 0.0065 |
| mcmc_mono | 3/3 | 26.155 ± 0.200 | 0.9413 ± 0.0023 | 0.1350 ± 0.0017 |
| mcmc_kr | 3/3 | 25.944 ± 0.133 | 0.9436 ± 0.0006 | 0.1334 ± 0.0013 |

| So sánh | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---|---|---|
| mcmc - default | +0.339 (dao động) | +0.0072 (tốt hơn) | -0.0080 (tốt hơn) |
| rbg - mcmc | -0.094 (dao động) | -0.0012 (dao động) | +0.0010 (dao động) |
| mono - mcmc | +0.457 (dao động) | -0.0031 (dao động) | +0.0009 (dao động) |
| kr - mcmc | +0.246 (dao động) | -0.0007 (dao động) | -0.0006 (dao động) |

### video9

d_r: PSNR 1.442 | SSIM 0.0162 | LPIPS 0.0173

| Cấu hình | Seed OK | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| default | 3/3 | 18.585 ± 1.372 | 0.8639 ± 0.0153 | 0.2411 ± 0.0169 |
| mcmc | 3/3 | 22.509 ± 0.444 | 0.8853 ± 0.0054 | 0.2038 ± 0.0039 |
| mcmc_rbg | 3/3 | 22.441 ± 0.499 | 0.8874 ± 0.0009 | 0.2047 ± 0.0025 |
| mcmc_mono | 3/3 | 22.425 ± 0.315 | 0.8844 ± 0.0047 | 0.2005 ± 0.0045 |
| mcmc_kr | 3/3 | 22.162 ± 0.494 | 0.8774 ± 0.0121 | 0.2112 ± 0.0083 |

| So sánh | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---|---|---|
| mcmc - default | +3.923 (tốt hơn) | +0.0214 (tốt hơn) | -0.0373 (tốt hơn) |
| rbg - mcmc | -0.067 (dao động) | +0.0021 (dao động) | +0.0009 (dao động) |
| mono - mcmc | -0.084 (dao động) | -0.0009 (dao động) | -0.0032 (dao động) |
| kr - mcmc | -0.347 (dao động) | -0.0079 (dao động) | +0.0074 (dao động) |

### video10

d_r: PSNR 0.274 | SSIM 0.0036 | LPIPS 0.0025

| Cấu hình | Seed OK | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| default | 3/3 | 19.532 ± 0.199 | 0.8210 ± 0.0035 | 0.2990 ± 0.0021 |
| mcmc | 3/3 | 21.816 ± 0.188 | 0.8709 ± 0.0009 | 0.2658 ± 0.0014 |
| mcmc_rbg | 3/3 | 21.928 ± 0.241 | 0.8730 ± 0.0008 | 0.2637 ± 0.0008 |
| mcmc_mono | 3/3 | 22.365 ± 0.515 | 0.8761 ± 0.0040 | 0.2589 ± 0.0081 |
| mcmc_kr | 3/3 | 21.356 ± 0.296 | 0.8625 ± 0.0012 | 0.2741 ± 0.0024 |

| So sánh | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---|---|---|
| mcmc - default | +2.285 (tốt hơn) | +0.0499 (tốt hơn) | -0.0332 (tốt hơn) |
| rbg - mcmc | +0.111 (dao động) | +0.0021 (dao động) | -0.0022 (dao động) |
| mono - mcmc | +0.549 (tốt hơn) | +0.0052 (tốt hơn) | -0.0069 (tốt hơn) |
| kr - mcmc | -0.460 (KÉM) | -0.0084 (KÉM) | +0.0083 (KÉM) |

## 4. Kết luận theo luật mục 3.8 (so với mcmc)

| Biến thể | Phòng tốt hơn quá d_r | Phòng kém quá d_r | Kết luận |
|---|---|---|---|
| rbg | không | không | **KHÔNG ĐẠT**: chỉ tốt hơn quá d_r ở 0 phòng (cần 2) |
| mono | video10 (PSNR, SSIM, LPIPS) | không | **KHÔNG ĐẠT**: chỉ tốt hơn quá d_r ở 1 phòng (cần 2) |
| kr | không | video10 (PSNR, SSIM, LPIPS) | **KHÔNG ĐẠT**: kém quá d_r ở video10 (PSNR, SSIM, LPIPS); chỉ tốt hơn quá d_r ở 0 phòng (cần 2) |

ĐẠT: đủ cả ba điều của luật. Tên gọi khi báo cáo theo mục 3.8: `rbg` là chọn cấu hình, `mono` là tích hợp kỹ thuật đã công bố, `kr` là heuristic của nhóm; không gọi là thuật toán mới.
