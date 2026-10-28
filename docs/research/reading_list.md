# Danh sách tài liệu cần khảo sát

_Phiên bản 1 · 09/10/2026 · thêm FSGS, ghi chú tài liệu dùng cho các biến thể của `tune_mcmc` ngày 28/10/2026_

Đây là danh sách khởi đầu, chưa phải literature matrix. Ở mốc M1, mỗi mục sẽ được đối chiếu với nguồn chính thức (CVF Open Access, ACM DL, ECVA, OpenReview, arXiv) và bổ sung link/DOI đã kiểm chứng trước khi đưa vào tài liệu tham khảo của báo cáo.

**Ký hiệu**

- ★ đọc kỹ trước khi chốt phương pháp
- ☆ đọc có chọn lọc
- (†) chưa xác minh venue chính thức

## A. Nền tảng

- ★ Kerbl et al. — 3D Gaussian Splatting for Real-Time Radiance Field Rendering, ACM TOG (SIGGRAPH) 2023
- ★ Zwicker et al. — EWA Volume Splatting, IEEE Visualization 2001 (cơ sở toán học của splatting)
- ★ Schönberger & Frahm — Structure-from-Motion Revisited, CVPR 2016 (COLMAP)
- ☆ Schönberger et al. — Pixelwise View Selection for Unstructured Multi-View Stereo, ECCV 2016
- ★ Mildenhall et al. — NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020
- ★ Barron et al. — Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields, CVPR 2022
- ☆ Đối chứng nhánh NeRF:
  - Müller et al. — Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, ACM TOG 2022
  - Barron et al. — Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields, ICCV 2023

## B. Thư viện

- ★ Ye et al. — gsplat: An Open-Source Library for Gaussian Splatting, JMLR 2025
- ☆ Tancik et al. — Nerfstudio: A Modular Framework for Neural Radiance Field Development, SIGGRAPH 2023

## C. Cải tiến chất lượng và hiệu năng 3DGS

- ★ Yu et al. — Mip-Splatting: Alias-free 3D Gaussian Splatting, CVPR 2024
- ★ Kerbl et al. — A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets, ACM TOG 2024 (nguồn gốc depth regularization và bù phơi sáng trong repo Inria)
- ★ Mallick et al. — Taming 3DGS: High-Quality Radiance Fields with Limited Resources, SIGGRAPH Asia 2024
- ☆ Tối ưu densification và chất lượng:
  - Kheradmand et al. — 3D Gaussian Splatting as Markov Chain Monte Carlo, NeurIPS 2024
  - Rota Bulò et al. — Revising Densification in Gaussian Splatting, ECCV 2024
  - Ye et al. — AbsGS: Recovering Fine Details for 3D Gaussian Splatting, ACM MM 2024
  - Lu et al. — Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering, CVPR 2024
- Cải tiến 3DGS-MCMC (chỉ ghi những gì đã đọc được):
  - ☆ Huang, Chen, Liu, Ji — Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation, AAAI 2026 ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37438); arXiv 2511.13571). Xây trên 3DGS-MCMC; đổi cách tối ưu (Adaptive Weighted SGLD, rồi quasi-Newton cho vị trí), không đổi cách đặt Gaussian. Chưa thấy code công bố
  - ☆ Hofer, Köhler, Windisch, Kerbl, Steinberger, Radl — Efficient Confidence-Aware Densification for 3DGS (†, arXiv 2610.13541, 09/10/2026): đặt và dời Gaussian theo sai số ảnh và độ tin cậy học được, trong khung ngân sách cố định của MCMC. Mới đọc tóm tắt; chưa thấy code
  - ☆ Radl et al. — Confidence-Based Mesh Extraction from 3D Gaussians (†, arXiv 2603.24725): độ tin cậy học được cho mỗi Gaussian (nguồn của ý tưởng độ tin cậy ở mục trên)
  - ☆ Kim, Jung, Park — Metropolis-Hastings Sampling for 3D Gaussian Reconstruction, NeurIPS 2025 ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d25ed2a375a33257d33e83fd9010ec60-Abstract-Conference.html)): thêm Gaussian theo sai số nhiều góc nhìn với phép thử chấp nhận; so với 3DGS-MCMC ở cùng số Gaussian, kết quả lẫn lộn (Deep Blending tốt hơn, Mip-NeRF 360 kém hơn)
- ☆ Nén mô hình:
  - Fan et al. — LightGaussian, NeurIPS 2024
  - Niedermayr et al. — Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis, CVPR 2024

## D. Nội thất và prior hình học

- ★ Turkulainen et al. — DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing, WACV 2025
- ☆ Zhu, Fan, Jiang, Wang — FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting, ECCV 2024 ([ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5583_ECCV_2024_paper.php); arXiv 2312.00451). Loss Pearson giữa độ sâu render và độ sâu đơn ảnh (DPT), không phụ thuộc tỉ lệ / độ dịch: dạng loss của biến thể `mono` trong `tune_mcmc` (`docs/design/experiments.md` mục 3.7)
- ☆ Xiang et al. — GaussianRoom: Improving 3D Gaussian Splatting with SDF Guidance and Monocular Cues for Indoor Scene Reconstruction (†, arXiv 2405.19671)
- ☆ Zhang et al. — 2DGS-Room: Seed-Guided 2D Gaussian Splatting with Geometric Constrains for High-Fidelity Indoor Scene Reconstruction (†, arXiv 2412.03428)
- ☆ Yu et al. — MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction, NeurIPS 2022 (đối chứng nhánh SDF)
- ☆ Mô hình prior:
  - Yang et al. — Depth Anything V2, NeurIPS 2024. Bản Small (Apache-2.0) tạo depth prior cho biến thể `mono` (`src/training/mono_depth.py`)
  - Bae & Davison — Rethinking Inductive Biases for Surface Normal Estimation (DSINE), CVPR 2024
  - Hu et al. — Metric3D v2, TPAMI 2024

## E. Hình học bề mặt

Đọc kỹ nếu GVHD yêu cầu hình học đo được (Q1).

- ☆ Huang et al. — 2D Gaussian Splatting for Geometrically Accurate Radiance Fields, SIGGRAPH 2024
- ☆ Yu et al. — Gaussian Opacity Fields, ACM TOG (SIGGRAPH Asia) 2024
- ☆ Chen et al. — PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction, IEEE TVCG 2024
- ☆ Guédon & Lepetit — SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction, CVPR 2024

## F. Video quay tay thực tế

- ★ Seiskari et al. — Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion, ECCV 2024
  - Code dựng trên bản fork cũ của Nerfstudio và gsplat; dùng với gsplat mới cần port.
  - Từ video thường chỉ bù được mờ hoặc rolling shutter; bù đồng thời cả hai cần biết thời gian đọc và phơi sáng.
- ☆ Khử mờ:
  - Lee et al. — Deblurring 3D Gaussian Splatting, ECCV 2024
  - Zhao et al. — BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting, ECCV 2024
- ☆ Ngoại hình thay đổi giữa các ảnh:
  - Wang et al. — Bilateral Guided Radiance Field Processing, ACM TOG 2024 (SIGGRAPH 2024). Có sẵn trong gsplat (`use_bilateral_grid`): biến thể `bilagrid` của `tune_mcmc`
  - Martin-Brualla et al. — NeRF in the Wild, CVPR 2021

## G. Ước lượng camera

- ★ Pan et al. — Global Structure-from-Motion Revisited (GLOMAP), ECCV 2024
- ★ Wang et al. — VGGT: Visual Geometry Grounded Transformer, CVPR 2025
- ☆ Đặc trưng và matching:
  - Lindenberger et al. — LightGlue: Local Feature Matching at Light Speed, ICCV 2023
  - Sarlin et al. — From Coarse to Fine: Robust Hierarchical Localization at Large Scale (hloc), CVPR 2019
  - DeTone et al. — SuperPoint: Self-Supervised Interest Point Detection and Description, CVPRW 2018
- ☆ Hướng không cần hoặc giảm phụ thuộc COLMAP:
  - Wang et al. — DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024
  - Leroy et al. — Grounding Image Matching in 3D with MASt3R, ECCV 2024
  - Fu et al. — COLMAP-Free 3D Gaussian Splatting, CVPR 2024
  - Fan et al. — InstantSplat (†)
- ☆ Mô hình mới:
  - Keetha et al. — MapAnything: Universal Feed-Forward Metric 3D Reconstruction (†, arXiv 2509.13414)
  - Depth Anything 3 (†, arXiv 2511.10647)

## H. SLAM và feed-forward

Đọc để biết giới hạn; nhiều khả năng không làm hướng chính.

- ☆ SLAM:
  - Matsuki et al. — Gaussian Splatting SLAM, CVPR 2024
  - Keetha et al. — SplaTAM, CVPR 2024 (cần RGB-D, không phù hợp ràng buộc đề tài)
- ☆ Feed-forward:
  - Charatan et al. — pixelSplat, CVPR 2024
  - Chen et al. — MVSplat, ECCV 2024

## I. Tổng quan

- ★ Chen & Wang — A Survey on 3D Gaussian Splatting (arXiv 2401.03890)
- ☆ Fei et al. — 3D Gaussian Splatting as New Era: A Survey, IEEE TVCG 2024
- ☆ Wu et al. — Recent Advances in 3D Gaussian Splatting, Computational Visual Media 2024

## J. Dataset và chỉ số

- ★ Yeshwanth et al. — ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes, ICCV 2023 (dùng bản v2 và benchmark NVS)
- ★ Cảnh trong nhà của Mip-NeRF 360 (room, counter, kitchen, bonsai)
- ★ Hedman et al. — Deep Blending for Free-Viewpoint Image-Based Rendering, ACM TOG 2018 (playroom, drjohnson)
- ☆ Dataset bổ sung:
  - Knapitsch et al. — Tanks and Temples, ACM TOG 2017
  - Dai et al. — ScanNet, CVPR 2017
  - Straub et al. — The Replica Dataset, arXiv 2019
  - Ling et al. — DL3DV-10K, CVPR 2024
- ★ Chỉ số:
  - Wang et al. — Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM), IEEE TIP 2004
  - Zhang et al. — The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS), CVPR 2018

## K. Tài liệu kỹ thuật chính thức

- README của graphdeco-inria/gaussian-splatting, nerfstudio-project/gsplat, facebookresearch/vggt
- Tài liệu và CHANGELOG của COLMAP
- Spark (renderer 3DGS cho THREE.js)
- PlayCanvas SuperSplat, đặc tả SOG, SplatTransform
- Niantic SPZ
- Đặc tả Khronos KHR_gaussian_splatting
