# Kiến trúc repository và dữ liệu

_Cập nhật: 28/10/2026_

Tài liệu này ghi:
- cây thư mục của repo và vai trò từng phần;
- bố cục `/workspace` trên network volume (code, `data/`, `outputs/`);
- ba luồng dữ liệu (cảnh tự chụp, Deep Blending, Mip-NeRF 360) và quy tắc không trộn;
- dữ liệu nằm ở đâu: Git, ổ của pod, network volume;
- cách khai báo đường dẫn dữ liệu (`configs/*.yaml`);
- pod mới, đổi GPU, chạy lại;
- nguồn gốc: repo này bắt đầu từ repo trước như thế nào.

## 1. Cây thư mục của repo

```
SpatialRoom3D/
├── app/                         ứng dụng
│   ├── cli/                       python -m app.cli <lệnh>: một lệnh cho mọi bước, gọi đúng script trong scripts/
│   ├── web/                       giao diện web: chưa làm (giai đoạn 5), README giữ chỗ
│   ├── notebooks/runpod.ipynb     notebook chạy trên pod: kiểm tra máy, khóa deploy, lấy code, cài môi trường, ...
│   └── __init__.py
├── src/                         mã nguồn chính, package `src` (chạy từ gốc repo: python -m src.<gói>.<module>)
│   ├── input/                     sources.py: nguồn là video hay thư mục ảnh, đủ ảnh không
│   │                              frames.py: frame nét từ video; ảnh chụp → input/ (không chọn frame)
│   ├── reconstruction/            colmap_runner.py, colmap_model.py, report.py (kiểm tra trước khi train)
│   ├── training/                  profiles.py, split.py, gsplat_launcher.py, run.py (một lượt train / đánh giá),
│   │                              experiment.py (kế hoạch thực nghiệm, configs/experiments/),
│   │                              depth_check.py (điểm SfM cho depth_loss), prune.py (lọc Gaussian, chấm lại),
│   │                              trainer_hooks.py (biến thể có code: bản sao có vá của trainer, KR, depth prior),
│   │                              mono_depth.py (depth prior bằng Depth Anything V2 Small)
│   ├── rendering/                 render.py: render lại từ checkpoint, xuất PLY
│   ├── evaluation/                gsplat_results.py, compare_modes.py, compare_reference.py, overview.py,
│   │                              splat_stats.py (thống kê PLY), panels.py (ảnh so sánh các lượt)
│   ├── datasets/                  registry.py: đọc configs/*.yaml, kiểm tra dữ liệu từng luồng; fetch.py: tải benchmark
│   ├── utils/                     gpu_lock.py: khóa GPU dùng chung với script bash
│   └── __init__.py
├── configs/
│   ├── base.yaml                  phần chung: code_dir, danh sách luồng, train_modes (protocol + default / mcmc)
│   ├── custom_scene.yaml          cảnh tự chụp: nguồn, nơi lưu COLMAP và kết quả
│   ├── mipnerf360.yaml            Mip-NeRF 360: dữ liệu, nguồn tải, kết quả, mục benchmark (protocol, số liệu công bố, ngưỡng)
│   ├── deep_blending.yaml         Deep Blending: như trên
│   ├── experiments/               kế hoạch thực nghiệm (main.yaml: default / mcmc, 3 seed, 6 cảnh;
│   │                              tune_mcmc.yaml: 16 biến thể của mcmc trên room, 1 seed, lọc Gaussian)
│   ├── environment/               phiên bản ghim của 3 env conda (gs-inria, gs-tools, gs-gsplat)
│   └── local.yaml                 (không vào Git) ghi đè đường dẫn trên một pod, mục 5
├── scripts/
│   ├── prepare_data.sh            chuẩn bị dữ liệu: custom <ảnh|video> <cảnh> | <benchmark>
│   ├── run_train.sh               huấn luyện: custom <cảnh> [chế độ ...] | <benchmark> [gsplat|b0]
│   ├── run_render.sh              render ảnh / video, xuất PLY
│   ├── run_evaluate.sh            bảng so sánh các chế độ / B0 / số liệu công bố
│   ├── run_filter.sh              lọc Gaussian của các lượt đã train xong rồi chấm lại (tùy chọn)
│   ├── run_panels.sh              ảnh so sánh các lượt trên cùng ảnh test, video quỹ đạo xếp chồng
│   ├── pipeline/                  phần thực thi: prepare_scene, prepare_benchmark, train_scene, train_gsplat_modes, train_inria_b0
│   ├── pod/                       run_all, run_scene (cả luồng một cảnh), run_experiment (thực nghiệm trọn vẹn),
│   │                              fetch_data (tải benchmark), job (chạy nền, xếp hàng GPU), results, pack_results,
│   │                              check_datasets
│   ├── setup/                     env_probe, setup_3dgs_inria, setup_data_tools, setup_gsplat, setup_depth_anything,
│   │                              smoke_test_*.py
│   └── lib/                       gpu_lock.sh, config.sh (đọc configs/*.yaml), benchmark.sh
├── tests/
│   ├── test_dataset.py            cấu hình và kiểm tra dữ liệu
│   ├── test_pipeline.py           luồng cảnh tự chụp qua run_scene.sh và bốn script chính
│   ├── unit/                      từng phần: input, COLMAP, huấn luyện, render, đánh giá, job, run_all, CLI, ...
│   ├── helpers/                   ShellEnv (bash thật + stub), viết mô hình COLMAP giả
│   └── conftest.py
├── docs/                        README (mục lục), ENVIRONMENT_AUDIT, USAGE_GUIDE, PROJECT_PLAN, PROJECT_STATUS; design/, data/, research/
├── experiments/
│   ├── deep_blending/README.md    protocol, lệnh, kết quả đã ghi nhận (dữ liệu và kết quả không vào Git)
│   └── mipnerf360/README.md
├── README.md                    giới thiệu dự án, chạy nhanh, kiểm thử
├── .gitignore, .gitattributes
└── requirements.txt             phụ thuộc để chạy test trên máy không có GPU (env GPU: scripts/setup/ + configs/environment/)
```

Luồng chính và nơi mỗi bước nằm:

| Bước | Code | Lệnh |
|---|---|---|
| Ảnh / Video → Validate Input | `src/input/sources.py` | `scripts/prepare_data.sh custom ...` (`scripts/pod/run_scene.sh` kiểm tra trước khi xếp hàng GPU) |
| Extract Frames (video) / chuẩn hóa ảnh | `src/input/frames.py` | như trên |
| COLMAP | `src/reconstruction/colmap_runner.py`, `colmap_model.py` | như trên |
| Prepare Dataset (kiểm tra trước khi train) | `src/reconstruction/report.py` | như trên; benchmark: `scripts/prepare_data.sh <benchmark>` |
| Train 3DGS | `src/training/` | `scripts/run_train.sh` |
| Export PLY / Checkpoint | `src/rendering/render.py` | `scripts/run_render.sh` |
| Lọc Gaussian, chấm lại (tùy chọn) | `src/training/prune.py` | `scripts/run_filter.sh` |
| Evaluate (PSNR / SSIM / LPIPS) | `src/evaluation/` | `scripts/run_evaluate.sh` |
| Ảnh so sánh các lượt | `src/evaluation/panels.py` | `scripts/run_panels.sh` |

Mọi lệnh Python chạy với thư mục gốc của repo trong `PYTHONPATH` (script tự đặt), ví dụ `python -m src.training.run`, `python -m src.reconstruction.colmap_runner`, `python -m src.datasets.registry`.

## 2. Bố cục `/workspace` trên network volume

```
/workspace/
├── SpatialRoom3D/                  code (repo Git, clone bằng deploy key: docs/ENVIRONMENT_AUDIT.md mục 2)
├── data/                           CHỈ dữ liệu gốc: bạn tự đưa lên, hoặc (benchmark) bước tải ghi khi bạn gọi; các bước khác
│   │                               không ghi vào (ngoại lệ duy nhất: mục 3)
│   ├── custom_scenes/<cảnh>/         ảnh JPG/JPEG/PNG tự chụp (nằm thẳng trong thư mục), hoặc file video
│   ├── benchmarks/
│   │   ├── deep_blending/{drjohnson,playroom}/
│   │   └── mipnerf360/<cảnh>/        (.sr3d_fetch.json: nguồn, SHA-256 của các lần tải)
│   └── downloads/                    file zip đang tải / giải nén (scripts/pod/fetch_data.sh), xóa sau khi xong
├── outputs/                        mọi thứ sinh ra
│   ├── custom_scenes/<cảnh>/
│   │   ├── colmap/                   ảnh đã chuẩn hóa (input/), COLMAP (images/, sparse/0/), frames_report.json, sfm_report.json
│   │   └── <chế_độ>_s<seed>/         các lượt train (default_s42, mcmc_s42, ...); render_<quỹ_đạo>_<thời_gian>/ khi render lại,
│   │                                 filtered/ khi đã lọc Gaussian
│   ├── benchmarks/<benchmark>/
│   │   ├── inria_b0/<cảnh>/          B0 Inria
│   │   ├── gsplat/<cảnh>/<chế_độ>[_<tag>]_s<seed>/   tag: biến thể tham số của một kế hoạch (vd. mcmc_or0001_s42)
│   │   │                                            .mono_depth_<dấu COLMAP>/: depth prior của cảnh (biến thể mono)
│   │   └── gsplat_quick/<cảnh>/<chế_độ>_s<seed>/   chạy thử 3.000 vòng
│   ├── checkpoints/<luồng>/<cảnh>/<lượt>.ply   PLY đã xuất để tải về / xem 3D (liên kết cứng tới PLY của lượt)
│   ├── panels/<luồng>_<cảnh>_<thời_gian>/       ảnh so sánh các lượt, video xếp chồng (scripts/run_panels.sh)
│   ├── reports/                      báo cáo; reports/configs/: bản JSON của cấu hình benchmark đã dùng
│   ├── logs/                         log; logs/jobs/: log của việc chạy nền
│   └── jobs/                         trạng thái việc chạy nền
├── miniforge3/                     env conda gs-inria, gs-gsplat, gs-tools
├── code/                           gaussian-splatting (Inria 54c035f), gsplat (6e8c837),
│                                   depth_anything_v2 (a561b84 + trọng số bản Small, chỉ biến thể mono)
├── ket_qua_*.tar.gz                gói kết quả nhẹ để tải về
└── .ssh/                           deploy key chỉ đọc (không bao giờ gửi đi)
```

Tên báo cáo và log theo `benchmark.tag` của từng benchmark (`db`, `mipnerf360`): `outputs/reports/modes_db_*.md`, `outputs/reports/baseline_db_summary_*.json`, `outputs/reports/compare_db_*.md`. Thực nghiệm (`scripts/pod/run_experiment.sh`): `outputs/reports/experiment_<tên>_<thời_gian>.*`; tải dữ liệu: `outputs/reports/fetch_<benchmark>_<thời_gian>.json`.

## 3. Ba luồng dữ liệu, không trộn

| Luồng | Dữ liệu vào | Lệnh | Kết quả | Trạng thái |
|---|---|---|---|---|
| Cảnh tự chụp | `data/custom_scenes/<cảnh>/`: một thư mục ảnh, hoặc một video | `bash scripts/pod/run_scene.sh <nguồn> <cảnh> --background`, hoặc từng bước `prepare_data.sh custom` → `run_train.sh custom` → `run_render.sh custom` → `run_evaluate.sh custom` | `outputs/custom_scenes/<cảnh>/` | Đã viết, đã kiểm thử trên CPU; bước chuẩn bị đã chạy với COLMAP thật trên ảnh / video tổng hợp, bước train chỉ chạy với stub; **chưa chạy trên pod** |
| Deep Blending | `data/benchmarks/deep_blending/{drjohnson,playroom}` (tải: `scripts/pod/fetch_data.sh deep_blending`) | `bash scripts/run_train.sh deep_blending b0`, `... gsplat`; thực nghiệm chính: `scripts/pod/run_experiment.sh main` | `outputs/benchmarks/deep_blending/` | Repo trước đã chạy trên pod ngày 17/10/2026 (seed 42); **repo này chưa chạy** |
| Mip-NeRF 360 | `data/benchmarks/mipnerf360/<cảnh>` (đổi được, mục 5; tải: `scripts/pod/fetch_data.sh mipnerf360 [cảnh ...]`) | `bash scripts/run_train.sh mipnerf360 b0`, `... gsplat`; thực nghiệm chính (4 cảnh trong nhà) | `outputs/benchmarks/mipnerf360/` | **Đã cấu hình, chưa chạy; chưa có dữ liệu trên volume** |

**Không trộn dữ liệu:**
- Code dùng chung cho mọi luồng; mỗi luồng có file cấu hình, thư mục dữ liệu và thư mục kết quả riêng.
- `run_scene.sh` không chạy benchmark nào và chỉ ghi vào `outputs/custom_scenes/<cảnh>/`. Thư mục ảnh / video gốc không bị sửa; `--redo-colmap` chỉ làm lại `colmap/`.
- Script benchmark không đọc cảnh tự chụp; mỗi benchmark ghi vào `outputs/benchmarks/<benchmark>/`.
- Bước tải (`scripts/pod/fetch_data.sh`, chỉ khi bạn gọi) chỉ tạo thư mục cảnh còn thiếu trong thư mục gốc của benchmark (giải nén vào thư mục tạm `.fetch_tmp_<cảnh>` rồi đổi tên) và file zip tạm trong `data/downloads/`. Thư mục cảnh đã có thì không sửa, không ghi đè; có nhưng thiếu phần thì dừng.
- **Ngoại lệ duy nhất:** B0 chạy `train.py` của Inria đúng như bản gốc. Lần đầu mở một cảnh, `train.py` tạo `sparse/0/points3D.ply` (chuyển từ `points3D.bin`) ngay trong thư mục dữ liệu nếu file này chưa có (`scene/dataset_readers.py` @54c035f). Bộ Mip-NeRF 360 không có sẵn file này. Dấu mô hình COLMAP chỉ tính trên các file `.bin`, nên không đổi. Không đổi cách gọi B0 để giữ đúng baseline đã tái lập.
- gsplat với Mip-NeRF 360 (`data_factor` 2 / 4) đọc một bản xem dữ liệu gồm các liên kết, ghi ảnh PNG thu nhỏ vào `outputs/benchmarks/mipnerf360/gsplat/<cảnh>/.images_<N>_png_<dấu mô hình>/`, không ghi gì vào `data/`.
- `run_all.sh` chỉ cài / kiểm tra môi trường: không train, không chạy benchmark, không đọc luồng dữ liệu nào. Chạy luồng nào là do người dùng chọn.
- Mỗi lượt train có thư mục riêng `<cảnh>/<chế_độ>_s<seed>`. Lượt cũ khác cấu hình không bị ghi đè: báo lỗi, hoặc đổi tên thành `.prev_<thời_gian>` khi được phép thay. Riêng B0: cảnh đã có `results.json` thì bỏ qua (không so cấu hình); `FORCE=1` chạy lại, lượt cũ được đổi tên `<cảnh>.prev_<thời_gian>`.
- Render lại (`run_render.sh`) thêm một thư mục mới `render_<quỹ_đạo>_<thời_gian>/` trong lượt đó, không sửa file nào của lượt; dừng nếu mô hình COLMAP của cảnh đã đổi sau khi train. Lọc Gaussian (`run_filter.sh`) cũng vậy, với thư mục `filtered/` (checkpoint và PLY đã lọc, lượt chấm lại, `filter_meta.json`); lọc lại với ngưỡng khác thì `filtered/` cũ đổi tên `filtered.prev_<thời_gian>`. Xuất PLY không ghi đè: tên đã là một PLY khác thì xuất thành `<lượt>_<thời điểm train xong>.ply`; lượt chạy thử của benchmark xuất vào `checkpoints/<benchmark>_quick/`.

## 4. Dữ liệu nằm ở đâu

| Nơi | Có gì | Còn khi tắt / đổi pod? |
|---|---|---|
| Git (GitHub, repo private `truong0809/SpatialRoom3D`) | Code, cấu hình, tài liệu, test, `app/notebooks/runpod.ipynb`, notebook demo tham khảo. **Không có** dataset, video, ảnh, checkpoint, PLY, log, `configs/local.yaml`, khóa (`.gitignore`) | Có |
| Ổ của pod (container disk, `/`) | Khóa GPU `/run/lock/spatialroom3d-gpu.lock` | Không: mất khi Stop / Terminate |
| Network volume (`/workspace` trên Runpod) | Mọi thứ ở mục 2 | Có, nếu pod mới gắn cùng volume (cùng data center) |

Chi tiết từng mục trên một pod: `bash scripts/pod/check_datasets.sh` (chỉ đọc).

## 5. Khai báo đường dẫn dữ liệu (`configs/*.yaml`)

- `configs/base.yaml` liệt kê các luồng; mỗi luồng có một file `configs/<tên>.yaml`. Đây là nơi duy nhất ghi dữ liệu và kết quả của từng luồng nằm ở đâu. Đường dẫn viết theo `${WS}`.
- `WS` mặc định `/workspace`. Volume gắn ở chỗ khác thì đặt `WS=<đường dẫn>` cho mọi lệnh.
- Đọc bằng `src/datasets/registry.py` (`yaml.safe_load`). Script bash gọi qua `scripts/lib/config.sh`: `registry_value <luồng> <khóa>` (vd. `registry_value custom_scene results.gsplat`). Cần PyYAML: dùng `python3` của máy nếu có, nếu không thì python của env `gs-gsplat`.
- Env `gs-inria` không có PyYAML: mỗi lần chạy, script benchmark ghi phần `benchmark` của `configs/<benchmark>.yaml` ra `outputs/reports/configs/<benchmark>_<thời_gian>.json` (`--export-benchmark`) và chỉ đọc bản JSON đó. Bản JSON ở lại làm bằng chứng cấu hình đã dùng.
- Cảnh tự chụp: COLMAP nằm trong thư mục kết quả của cảnh, nên `prepared` trong `configs/custom_scene.yaml` phải đúng bằng `<results.gsplat>/<cảnh>/colmap`. Đổi một khóa thì đổi cả hai; lệch thì mọi lệnh báo lỗi, không đoán. Biến `OUT_ROOT` vẫn thắng cấu hình.
- Dữ liệu nằm chỗ khác trên một pod (ví dụ Mip-NeRF 360 đặt tên `MipNeRF360`), không cần sửa file trong Git, chọn một trong hai:
  - tạo `configs/local.yaml` (đã nằm trong `.gitignore`), cùng cấu trúc, chỉ ghi khóa cần đổi:
    ```yaml
    datasets:
      mipnerf360:
        root: ${WS}/data/MipNeRF360
    ```
    Tên luồng không có trong `configs/base.yaml` thì báo lỗi (không bỏ qua lặng lẽ);
  - hoặc đặt biến môi trường `SR3D_DATA_MIPNERF360=/workspace/data/MipNeRF360`. Biến môi trường thắng file cục bộ.
- Dữ liệu benchmark: tải bằng `scripts/pod/fetch_data.sh` (mục `download` của `configs/<benchmark>.yaml`: URL, SHA-256 khi có, thư mục chứa cảnh trong file zip, các thư mục cần giải nén), hoặc tự đưa lên đúng thư mục gốc trong cấu hình (hoặc trỏ cấu hình tới chỗ đã có, như trên). Lệnh train không tự tải: thiếu thì dừng trước khi train, báo `dataset not found` kèm thư mục, các cảnh cần có và lệnh tải (`locate_dataset` trong `scripts/lib/benchmark.sh`).

## 6. Pod mới, đổi GPU, chạy lại

1. Tạo network volume và pod (`docs/USAGE_GUIDE.md` mục 1).
2. Tải `app/notebooks/runpod.ipynb` lên `/workspace`. Chạy theo thứ tự:
   - ô 1: kiểm tra máy;
   - ô 2: khóa deploy chỉ đọc, thêm vào GitHub (một lần cho mỗi volume);
   - ô 3: clone code vào `/workspace/SpatialRoom3D` (lần sau: `git pull --ff-only`);
   - ô 4: `bash scripts/pod/run_all.sh --background` (chỉ cài môi trường);
   - đưa dữ liệu của bạn lên, rồi ô 6: kiểm tra dữ liệu (chỉ đọc); dữ liệu benchmark tải được ở ô 7;
   - ô 7 (thực nghiệm chính, tải dữ liệu, benchmark) hoặc ô 8 (cảnh tự chụp) với dữ liệu bạn chọn.
3. Pod mới gắn lại cùng volume: chạy lại ô 1, 3 (cập nhật code) và 4 (env đã có thì chỉ kiểm tra lại).
4. GPU khác loại:
   - `setup_gsplat.sh` tự build lại gsplat và fused-ssim khi compute capability của GPU khác lần build trước (ghi trong dấu build của env);
   - GPU không có trong `TORCH_CUDA_ARCH_LIST` (`configs/environment/*.conf`) thì script cài đặt dừng và liệt kê các GPU dùng được.
5. Chạy lại không train lại: quyết định bỏ qua một lượt nằm ở `src.training.run`. Lượt OK cũ được bỏ qua khi yêu cầu mới giống hệt: cùng tên cảnh, cùng mô hình COLMAP (SHA-256 của `sparse/0`), cùng chế độ, protocol (cả protocol gốc), tham số, seed, ngân sách Gaussian và tập test. **Không so đường dẫn** hay tên module. B0 được bỏ qua khi `results.json` của cảnh có `ours_30000`.
   - Điều kiện: không sửa mục `train_modes` của `configs/base.yaml` hay mục `benchmark` của `configs/<benchmark>.yaml` giữa hai lần chạy (sửa thì lượt cũ khác cấu hình và bị báo lỗi, không bị ghi đè).

## 7. Nguồn gốc

Repo này bắt đầu ngày 19/10/2026 bằng một commit duy nhất, từ bản cấu trúc hoàn chỉnh của repo trước `truong0809/3D-Indoor-Scene-Reconstruction-from-Images` (commit `8037a66` cùng các thay đổi chưa commit ngày 17–19/10/2026). Lịch sử trước đó nằm ở repo cũ; repo cũ và volume cũ không bị sửa.

- Phần đã chạy trên pod ngày 17/10/2026 (cài 3 env, B0 Inria, gsplat `default` / `mcmc` trên Deep Blending, chạy nền và khóa GPU) được dời chỗ và đổi tên gói, không viết lại thuật toán. Hai chế độ, protocol và phiên bản CUDA / PyTorch / gsplat không đổi. Lệnh huấn luyện sinh ra cho Deep Blending giống lệnh đã chạy ngày 17/10/2026, chỉ khác đường dẫn và tên module (`tests/unit/test_benchmarks.py`).
- Phần viết sau ngày 17/10/2026, **chưa chạy trên pod**: đầu vào thư mục ảnh và các kiểm tra trước khi train, kết luận FAIL khi mô hình COLMAP không có điểm 3D, cấu hình YAML, bản xem dữ liệu cho `data_factor` > 1 (Mip-NeRF 360), render lại / xuất PLY, CLI, notebook cho volume trống, tải dữ liệu và thực nghiệm trọn vẹn, biến thể tham số, kiểm tra điểm SfM, lọc Gaussian. Các phần này chỉ được kiểm thử trên CPU (stub, runner giả, COLMAP bản CPU trên dữ liệu tổng hợp).
- Đã bỏ: phần dời dữ liệu của volume cũ, mọi phần tương thích bố cục cũ, lấy code từ file ZIP. Repo này dùng volume mới, trống.

Đối chiếu tên khi đọc báo cáo / log cũ:

| Lượt chạy trên pod 17/10/2026 (`8037a66` + bản vá v3) | Bản cấu trúc trước (chưa commit, ZIP 20261019) | Repo này |
|---|---|---|
| `src/indoor3d/{data,sfm,train,eval}/`, `python -m indoor3d.train.run` | `src/indoor3d/{input,reconstruction,training,evaluation}/` | `src/{input,reconstruction,training,evaluation}/`, `python -m src.training.run` |
| `configs/train_modes.json` | như cột trước | mục `train_modes` của `configs/base.yaml` |
| `configs/baseline_deepblending.json` | `experiments/deep_blending/benchmark.json` | mục `benchmark` của `configs/deep_blending.yaml` |
| — | `configs/datasets.json`, `src/indoor3d/datasets.py` (luồng `custom_scenes`) | `configs/<luồng>.yaml`, `src/datasets/registry.py` (luồng `custom_scene`) |
| `scripts/run_baseline_deepblending.sh` | `experiments/run_inria_b0.sh deep_blending` | `scripts/run_train.sh deep_blending b0` |
| `scripts/run_modes_deepblending.sh` (`MCMC_DB_SET`) | `experiments/run_gsplat_modes.sh deep_blending` (`MCMC_SET`) | `scripts/run_train.sh deep_blending gsplat` (`MCMC_SET`) |
| — | `experiments/comparisons/compare.sh` | `scripts/run_evaluate.sh` |
| `scripts/prepare_scene.sh`, `scripts/train_scene.sh` | như cột trước | `scripts/prepare_data.sh custom`, `scripts/run_train.sh custom` (phần thực thi: `scripts/pipeline/`) |
| `scripts/{run_all,run_scene,job,results,pack_results}.sh` | như cột trước, thêm `scripts/check_datasets.sh` | `scripts/pod/...` |
| `scripts/{env_probe,setup_*}.sh`, `scripts/smoke_test_*.py`, `environment/` | như cột trước | `scripts/setup/...`, `configs/environment/` |
| `chay_runpod_v3.ipynb` (ngoài Git) | `app/runpod.ipynb`, `app/README.md` | `app/notebooks/runpod.ipynb`, `docs/USAGE_GUIDE.md` |
| `/workspace/repo` | `/workspace/SpatialRoom3D` | `/workspace/SpatialRoom3D` |
| `data/tandt_db/db`, `outputs/baseline_inria/db`, `outputs/modes_db`, `outputs/modes_db_quick` | `data/benchmarks/deep_blending`, `outputs/benchmarks/deep_blending/{inria_b0,gsplat,gsplat_quick}` | như cột trước |
| `data/raw/`, `data/scenes/<cảnh>`, `outputs/modes/<cảnh>` | `data/custom_scenes/<cảnh>/`, `outputs/custom_scenes/<cảnh>/{colmap,<chế_độ>_s<seed>}` | như cột trước |
| `/workspace/{reports,logs,jobs}` | như cột trước | `/workspace/outputs/{reports,logs,jobs}` |
| `INDOOR3D_*`, `/run/lock/indoor3d-gpu.lock` | như cột trước | `SR3D_*`, `/run/lock/spatialroom3d-gpu.lock` |
