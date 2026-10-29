# So sánh các chế độ huấn luyện (`default`, `mcmc`, chế độ tự thêm)

`scripts/run_evaluate.sh` không train, không dùng GPU, không sửa file nào của lượt: chỉ đọc kết quả đã có, và ghi thêm `<lượt>/splat_stats.json` (thống kê PLY cuối, xem dưới). Script gọi `src.evaluation.compare_modes` với đúng thư mục kết quả của từng luồng, lấy từ `configs/<luồng>.yaml`.

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
- `BASELINE`: chế độ làm mốc (mặc định `default`). Khác `default` thì không đối chiếu với Inria, vì phần đó chỉ dành cho baseline `default`. Tên chế độ chỉ có một biến thể là đủ, vd. `mcmc` cho biến thể `mcmc[opacity-reg=0.001]` của Deep Blending. Chế độ có nhiều biến thể (vd. `mcmc` cùng các biến thể có tag của `tune_mcmc`) thì mốc là lượt không có tag.
- `TAGS`: đưa cả các lượt có tag này vào bảng (biến thể của một kế hoạch thực nghiệm, vd. `TAGS="or0001 rbg"` cho `tune_mcmc`). Mặc định lượt có tag bị để ngoài, kèm một dòng cảnh báo ghi số lượt và các tag: bảng của thực nghiệm chính không lẫn các lượt dò tham số nằm cùng thư mục.
- `REPORT`: tên file báo cáo (không có đuôi).
- `SPLAT_STATS=0`: không tính thống kê PLY; bảng cấu trúc mô hình chỉ có các lượt đã tính trước đó.
- `CHARTS=0`: không vẽ biểu đồ (mục Biểu đồ, dưới).

Báo cáo: `$WS/outputs/reports/compare_<tên>_<thời_gian>.{md,json}`, biểu đồ trong `compare_<tên>_<thời_gian>_charts/`. Với benchmark, phần cấu hình benchmark đã dùng (số liệu công bố, ngưỡng) được lưu kèm ở `$WS/outputs/reports/configs/<benchmark>_<thời_gian>.json`.

Bảng gồm:
- số liệu từng cảnh × biến thể (trung bình theo seed);
- chênh lệch so với baseline (`default`, hoặc chế độ trong `BASELINE`), chỉ tính trên các seed chung;
- các mục điều kiện: protocol, tập test, commit gsplat, GPU, ảnh gsplat thực dùng, số seed, ngân sách Gaussian;
- với benchmark và baseline `default`: đối chiếu với số liệu công bố (khi đủ mọi cảnh của benchmark), và với B0 tự chạy ở từng cảnh;
- khi có lượt đã lọc Gaussian (`<lượt>/filtered/`, `scripts/run_filter.sh`): bảng sau khi lọc, gồm chỉ số của bản đã lọc, chênh lệch so với chính lượt đó trước khi lọc (cùng ảnh test), số Gaussian giữ lại, PLY trước → sau, thời gian render mỗi ảnh. Mọi bảng khác dùng mô hình chưa lọc (`docs/design/experiments.md` mục 4);
- số cặp (cảnh, seed) và số cảnh mà mỗi biến thể tốt hơn baseline ở từng chỉ số (PSNR / SSIM cao hơn, LPIPS thấp hơn; bằng nhau không tính). Cảnh: chênh lệch trung bình trên các seed chung của cảnh, như bảng chênh lệch. Kèm chênh lệch trung bình ± độ lệch chuẩn theo cặp, và các cảnh chưa đủ điều kiện (cùng các mục của bảng chênh lệch). Đọc cùng độ lệch chuẩn: nhiều cặp tốt hơn với chênh lệch nhỏ hơn độ lệch chuẩn chưa phải "cải thiện";
- chất lượng theo dung lượng: mọi bản của mô hình trong một cảnh (mỗi biến thể, và bản đã lọc của nó), xếp theo dung lượng PLY, kèm byte mỗi Gaussian (theo bậc SH: 236 byte ở bậc 3), số Gaussian và chỉ số. Mọi dòng của một cảnh tính trên cùng các seed chung (cột Seed); bản đã lọc chỉ có khi mọi lượt trên các seed đó đã lọc xong, nếu không thì một dòng ghi chú nói vì sao (số lượt đã lọc xong, lọc lỗi, chỉ lọc ở seed ngoài các seed chung). Cảnh mà các biến thể không có seed chung cũng có ghi chú thay cho các dòng. Cột "Không bị vượt": ✓ khi không bản nào nhẹ hơn hoặc bằng có PSNR và LPIPS đều tốt hơn hoặc bằng (hơn hẳn ở ít nhất một mục), tức những bản đáng cân nhắc khi cần file nhẹ; `-` khi dòng thiếu dung lượng hoặc chỉ số;
- khi có `splat_stats.json`: cấu trúc mô hình (bậc SH; tỉ lệ Gaussian có opacity < 0,005 / 0,02 / 0,1; opacity trung vị; tỉ lệ Gaussian có trục lớn nhất > 0,1 × `scene_scale`; trục lớn nhất trung vị), trên các seed mà mọi biến thể có thống kê của cảnh đều có. Đây là thước đo gián tiếp của lớp mờ và floater: Gaussian gần trong suốt hay rất to chưa chắc là floater, và floater thật chỉ thấy qua ảnh, video độ sâu (ảnh so sánh, dưới).

Cột `Render (ms/ảnh)` (bảng chính, bảng sau khi lọc) là thời gian gsplat đo khi chấm ảnh test, trong tiến trình chấm riêng, gồm cả lần gọi đầu (nạp CUDA). Ở repo trước, trên RTX 4090 (Deep Blending, gói kết quả ngày 17/10/2026), con số này khoảng 60–95 ms/ảnh cho mọi mô hình từ 0,12 đến 3,7 triệu Gaussian: nó không phản ánh tốc độ render, nên không đổi ra FPS. Tốc độ thật cần một phép đo riêng (làm nóng rồi đo nhiều lần), chưa có; FPS của trình xem web là mục tiêu O5 của `docs/PROJECT_PLAN.md`, đo trên máy demo.

**Thống kê PLY** (`src/evaluation/splat_stats.py`): `run_evaluate.sh` tính cho mọi lượt dưới các thư mục kết quả của lần lập bảng (với `SCENES`: chỉ các cảnh đó), trước khi lập bảng, bằng python của env `gs-gsplat` (cần numpy; không có thì python3 của máy nếu có numpy; không có nữa thì in `NOTE` và bỏ qua).
- Lượt đã có thống kê, PLY cuối không đổi (cùng kích thước, cùng thời điểm sửa) và cùng `scene_scale` thì bỏ qua; `SPLAT_STATS=force` tính lại tất cả. Lần đầu phải đọc mọi PLY cuối (vài GB mỗi cảnh), các lần sau dùng lại.
- `scene_scale` lấy từ `filtered/filter_meta.json` nếu có, nếu không từ dòng `Scene scale:` trong `train.log`; không có thì các cột theo `scene_scale` để trống.
- Gói kết quả nhẹ có `splat_stats.json` mà không có PLY: thống kê vẫn được dùng (là của PLY lúc tính), bảng ghi số lượt không kiểm lại được. PLY còn đó mà đã đổi thì thống kê cũ bị bỏ.
- PLY đọc lỗi thì in `WARN`, bảng vẫn được lập, lượt đó không có dòng cấu trúc.

## Biểu đồ: `src/evaluation/charts.py`

Biểu đồ tổng, vẽ từ chính file JSON của báo cáo (không đọc lại lượt chạy): mỗi phương pháp hơn hay kém mốc bao nhiêu khi gộp mọi cảnh và seed, và mốc có khớp Inria không. Số từng cảnh, chi phí, dung lượng: ở các bảng của báo cáo.

| Ảnh (theo thứ tự) | Vẽ khi | Nội dung |
|---|---|---|
| `01_doi_chieu_inria.png`: đối chiếu `default` với Inria | báo cáo đã đối chiếu (PASS / FAIL) | chênh lệch của `default` so với B0 tự chạy từng cảnh và với số công bố, vùng ngưỡng, ✓ / ✗; bảng ghi Inria → mốc. Đứng đầu: đây là điều kiện để dùng `default` làm mốc. Gộp nhiều báo cáo: mỗi báo cáo đã đối chiếu một khung; báo cáo không đối chiếu được (vd. 4 cảnh của Mip-NeRF 360: SKIPPED) ghi lý do ở đầu ảnh |
| `NN_so_voi_<mốc>.png`: chênh lệch tổng so với `<mốc>` | có phương pháp so được với mốc; mỗi mốc một ảnh (`default`; `mcmc` khi có bảng so với `mcmc`, vd. `tune_mcmc`) | mỗi phương pháp một thanh: chênh lệch trung bình trên mọi cặp (cảnh, seed) chung với mốc; chấm đen: từng cặp; số ghi: chênh lệch trung bình · số cặp tốt hơn / số cặp. Một báo cáo: đúng các số ở bảng số cặp tốt hơn của nó. Gộp nhiều benchmark: dòng `<phương pháp>: tổng` gộp mọi cặp, thêm một dòng (thanh nhạt) cho mỗi benchmark |
| `01_b0_so_voi_cong_bo.png`: B0 so với số liệu Inria công bố | báo cáo chỉ B0 (`compare_reference`) | B0 từng cảnh, trung bình (kèm chênh lệch) và số công bố với vạch ngưỡng |

- Nơi vẽ: `run_evaluate.sh`, `scripts/pipeline/train_gsplat_modes.sh`, `scripts/pipeline/train_inria_b0.sh` vẽ ngay sau khi lập bảng, vào `<báo cáo>_charts/` cạnh file JSON. `scripts/pod/run_experiment.sh` vẽ một lần ở cuối cho cả lần chạy, từ mọi bảng `<tên bản>_<thời_gian>_<benchmark>[_vs_<mốc>].json` (bảng cùng mốc được gộp), vào `<tên bản>_<thời_gian>_charts/`; từng bảng, kể cả bảng của các bước B0 và gsplat bên trong (`baseline_*`, `modes_*`), không có biểu đồ riêng. Báo cáo `.md` (của thực nghiệm: `<tên bản>_<thời_gian>.md`) có thêm mục "Biểu đồ" ở cuối, dẫn tới các ảnh bằng đường dẫn tương đối.
- Gộp theo nhãn của phương pháp (tên thư mục lượt bỏ `_s<seed>`): `mcmc` của Deep Blending (`mcmc[opacity_reg=0.001]`) và của Mip-NeRF 360 cùng là `mcmc`; `index.md` ghi nhãn ứng với biến thể nào ở báo cáo nào. Hai báo cáo có chung lượt thì không gộp (mỗi cặp chỉ tính một lần). Dòng tổng tính mọi cặp như nhau, nên benchmark nhiều cảnh hơn nặng hơn: đọc cùng các dòng của từng benchmark.
- `index.md` của mỗi thư mục: cách đọc và bảng số của từng ảnh (kèm độ lệch chuẩn theo cặp), các ảnh không vẽ kèm lý do (thiếu dữ liệu; phần báo cáo cần cho ảnh hỏng hoặc lập bằng bản cũ của `compare_modes`; vẽ lỗi). Một ảnh không vẽ được thì ảnh khác vẫn vẽ. Lập lại thì ảnh và mục "Biểu đồ" được thay, không nhân đôi; chỉ ảnh và `index.md` do lần trước ghi bị thay, thư mục có `index.md` không do module này ghi thì dừng.
- Cần matplotlib: python của env `gs-gsplat` (`matplotlib==3.10.5`, ghim trong `configs/environment/gs-gsplat.lock.txt`), không có thì python3 của máy nếu có; luôn chạy với `MPLBACKEND=Agg` (biến `MPLBACKEND` mà Jupyter đặt cho tiến trình con làm matplotlib báo lỗi khi import). Không có matplotlib thì in `NOTE` và chỉ có bảng. `CHARTS=0`: không vẽ. Vẽ lỗi chỉ in `WARN`, bảng không đổi.
- Trong notebook: ô "Bảng" của `b0_inria.ipynb`, `gsplat_default.ipynb`, `gsplat_mcmc.ipynb` và ô 7 của `runpod.ipynb` (`WHAT = "report"`: báo cáo mới nhất của kế hoạch `EXP`; `WHAT = "evaluate"`: ngay sau bảng vừa lập) hiện bảng kẻ sẵn rồi các ảnh. Gói kết quả nhẹ có cả các ảnh; `scripts/pod/results.sh` in đường dẫn thư mục biểu đồ của thực nghiệm gần nhất.

Quy ước trên ảnh:
- Mọi khung chênh lệch đều "phải là tốt hơn": trục LPIPS đảo chiều (tiêu đề khung ghi "trục đảo"), số ghi vẫn là chênh lệch thật (ΔLPIPS âm là tốt hơn); thanh bắt đầu từ 0. Khung đối chiếu Inria không đảo chiều: ở đó cần gần 0.
- Số thập phân dấu phẩy, số âm dấu trừ "−", số chữ số như bảng (PSNR 3, SSIM và LPIPS 4); chênh lệch luôn có dấu như bảng.
- Màu theo phương pháp: `mcmc` cam, các biến thể khác xanh ngọc (`default` xanh dương, B0 xám đậm), dòng của một benchmark nhạt hơn dòng tổng; xanh lá / đỏ chỉ cho trong / ngoài ngưỡng, luôn kèm ✓ / ✗. Màu đã qua bộ kiểm tra bảng màu của skill dataviz (phân biệt được cả với người mù màu); xanh ngọc tương phản thấp với nền nên mọi thanh đều có nhãn số.
- Đầu mỗi ảnh ghi mốc, các cảnh, seed, các mục điều kiện so với mốc chưa đạt và file báo cáo nguồn, để ảnh tách khỏi báo cáo (vd. chèn vào báo cáo đồ án) vẫn đọc đúng. Biểu đồ không tự kết luận phương pháp nào tốt hơn.
- Biểu đồ của `main` hiện chênh lệch `default` / `mcmc` trên cả các cảnh dùng để kết luận, nổi hơn bảng; khi in báo cáo của `main`, notebook in thêm một dòng nhắc. Trước khi chọn `mcmc_indoor` (`docs/design/experiments.md` mục 3.6, bước 5), phần của `main` chỉ dùng để kiểm điều kiện: các bước OK (bảng các bước), `default` so với B0 từng cảnh và với số công bố (`01_doi_chieu_inria.png`), B0 so với số công bố (`01_b0_so_voi_cong_bo.png` của lần chạy `--only b0`, hoặc bước `b0_<benchmark>` OK), đủ seed (cột n và cột Seed của bảng), độ lệch chuẩn của `default`, `mcmc` trên `room` (bảng đầu). Chênh lệch trên các cảnh khác đọc ở P9 (`docs/PROJECT_STATUS.md`).
- Khi `main` có `mcmc_indoor` (chọn bằng `room`): các dòng `mcmc_indoor` của ảnh tổng gộp cả `room`. Kết luận về `mcmc_indoor` dùng bảng và biểu đồ không có `room` (`docs/design/experiments.md` mục 3.1).

## Ảnh so sánh: `scripts/run_panels.sh`

Ảnh so sánh nhiều lượt trên cùng ảnh test và video quỹ đạo của các lượt xếp chồng (`src/evaluation/panels.py`). Không train, không dùng GPU: chỉ đọc ảnh và video mà gsplat đã ghi khi chấm (`<lượt>/renders/`, `<lượt>/videos/`).

```bash
bash scripts/run_panels.sh mipnerf360 counter                          # các lượt seed 42 không có tag của cảnh
bash scripts/run_panels.sh mipnerf360 room mcmc_s42 mcmc_or005_s42     # chọn lượt (vd. biến thể của tune_mcmc)
bash scripts/run_panels.sh mipnerf360 room mcmc_s42 mcmc_s42/filtered  # một lượt và bản đã lọc của nó
IMAGES="0 12 24" CROP=0.5,0.4,0.25,0.25 VIDEO=1 bash scripts/run_panels.sh custom phong1
```

- Mỗi panel: hàng đầu là ảnh gốc (khung đỏ là vùng cắt) và vùng cắt; mỗi lượt một hàng: ảnh render kèm PSNR, vùng cắt phóng to (giữ nguyên điểm ảnh), bản đồ sai số |render − gốc| (trung bình 3 kênh, × 4, cùng thang cho mọi lượt).
- Lượt mặc định: các lượt đã xong (status OK), đúng protocol chung (như bảng so sánh mặc định), không có tag, của seed `SEED` (42), có ảnh render; bỏ thư mục cũ `.prev_*`. Lượt có tag, lượt khác protocol và bản đã lọc (`<lượt>/filtered`) chỉ vào khi chỉ rõ tên. `QUICK=1`: các lượt chạy thử của benchmark.
- Các lượt phải có cùng các ảnh test với cùng ảnh gốc (cùng cảnh, cùng tập test, cùng độ phân giải); nếu không thì dừng, không lặng lẽ bỏ bớt ảnh. Chấm ở bước khác nhau (vd. một chế độ 40.000 vòng chỉ rõ tên) thì chỉ cảnh báo; `panels.md` ghi bước của từng lượt.
- Vùng cắt: `CROP` (tỉ lệ x, y, rộng, cao), hoặc tự chọn theo một quy tắc cố định: ô 1/4 × 1/4 khung hình có tổng sai số (cộng trên mọi lượt) lớn nhất. Ô đó thường là chỗ hỏng nặng nhất của lượt kém nhất, và thêm hay bớt lượt có thể làm đổi ô; khi so một chỗ cụ thể thì đặt `CROP`. Quy tắc chọn ảnh và vùng cắt ghi trong `panels.json`. Ảnh mặc định: ảnh test đầu, giữa và cuối.
- PSNR trong panel và `panels.md` tính lại từ PNG 8 bit, chỉ để chọn và chú thích ảnh; chỉ số chính thức là bảng của `run_evaluate.sh`. `panels.md` có PSNR (PNG) của mọi ảnh test.
- `VIDEO=1`: ghép `videos/traj_<bước>.mp4` (nửa trái màu, nửa phải độ sâu) của các lượt thành `traj_compare.mp4`, mỗi lượt một hàng có nhãn, dừng ở video ngắn nhất; video quá cao thì thu nhỏ để không quá 4.096 px. Ghi bằng H.264 (imageio-ffmpeg của env `gs-gsplat`, phát được trong trình duyệt), không có thì mp4v của OpenCV. Floater thường thấy rõ nhất ở nửa độ sâu, nhưng độ sâu được gsplat chuẩn hóa theo từng khung của từng lượt: so hình dạng, không so giá trị giữa các hàng. Lượt thiếu video hoặc không ghi được video thì chỉ cảnh báo, ảnh so sánh vẫn giữ.
- Kết quả: `$WS/outputs/panels/<luồng>[_quick]_<cảnh>_<thời gian>/` (`panel_<i>.png`, `panels.json`, `panels.md`, `traj_compare.mp4`). Thư mục này không nằm trong gói kết quả nhẹ (gói chỉ lấy `panels.json`): tải ảnh và video qua thanh file của Jupyter. Trong notebook: ô 7 với `WHAT = "panels"` (benchmark), ô 9 với `SO_SANH = True` (cảnh tự chụp).

Báo cáo không tự kết luận chế độ nào tốt hơn. Chỉ gọi là "cải thiện" khi có ít nhất 3 seed, mọi điều kiện đạt (ngân sách Gaussian EQUAL) và kết quả nhất quán trên các cảnh (`docs/design/training_modes.md` mục 4.3).

Đã kiểm tra trong repo trước, trên kết quả thật của pod ngày 17/10/2026: cùng mã so sánh cho ra báo cáo giống hệt `reports/modes_db_20261017_162815.md`. Repo này chưa có kết quả nào trên pod (`docs/PROJECT_STATUS.md`).
