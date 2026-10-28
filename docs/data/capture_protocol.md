# Quy trình quay video và chụp ảnh không gian nội thất

_Phiên bản 1 · 09/10/2026 (mục 8 về chụp ảnh: 18/10/2026) · Áp dụng cho dữ liệu tự thu (mục tiêu O3)._

Mục tiêu: có video đủ nét, đủ góc nhìn và đủ ổn định để ước lượng camera (SfM) thành công và huấn luyện 3DGS. Các con số dưới đây là điểm xuất phát; chúng sẽ được hiệu chỉnh sau lần quay thử đầu tiên (mốc M2) và cập nhật lại tài liệu này.

## 1. Cài đặt điện thoại

| Hạng mục | Khuyến nghị | Lý do |
|---|---|---|
| Ống kính | Camera chính (1x) | Ít méo và ít nhiễu hơn ống góc siêu rộng |
| Độ phân giải / tốc độ | 4K 30 fps (hoặc 1080p 30 fps nếu máy yếu) | Đủ chi tiết; 3DGS gốc tự giảm ảnh về khoảng 1,6K chiều ngang |
| Phơi sáng, lấy nét | **Khóa** (AE/AF lock hoặc chế độ Pro) | Phơi sáng thay đổi tạo vệt màu; lấy nét thay đổi làm đổi tham số camera |
| Cân bằng trắng | Cố định (không để Auto nếu chỉnh được) | Màu ổn định giữa các frame |
| Chống rung điện tử | **Tắt** nếu có thể (giữ chống rung quang học) | Chống rung điện tử cắt và bóp méo ảnh theo từng frame, gây sai cho SfM |
| HDR video, chế độ chân dung, bộ lọc | Tắt | Xử lý ảnh khác nhau giữa các frame |
| Định dạng ảnh (nếu chụp ảnh) | JPEG ("Most Compatible" trên iPhone) | Pipeline chỉ nhận JPG/JPEG/PNG; ảnh HEIC bị bỏ qua (xuất sang JPG trước) |

## 2. Chuẩn bị không gian

- Bật đủ đèn; giữ nguyên rèm và đèn trong suốt lượt quay. Tránh nắng chiếu trực tiếp đang di chuyển.
- Không có người hoặc vật di chuyển trong khung hình (kể cả quạt đang quay, rèm bay).
- Tắt màn hình TV/máy tính nếu được; hạn chế gương và kính phản chiếu người quay.
- **Quyền riêng tư:** chỉ quay nơi được phép; cất giấy tờ, ảnh cá nhân, màn hình có thông tin riêng.
- Không thay đổi đồ đạc giữa lượt quay chính và lượt quay kiểm tra.
- Tùy chọn — tham chiếu tỉ lệ: đặt một vật biết kích thước (tờ A4, thước) trong cảnh và đo bằng thước một kích thước của phòng. Dùng để kiểm tra tỉ lệ, vì SfM từ camera thường không biết đơn vị mét.

## 3. Cách di chuyển khi quay (lượt quay chính)

1. **Đi chậm và đều**, như đi bộ thong thả. Xoay người từ từ, không lia nhanh.
2. **Luôn có dịch chuyển:** không đứng một chỗ xoay tròn — SfM cần thị sai để tính độ sâu.
3. **Vòng ngoài:** đi dọc theo tường, quanh phòng, camera hướng vào giữa phòng và phía đối diện.
4. **Vòng trong:** đi vòng ở giữa phòng, camera hướng ra tường và đồ đạc.
5. **Nhiều độ cao:** lặp lại một vòng ở độ cao ngang ngực hoặc thấp hơn tầm mắt, để thấy được mặt bàn, gầm ghế.
6. **Khép vòng:** kết thúc ở gần vị trí bắt đầu, nhìn lại cảnh ban đầu.
7. Hai frame liên tiếp phải chồng lấp nhiều (phần lớn khung hình vẫn còn thấy ở frame sau).
8. Với tường trơn: để khung hình luôn có thêm đồ vật hoặc cạnh tường, góc trần, tránh quay sát một mảng tường trống.
9. Thời lượng gợi ý: 1–3 phút cho một phòng.

## 4. Lượt quay kiểm tra (bắt buộc cho đánh giá)

Quay thêm một đoạn ngắn 10–20 giây, **đi theo đường khác** với lượt chính (vị trí và độ cao khác), trong cùng điều kiện ánh sáng.

Đoạn này chỉ dùng làm tập test. Lý do: nếu lấy frame test xen giữa các frame của cùng một video, frame test gần như trùng frame train, điểm số sẽ bị thổi phồng (xem `docs/PROJECT_PLAN.md`, mục 4.2).

## 5. Quản lý file

- **Không cắt, nén lại hay chỉnh sửa video**; giữ file gốc từ điện thoại.
- Đặt tên: `<mã_cảnh>_<yyyymmdd>_<main|test>_<lần>.mp4`, ví dụ `bedroom01_20261012_main_1.mp4`.
- Ghi thông tin mỗi cảnh vào bảng metadata:

| Trường | Ví dụ |
|---|---|
| Mã cảnh | bedroom01 |
| Mô tả, diện tích ước lượng | Phòng ngủ, khoảng 12 m² |
| Thiết bị, ống kính | Điện thoại X, camera chính 1x |
| Cài đặt | 4K30, khóa AE/AF, tắt chống rung điện tử |
| Ánh sáng | Đèn trần + đèn bàn, rèm đóng |
| Thời lượng | main 2:10, test 0:15 |
| Tham chiếu tỉ lệ | Tờ A4 trên bàn; chiều dài phòng 3,62 m |
| Ghi chú | Gương tủ quần áo ở góc phải |

- Dữ liệu thô không đưa lên Git; lưu trên network volume (`/workspace/data/custom_scenes/<tên_cảnh>/`) và một bản sao riêng của bạn.

## 6. Tự kiểm tra ngay sau khi quay

- Tua ngẫu nhiên vài chỗ: ảnh có nét không, có bị nhòe khi xoay không?
- Độ sáng có nhảy giữa các đoạn không?
- Đã phủ hết các bức tường, góc phòng và đồ đạc chính chưa?
- Có người hoặc vật chuyển động lọt vào không?

Nếu một trong các mục trên không đạt, quay lại ngay khi còn ở hiện trường sẽ rẻ hơn nhiều so với phát hiện sau khi đã chạy SfM.

## 7. Checklist nhanh

- [ ] Camera chính, 4K/1080p 30 fps
- [ ] Khóa phơi sáng, lấy nét; cân bằng trắng cố định
- [ ] Tắt chống rung điện tử, HDR, bộ lọc
- [ ] Không người/vật di chuyển; đã cất vật riêng tư
- [ ] Vòng ngoài + vòng trong + một vòng thấp, đi chậm, khép vòng
- [ ] Lượt quay kiểm tra 10–20 giây theo đường khác
- [ ] File gốc, đặt tên đúng quy ước, đã ghi metadata

## 8. Chụp ảnh thay cho quay video

Pipeline nhận một **thư mục nhiều ảnh** JPG/JPEG/PNG chụp cùng một căn phòng. Ảnh được dùng hết, không qua bước
chọn frame, rồi đi chung bước COLMAP và huấn luyện với video. Mục 1, 2 và 3 vẫn áp dụng; khác biệt:

| Hạng mục | Khuyến nghị | Lý do |
|---|---|---|
| Số ảnh | Tối thiểu 10 (ít hơn bị từ chối, một ảnh luôn bị từ chối); nên 30–100 ảnh cho một phòng | Cần nhiều góc nhìn chồng lấp để ước lượng camera và phủ hết căn phòng |
| Chồng lấp | Mỗi ảnh trùng khoảng 60–80 % khung hình với ảnh trước | COLMAP ghép ảnh qua các điểm chung |
| Di chuyển | Bước 30–50 cm giữa hai lần chụp; **không** đứng một chỗ xoay máy | Cần thị sai để tính độ sâu (báo cáo cảnh báo khi camera gần như đứng yên) |
| Chiều máy, ống kính | Cùng một chiều (toàn ngang hoặc toàn dọc), cùng ống kính 1x, không zoom, không chế độ chân dung / HDR | COLMAP dùng một camera chung cho cả cảnh; ảnh khác kích thước bị từ chối, ảnh khác ống kính làm ghép kém |
| Phạm vi | Phủ đủ mọi bức tường, góc phòng, đồ đạc chính; chụp cả vòng ngoài và vòng trong như mục 3 | Báo cáo cảnh báo khi có khoảng hướng ≥ 90 độ không ảnh nào nhìn tới |
| Nhiều phòng | Mỗi phòng một thư mục, một tên cảnh (job) | Ảnh của hai phòng không nối được sẽ bị COLMAP tách nhóm hoặc bỏ |
| Tên file | Giữ tên gốc của máy (IMG_0001...) | Thứ tự tên là thứ tự chụp; tập test lấy cứ 8 ảnh 1 theo thứ tự này |
| File ảnh | Dùng **file gốc** của máy (chép qua cáp, Google Drive / iCloud tải bản gốc), không dùng ảnh đã gửi qua ứng dụng nhắn tin, ảnh chụp màn hình hay ảnh đã chỉnh sửa | File gốc còn EXIF tiêu cự: pipeline giữ lại để COLMAP có tiêu cự ban đầu (thiếu thì báo cáo cảnh báo và COLMAP dễ ghép thiếu ảnh hơn) |

**Tải lên pod:**
1. Tạo thư mục `/workspace/data/custom_scenes/<tên_cảnh>/`.
2. Mở thư mục đó trong thanh file của Jupyter, rồi chọn nhiều ảnh cùng lúc trong hộp Upload (hoặc kéo thả nhiều file).
3. Chỉ ảnh nằm trực tiếp trong thư mục được đọc. Thư mục con, file ẩn và file không phải JPG/JPEG/PNG bị bỏ qua và được liệt kê trong báo cáo.
4. Chạy `bash scripts/pod/run_scene.sh /workspace/data/custom_scenes/<tên_cảnh> <tên_cảnh> --background`, hoặc ô 8 của `app/notebooks/runpod.ipynb` (hoặc ô 7 của `gsplat_default.ipynb`: chỉ `default`; của `gsplat_mcmc.ipynb`: `default` nếu chưa có, rồi `mcmc`). Ảnh gốc không bị sửa; mọi thứ sinh ra nằm trong `/workspace/outputs/custom_scenes/<tên_cảnh>/`.

**Sau khi chạy**, xem `outputs/custom_scenes/<tên_cảnh>/colmap/sfm_report.json`, hoặc ô 9 của notebook:
- kết luận COLMAP;
- ảnh không ghép được (kèm tên file gốc);
- nhóm ảnh bị tách;
- độ phủ góc nhìn.

Nếu kết luận là FAIL, hệ thống không train: chụp bổ sung theo gợi ý trong báo cáo rồi chạy lại với một tên cảnh mới.
