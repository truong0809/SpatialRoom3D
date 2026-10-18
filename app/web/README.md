# app/web — giao diện web (chưa làm)

Trạng thái: **chưa có code**. Thư mục giữ chỗ cho giai đoạn 5 của kế hoạch (`docs/PROJECT_PLAN.md`): ứng dụng demo với trình xem 3D tương tác.

Dự kiến, chỉ làm khi luồng chính đã chạy ổn định trên pod:
- tải ảnh / video lên và theo dõi tiến độ xử lý;
- xem mô hình 3D (file PLY của một lượt train) trong trình duyệt.

Hiện tại:
- chạy bằng notebook `app/notebooks/runpod.ipynb` hoặc giao diện dòng lệnh `python -m app.cli`;
- mô hình 3D của một lượt: `bash scripts/run_render.sh custom <cảnh>` đưa file PLY vào `/workspace/outputs/checkpoints/custom_scene/<cảnh>/`, tải file đó về rồi mở bằng một trình xem 3D Gaussian Splatting.
