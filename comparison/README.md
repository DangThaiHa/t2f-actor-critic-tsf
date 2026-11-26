# Đối chứng ETTm, dự báo 30 bước

Hai trạm ETTm1 và ETTm2, 464 cửa sổ test mỗi trạm, 30 bước mỗi cửa sổ (15 phút/bước). Mỗi dòng dự báo có khóa trạm, thời điểm xuất phát và bước dự báo. Em kiểm tra thời gian và giá trị OT thực với dữ liệu CSV gốc trước khi chấm trên thang nhiệt độ gốc.

T2f dùng lớp T2f của tác giả với lớp tích hợp src/model.py của dự án (điều chỉnh phép tính reward), năm mô hình thành phần, chế độ level và 8 epoch. Mỗi trạm dùng 512 mốc cuối của tập train. Mỗi lần dự báo dùng 32 mốc đầu vào như cấu hình giữa kỳ. Em chạy ba seed 42, 43 và 44, chấm từng seed rồi lấy trung bình chỉ số.

Stacking-full dùng dự báo đã lưu của dự án. Các mô hình thành phần dùng tập train 48.776 mốc/trạm. Đối chứng Stacking-512 giới hạn các cửa sổ nhãn huấn luyện vào 512 mốc cuối, vẫn được dùng lịch sử trước đó cho biến trễ. Cả hai cấu hình Stacking học trọng số trên phần 5% sau tập train. Stacking có lịch sử OT, đặc trưng trễ và lịch. T2f nhận sáu biến tải tại thời điểm dự báo. So sánh này giữ chung tập test và chân trời, còn lượng thông tin và cách huấn luyện khác nhau.

Để chạy lại, đặt compare_ettm.py ở thư mục riêng, mở terminal tại thư mục gốc bản t2f-final đã giải nén và cài requirements.txt. Trên Linux dùng lệnh sau (thay đường dẫn tới script):

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /path/to/compare_ettm.py
```

predictions.csv.gz chứa dự báo theo từng điểm. scores.csv chứa MAE, RMSE theo trạm và trên toàn bộ test. summary.csv ghi MAE, RMSE và trung vị sai số tuyệt đối của mỗi lượt chạy. Thí nghiệm sử dụng PyTorch 2.6.0+cpu.
