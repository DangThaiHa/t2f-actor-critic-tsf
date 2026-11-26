# Kết quả thực nghiệm cuối kỳ

Em đánh giá theo hai cách. Với `bridge`, em dùng lại 48 gốc dự báo của bài giữa kỳ để so sánh trực tiếp. Với `full`, em đánh giá trên toàn bộ lịch sử của bốn chuỗi ETT và 370 khách hàng ELD.

## Dữ liệu

| Dữ liệu gốc | Số chuỗi × mốc | Đích | Mốc train cộng các chuỗi | Số giá trị test ở chân trời chính |
|---|---:|---|---:|---:|
| ETTh1/2 | 2 × 17.420 | OT, H=2 giờ | 24.388 | 6.968 |
| ETTm1/2 | 2 × 69.680 | OT, H=30 bước 15 phút | 97.552 | 27.840 |
| ELD UCI | 370 × 140.256 lần đo 15 phút | Công suất trung bình 6 giờ, H=2 | 1.513.300 mốc 6 giờ | 432.160 |

Bốn CSV ETT chính thức có tổng cộng 174.200 dòng. Bản Zenodo giữa kỳ thiếu phần cuối ETTm2, nên giao thức full đọc trực tiếp dữ liệu từ nhóm phát hành ETT. ELD gốc gồm 51.894.720 giá trị; em gộp 24 phép đo thành một giá trị trung bình, giữ đủ 5.844 mốc/khách hàng. Giữ nguyên các số 0, kể cả thời gian trước khi khách hàng bắt đầu hoạt động. UCI biểu diễn công suất bằng kW. Các biến nhiệt độ và giá điện trong bản Zenodo không có trong nguồn UCI; mô hình full ELD chỉ dùng lịch sử phụ tải và lịch. Điểm ELD của hai giao thức không đặt cạnh nhau để tính phần trăm cải thiện.

## Cách chạy và chọn mô hình

Em chia mỗi chuỗi theo thứ tự thời gian. Cửa sổ dự báo chỉ được dùng khi toàn bộ nhãn nằm trong cùng một tập.

| Phần dữ liệu | Cách dùng |
|---|---|
| 70% đầu | Huấn luyện các mô hình thành phần |
| 5% tiếp theo | Học trọng số tổ hợp hoặc chọn epoch cho Transformer |
| 5% tiếp theo | Chọn phương pháp có MAE thấp nhất |
| 20% cuối | Đánh giá trên dữ liệu test |

Ở mỗi gốc dự báo, mô hình được dùng các quan sát đã có đến thời điểm đó. Trong `full`, các cửa sổ test không chồng nhau. Em bỏ đoạn cuối nếu số mốc còn lại ngắn hơn chân trời dự báo.

**Các mô hình thành phần.** Em dùng ba dự báo đơn giản (giá trị cuối, cùng thời điểm hôm trước và tuần trước) cùng NLinear, DLinear và Ridge lịch. Các mô hình tuyến tính học riêng cho từng chuỗi, với bước trượt bằng 1.

- NLinear lấy 96 giá trị gần nhất, trừ giá trị cuối trước khi dự báo rồi cộng lại. DLinear dùng trung bình trượt 25 bước để tách xu hướng và phần dư. Em triển khai hai mô hình theo Zeng và cộng sự, với hệ số phạt tương đối 10⁻⁶.
- Ridge lịch dự báo chênh lệch so với quan sát cuối. Đầu vào gồm lịch sử của chuỗi, độ trễ theo ngày và tuần, giờ và thứ trong tuần. Trên ETT, Ridge còn dùng sáu biến tải đã quan sát. Để kiểm tra ảnh hưởng của lượng dữ liệu, em so sánh Ridge học trên 512 mốc cuối với Ridge học trên toàn bộ tập train. Hai bản dùng cùng đặc trưng và cùng tập test.

**Cách kết hợp dự báo.** Uniform lấy trung bình sáu dự báo. Stacking học sáu trọng số không âm có tổng bằng 1, sao cho MAE sau khi chuẩn hóa theo độ lệch chuẩn tập train của từng chuỗi là nhỏ nhất. Actor-MAE dùng lớp Actor của tác giả, nhận thông tin về biến động, chênh lệch giữa các dự báo và lịch. Softmax chuyển đầu ra thành sáu trọng số. Em dùng 250 lần cập nhật với batch 128.

**Hai mô hình đối chứng.** TD3 dùng actor và hai critic của tác giả, với 400 lần cập nhật, batch 128 và hệ số chiết khấu 0,9. Phần thưởng là âm MAE đã chuẩn hóa. Hành động quyết định trọng số ở bước hiện tại. Chuỗi quan sát tiếp theo vẫn lấy từ dữ liệu thực. TD3 dùng cập nhật actor trễ, làm trơn hành động đích và cập nhật mềm.

PatchTST-small chỉ dùng OT của hai trạm ETT. Mỗi cửa sổ 96 bước được chuẩn hóa rồi chia thành 11 patch dài 16, dịch 8 bước mỗi lần. Encoder có một tầng, 32 chiều và 4 đầu attention. Mỗi seed chạy 6 epoch với batch 128, không qua tiền huấn luyện. Ridge lịch có thêm biến tải, nên chênh lệch MAE giữa hai mô hình còn chịu ảnh hưởng của đầu vào.

**Cách đọc dòng `selected`.** Em chọn giữa sáu mô hình thành phần, Uniform, Stacking và Actor-MAE bằng MAE ở phần validation thứ hai. TD3 và PatchTST-small được chấm riêng. Với Actor-MAE và TD3, em lấy trung bình dự báo từ ba seed 42, 43, 44 rồi mới tính sai số.

## Cùng tập kiểm tra với giữa kỳ

| Bộ dữ liệu | T2f sai phân + Ridge giữa kỳ | Sau hiệu chỉnh giữa kỳ | Đối chứng ngày | Cuối kỳ chọn trên validation | MAE cuối kỳ |
|---|---:|---:|---:|---|---:|
| ETTh | 1,1271 | 1,1216 | 2,4734 | NLinear | 0,6148 |
| ETTm | 1,1023 | 1,0657 | 2,2802 | Stacking | 0,8547 |
| ELD MT_211 | 67,7599 | 62,3781 | 14,8353 | Actor-MAE | 12,0027 |

Trước khi tính MAE, em ghép dự báo theo trạm, thời điểm xuất phát và bước dự báo, rồi kiểm tra giá trị thực ở hai bản. Hai cột giữa kỳ lấy trung bình MAE ba seed. ETTm giảm 19,8% so với bản giữa kỳ đã hiệu chỉnh. Trên ELD, so với baseline ngày mạnh nhất trong báo cáo cũ, mức giảm là 19,1%. Những con số này phản ánh thay đổi đồng thời về dữ liệu train, đầu vào và phương pháp. Để tách tác dụng của dữ liệu, xem cặp Ridge-512/Ridge-full trong bảng tiếp theo.

## Trên toàn bộ chuỗi gốc

Bảng sau dùng MAE test. ETTh và ETTm giữ đơn vị OT, ELD dùng kW. Các mô hình theo từng bộ được chấm trên cùng khóa dự báo.

| Phương pháp | ETTh H2 | ETTm H30 | ELD H2 | ETTh H96 | ETTm H96 |
|---|---:|---:|---:|---:|---:|
| last | 0.994 | 2.047 | 328.046 | 5.689 | 4.109 |
| daily | 2.620 | 2.623 | 40.520 | 3.601 | 2.623 |
| weekly | 4.561 | 4.560 | 55.541 | 4.574 | 4.560 |
| nlinear | 0.625 | 1.195 | 36.321 | 3.184 | 1.884 |
| dlinear | 0.623 | 1.192 | 40.633 | 3.138 | 1.918 |
| ridge_512 | 1.187 | 3.337 | 37.872 | 6.984 | 6.434 |
| ridge_calendar | 0.612 | 1.115 | 33.620 | 3.100 | 1.824 |
| uniform | 1.178 | 1.504 | 67.111 | 3.143 | 1.949 |
| stacked | 0.609 | 1.114 | 38.780 | 3.090 | 1.811 |
| actor_mean | 0.626 | 1.196 | 33.830 | 3.119 | 2.110 |
| TD3 | 0.623 | 1.144 | 35.706 | — | — |
| PatchTST-small | 0.649 | 1.176 | — | 3.753 | 2.013 |
| selected | 0.625 | 1.114 | 33.620 | 3.184 | 1.811 |

Trên ETTh H2, validation chọn NLinear. Stacking và Ridge lịch có MAE test thấp hơn NLinear một chút nhưng giữ nguyên quyết định chọn ban đầu. Trên ETTm H30, stacking đạt MAE 1,114 so với 2,047 của persistence. Tăng số mốc train cho cùng Ridge lịch đưa MAE từ 3,337 về 1,115. Trên ELD, Ridge lịch giảm MAE từ 40,520 của baseline ngày xuống 33,620 và thắng trên 330/370 khách hàng. Trong cấu hình ELD này, Ridge lịch cũng có MAE thấp hơn Actor-MAE (33,830 kW) và TD3 (35,706 kW).

Với H96, ETTh dự báo 4 ngày, ETTm dự báo 24 giờ. Đây là phép thử riêng về khả năng dự báo xa. Số liệu trong CSV gồm RMSE và sMAPE (thang 0–2, 0/0=0), ngoài MAE. Điểm tổng hợp trên ELD ở đơn vị gốc chịu tác động lớn từ khách hàng có công suất cao. Có thể xem toàn bộ điểm từng khách hàng trong by_series.csv.

## Kiểm tra và giới hạn

- Chênh lệch MAE được tính theo từng gốc rồi lấy trung bình cùng thời điểm qua các trạm. Khoảng bootstrap 95% dùng 1.000 lần lấy mẫu khối thời gian tròn, độ dài khoảng 7 ngày. So với persistence, khoảng cho ETTh H2 là [−0,439; −0,302], cho ETTm H30 là [−1,116; −0,752]. So với seasonal naïve ngày, ELD H2 có khoảng [−8,230; −5,713] kW. Khoảng này có điều kiện trên tập test đã chọn; chưa đánh giá một kịch bản drift sang giai đoạn khác.
- Kiểm tra năm local gốc trên cả ba tập giữa kỳ ở seed 42, chế độ level và delta_ridge, gồm cả trọng số từng gốc. Trên ELD level, SimpleVAR tạo đỉnh trị tuyệt đối khoảng 5,24 tỷ. Bốn mạng neural có đỉnh khoảng 200–216. MAE riêng VAR khoảng 112 triệu, gần bằng MAE của tổ hợp gốc. Ở lượt chạy này, sai số lớn chủ yếu đến từ SimpleVAR. Đối với delta_ridge, đỉnh của VAR là 288 và MAE 63,24. Hai chế độ thay đổi cả mục tiêu và ridge, vì vậy không quy toàn bộ chênh lệch cho riêng ridge.
- T2f giữa kỳ dùng dữ liệu và code Zenodo. Sáu phương pháp thành phần của bản cuối kỳ và lớp chuẩn hóa trọng số là thay đổi có chủ đích. Kết quả mới nên gọi là mở rộng và đối chứng của T2f. Chênh lệch với số đã công bố trong Bảng 3 của Sousa–Henriques không được quy đổi sang phần trăm do giao thức và thang đo khác nhau.
- Giới hạn còn lại: ELD gộp xuống 6 giờ, chưa thử 370 khách hàng trực tiếp ở tần suất 15 phút; ETT chỉ có hai trạm mỗi tần suất; chưa thực hiện nhiều mốc chia theo năm. Chưa đối chiếu benchmark long-horizon chuẩn trên các mốc 192/336/720.

## Tài liệu và dữ liệu

1. Sousa & Henriques, *T2f: Actor-critic reinforcement learning for time-series forecasting*, Neural Computing and Applications (2026). https://doi.org/10.1007/s00521-026-12209-6
2. Zeng và cộng sự, *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023. https://arxiv.org/abs/2205.13504 · https://github.com/cure-lab/LTSF-Linear
3. Nie và cộng sự, *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023. https://arxiv.org/abs/2211.14730
4. ETT: https://github.com/zhouhaoyi/ETDataset · ELD: https://doi.org/10.24432/C58C86 · Mã T2f: https://github.com/jfpsousa/t2f
