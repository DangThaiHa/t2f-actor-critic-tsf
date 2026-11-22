# Dự báo điện năng và nhiệt độ máy biến áp

Bài cuối kỳ phát triển từ lần tái lập T2f. Mã giữa kỳ ở `src/`, mã cuối kỳ ở `final/`. Bảng kết quả và cách đọc nằm trong [EXPERIMENTS.md](EXPERIMENTS.md). ZIP chứa đủ dữ liệu đã chuẩn bị để chạy offline.

## Chạy

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m final.compare --dataset ETTh
python -m final.compare --dataset ETTm
python -m final.compare --dataset ELD
```

Dùng `--bridge` để chấm trên đúng 48 gốc cũ, hoặc `--horizon 96` cho ETT trên toàn bộ bản gốc. Luôn đặt OPENBLAS_NUM_THREADS=1 và OMP_NUM_THREADS=1 trên CPU để tránh tốn thời gian cho ma trận nhỏ. Có thể dùng `--output THU_MUC` để giữ một bộ kết quả khác. Các lệnh riêng:

```bash
python -m final.td3 --dataset ETTm
python -m final.patch_transformer --dataset ETTm
python -m final.local_audit --dataset eld
python -m final.summarize
```

`td3` chạy đối chứng TD3 với sáu dự báo thành phần. `patch_transformer` hỗ trợ ETTh và ETTm (có thể thêm `--horizon 96`). `local_audit` tách năm local gốc tại seed 42 và kiểm tra với kết quả giữa kỳ. `summarize` đọc tệp dự báo đã chạy để kiểm tra MAE, RMSE, khóa trạm/thời điểm và xuất bảng tổng. Lệnh audit nhận tên `eld` viết thường theo code giữa kỳ.

Nếu clone từ GitHub, làm theo [MIDTERM.md](MIDTERM.md) để khôi phục `upstream/` và bộ dữ liệu giữa kỳ từ ZIP Zenodo. Chạy `python final/prepare.py` để lấy bốn CSV gốc ETT và tạo Parquet ELD từ UCI; có thể truyền `--eld-zip /duong/dan/uci.zip`.

## Kết quả chi tiết

- `results/final/comparison.csv`: điểm test ở từng phương pháp, bộ dữ liệu, chân trời và giao thức.
- `paired_midterm.csv`, `paired_blocks.csv`, `coverage.csv` trong cùng thư mục: so sánh ghép cặp, khoảng tin cậy và phạm vi dữ liệu.
- `full/`, `bridge/`: điểm validation/test, điểm theo chuỗi, cấu hình chọn mô hình, dự báo theo gốc và trọng số actor. Trong `forecasts.npz`, trục thứ nhất là gốc dự báo, trục thứ hai là bước tương lai. `origin` là thời điểm quan sát cuối. `selection.json` cho biết mảng nào chứa dự báo selected.
- `local_audit/`: dự báo và sai số của năm local gốc, trọng số do actor gốc chọn, trung bình đều và dự báo của T2f giữa kỳ.
- `data/final/sources.json`: nguồn và SHA-256 của bốn tệp ETT và ZIP UCI đã dùng.
