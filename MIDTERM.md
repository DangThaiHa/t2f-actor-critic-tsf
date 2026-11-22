# T2f forecasting

Thử nghiệm dự báo trên ETTh, ETTm và một khách hàng trong ELD bằng mã nguồn T2f của Sousa và Henriques. Ba cấu hình gồm dự báo mức (`level`), dự báo sai phân (`delta`) và sai phân có Ridge cho VAR (`delta_ridge`).

## Chạy thử

Giải nén gói dự án, mở terminal ở thư mục chứa README và chạy:

```bash
python -m pip install -r requirements.txt
python run_experiment.py --datasets ETTh --modes level --seeds 42 --profile quick
```

Chạy cấu hình chính trên ETTh và ETTm với ba seed:

```bash
python run_experiment.py
```

Trong Kaggle hoặc Colab, mở `notebooks/forecasting.ipynb` và thêm gói `t2f-forecasting.zip` làm input. Notebook ghi kết quả vào vùng làm việc của phiên và tạo ZIP ở ô cuối. Runtime cần PyTorch, pandas, NumPy, scikit-learn và pyarrow.

## Mã nguồn

| Tệp | Nội dung |
|---|---|
| `src/data.py` | Đọc dữ liệu và chuẩn hóa theo từng trạm |
| `src/model.py` | Tích hợp T2f, chỉnh reward và thêm RidgeVAR |
| `src/experiment.py` | Chia dữ liệu và huấn luyện theo từng seed |
| `src/evaluation.py` | Dự báo cuốn chiếu, baseline và tính sai số |
| `results/reference/summary.csv` | Điểm trung bình và độ lệch chuẩn của các lượt đã chạy |

Mỗi lượt mới lưu cấu hình, dự báo và bảng điểm trong `results/runs/<profile>/<dataset>/<mode>/<seed>/`. Dataset giữ tên ETTh, ETTm và eld ở tham số dòng lệnh. Tên tệp Parquet trong `data/` viết thường.

## Dữ liệu và mã của tác giả

ZIP đi kèm có sẵn dữ liệu trong `data/` và mã của tác giả trong `upstream/`. Git bỏ qua hai phần này cùng kết quả chạy mới; các bảng tham chiếu trong `results/reference/` được theo dõi. Khi clone từ GitHub, tải bản phát hành Zenodo của tác giả rồi khôi phục input một lần:

```bash
python -m pip install py7zr
python scripts/prepare_inputs.py /path/to/t2f-Zenodo.zip
```

Bộ chuẩn bị dữ liệu chỉnh import nội bộ và năm lệnh chuyển mảng trong replay buffer để chạy được với NumPy 2. Bản gốc sử dụng năm local model; các mạng cục bộ được gọi trong `no_grad` khi chạy `T2f.fit`.

## Thiết lập chính

Mỗi trạm dùng 512 mốc train và 8 epoch. Các mốc chia train/validation/test là 70/10/20 theo thời gian. ETTh dự báo 2 bước, ETTm dự báo 30 bước, dùng 48 gốc dự báo mỗi tập đánh giá. ELD dùng MT_211, lấy trung bình theo khung 6 giờ và dự báo 2 bước. MAE và RMSE được tính theo đơn vị gốc; kết quả thuộc thí nghiệm rút gọn.

[Bài báo](https://doi.org/10.1007/s00521-026-12209-6) · [Mã gốc](https://github.com/jfpsousa/t2f) · [Bản phát hành Zenodo](https://zenodo.org/records/20718388)
