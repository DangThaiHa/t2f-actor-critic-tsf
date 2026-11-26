# Dự báo nhiệt độ máy biến áp và phụ tải điện với T2f

Ở phần giữa kỳ, em chạy lại T2f trên ETTh, ETTm và một khách hàng trong bộ ELD. Ở phần cuối kỳ, em so sánh các cách kết hợp dự báo trên toàn bộ bốn chuỗi ETT và 370 khách hàng ELD. [Bảng kết quả thực nghiệm](EXPERIMENTS.md) ghi lại cách chia dữ liệu và MAE của từng phương pháp.

## Demo cuối kỳ

Sau khi clone repo, chạy tại thư mục chứa README:

```bash
python -m pip install numpy
python -m demo.forecast final --station 1 --window 1 --step 30
python -m demo.forecast --verify
```

Demo tính lại dự báo Stacking từ sáu đầu ra đã lưu, sau đó giải thích từng đóng góp so với OT tại gốc. Đổi trạm (1–2), cửa sổ (1–464) hoặc bước (1–30) để xem trường hợp khác. Lệnh kiểm tra đối chiếu cả 27.840 kết quả với dự báo đã lưu.

[Báo cáo cuối kỳ](submission/Bao_cao_T2f.pdf) · [Slide PowerPoint](submission/Slide_T2f.pptx) · [Hỏi đáp](submission/Hoi_dap.md) · [Kết quả đối chứng ETTm](comparison/summary.csv).

## Chuẩn bị dữ liệu

Từ thư mục gốc của dự án, cài thư viện rồi khôi phục mã T2f và dữ liệu giữa kỳ từ [bản phát hành Zenodo của tác giả](https://zenodo.org/records/20718388). Thay đường dẫn ở lệnh thứ ba bằng vị trí tệp ZIP đã tải.

```bash
python -m pip install -r requirements.txt
python -m pip install py7zr
python scripts/prepare_inputs.py /path/to/t2f-Zenodo.zip
python final/prepare.py
```

Lệnh cuối tải bốn tệp ETT và gộp phụ tải ELD theo khung 6 giờ. Nếu đã tải ZIP của UCI, dùng `python final/prepare.py --eld-zip /path/to/uci.zip`.

## Chạy thực nghiệm

```bash
python -m final.compare --dataset ETTh
python -m final.compare --dataset ETTm
python -m final.compare --dataset ELD
```

Trên CPU, đặt `OPENBLAS_NUM_THREADS=1` và `OMP_NUM_THREADS=1` để giảm chi phí khi nhân các ma trận nhỏ. Mặc định, chương trình ghi kết quả vào `results/final/`. Có thể đổi thư mục bằng `--output`.

| Mục đích | Lệnh ví dụ |
|---|---|
| So sánh trên cùng 48 gốc dự báo với giữa kỳ | `python -m final.compare --dataset ETTm --bridge` |
| Dự báo trước 96 bước trên ETT | `python -m final.compare --dataset ETTh --horizon 96` |
| Chạy đối chứng TD3 | `python -m final.td3 --dataset ETTm` |
| Chạy PatchTST-small trên ETT | `python -m final.patch_transformer --dataset ETTm` |
| Kiểm tra năm mô hình thành phần của T2f gốc | `python -m final.local_audit --dataset eld` |
| Tổng hợp các lượt đã chạy | `python -m final.summarize` |

Lệnh tổng hợp đọc dự báo trong các tệp `.npz` do các lượt chạy tạo ra. Để thử riêng một cấu hình, dùng cùng một thư mục `--output` mới cho cả lệnh chạy và lệnh tổng hợp.

## Các tệp chính

- `src/` và [MIDTERM.md](MIDTERM.md) chứa mã và hướng dẫn chạy phần giữa kỳ.
- `final/` chứa các mô hình và phần đánh giá cuối kỳ. Hai notebook trong `notebooks/` dùng để chạy trên Kaggle hoặc Colab sau khi chuẩn bị dữ liệu.
- `results/final/comparison.csv` lưu MAE, RMSE và sMAPE. Các tệp `paired_midterm.csv`, `paired_blocks.csv` và `coverage.csv` lưu so sánh với giữa kỳ, khoảng tin cậy và số mốc đánh giá.
- `data/final/sources.json` ghi nguồn và mã SHA-256 của các tệp dữ liệu.

## Demo trong buổi trình bày

Tại thư mục chứa README, chạy một trong hai lệnh bằng Python 3. Demo đọc các dự báo test đi kèm, có thể dùng ngay sau khi giải nén.

```bash
python demo/forecast.py mid --station 1 --window 1 --step 30
python demo/forecast.py final --station 1 --window 1 --step 30
```

Đổi trạm (1–2), cửa sổ (1–48 ở giữa kỳ, 1–464 ở cuối kỳ) hoặc bước (1–30). Bản cuối kỳ tính tổ hợp và đóng góp Shapley từ sáu dự báo thành phần. Mốc tham chiếu là OT cuối cùng. Đối chứng T2f chạy lại được mô tả trong comparison/README.md.
