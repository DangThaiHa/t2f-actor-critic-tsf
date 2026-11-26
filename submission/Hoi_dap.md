# T2f · 5 phút hỏi đáp cuối kỳ

1. **Mức giảm 22,5% tính so với đâu?** Trên cùng 48 gốc/trạm của ETTm, MAE giảm từ 1,102 của T2f sai phân + Ridge giữa kỳ xuống 0,855 của Stacking. Tỷ lệ này chỉ thuộc giao thức bridge.
2. **Hai đoạn 5% dùng làm gì?** Đoạn đầu học trọng số hoặc chọn epoch. Đoạn sau chọn phương pháp bằng MAE. 20% cuối dùng để chấm sau khi chốt lựa chọn.
3. **Tại sao T2f level có MAE rất cao ở đối chứng full?** Ở seed 42, 30 sai số lớn nhất chiếm gần 96% tổng sai số tuyệt đối. Trung vị 5,583 là trung bình trung vị của ba seed, cho biết mức sai số ở giữa phân phối. Chưa đủ bằng chứng để quy mọi điểm cực đoan trên ETTm cho riêng một thành phần.
4. **Trọng số có phải đóng góp ở một thời điểm không?** Trọng số là hệ số cố định. Đóng góp so với mốc giữ nguyên OT còn phụ thuộc mức từng mô hình dự báo cao hay thấp hơn mốc đó. Với tổ hợp tuyến tính, giá trị Shapley ở tầng tổ hợp bằng trọng số nhân chênh lệch này.
5. **Có thể kết luận tốt hơn bài báo không?** Em kết luận theo đối chứng cùng cửa sổ trong dự án. Bảng 3 của tác giả dùng cấu hình và cách tổng hợp khác. XAI cũng chỉ giải thích cách tạo đầu ra của tổ hợp, chưa xác định nguyên nhân vật lý làm nhiệt độ thay đổi.
