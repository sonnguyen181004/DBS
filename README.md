# Phát hiện SQL Injection sử dụng Học máy (So sánh 5 Mô hình)

Dự án này thực hiện quy trình trích xuất đặc trưng cú pháp-ngữ nghĩa (Feature Engineering) cho các câu lệnh SQL thô và huấn luyện 5 mô hình phân lớp Học máy: **Support Vector Machine (SVM)**, **XGBoost**, **LightGBM**, **AdaBoost**, và **Random Forest** để tự động phát hiện mã độc tấn công SQL Injection.

Mô hình được huấn luyện hoàn chỉnh trên tập dữ liệu Kaggle thô và kiểm thử độc lập trực tiếp trên tập dữ liệu Mendeley Data nhằm đánh giá năng lực tổng quát hóa Out-of-Distribution (OOD) thực tế.

---

## 1. Cấu trúc thư mục dự án

```text
DBS/
├── dataset/
│   ├── clean_sql_dataset.csv     # Tập dữ liệu huấn luyện từ Kaggle (148,326 dòng)
│   └── SQLI_Dataset.csv          # Tập dữ liệu kiểm thử độc lập từ Mendeley (47,463 dòng)
├── train_mendeley_only.py        # Script huấn luyện 5 mô hình trên 100% Kaggle và kiểm thử trên Mendeley
├── train_eval.py                 # Script huấn luyện cũ (chia 80/20 trên Kaggle và kiểm thử chéo)
├── predict_interactive.py        # Công cụ kiểm thử câu lệnh SQL tương tác trực quan với cả 5 mô hình
├── RBLDBS_18.docx                # Báo cáo tiến độ nghiên cứu chính thức (đạt chuẩn định dạng IEEE)
├── Giai_Thich_Du_An_DBS.docx     # Cẩm nang đọc hiểu chi tiết, giải thích toàn bộ thuật ngữ của dự án
├── results_summary.txt           # File text lưu kết quả số liệu thô từ thực nghiệm huấn luyện
├── adaboost_model.pkl            # Bộ não lưu trữ của mô hình AdaBoost đã huấn luyện
├── lightgbm_model.pkl            # Bộ não lưu trữ của mô hình LightGBM đã huấn luyện
├── random_forest_model.pkl       # Bộ não lưu trữ của mô hình Random Forest đã huấn luyện
├── svm_model.pkl                 # Bộ não lưu trữ của mô hình SVM đã huấn luyện
├── xgboost_model.pkl             # Bộ não lưu trữ của mô hình XGBoost đã huấn luyện
├── fig1_performance.png          # Biểu đồ so sánh 4 chỉ số hiệu năng (Accuracy, Precision, Recall, F1)
├── fig2_time.png                 # Biểu đồ so sánh thời gian xử lý (huấn luyện & dự đoán - log scale)
├── fig3_roc.png                  # Biểu đồ đường cong ROC-AUC so sánh khả năng phân tách của 5 mô hình
└── fig4_confusion.png            # Lưới ma trận nhầm lẫn (Confusion Matrix) của cả 5 mô hình
```

---

## 2. Quy trình Trích xuất 18 Đặc trưng (Feature Engineering)

Mỗi truy vấn SQL được chuyển đổi tự động thành vectơ 18 chiều đặc trưng tương thích 100% với cấu trúc tập kiểm thử Mendeley:

1.  **`length`**: Độ dài chuỗi truy vấn (số lượng ký tự).
2.  **`no_letter`**: Số lượng chữ cái `[a-zA-Z]`.
3.  **`no_digit`**: Số lượng chữ số `[0-9]`.
4.  **`no_semicolon`**: Số lượng dấu chấm phẩy `;`.
5.  **`no_single_ qout`**: Số lượng dấu nháy đơn `'`.
6.  **`no_double_ qout`**: Số lượng dấu nháy kép `"`.
7.  **`no_percentage`**: Số lượng ký tự `%`.
8.  **`whit_ space`**: Số lượng khoảng trắng ` ` cộng thêm 1.
9.  **`entropy`**: Chỉ số Entropy Shannon đo lường độ hỗn loạn thông tin.
10. **`no_special _char`**: Ký tự đặc biệt (không phải chữ cái, số hay khoảng trắng).
11. **`no_punctuation`**: Ký tự dấu câu (`string.punctuation`).
12. **`no_keyword`**: Đếm số từ khóa SQL thông dụng: `select`, `update`, `insert`, `delete`, `drop`, `create`, `alter`, `where`, `from`, `join`, `into`, `table`, `union`, `all`, `group`, `order`, `by`, `having`, `limit`, `exec`, `execute`.
13. **`no_logical _operat`**: Số lượng toán tử logic SQL: `and`, `or`, `not`, `xor`, `like`, `between`, `in`, `is`.
14. **`no_operat`**: Các toán tử cơ bản: `+`, `-`, `*`, `/`, `=`, `<`, `>`.
15. **`no_or`**: Từ khóa `or` đứng độc lập.
16. **`no_and`**: Từ khóa `and` đứng độc lập.
17. **`no_comment`**: Chú thích SQL (`--`, `/*`, `#`).
18. **`no_null Value in the SQL Query`**: Đếm số lượng từ khóa `null` độc lập.

---

## 3. Cách chạy dự án

### Yêu cầu cài đặt môi trường
Đảm bảo máy bạn đã cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn python-docx
```

### Chạy huấn luyện và đánh giá mô hình
Chạy script chính để thực hiện huấn luyện trên 100% tập dữ liệu Kaggle và kiểm thử trực tiếp trên Mendeley:
```bash
python train_mendeley_only.py
```
Sau khi chạy xong, chương trình sẽ lưu lại kết quả thô vào `results_summary.txt` và cập nhật các biểu đồ `fig1_performance.png` đến `fig4_confusion.png`.

### Chạy kiểm thử tương tác
Để kiểm tra khả năng dự đoán của mô hình với một câu lệnh SQL bất kỳ do bạn tự nhập, chạy lệnh:
```bash
python predict_interactive.py
```

---

## 4. Kết quả Thực nghiệm & So sánh Hiệu năng

Số liệu thực tế thu được từ thực nghiệm:
- **Tập Huấn luyện (Kaggle)**: 148,326 mẫu (SVM huấn luyện trên tập con ngẫu nhiên 30,000 mẫu).
- **Tập Kiểm thử (Mendeley)**: 47,463 mẫu (Benign: 25,799 | SQLi: 21,664).

### Bảng so sánh kết quả:

| Mô hình | Accuracy | Precision (SQLi) | Recall (SQLi) | F1-Score (SQLi) | AUC-ROC | Thời gian huấn luyện | Thời gian dự đoán (47k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | 87.55% | 96.64% | 75.34% | 84.67% | 0.9739 | 135.68s | 382.96s |
| **XGBoost** | **96.46%** | 95.23% | 97.10% | **96.16%** | 0.9943 | **0.59s** | **0.035s** |
| **LightGBM** | 95.97% | 94.12% | **97.25%** | 95.66% | **0.9946** | 0.67s | 0.062s |
| **AdaBoost** | 95.04% | **99.13%** | 89.91% | 94.30% | 0.9856 | 6.06s | 0.316s |
| **Random Forest** | 95.13% | 94.46% | 94.90% | 94.68% | 0.9888 | 2.07s | 0.144s |

### Đánh giá sự đánh đổi (Trade-off):
1.  **Mô hình tối ưu thực tế**: **XGBoost** là mô hình tốt nhất để triển khai cho hệ thống lọc Web Application Firewall (WAF) vì nó đạt F1-Score cao nhất (96.16%) đồng thời có tốc độ xử lý nhanh nhất (~1.35 triệu yêu cầu/giây, đáp ứng yêu cầu độ trễ < 1ms).
2.  **Khả năng phát hiện tối đa**: **LightGBM** đạt độ nhạy (Recall) cao nhất (97.25%), thích hợp nếu hệ thống muốn hạn chế tối đa việc bỏ sót mã độc.
3.  **Hạn chế của SVM**: SVM chạy quá chậm (382.96s để dự đoán) và bỏ sót nhiều cuộc tấn công (Recall chỉ đạt 75.34%), không khả thi để triển khai trong thực tế.
