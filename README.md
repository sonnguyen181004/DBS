# Phát hiện SQL Injection sử dụng Machine Learning (SVM & XGBoost)

Dự án này thực hiện quy trình trích xuất đặc trưng (Feature Engineering) cho các câu lệnh SQL thô và huấn luyện hai mô hình phân lớp Học máy: **Support Vector Machine (SVM)** và **XGBoost Classifier** để tự động phát hiện mã độc tấn công SQL Injection.

Mô hình được huấn luyện chéo trên tập dữ liệu Kaggle thô và kiểm thử độc lập trực tiếp trên tập dữ liệu số của Mendeley Data.

---

## 1. Cấu trúc thư mục dự án

```text
DBS/
├── dataset/
│   ├── clean_sql_dataset.csv  # Tập dữ liệu huấn luyện thô từ Kaggle (148,326 dòng)
│   └── SQLI_Dataset.csv       # Tập dữ liệu kiểm thử dạng số sẵn từ Mendeley (47,463 dòng)
├── train_eval.py              # Script huấn luyện hai mô hình SVM và XGBoost
├── RBL DBS (1).docx           # Báo cáo Word tiến độ nghiên cứu (đã được cập nhật phần kết quả thực nghiệm)
├── model_comparison.png       # Biểu đồ cột so sánh Accuracy, Precision, Recall, F1-Score
├── confusion_matrices.png     # Ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap của 2 mô hình
├── roc_curves.png             # Đường cong ROC-AUC so sánh khả năng phân tách
└── README.md                  # Hướng dẫn dự án này
```

---

## 2. Quy trình Trích xuất 18 Đặc trưng (Feature Engineering)

Cột `Query` của tập Kaggle được hóa số thành 18 chiều đặc trưng tương thích 100% với cấu trúc tập kiểm thử Mendeley:

1.  **`length`**: Độ dài chuỗi truy vấn.
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
Đảm bảo bạn đã cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn python-docx
```

### Chạy huấn luyện và đánh giá mô hình
Chạy file python chính để thực hiện trích xuất và hiển thị báo cáo chi tiết:
```bash
python train_eval.py
```

Sau khi chạy xong, chương trình sẽ in báo cáo `classification_report` và lưu các mô hình.

---

## 4. Kết quả Thực nghiệm & So sánh Hiệu năng

### Thống kê thời gian xử lý:
-   **Trích xuất đặc trưng (148k dòng)**: 23.78 giây.
-   **Huấn luyện SVM (30k mẫu)**: 133.22 giây.
-   **Huấn luyện XGBoost (148k mẫu)**: 1.40 giây.

### Bảng kết quả đánh giá (Kiểm thử trên 47,463 mẫu Mendeley):

| Chỉ số đánh giá | SVM Model | XGBoost Model |
| :--- | :---: | :---: |
| **Độ chính xác (Accuracy)** | 87.68% | **96.46%** |
| **Precision** (Lớp SQLi) | **96.60%** | 95.23% |
| **Recall** (Lớp SQLi) | 75.67% | **97.10%** |
| **F1-Score** (Lớp SQLi) | 84.86% | **96.16%** |
| **ROC-AUC** | 0.9575 | **0.9922** |

---

## 5. Các biểu đồ xuất ra dự án

-   **`model_comparison.png`**: Biểu đồ so sánh trực quan các thông số.
-   **`confusion_matrices.png`**: Ma trận nhầm lẫn đo lường tỉ lệ dự đoán đúng/sai.
-   **`roc_curves.png`**: Đánh giá năng lực phân loại của mô hình tại các ngưỡng khác nhau.
