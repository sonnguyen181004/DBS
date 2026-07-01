import re
import math
import string
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import time

def extract_query_features(query):
    """
    Trích xuất 18 đặc trưng số học từ một câu lệnh SQL thô.
    Hàm này xử lý cả trường hợp chuỗi trống hoặc giá trị NaN.
    """
    if not isinstance(query, str):
        query = ""
    
    # 1. Độ dài câu lệnh SQL
    length = len(query)
    
    # 2. Số lượng chữ cái (a-z, A-Z)
    no_letter = sum(1 for c in query if c.isalpha())
    
    # 3. Số lượng chữ số (0-9)
    no_digit = sum(1 for c in query if c.isdigit())
    
    # 4. Số lượng dấu chấm phẩy (;)
    no_semicolon = query.count(';')
    
    # 5. Số lượng dấu nháy đơn (')
    no_single_qout = query.count("'")
    
    # 6. Số lượng dấu nháy kép (")
    no_double_qout = query.count('"')
    
    # 7. Số lượng dấu phần trăm (%)
    no_percentage = query.count('%')
    
    # 8. Khoảng trắng (số khoảng trắng ' ' + 1)
    whit_space = query.count(' ') + 1
    
    # 9. Entropy Shannon của chuỗi
    if length == 0:
        entropy = 0.0
    else:
        counts = Counter(query)
        entropy = 0.0
        for val in counts.values():
            p = val / length
            entropy -= p * math.log2(p)
            
    # 10. Ký tự đặc biệt (không phải chữ cái, chữ số, hay khoảng trắng)
    no_special_char = sum(1 for c in query if not c.isalnum() and not c.isspace())
    
    # 11. Dấu câu (punctuation) theo string.punctuation
    no_punctuation = sum(1 for c in query if c in string.punctuation)
    
    query_lower = query.lower()
    
    # 12. Từ khóa SQL thông dụng
    keywords_list = ["select", "update", "insert", "delete", "drop", "create", "alter", "where", 
                     "from", "join", "into", "table", "union", "all", "group", "order", "by", 
                     "having", "limit", "exec", "execute"]
    no_keyword = 0
    for kw in keywords_list:
        no_keyword += len(re.findall(rf'\b{kw}\b', query_lower))
        
    # 13. Toán tử logic
    logical_ops = ["and", "or", "not", "xor", "like", "between", "in", "is"]
    no_logical_operat = 0
    for op in logical_ops:
        no_logical_operat += len(re.findall(rf'\b{op}\b', query_lower))
        
    # 14. Toán tử cơ bản (+ - * / = < >)
    no_operat = len(re.findall(r'[+\-*/=<>]', query))
    
    # 15. Từ khóa OR đứng độc lập
    no_or = len(re.findall(r'\bor\b', query_lower))
    
    # 16. Từ khóa AND đứng độc lập
    no_and = len(re.findall(r'\band\b', query_lower))
    
    # 17. Chú thích SQL (--, /*, #)
    no_comment = query.count('--') + query.count('/*') + query.count('#')
    
    # 18. Giá trị NULL độc lập
    no_null = len(re.findall(r'\bnull\b', query_lower))
    
    # Trả về dưới dạng dictionary khớp chính xác với tên cột của tập Mendeley
    return {
        'no_letter': float(no_letter),
        'no_digit': float(no_digit),
        'no_special _char': float(no_special_char),
        'no_keyword': float(no_keyword),
        'length': float(length),
        'entropy': float(entropy),
        'no_semicolon': float(no_semicolon),
        'no_single_ qout': float(no_single_qout),
        'no_double_ qout': float(no_double_qout),
        'no_percentage': float(no_percentage),
        'whit_ space': float(whit_space),
        'no_punctuation': float(no_punctuation),
        'no_logical _operat': float(no_logical_operat),
        'no_operat': float(no_operat),
        'no_or': float(no_or),
        'no_and': float(no_and),
        'no_comment': float(no_comment),
        'no_null Value in the SQL Query': float(no_null)
    }

def process_batch(queries):
    """Xử lý trích xuất đặc trưng cho một lô các câu truy vấn."""
    return [extract_query_features(q) for q in queries]

if __name__ == '__main__':
    import sys
    import io
    # Cấu hình stdout/stderr dùng UTF-8 để tránh lỗi hiển thị tiếng Việt trên Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("======================================================================")
    print("BẮT ĐẦU QUY TRÌNH HÓA SỐ HÓA, HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH")
    print("======================================================================")
    
    # --- BƯỚC 1: Đọc và Trích xuất Đặc trưng từ Tập dữ liệu Kaggle (Train) ---
    print("\n[Bước 1] Đang tải tập dữ liệu Kaggle (clean_sql_dataset.csv)...")
    train_df = pd.read_csv("c:/Users/LOQ/Downloads/DBS/dataset/clean_sql_dataset.csv")
    print(f"-> Tập Kaggle chứa {len(train_df)} dòng.")
    
    # Làm sạch các giá trị trống (NaN)
    train_df['Query'] = train_df['Query'].fillna("")
    
    # Sử dụng Multiprocessing để tăng tốc quá trình trích xuất đặc trưng
    print("-> Đang trích xuất đặc trưng bằng Multiprocessing...")
    queries = train_df['Query'].tolist()
    num_workers = 4 # Số nhân CPU sử dụng
    batch_size = math.ceil(len(queries) / num_workers)
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    start_time = time.time()
    extracted_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_batch, batches)
        for res in results:
            extracted_features.extend(res)
    print(f"-> Hoàn thành trích xuất đặc trưng trong {time.time() - start_time:.2f} giây.")
    
    # Chuyển đặc trưng thành DataFrame
    X_train_full = pd.DataFrame(extracted_features)
    y_train_full = train_df['Label'].values
    
    # --- BƯỚC 2: Đọc tập dữ liệu kiểm thử Mendeley (Test) ---
    print("\n[Bước 2] Đang tải tập dữ liệu Mendeley (SQLI_Dataset.csv) để làm tập Test...")
    test_df = pd.read_csv("c:/Users/LOQ/Downloads/DBS/dataset/SQLI_Dataset.csv")
    print(f"-> Tập Mendeley chứa {len(test_df)} dòng.")
    
    # Đảm bảo thứ tự các cột đặc trưng khớp hoàn toàn
    feature_cols = [
        'no_letter', 'no_digit', 'no_special _char', 'no_keyword', 'length', 'entropy', 
        'no_semicolon', 'no_single_ qout', 'no_double_ qout', 'no_percentage', 'whit_ space', 
        'no_punctuation', 'no_logical _operat', 'no_operat', 'no_or', 'no_and', 'no_comment', 
        'no_null Value in the SQL Query'
    ]
    
    # Làm sạch NaNs nếu có trong tập test
    test_df = test_df.dropna(subset=['label'])
    test_df[feature_cols] = test_df[feature_cols].fillna(0.0)
    
    X_test = test_df[feature_cols]
    y_test = test_df['label'].values
    
    # --- BƯỚC 3: Huấn luyện mô hình ---
    print("\n[Bước 3] Đang tiến hành huấn luyện các mô hình...")
    
    # Lưu ý: Do thuật toán SVM (SVC) chạy với độ phức tạp O(N^2) đến O(N^3),
    # huấn luyện trên toàn bộ 156k mẫu sẽ tốn rất nhiều thời gian (vài tiếng).
    # Để tối ưu hóa tốc độ mà vẫn giữ độ chính xác ổn định, ta sẽ huấn luyện SVM trên một tập con ngẫu nhiên (30,000 mẫu).
    # XGBoost Classifier huấn luyện rất nhanh nên sẽ sử dụng toàn bộ tập dữ liệu (156k mẫu).
    
    # Trích xuất tập con cho SVM
    svm_subset_size = 30000
    if len(X_train_full) > svm_subset_size:
        indices = np.random.choice(len(X_train_full), svm_subset_size, replace=False)
        X_train_svm = X_train_full.iloc[indices]
        y_train_svm = y_train_full[indices]
        print(f"-> Sử dụng tập con ngẫu nhiên gồm {svm_subset_size} mẫu để huấn luyện SVM.")
    else:
        X_train_svm = X_train_full
        y_train_svm = y_train_full
        print(f"-> Huấn luyện SVM trên toàn bộ {len(X_train_full)} mẫu.")
        
    # Huấn luyện SVM
    print("-> Đang huấn luyện mô hình Support Vector Machine (SVM)...")
    svm_start = time.time()
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_svm, y_train_svm)
    print(f"-> Hoàn thành huấn luyện SVM trong {time.time() - svm_start:.2f} giây.")
    
    # Huấn luyện XGBoost trên toàn bộ dữ liệu
    print("-> Đang huấn luyện mô hình XGBoost Classifier...")
    xgb_start = time.time()
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_full, y_train_full)
    print(f"-> Hoàn thành huấn luyện XGBoost trong {time.time() - xgb_start:.2f} giây.")
    
    # --- BƯỚC 4: Dự đoán và Kiểm thử (Test) trực tiếp trên tập Mendeley ---
    print("\n[Bước 4] Đang dự đoán và đánh giá trên tập dữ liệu Mendeley...")
    
    # Đánh giá SVM
    print("\n" + "="*50)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH: Support Vector Machine (SVM)")
    print("="*50)
    y_pred_svm = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred_svm, target_names=["Benign (0)", "SQLi (1)"], digits=4))
    
    # Đánh giá XGBoost
    print("\n" + "="*50)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH: XGBoost Classifier")
    print("="*50)
    y_pred_xgb = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred_xgb, target_names=["Benign (0)", "SQLi (1)"], digits=4))
    print("\nQuy trình hoàn tất thành công!")
