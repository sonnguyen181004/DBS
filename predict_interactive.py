import re
import math
import string
import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os
import sys

# Đảm bảo in tiếng Việt không lỗi font trên console Windows
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 1. Định nghĩa lại hàm trích xuất đặc trưng (phải khớp hoàn toàn với mô hình)
def extract_query_features(query):
    if not isinstance(query, str):
        query = ""
    length = len(query)
    no_letter = sum(1 for c in query if c.isalpha())
    no_digit = sum(1 for c in query if c.isdigit())
    no_semicolon = query.count(';')
    no_single_qout = query.count("'")
    no_double_qout = query.count('"')
    no_percentage = query.count('%')
    whit_space = query.count(' ') + 1
    
    if length == 0:
        entropy = 0.0
    else:
        counts = Counter(query)
        entropy = 0.0
        for val in counts.values():
            p = val / length
            entropy -= p * math.log2(p)
            
    no_special_char = sum(1 for c in query if not c.isalnum() and not c.isspace())
    no_punctuation = sum(1 for c in query if c in string.punctuation)
    
    query_lower = query.lower()
    keywords_list = ["select", "update", "insert", "delete", "drop", "create", "alter", "where", 
                     "from", "join", "into", "table", "union", "all", "group", "order", "by", 
                     "having", "limit", "exec", "execute"]
    no_keyword = 0
    for kw in keywords_list:
        no_keyword += len(re.findall(rf'\b{kw}\b', query_lower))
        
    logical_ops = ["and", "or", "not", "xor", "like", "between", "in", "is"]
    no_logical_operat = 0
    for op in logical_ops:
        no_logical_operat += len(re.findall(rf'\b{op}\b', query_lower))
        
    no_operat = len(re.findall(r'[+\-*/=<>]', query))
    no_or = len(re.findall(r'\bor\b', query_lower))
    no_and = len(re.findall(r'\band\b', query_lower))
    no_comment = query.count('--') + query.count('/*') + query.count('#')
    no_null = len(re.findall(r'\bnull\b', query_lower))
    
    # Trả về các giá trị đặc trưng
    features = {
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
    return features

# Tên file lưu trữ mô hình
svm_path = "c:/Users/LOQ/Downloads/DBS/svm_model.pkl"
xgb_path = "c:/Users/LOQ/Downloads/DBS/xgb_model.pkl"

# Kiểm tra sự tồn tại của mô hình, nếu không có thì huấn luyện và lưu lại
if not os.path.exists(svm_path) or not os.path.exists(xgb_path):
    print("Không tìm thấy tệp mô hình đã lưu. Đang tự động chạy huấn luyện lại trên tập dữ liệu...")
    from train_eval import extract_query_features as train_extract
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from concurrent.futures import ProcessPoolExecutor
    
    # Đọc dữ liệu huấn luyện
    train_df = pd.read_csv("c:/Users/LOQ/Downloads/DBS/dataset/clean_sql_dataset.csv")
    train_df['Query'] = train_df['Query'].fillna("")
    queries = train_df['Query'].tolist()
    
    # Trích xuất song song
    num_workers = 4
    batch_size = math.ceil(len(queries) / num_workers)
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    def process_batch(queries):
        return [train_extract(q) for q in queries]
        
    extracted_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_batch, batches)
        for res in results:
            extracted_features.extend(res)
            
    X_train_full = pd.DataFrame(extracted_features)
    y_train_full = train_df['Label'].values
    
    # Trích xuất tập con cho SVM
    svm_subset_size = 30000
    indices = np.random.choice(len(X_train_full), svm_subset_size, replace=False)
    X_train_svm = X_train_full.iloc[indices]
    y_train_svm = y_train_full[indices]
    
    # Train SVM
    print("-> Đang huấn luyện mô hình SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_svm, y_train_svm)
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_model, f)
        
    # Train XGBoost
    print("-> Đang huấn luyện mô hình XGBoost...")
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_full, y_train_full)
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print("-> Đã lưu mô hình thành công.")

# Tải các mô hình từ file
with open(svm_path, 'rb') as f:
    svm_model = pickle.load(f)
with open(xgb_path, 'rb') as f:
    xgb_model = pickle.load(f)

# Thứ tự cột đặc trưng
feature_cols = [
    'no_letter', 'no_digit', 'no_special _char', 'no_keyword', 'length', 'entropy', 
    'no_semicolon', 'no_single_ qout', 'no_double_ qout', 'no_percentage', 'whit_ space', 
    'no_punctuation', 'no_logical _operat', 'no_operat', 'no_or', 'no_and', 'no_comment', 
    'no_null Value in the SQL Query'
]

print("\n" + "="*70)
print(" CHƯƠNG TRÌNH KIỂM THỬ TRỰC QUAN MÔ HÌNH PHÁT HIỆN SQL INJECTION")
print("="*70)
print("Mô tả: Nhập câu lệnh SQL bên dưới để kiểm tra xem mô hình dự đoán thế nào.")
print("Gõ 'exit' hoặc 'quit' để thoát chương trình.\n")

while True:
    try:
        user_input = input("Nhập câu lệnh SQL cần test > ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Đang thoát chương trình. Tạm biệt!")
            break
            
        if not user_input.strip():
            continue
            
        # Trích xuất đặc trưng
        feat_dict = extract_query_features(user_input)
        feat_df = pd.DataFrame([feat_dict])[feature_cols]
        
        # Dự đoán bằng SVM
        svm_pred = svm_model.predict(feat_df)[0]
        svm_prob = svm_model.predict_proba(feat_df)[0][1]
        
        # Dự đoán bằng XGBoost
        xgb_pred = xgb_model.predict(feat_df)[0]
        xgb_prob = xgb_model.predict_proba(feat_df)[0][1]
        
        # In kết quả
        print("\n" + "-"*50)
        print(f"Câu lệnh SQL kiểm tra: {repr(user_input)}")
        print("-"*50)
        
        print("KẾT QUẢ DỰ ĐOÁN:")
        print(f"1. Mô hình Support Vector Machine (SVM):")
        label_svm = "ĐỘC HẠI (SQL Injection) ⚠" if svm_pred == 1 else "LÀNH TÍNH (An toàn) ✔"
        print(f"   -> Kết quả: {label_svm}")
        print(f"   -> Xác suất độc hại: {svm_prob*100:.2f}%")
        
        print(f"2. Mô hình XGBoost Classifier:")
        label_xgb = "ĐỘC HẠI (SQL Injection) ⚠" if xgb_pred == 1 else "LÀNH TÍNH (An toàn) ✔"
        print(f"   -> Kết quả: {label_xgb}")
        print(f"   -> Xác suất độc hại: {xgb_prob*100:.2f}%")
        
        # Giải thích đặc trưng được phát hiện (Explainable AI)
        print("\nPHÂN TÍCH ĐẶC TRƯNG CHÍNH ĐƯỢC PHÁT HIỆN:")
        print(f"   - Độ dài câu lệnh: {int(feat_dict['length'])} ký tự")
        print(f"   - Chỉ số hỗn loạn (Entropy): {feat_dict['entropy']:.4f}")
        if feat_dict['no_single_ qout'] > 0:
            print(f"   - Dấu nháy đơn (') xuất hiện: {int(feat_dict['no_single_ qout'])} lần (Kỹ thuật phá vỡ ranh giới chuỗi)")
        if feat_dict['no_double_ qout'] > 0:
            print(f"   - Dấu nháy kép (\") xuất hiện: {int(feat_dict['no_double_ qout'])} lần")
        if feat_dict['no_keyword'] > 0:
            print(f"   - Từ khóa SQL được sử dụng: {int(feat_dict['no_keyword'])} từ")
        if feat_dict['no_logical _operat'] > 0:
            print(f"   - Toán tử logic (AND/OR/LIKE...): {int(feat_dict['no_logical _operat'])} lần")
        if feat_dict['no_comment'] > 0:
            print(f"   - Ký tự chú thích (--, /*, #): {int(feat_dict['no_comment'])} lần (Kỹ thuật vô hiệu hóa câu lệnh phía sau)")
        if feat_dict['no_special _char'] > 0:
            print(f"   - Số lượng ký tự đặc biệt khác: {int(feat_dict['no_special _char'])} ký tự")
            
        print("-"*50 + "\n")
        
    except KeyboardInterrupt:
        print("\nĐang thoát chương trình. Tạm biệt!")
        break
    except Exception as e:
        print(f"Lỗi: {e}\n")
