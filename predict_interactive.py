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

# 1. Định nghĩa hàm trích xuất đặc trưng
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

# Load 5 mô hình đã lưu
model_names = ['SVM', 'XGBoost', 'LightGBM', 'AdaBoost', 'Random Forest']
models = {}

for name in model_names:
    pkl_name = f"{name.lower().replace(' ', '_')}_model.pkl"
    pkl_path = os.path.join("c:/Users/LOQ/Downloads/DBS", pkl_name)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            models[name] = pickle.load(f)
    else:
        print(f"Lưu ý: Không tìm thấy file mô hình {pkl_name}. Vui lòng chạy train_eval.py trước.")
        sys.exit(1)

# Thứ tự đặc trưng
feature_cols = [
    'no_letter', 'no_digit', 'no_special _char', 'no_keyword', 'length', 'entropy', 
    'no_semicolon', 'no_single_ qout', 'no_double_ qout', 'no_percentage', 'whit_ space', 
    'no_punctuation', 'no_logical _operat', 'no_operat', 'no_or', 'no_and', 'no_comment', 
    'no_null Value in the SQL Query'
]

print("\n" + "="*70)
print(" CHƯƠNG TRÌNH KIỂM THỬ TRỰC QUAN 5 MÔ HÌNH PHÁT HIỆN SQL INJECTION")
print("="*70)
print("Nhập câu lệnh SQL để xem kết quả dự đoán đồng thời của cả 5 mô hình.")
print("Gõ 'exit' hoặc 'quit' để thoát.\n")

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
        
        # In kết quả
        print("\n" + "-"*65)
        print(f"Câu lệnh SQL kiểm tra: {repr(user_input)}")
        print("-"*65)
        
        print("KẾT QUẢ DỰ ĐOÁN TỪ 5 MÔ HÌNH:")
        for name, model in models.items():
            pred = model.predict(feat_df)[0]
            prob = model.predict_proba(feat_df)[0][1] if hasattr(model, 'predict_proba') else 0.0
            
            label = "ĐỘC HẠI (SQL Injection) ⚠" if pred == 1 else "LÀNH TÍNH (An toàn) ✔"
            prob_str = f" (Xác suất độc hại: {prob*100:.2f}%)" if prob > 0.0 else ""
            print(f"· {name:<15}: {label}{prob_str}")
            
        # Giải thích đặc trưng chính
        print("\nPHÂN TÍCH ĐẶC TRƯNG CHÍNH ĐƯỢC PHÁT HIỆN:")
        print(f"   - Độ dài câu lệnh: {int(feat_dict['length'])} ký tự")
        print(f"   - Chỉ số hỗn loạn (Entropy): {feat_dict['entropy']:.4f}")
        if feat_dict['no_single_ qout'] > 0:
            print(f"   - Dấu nháy đơn (') xuất hiện: {int(feat_dict['no_single_ qout'])} lần")
        if feat_dict['no_double_ qout'] > 0:
            print(f"   - Dấu nháy kép (\") xuất hiện: {int(feat_dict['no_double_ qout'])} lần")
        if feat_dict['no_keyword'] > 0:
            print(f"   - Từ khóa SQL được sử dụng: {int(feat_dict['no_keyword'])} từ")
        if feat_dict['no_logical _operat'] > 0:
            print(f"   - Toán tử logic (AND/OR/LIKE...): {int(feat_dict['no_logical _operat'])} lần")
        if feat_dict['no_comment'] > 0:
            print(f"   - Ký tự chú thích (--, /*, #): {int(feat_dict['no_comment'])} lần")
        if feat_dict['no_special _char'] > 0:
            print(f"   - Ký tự đặc biệt khác: {int(feat_dict['no_special _char'])} ký tự")
            
        print("-"*65 + "\n")
        
    except KeyboardInterrupt:
        print("\nĐang thoát chương trình. Tạm biệt!")
        break
    except Exception as e:
        print(f"Lỗi: {e}\n")
