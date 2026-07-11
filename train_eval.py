import re
import math
import string
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import time
import pickle
import os

# Học máy & Đánh giá
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix, classification_report

# Vẽ đồ thị
import matplotlib.pyplot as plt
import seaborn as sns

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

def process_batch(queries):
    return [extract_query_features(q) for q in queries]

if __name__ == '__main__':
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("======================================================================")
    print("QUY TRÌNH HÓA SỐ HÓA, HUẤN LUYỆN VÀ ĐÁNH GIÁ 5 MÔ HÌNH TRÊN 2 TẬP DỮ LIỆU")
    print("======================================================================")
    
    # 1. Đọc và trích xuất đặc trưng Kaggle
    print("\n[Bước 1] Đang tải tập dữ liệu Kaggle (clean_sql_dataset.csv)...")
    kaggle_df = pd.read_csv("c:/Users/LOQ/Downloads/DBS/dataset/clean_sql_dataset.csv")
    print(f"-> Tập Kaggle chứa {len(kaggle_df)} dòng.")
    kaggle_df['Query'] = kaggle_df['Query'].fillna("")
    
    # Sử dụng Multiprocessing
    print("-> Đang trích xuất đặc trưng bằng Multiprocessing...")
    queries = kaggle_df['Query'].tolist()
    num_workers = 4
    batch_size = math.ceil(len(queries) / num_workers)
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    start_time = time.time()
    extracted_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_batch, batches)
        for res in results:
            extracted_features.extend(res)
    print(f"-> Hoàn thành trích xuất đặc trưng trong {time.time() - start_time:.2f} giây.")
    
    X_kaggle_all = pd.DataFrame(extracted_features)
    y_kaggle_all = kaggle_df['Label'].values
    
    # Chia tập Kaggle thành 80% Train và 20% Test
    X_train_full, X_test_kaggle, y_train_full, y_test_kaggle = train_test_split(
        X_kaggle_all, y_kaggle_all, test_size=0.2, random_state=42, stratify=y_kaggle_all
    )
    print(f"-> Tập Kaggle Train (80%): {len(X_train_full)} dòng.")
    print(f"-> Tập Kaggle Test (20%): {len(X_test_kaggle)} dòng.")

    # 2. Đọc tập dữ liệu Mendeley
    print("\n[Bước 2] Đang tải tập dữ liệu Mendeley (SQLI_Dataset.csv)...")
    mendeley_df = pd.read_csv("c:/Users/LOQ/Downloads/DBS/dataset/SQLI_Dataset.csv")
    print(f"-> Tập Mendeley chứa {len(mendeley_df)} dòng.")
    
    feature_cols = [
        'no_letter', 'no_digit', 'no_special _char', 'no_keyword', 'length', 'entropy', 
        'no_semicolon', 'no_single_ qout', 'no_double_ qout', 'no_percentage', 'whit_ space', 
        'no_punctuation', 'no_logical _operat', 'no_operat', 'no_or', 'no_and', 'no_comment', 
        'no_null Value in the SQL Query'
    ]
    
    mendeley_df = mendeley_df.dropna(subset=['label'])
    mendeley_df[feature_cols] = mendeley_df[feature_cols].fillna(0.0)
    X_test_mendeley = mendeley_df[feature_cols]
    y_test_mendeley = mendeley_df['label'].values

    # 3. Định nghĩa và Huấn luyện 5 mô hình
    print("\n[Bước 3] Tiến hành huấn luyện 5 mô hình...")
    
    # Tối ưu SVM: huấn luyện trên 30k mẫu ngẫu nhiên từ tập Kaggle Train (80%)
    svm_subset_size = 30000
    indices = np.random.choice(len(X_train_full), svm_subset_size, replace=False)
    X_train_svm = X_train_full.iloc[indices]
    y_train_svm = y_train_full[indices]
    
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results_dict = {}
    
    for name, model in models.items():
        print(f"\n-> Đang huấn luyện: {name}...")
        X_tr = X_train_svm if name == 'SVM' else X_train_full
        y_tr = y_train_svm if name == 'SVM' else y_train_full
        
        # Đo thời gian huấn luyện
        t_start = time.time()
        model.fit(X_tr, y_tr)
        t_train = time.time() - t_start
        print(f"   - Hoàn thành huấn luyện trong {t_train:.2f} giây.")
        
        # 3.1. Dự đoán trên Kaggle Test (20%)
        t_start = time.time()
        y_pred_k = model.predict(X_test_kaggle)
        y_prob_k = model.predict_proba(X_test_kaggle)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_kaggle)
        t_test_k = time.time() - t_start
        
        acc_k = accuracy_score(y_test_kaggle, y_pred_k)
        prec_k, rec_k, f1_k, _ = precision_recall_fscore_support(y_test_kaggle, y_pred_k, average='binary', pos_label=1)
        fpr_k, tpr_k, _ = roc_curve(y_test_kaggle, y_prob_k)
        auc_k = auc(fpr_k, tpr_k)
        cm_k = confusion_matrix(y_test_kaggle, y_pred_k)
        
        # 3.2. Dự đoán trên Mendeley Test
        t_start = time.time()
        y_pred_m = model.predict(X_test_mendeley)
        y_prob_m = model.predict_proba(X_test_mendeley)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_mendeley)
        t_test_m = time.time() - t_start
        
        acc_m = accuracy_score(y_test_mendeley, y_pred_m)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test_mendeley, y_pred_m, average='binary', pos_label=1)
        fpr_m, tpr_m, _ = roc_curve(y_test_mendeley, y_prob_m)
        auc_m = auc(fpr_m, tpr_m)
        cm_m = confusion_matrix(y_test_mendeley, y_pred_m)
        
        results_dict[name] = {
            'train_time': t_train,
            'test_time_kaggle': t_test_k,
            'test_time_mendeley': t_test_m,
            'kaggle': {
                'accuracy': acc_k, 'precision': prec_k, 'recall': rec_k, 'f1': f1_k, 'auc': auc_k,
                'fpr': fpr_k, 'tpr': tpr_k, 'cm': cm_k,
                'report': classification_report(y_test_kaggle, y_pred_k, target_names=["Benign (0)", "SQLi (1)"], digits=4)
            },
            'mendeley': {
                'accuracy': acc_m, 'precision': prec_m, 'recall': rec_m, 'f1': f1_m, 'auc': auc_m,
                'fpr': fpr_m, 'tpr': tpr_m, 'cm': cm_m,
                'report': classification_report(y_test_mendeley, y_pred_m, target_names=["Benign (0)", "SQLi (1)"], digits=4)
            }
        }
        
        # Lưu file mô hình
        pkl_path = f"c:/Users/LOQ/Downloads/DBS/{name.lower().replace(' ', '_')}_model.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
            
    # 4. Vẽ đồ thị rõ nét (DPI=300, kích thước lớn, nhãn rõ ràng)
    print("\n[Bước 4] Đang vẽ và lưu các biểu đồ so sánh rõ nét...")
    sns.set_theme(style="whitegrid", context="talk") # Tự động tăng cỡ chữ và nét vẽ
    
    models_names = list(models.keys())
    
    # 4.1. model_comparison_kaggle.png
    plt.figure(figsize=(12, 8))
    kaggle_metrics = []
    for name in models_names:
        kaggle_metrics.append([name, 'Accuracy', results_dict[name]['kaggle']['accuracy']])
        kaggle_metrics.append([name, 'Precision', results_dict[name]['kaggle']['precision']])
        kaggle_metrics.append([name, 'Recall', results_dict[name]['kaggle']['recall']])
        kaggle_metrics.append([name, 'F1-Score', results_dict[name]['kaggle']['f1']])
    df_k = pd.DataFrame(kaggle_metrics, columns=['Model', 'Metric', 'Value'])
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_k, palette='Blues_r')
    plt.title("Hiệu năng phân loại của 5 mô hình trên Tập Kaggle (20% Test)", fontsize=16, fontweight='bold', pad=15)
    plt.ylim(0.7, 1.01)
    plt.ylabel("Điểm số", fontsize=13)
    plt.xlabel("Chỉ số hiệu năng", fontsize=13)
    plt.legend(title="Mô hình", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("c:/Users/LOQ/Downloads/DBS/model_comparison_kaggle.png", dpi=300)
    plt.close()

    # 4.2. model_comparison_mendeley.png
    plt.figure(figsize=(12, 8))
    mendeley_metrics = []
    for name in models_names:
        mendeley_metrics.append([name, 'Accuracy', results_dict[name]['mendeley']['accuracy']])
        mendeley_metrics.append([name, 'Precision', results_dict[name]['mendeley']['precision']])
        mendeley_metrics.append([name, 'Recall', results_dict[name]['mendeley']['recall']])
        mendeley_metrics.append([name, 'F1-Score', results_dict[name]['mendeley']['f1']])
    df_m = pd.DataFrame(mendeley_metrics, columns=['Model', 'Metric', 'Value'])
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df_m, palette='Oranges_r')
    plt.title("Hiệu năng phân loại của 5 mô hình trên Tập Mendeley (Test độc lập)", fontsize=16, fontweight='bold', pad=15)
    plt.ylim(0.7, 1.01)
    plt.ylabel("Điểm số", fontsize=13)
    plt.xlabel("Chỉ số hiệu năng", fontsize=13)
    plt.legend(title="Mô hình", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("c:/Users/LOQ/Downloads/DBS/model_comparison_mendeley.png", dpi=300)
    plt.close()

    # 4.3. time_comparison_all.png (So sánh thời gian xử lý)
    plt.figure(figsize=(12, 8))
    time_data = []
    for name in models_names:
        time_data.append([name, 'Huấn luyện (Kaggle Train)', results_dict[name]['train_time']])
        time_data.append([name, 'Kiểm thử (Kaggle Test)', results_dict[name]['test_time_kaggle']])
        time_data.append([name, 'Kiểm thử (Mendeley Test)', results_dict[name]['test_time_mendeley']])
    df_time = pd.DataFrame(time_data, columns=['Model', 'Task', 'Seconds'])
    
    # Do thời gian SVM quá lớn so với các cây quyết định, ta sử dụng thang đo Logarithmic
    ax = sns.barplot(x='Model', y='Seconds', hue='Task', data=df_time, palette='Set2')
    ax.set_yscale('log')
    plt.title("So sánh thời gian xử lý của 5 mô hình (Thang đo logarit)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Thời gian (giây) - Cực log", fontsize=13)
    plt.xlabel("Mô hình học máy", fontsize=13)
    plt.legend(title="Tác vụ")
    plt.tight_layout()
    plt.savefig("c:/Users/LOQ/Downloads/DBS/time_comparison_all.png", dpi=300)
    plt.close()

    # 4.4. roc_curves_combined.png (ROC curve song song 1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Kaggle ROC
    for name in models_names:
        fpr = results_dict[name]['kaggle']['fpr']
        tpr = results_dict[name]['kaggle']['tpr']
        auc_val = results_dict[name]['kaggle']['auc']
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_title('Đường cong ROC trên tập Kaggle Test', fontsize=15, fontweight='bold')
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--')

    # Mendeley ROC
    for name in models_names:
        fpr = results_dict[name]['mendeley']['fpr']
        tpr = results_dict[name]['mendeley']['tpr']
        auc_val = results_dict[name]['mendeley']['auc']
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})', lw=2)
    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_title('Đường cong ROC trên tập Mendeley Test độc lập', fontsize=15, fontweight='bold')
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--')

    plt.suptitle("So sánh đường cong ROC và chỉ số AUC của 5 mô hình", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("c:/Users/LOQ/Downloads/DBS/roc_curves_combined.png", dpi=300)
    plt.close()

    # 4.5. confusion_matrices_combined.png (Lưới 2x5)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Hàng 1: Kaggle Test
    for idx, name in enumerate(models_names):
        cm = results_dict[name]['kaggle']['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx], cbar=False,
                    xticklabels=['Benign', 'SQLi'], yticklabels=['Benign', 'SQLi'],
                    annot_kws={"size": 14, "weight": "bold"})
        axes[0, idx].set_title(f'Kaggle - {name}', fontsize=14, fontweight='bold')
        axes[0, idx].set_ylabel('Thực tế', fontsize=11)
        axes[0, idx].set_xlabel('Dự đoán', fontsize=11)
        
    # Hàng 2: Mendeley Test
    for idx, name in enumerate(models_names):
        cm = results_dict[name]['mendeley']['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1, idx], cbar=False,
                    xticklabels=['Benign', 'SQLi'], yticklabels=['Benign', 'SQLi'],
                    annot_kws={"size": 14, "weight": "bold"})
        axes[1, idx].set_title(f'Mendeley - {name}', fontsize=14, fontweight='bold')
        axes[1, idx].set_ylabel('Thực tế', fontsize=11)
        axes[1, idx].set_xlabel('Dự đoán', fontsize=11)
        
    plt.suptitle("Ma trận nhầm lẫn của 5 mô hình trên 2 tập dữ liệu", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("c:/Users/LOQ/Downloads/DBS/confusion_matrices_combined.png", dpi=300)
    plt.close()
    
    print("-> Đã tạo và lưu các biểu đồ so sánh rõ nét mới.")

    # 5. In bảng kết quả so sánh
    for dataset_name in ['kaggle', 'mendeley']:
        print(f"\n========================================================")
        print(f"BẢNG SO SÁNH HIỆU NĂNG TRÊN TẬP DỮ LIỆU: {dataset_name.upper()}")
        print(f"========================================================")
        print("| Mô hình | Accuracy | Precision | Recall | F1-Score | AUC | Test Time (s) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for name in models_names:
            d = results_dict[name]
            dm = d[dataset_name]
            tt = d['test_time_kaggle'] if dataset_name == 'kaggle' else d['test_time_mendeley']
            print(f"| {name} | {dm['accuracy']*100:.2f}% | {dm['precision']*100:.2f}% | {dm['recall']*100:.2f}% | {dm['f1']*100:.2f}% | {dm['auc']:.4f} | {tt:.4f}s |")
            
    print("\nToàn bộ quy trình huấn luyện và đánh giá trên 2 tập dữ liệu hoàn tất!")
