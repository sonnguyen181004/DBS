"""
Train 5 mô hình trên TOÀN BỘ Kaggle (148,326 mẫu)
Test HOÀN TOÀN trên Mendeley (47,463 mẫu)
Xuất số liệu thật + 5 biểu đồ rõ nét
"""
import re, math, string, time, pickle, os, sys, io
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_curve, auc, confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

IMG_DIR = "c:/Users/LOQ/Downloads/DBS"

# ─── Feature extraction ───────────────────────────────────────────────────────
def extract_features(query):
    if not isinstance(query, str): query = ""
    L = len(query); ql = query.lower()
    counts = Counter(query)
    entropy = -sum((v/L)*math.log2(v/L) for v in counts.values()) if L else 0.0
    kw = ["select","update","insert","delete","drop","create","alter","where",
          "from","join","into","table","union","all","group","order","by",
          "having","limit","exec","execute"]
    lops = ["and","or","not","xor","like","between","in","is"]
    return {
        'no_letter':          float(sum(c.isalpha()  for c in query)),
        'no_digit':           float(sum(c.isdigit()  for c in query)),
        'no_special _char':   float(sum(not c.isalnum() and not c.isspace() for c in query)),
        'no_keyword':         float(sum(len(re.findall(rf'\b{k}\b', ql)) for k in kw)),
        'length':             float(L),
        'entropy':            entropy,
        'no_semicolon':       float(query.count(';')),
        'no_single_ qout':    float(query.count("'")),
        'no_double_ qout':    float(query.count('"')),
        'no_percentage':      float(query.count('%')),
        'whit_ space':        float(query.count(' ')+1),
        'no_punctuation':     float(sum(c in string.punctuation for c in query)),
        'no_logical _operat': float(sum(len(re.findall(rf'\b{o}\b', ql)) for o in lops)),
        'no_operat':          float(len(re.findall(r'[+\-*/=<>]', query))),
        'no_or':              float(len(re.findall(r'\bor\b',  ql))),
        'no_and':             float(len(re.findall(r'\band\b', ql))),
        'no_comment':         float(query.count('--')+query.count('/*')+query.count('#')),
        'no_null Value in the SQL Query': float(len(re.findall(r'\bnull\b', ql))),
    }

def batch_extract(queries):
    return [extract_features(q) for q in queries]

FEAT_COLS = [
    'no_letter','no_digit','no_special _char','no_keyword','length','entropy',
    'no_semicolon','no_single_ qout','no_double_ qout','no_percentage',
    'whit_ space','no_punctuation','no_logical _operat','no_operat',
    'no_or','no_and','no_comment','no_null Value in the SQL Query'
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def feat_df(queries, n_workers=4):
    qs = list(queries)
    bs = math.ceil(len(qs)/n_workers)
    batches = [qs[i:i+bs] for i in range(0, len(qs), bs)]
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(batch_extract, batches):
            rows.extend(r)
    return pd.DataFrame(rows)[FEAT_COLS]

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print("TRAIN: 100% KAGGLE  |  TEST: 100% MENDELEY")
    print("="*70)

    # 1. Load Kaggle (full train)
    print("\n[1] Đang tải Kaggle (full training set)...")
    kg = pd.read_csv(f"{IMG_DIR}/dataset/clean_sql_dataset.csv")
    kg['Query'] = kg['Query'].fillna("")
    print(f"    → {len(kg):,} dòng Kaggle")

    print("    → Trích xuất đặc trưng (multiprocessing)...")
    t0 = time.time()
    X_train = feat_df(kg['Query'])
    t_feat_train = time.time()-t0
    y_train = kg['Label'].values
    print(f"    → Xong trong {t_feat_train:.1f}s")

    # 2. Load Mendeley (full test)
    print("\n[2] Đang tải Mendeley (independent test set)...")
    md = pd.read_csv(f"{IMG_DIR}/dataset/SQLI_Dataset.csv")
    md = md.dropna(subset=['label'])
    md[FEAT_COLS] = md[FEAT_COLS].fillna(0.0)
    X_test  = md[FEAT_COLS]
    y_test  = md['label'].values
    print(f"    → {len(md):,} dòng Mendeley")
    print(f"    → Benign: {(y_test==0).sum():,} | SQLi: {(y_test==1).sum():,}")

    # 3. Define models
    MODELS = {
        'SVM':          SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost':      XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
        'LightGBM':     LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
        'AdaBoost':     AdaBoostClassifier(n_estimators=50, random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    # SVM subset (30k vì O(n²))
    rng = np.random.default_rng(42)
    svm_idx = rng.choice(len(X_train), 30_000, replace=False)
    X_svm = X_train.iloc[svm_idx]; y_svm = y_train[svm_idx]

    # 4. Train & Evaluate
    print("\n[3] Huấn luyện và đánh giá...")
    results = {}

    for name, model in MODELS.items():
        print(f"\n    ── {name} ──")
        Xtr = X_svm  if name == 'SVM' else X_train
        ytr = y_svm  if name == 'SVM' else y_train
        n_tr= len(Xtr)

        t0 = time.time()
        model.fit(Xtr, ytr)
        t_train = time.time()-t0
        print(f"       Train ({n_tr:,} mẫu): {t_train:.4f}s")

        t0 = time.time()
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:,1]
                  if hasattr(model,'predict_proba')
                  else model.decision_function(X_test))
        t_test = time.time()-t0
        print(f"       Test  ({len(X_test):,} mẫu): {t_test:.4f}s")

        acc  = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        cm  = confusion_matrix(y_test, y_pred)
        rep = classification_report(y_test, y_pred,
                                    target_names=['Benign (0)','SQLi (1)'], digits=4)

        results[name] = dict(
            n_train=n_tr, t_train=t_train, t_test=t_test,
            acc=acc, prec=prec, rec=rec, f1=f1, auc=auc_val,
            fpr=fpr, tpr=tpr, cm=cm, report=rep
        )

        # Save pkl
        with open(f"{IMG_DIR}/{name.lower().replace(' ','_')}_model.pkl",'wb') as f:
            pickle.dump(model, f)

    # 5. Print results table
    print("\n" + "="*80)
    print("KẾT QUẢ THỰC NGHIỆM (Train: Kaggle full | Test: Mendeley)")
    print("="*80)
    print(f"{'Mô hình':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9} {'AUC':>7} {'Train(s)':>9} {'Test(s)':>8}")
    print("-"*80)
    for name, r in results.items():
        print(f"{name:<16} {r['acc']*100:>8.2f}% {r['prec']*100:>9.2f}% "
              f"{r['rec']*100:>7.2f}% {r['f1']*100:>8.2f}% "
              f"{r['auc']:>7.4f} {r['t_train']:>9.4f}s {r['t_test']:>7.4f}s")

    for name, r in results.items():
        print(f"\n── Classification Report: {name} ──")
        print(r['report'])

    # 6. Plot 1: Performance bar chart
    print("\n[4] Vẽ biểu đồ...")
    sns.set_theme(style='whitegrid', context='talk', font_scale=1.0)
    PALETTE = ['#2980B9','#E74C3C','#27AE60','#F39C12','#8E44AD']
    names   = list(results.keys())

    # Fig 1: 4-metric grouped bar
    metrics_map = {'Accuracy':'acc','Precision':'prec','Recall':'rec','F1-Score':'f1'}
    x = np.arange(len(metrics_map))
    w = 0.15
    fig, ax = plt.subplots(figsize=(13,7))
    for i, (name, r) in enumerate(results.items()):
        vals = [r[k]*100 for k in metrics_map.values()]
        bars = ax.bar(x + i*w - 2*w, vals, w, label=name, color=PALETTE[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(list(metrics_map.keys()), fontsize=13)
    ax.set_ylabel('Điểm số (%)', fontsize=13)
    ax.set_ylim(70, 105)
    ax.set_title('Hiệu năng phân loại của 5 mô hình\n(Train: Kaggle full | Test: Mendeley)', fontsize=15, fontweight='bold')
    ax.legend(title='Mô hình', fontsize=11, title_fontsize=12, loc='lower right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig1_performance.png", dpi=300, bbox_inches='tight')
    plt.close(); print("    → fig1_performance.png")

    # Fig 2: Time comparison (log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    t_trains = [results[n]['t_train'] for n in names]
    t_tests  = [results[n]['t_test']  for n in names]
    short_names = ['SVM','XGBoost','LightGBM','AdaBoost','Rand.\nForest']

    bars1 = ax1.bar(short_names, t_trains, color=PALETTE, edgecolor='white')
    for b, v in zip(bars1, t_trains):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()*1.05, f'{v:.2f}s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_yscale('log'); ax1.set_ylabel('Thời gian (giây) – log scale', fontsize=12)
    ax1.set_title('Thời gian Huấn luyện\n(Train: Kaggle full)', fontsize=13, fontweight='bold')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6); ax1.set_axisbelow(True)

    bars2 = ax2.bar(short_names, t_tests, color=PALETTE, edgecolor='white')
    for b, v in zip(bars2, t_tests):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()*1.05, f'{v:.4f}s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_yscale('log'); ax2.set_ylabel('Thời gian (giây) – log scale', fontsize=12)
    ax2.set_title('Thời gian Dự đoán\n(Test: Mendeley – 47,463 mẫu)', fontsize=13, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.6); ax2.set_axisbelow(True)

    plt.suptitle('So sánh Chi phí Tính toán của 5 Mô hình (thang đo logarit)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig2_time.png", dpi=300, bbox_inches='tight')
    plt.close(); print("    → fig2_time.png")

    # Fig 3: ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, r) in enumerate(results.items()):
        ax.plot(r['fpr'], r['tpr'], color=PALETTE[i], lw=2.5,
                label=f"{name} (AUC = {r['auc']:.4f})")
    ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random classifier')
    ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=13)
    ax.set_ylabel('True Positive Rate (TPR) / Recall', fontsize=13)
    ax.set_title('Đường cong ROC – AUC Score\n(Test: Mendeley – 47,463 mẫu)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig3_roc.png", dpi=300, bbox_inches='tight')
    plt.close(); print("    → fig3_roc.png")

    # Fig 4: Confusion matrices (1 row x 5 cols)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for i, (name, r) in enumerate(results.items()):
        cm = r['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Benign','SQLi'], yticklabels=['Benign','SQLi'],
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size':14, 'weight':'bold'}, cbar=False)
        tn,fp,fn,tp = cm.ravel()
        axes[i].set_title(f'{name}\nAcc={r["acc"]*100:.2f}% | F1={r["f1"]*100:.2f}%',
                          fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Dự đoán', fontsize=11); axes[i].set_ylabel('Thực tế', fontsize=11)
        # Annotate FP / FN
        axes[i].text(0.5, -0.22, f'FP={fp:,}  FN={fn:,}',
                     ha='center', transform=axes[i].transAxes, fontsize=10, color='#C0392B')
    plt.suptitle('Ma trận Nhầm lẫn (Confusion Matrix) của 5 Mô hình\n(Test: Mendeley – 47,463 mẫu)',
                 fontsize=16, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig4_confusion.png", dpi=300, bbox_inches='tight')
    plt.close(); print("    → fig4_confusion.png")

    # Fig 5: Radar / Spider chart
    from matplotlib.patches import FancyArrowPatch
    radar_metrics = ['Accuracy','Precision','Recall','F1-Score','AUC (×100)']
    radar_keys    = ['acc','prec','rec','f1','auc']
    N = len(radar_metrics)
    angles = [n/float(N)*2*math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for i, (name, r) in enumerate(results.items()):
        vals = [r[k]*100 for k in radar_keys] + [r[radar_keys[0]]*100]
        ax.plot(angles, vals, color=PALETTE[i], lw=2.5, label=name)
        ax.fill(angles, vals, color=PALETTE[i], alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=12)
    ax.set_ylim(70, 100)
    ax.set_yticks([75,80,85,90,95,100])
    ax.set_yticklabels(['75','80','85','90','95','100'], fontsize=9)
    ax.grid(color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Biểu đồ Radar so sánh toàn diện 5 Mô hình\n(Test: Mendeley)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig5_radar.png", dpi=300, bbox_inches='tight')
    plt.close(); print("    → fig5_radar.png")

    print("\n✅ Hoàn tất! Lưu kết quả vào results_summary.txt...")

    # Save summary to file for docx script
    with open(f"{IMG_DIR}/results_summary.txt", 'w', encoding='utf-8') as f:
        f.write("MODEL|ACC|PREC|REC|F1|AUC|TTRAIN|TTEST|NTRAIN\n")
        for name, r in results.items():
            f.write(f"{name}|{r['acc']:.6f}|{r['prec']:.6f}|{r['rec']:.6f}|"
                    f"{r['f1']:.6f}|{r['auc']:.6f}|{r['t_train']:.4f}|{r['t_test']:.4f}|{r['n_train']}\n")
        f.write("\n===REPORTS===\n")
        for name, r in results.items():
            f.write(f"---{name}---\n{r['report']}\n")
        for name, r in results.items():
            cm = r['cm']
            tn,fp,fn,tp = cm.ravel()
            f.write(f"CM_{name}|TN={tn}|FP={fp}|FN={fn}|TP={tp}\n")

    print("✅ Tất cả hoàn tất!")
