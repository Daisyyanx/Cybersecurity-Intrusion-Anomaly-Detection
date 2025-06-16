# ✅ train_xgb.py

import pandas as pd
import xgboost as xgb
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(df):
    """严格和 BETHDataset 一样的预处理"""
    df = df.copy()
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["parentProcessId"] = df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
    df["mountNamespace"] = df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)
    df["returnValue"] = df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))
    return df

# === 1. 读 CSV ===
train_df = pd.read_csv("data/labelled_training_data.csv")
val_df = pd.read_csv("data/labelled_validation_data.csv")
test_df = pd.read_csv("data/labelled_testing_data.csv")

# === 2. 特征列 & Label ===
features = [
    "processId", "parentProcessId", "userId",
    "mountNamespace", "eventId", "argsNum", "returnValue"
]
label = "sus"

X_train = preprocess(train_df[features])
y_train = train_df[label]

X_val = preprocess(val_df[features])
y_val = val_df[label]

X_test = preprocess(test_df[features])
y_test = test_df[label]

# === 3. 训练 XGB ===
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# === 4. 保存模型 ===
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ XGBoost 模型训练并保存为 xgb_model.pkl")

# === 5. 在验证集 & 测试集上评估 ===
for name, X, y in [
    ("val", X_val, y_val),
    ("test", X_test, y_test)
]:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "auroc": roc_auc_score(y, probs)
    }

    # 保存 JSON
    os.makedirs("results", exist_ok=True)
    with open(f"results/xgb_{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Saved {name} metrics:", metrics)

    # 画柱状图
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.ylim(0, 1)
    plt.title(f"XGB {name} Metrics")
    plt.tight_layout()
    plt.savefig(f"results/xgb_{name}_metrics.png")
    plt.close()
    print(f"✅ Saved {name} metrics plot: results/xgb_{name}_metrics.png")