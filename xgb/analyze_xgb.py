# ✅ analyze_xgb.py

import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 加载模型 ===
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ 已加载模型")

# === 加载数据（这里举例用测试集）===
test_df = pd.read_csv("data/labelled_testing_data.csv")

# === 跟 BETH 一样预处理 ===
def preprocess(df):
    df = df.copy()
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0,1,2] else 1)
    df["parentProcessId"] = df["parentProcessId"].map(lambda x: 0 if x in [0,1,2] else 1)
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
    df["mountNamespace"] = df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)
    df["returnValue"] = df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))
    return df

features = [
    "processId", "parentProcessId", "userId",
    "mountNamespace", "eventId", "argsNum", "returnValue"
]

X_test = preprocess(test_df[features])
y_test = test_df["sus"]

# === 预测 ===
probas = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

# === 打印样例 ===
print("\n=== 样例输出 ===")
print(pd.DataFrame({
    "score": probas[:10],
    "predict": preds[:10],
    "true_label": y_test.values[:10]
}))

# === 特征重要性 ===
importances = model.feature_importances_
print("\n=== 特征重要性 ===")
for feat, score in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{feat}: {score:.4f}")

# === 画前 5 大特征 分布对比 ===
top_features = [feat for feat, _ in sorted(zip(features, importances), key=lambda x: -x[1])[:5]]

os.makedirs("results", exist_ok=True)

for feat in top_features:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(X_test[ y_test==0 ][feat], label="Normal", fill=True)
    sns.kdeplot(X_test[ y_test==1 ][feat], label="Suspect", fill=True)
    plt.title(f"Feature: {feat} distribution")
    plt.legend()
    plt.savefig(f"results/{feat}_distribution.png")
    plt.close()

print(f"✅ 已保存前 5 大特征分布对比到 results/ 文件夹")

# === 额外保存分数 ===
result_df = pd.DataFrame({
    "proba": probas,
    "predict": preds,
    "true_label": y_test.values
})
result_df.to_csv("results/test_scores.csv", index=False)
print("✅ 已保存测试分数 results/test_scores.csv")