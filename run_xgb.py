#!/usr/bin/env python3
"""XGBoost training and testing functions based on the reference implementation."""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def preprocess(df):
    """严格和 BETHDataset 一样的预处理"""
    df = df.copy()
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["parentProcessId"] = df["parentProcessId"].map(
        lambda x: 0 if x in [0, 1, 2] else 1
    )
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
    df["mountNamespace"] = df["mountNamespace"].map(
        lambda x: 0 if x == 4026531840 else 1
    )
    df["returnValue"] = df["returnValue"].map(
        lambda x: 0 if x == 0 else (1 if x > 0 else 2)
    )
    return df


def get_model_path():
    """Get the path for saving/loading XGBoost models."""
    os.makedirs("models", exist_ok=True)
    return os.path.join("models", "xgb_beth_best.pkl")


def train(seed=42):
    """
    Train XGBoost model on the BETH dataset.

    Args:
        seed (int): Random seed for reproducibility

    Returns:
        dict: Training results including metrics
    """
    print(f"✅ 开始训练 XGBoost 模型...")

    # === 1. 读 CSV ===
    print("Loading data...")
    train_df = pd.read_csv("data/labelled_training_data.csv")
    val_df = pd.read_csv("data/labelled_validation_data.csv")
    test_df = pd.read_csv("data/labelled_testing_data.csv")

    print(f"Train data shape: {train_df.shape}")
    print(f"Val data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # === 2. 特征列 & Label ===
    features = [
        "processId",
        "parentProcessId",
        "userId",
        "mountNamespace",
        "eventId",
        "argsNum",
        "returnValue",
    ]
    label = "sus"

    X_train = preprocess(train_df[features])
    y_train = train_df[label]

    X_val = preprocess(val_df[features])
    y_val = val_df[label]

    X_test = preprocess(test_df[features])
    y_test = test_df[label]

    print(f"Train features shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Anomaly ratio in train: {y_train.mean():.4f}")
    print(f"Anomaly ratio in val: {y_val.mean():.4f}")
    print(f"Anomaly ratio in test: {y_test.mean():.4f}")

    # === 3. 训练 XGB ===
    print("Creating and fitting XGBoost model...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=seed,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    print("✅ XGBoost fitting successful!")

    # === 4. 保存模型 ===
    model_path = get_model_path()
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ XGBoost 模型训练并保存为 {model_path}")

    # === 5. 在验证集上评估 ===
    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_metrics = {
        "accuracy": accuracy_score(y_val, val_preds),
        "precision": precision_score(y_val, val_preds),
        "recall": recall_score(y_val, val_preds),
        "f1": f1_score(y_val, val_preds),
        "auroc": roc_auc_score(y_val, val_probs),
    }

    # === 6. 在测试集上评估 ===
    print("Evaluating on test set...")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "auroc": roc_auc_score(y_test, test_probs),
    }

    # Print results
    print(f"\nTraining Results:")
    print(f"Val AUROC: {val_metrics['auroc']:.4f}")
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Val Precision: {val_metrics['precision']:.4f}")
    print(f"Val Recall: {val_metrics['recall']:.4f}")
    print(f"Val F1: {val_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

    # Return results
    results = {
        "val_auroc": val_metrics["auroc"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "test_auroc": test_metrics["auroc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "model_path": model_path,
    }

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/xgb_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Saved metrics:", results)

    return results


def test(features):
    """
    Test XGBoost model on user input features.

    Args:
        features (pd.DataFrame or np.ndarray): Input features with columns:
            ["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]

    Returns:
        dict: Prediction results including scores and predictions
    """
    print(f"Testing XGBoost model on user input...")

    # Load the trained model
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first."
        )

    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")

    # Convert features to DataFrame if it's numpy array
    if isinstance(features, np.ndarray):
        feature_names = [
            "processId",
            "parentProcessId",
            "userId",
            "mountNamespace",
            "eventId",
            "argsNum",
            "returnValue",
        ]
        features = pd.DataFrame(features, columns=feature_names)

    # Preprocess features
    features_processed = preprocess(features)

    print(f"Input features shape: {features_processed.shape}")

    # Make predictions
    print("Making predictions...")
    probs = model.predict_proba(features_processed)[
        :, 1
    ]  # Probability of anomaly class
    preds = (probs >= 0.5).astype(int)  # Binary predictions

    print(f"✅ Prediction completed!")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(
        f"Predicted anomalies: {preds.sum()} out of {len(preds)} ({preds.mean():.4f})"
    )

    # Return results
    results = {"probabilities": probs, "predictions": preds, "model_path": model_path}

    return results


if __name__ == "__main__":
    # Example usage
    print("XGBoost Training and Testing Example")
    print("=" * 50)

    # Train the model
    print("\n1. Training XGBoost model...")
    train_results = train(seed=42)

    # Example of using test function with user input
    print("\n2. Example: Testing with user input features...")

    # Create example features (you can replace this with your own data)
    example_features = pd.DataFrame(
        {
            "processId": [1, 2, 3],
            "parentProcessId": [0, 1, 2],
            "userId": [1000, 500, 2000],
            "mountNamespace": [4026531840, 4026531841, 4026531840],
            "eventId": [1, 2, 3],
            "argsNum": [2, 3, 1],
            "returnValue": [0, 1, -1],
        }
    )

    print("Example input features:")
    print(example_features)

    test_results = test(example_features)

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Val AUROC: {train_results['val_auroc']:.4f}")
    print(f"Test AUROC: {train_results['test_auroc']:.4f}")
    print(f"Test F1: {train_results['test_f1']:.4f}")
    print(f"User input predictions: {test_results['predictions']}")
    print(f"User input probabilities: {test_results['probabilities']}")
