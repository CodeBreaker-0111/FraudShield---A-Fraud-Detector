"""
train_model.py
--------------
Trains a Random Forest fraud detection model.

Usage:
    python train_model.py               # uses synthetic data
    python train_model.py --dataset kaggle   # uses Kaggle dataset
"""

import argparse
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─── Feature sets ────────────────────────────────────────────────────────────

SYNTHETIC_FEATURES = [
    "amount", "hour", "distance_from_home", "txn_per_hour",
    "avg_monthly_spend", "new_device", "merchant_risk",
    "card_type", "international",
    "amount_ratio", "is_night", "high_velocity",
    "far_from_home", "risk_amount_combo",
]

KAGGLE_FEATURES = (
    [f"V{i}" for i in range(1, 29)]
    + ["amount_norm", "time_norm", "amount_log"]
)


# ─── Load data ────────────────────────────────────────────────────────────────

def load_data(dataset="synthetic"):
    if dataset == "kaggle":
        path = Path("data/kaggle_processed.csv")
        if not path.exists():
            raise FileNotFoundError(
                "Kaggle data not found. Run: python prepare_kaggle_data.py first."
            )
        df = pd.read_csv(path)
        features = KAGGLE_FEATURES
        print(f"Loaded Kaggle dataset: {df.shape}")
    else:
        path = Path("data/card_transactions.csv")
        if not path.exists():
            raise FileNotFoundError(
                "Synthetic data not found. Run: python generate_data.py first."
            )
        df = pd.read_csv(path)
        features = SYNTHETIC_FEATURES
        print(f"Loaded synthetic dataset: {df.shape}")

    return df, features


# ─── Train ────────────────────────────────────────────────────────────────────

def train(dataset="synthetic"):
    df, features = load_data(dataset)

    X = df[features].values
    y = df["is_fraud"].values

    print(f"\nClass distribution before SMOTE:")
    print(f"  Legit : {(y==0).sum():,}")
    print(f"  Fraud : {(y==1).sum():,}  ({y.mean()*100:.2f}%)")

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE — oversample minority (fraud) class in training set only
    print("\nApplying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE — Legit: {(y_train_sm==0).sum():,}  Fraud: {(y_train_sm==1).sum():,}")

    # Train Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_sm, y_train_sm)
    print("Training complete.")

    # Evaluate
    print("\n── Evaluation on held-out test set ──────────────────────")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_pr  = average_precision_score(y_test, y_proba)
    print(f"ROC-AUC Score     : {roc_auc:.4f}")
    print(f"Avg Precision     : {avg_pr:.4f}")

    # Save metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "roc_auc":         round(float(roc_auc), 4),
        "avg_precision":   round(float(avg_pr), 4),
        "dataset":         dataset,
        "n_features":      len(features),
        "n_train":         len(y_train_sm),
        "n_test":          len(y_test),
        "confusion_matrix": cm.tolist(),
        "feature_names":   features,
    }

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    metrics["feature_importance"] = [
        {"feature": f, "importance": round(float(i), 4)}
        for f, i in feat_imp
    ]

    # Save artifacts
    joblib.dump(model,  MODELS_DIR / "fraud_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "features.json", "w") as f:
        json.dump(features, f)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved  → models/fraud_model.pkl")
    print("Scaler saved → models/scaler.pkl")
    print("Metrics saved→ models/metrics.json")

    # Plots
    _plot_confusion_matrix(cm)
    _plot_roc_curve(y_test, y_proba, roc_auc)
    _plot_feature_importance(feat_imp[:10])
    print("\nPlots saved  → models/*.png")
    print("\nAll done! Now run: python app.py")


# ─── Plots ───────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()


def _plot_roc_curve(y_test, y_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "roc_curve.png", dpi=120)
    plt.close()


def _plot_feature_importance(feat_imp):
    names  = [f for f, _ in feat_imp]
    values = [v for _, v in feat_imp]
    plt.figure(figsize=(7, 4))
    plt.barh(names[::-1], values[::-1], color="steelblue")
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png", dpi=120)
    plt.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "kaggle"],
        default="synthetic",
        help="Which dataset to use (default: synthetic)",
    )
    args = parser.parse_args()
    train(args.dataset)