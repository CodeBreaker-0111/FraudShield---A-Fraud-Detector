"""
generate_data.py
----------------
Synthetic Credit & Debit Card Fraud Dataset Generator
Run this if you don't have the Kaggle dataset.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

NUM_TRANSACTIONS = 50000
FRAUD_RATIO = 0.03   # 3% fraud — realistic


def generate_dataset(n=NUM_TRANSACTIONS, fraud_ratio=FRAUD_RATIO):
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    def legit_transactions(count):
        return pd.DataFrame({
            "amount":            np.random.exponential(scale=1500, size=count).clip(10, 50000),
            "hour":              np.random.choice(range(24), size=count, p=_hour_distribution()),
            "distance_from_home": np.abs(np.random.normal(loc=8, scale=12, size=count)).clip(0, 200),
            "txn_per_hour":      np.random.choice([1, 1, 1, 2, 2, 3], size=count),
            "avg_monthly_spend": np.random.normal(loc=18000, scale=6000, size=count).clip(2000, 150000),
            "new_device":        np.random.choice([0, 0, 0, 0, 1], size=count),
            "merchant_risk":     np.random.choice([0.05, 0.07, 0.1, 0.2, 0.25, 0.3], size=count,
                                                   p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
            "card_type":         np.random.choice([0, 1], size=count),          # 0=credit 1=debit
            "international":     np.random.choice([0, 0, 0, 0, 1], size=count),
            "is_fraud":          np.zeros(count, dtype=int),
        })

    def fraud_transactions(count):
        return pd.DataFrame({
            "amount":            np.random.exponential(scale=8000, size=count).clip(500, 200000),
            "hour":              np.random.choice(range(24), size=count, p=_night_heavy_distribution()),
            "distance_from_home": np.abs(np.random.normal(loc=120, scale=80, size=count)).clip(0, 500),
            "txn_per_hour":      np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=count,
                                                   p=[0.05, 0.1, 0.2, 0.25, 0.2, 0.1, 0.07, 0.03]),
            "avg_monthly_spend": np.random.normal(loc=18000, scale=6000, size=count).clip(2000, 150000),
            "new_device":        np.random.choice([0, 1, 1, 1], size=count),
            "merchant_risk":     np.random.choice([0.25, 0.3, 0.35, 0.45, 0.55, 0.6], size=count,
                                                   p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1]),
            "card_type":         np.random.choice([0, 1], size=count),
            "international":     np.random.choice([0, 1, 1], size=count),
            "is_fraud":          np.ones(count, dtype=int),
        })

    df = pd.concat([legit_transactions(n_legit), fraud_transactions(n_fraud)], ignore_index=True)

    # Feature engineering
    df["amount_ratio"]      = df["amount"] / (df["avg_monthly_spend"] / 30)
    df["is_night"]          = df["hour"].apply(lambda h: 1 if 1 <= h <= 5 else 0)
    df["high_velocity"]     = (df["txn_per_hour"] >= 4).astype(int)
    df["far_from_home"]     = (df["distance_from_home"] > 100).astype(int)
    df["risk_amount_combo"] = df["merchant_risk"] * df["amount_ratio"]

    df = df.sample(frac=1).reset_index(drop=True)
    return df


def _hour_distribution():
    # Legit: mostly daytime
    weights = np.ones(24)
    for h in range(8, 21):
        weights[h] = 5
    for h in [1, 2, 3, 4, 5]:
        weights[h] = 0.3
    return (weights / weights.sum()).tolist()


def _night_heavy_distribution():
    # Fraud: skewed to night hours
    weights = np.ones(24) * 2
    for h in [1, 2, 3, 4, 5]:
        weights[h] = 8
    for h in range(8, 18):
        weights[h] = 1
    return (weights / weights.sum()).tolist()


if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    df = generate_dataset()

    out_path = DATA_DIR / "card_transactions.csv"
    df.to_csv(out_path, index=False)

    total  = len(df)
    frauds = df["is_fraud"].sum()
    print(f"  Total transactions : {total:,}")
    print(f"  Fraud transactions : {frauds:,} ({frauds/total*100:.2f}%)")
    print(f"  Legit transactions : {total - frauds:,}")
    print(f"  Saved to           : {out_path}")
    print("\nDone! Now run: python train_model.py")