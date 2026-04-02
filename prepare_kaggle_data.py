"""
prepare_kaggle_data.py
----------------------
Preprocesses the official Kaggle Credit Card Fraud dataset.

Steps to use:
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download creditcard.csv
3. Place it in the data/ folder
4. Run: python prepare_kaggle_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


def prepare_kaggle(input_file="data/creditcard.csv"):
    path = Path(input_file)
    if not path.exists():
        print(f"ERROR: '{input_file}' not found.")
        print("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("and place creditcard.csv inside the data/ folder.")
        return False

    print(f"Loading {input_file} ...")
    df = pd.read_csv(input_file)

    print(f"  Shape            : {df.shape}")
    print(f"  Fraud cases      : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"  Legit cases      : {(df['Class']==0).sum():,}")

    # Kaggle dataset uses PCA features V1..V28, Time, Amount
    # We rename Class -> is_fraud and keep all PCA features
    df = df.rename(columns={"Class": "is_fraud"})

    # Normalize Amount and Time
    df["amount_norm"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["time_norm"]   = (df["Time"] - df["Time"].mean()) / df["Time"].std()

    # Feature engineering on top of PCA features
    df["amount_log"]  = np.log1p(df["Amount"])

    out_path = DATA_DIR / "kaggle_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"\nProcessed data saved to: {out_path}")
    print("Now run: python train_model.py --dataset kaggle")
    return True


if __name__ == "__main__":
    prepare_kaggle()