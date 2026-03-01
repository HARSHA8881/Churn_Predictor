"""
Standalone model training script.
Usage:
    python model_train.py --data data/Churn_Modelling.csv
"""
import argparse
import pandas as pd
from utils.data_utils import load_data, clean_data, encode_features, scale_features
from src.model import train_models, save_models


def main(path: str):
    print(f"[1/4] Loading data from: {path}")
    with open(path, 'rb') as f:
        raw_df, df = load_data(f)

    print("[2/4] Cleaning and encoding...")
    X, Y, num_cols, cat_cols = clean_data(df)
    X, encoders = encode_features(X, cat_cols)
    X, minmax = scale_features(X, num_cols)

    print("[3/4] Training models...")
    log, dt, x_train, x_test, y_train, y_test = train_models(X, Y)

    print("[4/4] Saving models to models/...")
    save_models(log, dt, minmax)
    print("Done! Models saved to the models/ directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset.")
    args = parser.parse_args()
    main(args.data)
