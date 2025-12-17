#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


# 1. Load data
def load_data(path=None):
    if path is None:
        ROOT = Path(__file__).resolve().parents[3]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


# 2. Feature engineering: lags, differences, trend, fixed effects
def engineer_features(df):

    # Encode canton as categorical codes
    df["canton_id"] = df["canton"].astype("category").cat.codes
    # Sort
    df = df.sort_values(["canton_id", "year"]).copy()

    # 1) Lag of migration
    df["migration_lag1"] = df.groupby("canton_id")["migration_rate"].shift(1)

    # 2) Differences (Δ)
    for col in ["log_rent_avg", "log_avg_income", "log_unemployment", "log_schockexposure"]:
        df[f"d_{col}"] = df.groupby("canton_id")[col].diff()

    # 3) Trend
    df["t"] = df["year"] - df["year"].min()

    # 4) Canton FE (one-hot encoding)
    FE = pd.get_dummies(df["canton_id"], prefix="FE", drop_first=True)
    df = pd.concat([df, FE], axis=1)

    return df

# 3. Prepare dataframe: features and target

def prepare_dataframe(df):

    base_features = [
        "log_rent_avg",
        "log_avg_income",
        "log_unemployment",
        "log_schockexposure",
        "migration_lag1",
        "d_log_rent_avg",
        "d_log_avg_income",
        "d_log_unemployment",
        "d_log_schockexposure",
        "t",
    ]

    fe_cols = [c for c in df.columns if c.startswith("FE_")]

    feature_cols = base_features + fe_cols
    target_col = "migration_rate"

    df = df.dropna(subset=feature_cols + [target_col]).copy()

    return df, feature_cols, target_col



# 4. Time split

def time_split(df, feature_cols, target_col):

    df = df.sort_values("year")

    years = df["year"].unique()
    cut = int(0.8 * len(years))

    train_years = years[:cut]
    test_years  = years[cut:]

    df_train = df[df["year"].isin(train_years)]
    df_test  = df[df["year"].isin(test_years)]

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy()

    X_test  = df_test[feature_cols].to_numpy()
    y_test  = df_test[target_col].to_numpy()

    return X_train, X_test, y_train, y_test



# 5. Train scikit-learn boosting

def train_boosting(X_train, y_train):
    
    model = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=80,   
    learning_rate=0.05,      
    max_depth=2,             
    min_samples_split=10,    
    min_samples_leaf=5,      
    subsample=0.8,
    random_state=0,
    )

    model.fit(X_train, y_train)
    return model



# 6. Evaluation

def evaluate(model, X_train, y_train, X_test, y_test):

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test, y_pred_test)
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return train_r2, test_r2, rmse



# MAIN
if __name__ == "__main__":
    print("\n=== LOADING DATA ===")
    df = load_data()

    print("\n=== ENGINEERING FEATURES (LAGS, DIFF, FE) ===")
    df = engineer_features(df)

    df, features, target = prepare_dataframe(df)

    print("\n=== TIME SPLIT ===")
    X_train, X_test, y_train, y_test = time_split(df, features, target)

    print("\n=== TRAINING BOOSTING ===")
    model = train_boosting(X_train, y_train)

    print("\n=== RESULTS ===")
    train_r2, test_r2, rmse = evaluate(model, X_train, y_train, X_test, y_test)
    print(f"Train R² : {train_r2:.4f}")
    print(f"Test  R² : {test_r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    print("\n=== FEATURE IMPORTANCE ===")
    importance = dict(zip(features, model.feature_importances_))
    for k, v in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"{k:25s}: {v:.4f}")
