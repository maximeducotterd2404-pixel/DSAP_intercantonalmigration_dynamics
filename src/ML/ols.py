#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sklearn OLS baseline for migration_rate
Pattern: split -> model -> fit -> predict -> evaluate
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# data loading function
def load_data(path=None):
    if path is None:
        ROOT = Path(__file__).resolve().parents[2]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Dataset exists but is empty at {path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading dataset: {e}")


# Variables choice
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Encode canton as categorical codes
    df["canton_id"] = df["canton"].astype("category").cat.codes
    df["migration_lag1"] = df.groupby("canton_id")["migration_rate"].shift(1)   

    base_vars = [
        "log_rent_avg",
        "log_avg_income",
        "log_unemployment",
        "log_schockexposure",
        "CLUSTER1",
        "CLUSTER2",
        "migration_lag1",
    ]
    # check required columns before interactions
    required_before_interactions = [
        "migration_rate", "canton", "year",
        "log_rent_avg", "log_avg_income",
        "log_unemployment", "log_schockexposure",
        "CLUSTER1", "CLUSTER2"
    ]   

    missing = [c for c in required_before_interactions if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns before interactions: {missing}\n"
            f"Columns available: {list(df.columns)}"
        )

    # Interactions terms
    df["log_avg_income_x_log_rent_avg"] = df["log_avg_income"] * df["log_rent_avg"]
    df["log_unemployment_rate_x_log_avg_income"] = df["log_unemployment"] * df["log_avg_income"]
    df["log_schockexposure_x_CLUSTER1"] = df["log_schockexposure"] * df["CLUSTER1"]
    df["log_schockexposure_x_CLUSTER2"] = df["log_schockexposure"] * df["CLUSTER2"]

    interaction_vars = [
        "log_avg_income_x_log_rent_avg",
        "log_unemployment_rate_x_log_avg_income",
        "log_schockexposure_x_CLUSTER1",
        "log_schockexposure_x_CLUSTER2",
    ]

    required_cols = ["migration_rate", "canton", "year"] + base_vars + interaction_vars

    # check required columns after interactions
    missing_after = [c for c in required_cols if c not in df.columns]
    if missing_after:
        raise KeyError(
            f"Missing required columns after interactions: {missing_after}\n"
            f"Columns available: {list(df.columns)}"
        )

    df_model = df.dropna(subset=required_cols).copy()
    print(f"After initial cleaning: {len(df_model)} rows")

    # Dummies canton 
    df_model = pd.get_dummies(df_model, columns=["canton"], drop_first=True)

    # We have to ensure "year" is numeric
    df_model["year"] = pd.to_numeric(df_model["year"], errors="coerce")

    # Finales feature
    feature_cols = base_vars + interaction_vars + [c for c in df_model.columns if c.startswith("canton_")]

            # numeric conversion with error handling
    try:
        X_df = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
        y_ser = pd.to_numeric(df_model["migration_rate"], errors="coerce")
    except Exception as e:
        raise ValueError(f"Failed to convert features to numeric: {e}")

    # detect NaNs after conversion
    if X_df.isna().any().any() or y_ser.isna().any():
        raise ValueError(
            "NaN values detected after numeric conversion. "
            "Check dataset or preprocessing."
        )

    # Drop invalids lines
    mask_valid = (~X_df.isna().any(axis=1)) & (~y_ser.isna()) & (~df_model["year"].isna())
    before = len(df_model)
    X_df = X_df.loc[mask_valid]
    y_ser = y_ser.loc[mask_valid]
    df_model = df_model.loc[mask_valid].copy()
    after = len(df_model)

    print(f"Rows kept after numeric coercion: {after} (dropped {before-after})")
    return df_model, X_df, y_ser, feature_cols

# temporal train-test split based on years
def time_split(df: pd.DataFrame, feature_cols):
    df_sorted = df.sort_values(["year"]).reset_index(drop=True)
    years = df_sorted["year"].unique()
    cut = int(0.8 * len(years)) if len(years) > 1 else 1

    train_years = set(years[:cut])
    test_years = set(years[cut:]) if cut < len(years) else set()

    X_train = df_sorted.loc[df_sorted["year"].isin(train_years), feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y_train = pd.to_numeric(df_sorted.loc[df_sorted["year"].isin(train_years), "migration_rate"], errors="coerce").to_numpy()
    X_test  = df_sorted.loc[df_sorted["year"].isin(test_years),  feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y_test  = pd.to_numeric(df_sorted.loc[df_sorted["year"].isin(test_years),  "migration_rate"], errors="coerce").to_numpy()


    print("Split shapes ->",
        "X_train", X_train.shape, "X_test", X_test.shape,
        "y_train", y_train.shape, "y_test", y_test.shape)
    
    return X_train, X_test, y_train, y_test

# model creation
def run_ols(X_train, y_train, X_test, y_test):

    model = LinearRegression()

    # training the model with error handling
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"LinearRegression training failed: {e}")

    # prediction
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    #coefficients
    coefs = model.coef_
    intercept = model.intercept_

    # evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, mse, r2, coefs, intercept

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = ROOT / "data" / "databasecsv.csv"

    # 1. load
    df = load_data()

    # 2. prepare
    df_model, X_df, y_ser, feature_cols = prepare_dataframe(df)

    # 3. time split
    X_train, X_test, y_train, y_test = time_split(df_model, feature_cols)

    # 4. OLS
    model, y_pred, mse, r2, coefs, intercept = run_ols(X_train, y_train, X_test, y_test)

    # 5. printing results
    print("\n=== Sklearn OLS (LinearRegression) ===")
    print(f"MSE: {mse:.4f}")
    print(f"R^2: {r2:.4f}")

    print("\nFirst 5 predictions vs true:")
    for yp, yt in list(zip(y_pred[:5], y_test[:5])):
        print(f"pred={yp:.3f} | true={yt:.3f}")
    print("\n=== OLS COEFFICIENTS ===")
    for name, coef in zip(feature_cols, coefs):
        print(f"{name:40s}  {coef:.6f}")

    print(f"\nIntercept: {intercept:.6f}")


