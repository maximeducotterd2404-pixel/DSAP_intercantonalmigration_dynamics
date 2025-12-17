#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sklearn Ridge regression for migration_rate
Pattern: split -> scale -> model -> fit -> predict -> evaluate
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# data loading function
def load_data(path=None):
    if path is None:
        ROOT = Path(__file__).resolve().parents[3]
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


# Base variables
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Encode canton as categorical codes
    df["canton_id"] = df["canton"].astype("category").cat.codes
    df["migration_lag1"] = df.groupby("canton_id")["migration_rate"].shift(1)


    base_vars = [ 
        "Z_score_rent",
        "avg_income_zscore",
        "z-score_unemployment",
        "shockexposure_zscore",
        "CLUSTER1",
        "CLUSTER2",
        "migration_lag1",
    ]

    # Interactions
    df["avg_income_zscore_x_Z_score_rent"] = df["avg_income_zscore"] * df["Z_score_rent"]
    df["z-score_unemployment_x_avg_income_zscore"] = df["z-score_unemployment"] * df["avg_income_zscore"]
    df["schockexposure_x_CLUSTER1"] = df["shockexposure_zscore"] * df["CLUSTER1"]
    df["schockexposure_x_CLUSTER2"] = df["shockexposure_zscore"] * df["CLUSTER2"]

    interaction_vars = [
        "avg_income_zscore_x_Z_score_rent",
        "z-score_unemployment_x_avg_income_zscore",
        "schockexposure_x_CLUSTER1",
        "schockexposure_x_CLUSTER2",
    ]

    required_cols = ["migration_rate", "canton", "year"] + base_vars + interaction_vars
    
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_model= df.dropna(subset=required_cols).copy()
    print(f"After initial cleaning: {len(df_model)} rows")

    # Canton dummies 
    df_model = pd.get_dummies(df_model, columns=["canton"], drop_first=True)

    # Ensure year is numeric for chronological split
    df_model["year"] = pd.to_numeric(df_model["year"], errors="coerce")

    # Final feature list
    feature_cols = base_vars + interaction_vars + [c for c in df_model.columns if c.startswith("canton_")]

    # safe numeric conversion
    try:
       df_model[feature_cols] = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
       df_model["migration_rate"] = pd.to_numeric(df_model["migration_rate"], errors="coerce")

    except Exception as e:
        raise ValueError(f"Failed to convert features to numeric: {e}")

    # NaN check after coercion
    if df_model[feature_cols].isna().any().any() or df_model["migration_rate"].isna().any():
        raise ValueError(
            "Some features or target contain non-numeric or missing values "
        )
    return df_model

# times based train-test split
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

    return X_train, X_test, y_train, y_test

#standardized scaling
def run_ridge(X_train, y_train, X_test, y_test, alphas):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    best_alpha = None
    best_r2_test = -np.inf 
    best_model = None

    results = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)  # alpha = lambda
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test  = model.predict(X_test_scaled)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test  = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        
        results.append((alpha, mse_test, r2_train, r2_test))

        if r2_test > best_r2_test:
            best_r2_test = r2_test
            best_alpha = alpha
            best_model = model

    return best_model, best_alpha, best_r2_test, results, scaler

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = ROOT / "data" / "databasecsv.csv"

    df = load_data(DATA_PATH)
    df = prepare_dataframe(df)

    base_vars = [
        "Z_score_rent",
        "avg_income_zscore",
        "z-score_unemployment",
        "shockexposure_zscore",
        "CLUSTER1",
        "CLUSTER2",
        "migration_lag1",
    ]

    interaction_vars = [
        "avg_income_zscore_x_Z_score_rent",
        "z-score_unemployment_x_avg_income_zscore",
        "schockexposure_x_CLUSTER1",
        "schockexposure_x_CLUSTER2",
    ]

    canton_cols = [c for c in df.columns if c.startswith("canton_")]
    feature_cols = base_vars + interaction_vars + canton_cols

    X_train, X_test, y_train, y_test = time_split(df, feature_cols)

    best_model, best_alpha, best_r2_test, results, scaler = run_ridge(
    X_train, y_train, X_test, y_test,
    alphas=[0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
)

    # --- Recompute train/test predictions for the best model ---
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test  = best_model.predict(X_test_scaled)

    r2_train_best = r2_score(y_train, y_pred_train)
    r2_test_best  = r2_score(y_test, y_pred_test)

    print(f"\n=== BEST MODEL PERFORMANCE ===")
    print(f"Best λ: {best_alpha}")
    print(f"Train R²: {r2_train_best:.4f}")
    print(f"Test  R²: {r2_test_best:.4f}")
    # ---- Display Ridge coefficients ----
    print("\n=== RIDGE COEFFICIENTS (best model) ===")

    coefs = best_model.coef_
    coef_table = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs
    }).sort_values("coef", key=abs, ascending=False)

    print(coef_table.to_string(index=False))
    print("Feature columns:")
    for f in feature_cols:
        print(f)
    print(coef_table.head(50).to_string())



