#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


# 1. Load and clean data to keep the pipeline reproducible and source-consistent


def load_data(path=None):
    if path is None:
        # Keep a repo-relative default so results are reproducible across machines.
        ROOT = Path(__file__).resolve().parents[3]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        # The raw file uses ";" separators, so read without implicit parsing changes.
        df = pd.read_csv(path, sep=";")
        # Strip whitespace to avoid silent mismatches after source harmonization.
        df.columns = df.columns.str.strip()
        # Normalize cluster labels to keep feature names stable across exports.
        rename_map = {
            c: c.replace("CLUSTER ", "CLUSTER")
            for c in df.columns
            if c.startswith("CLUSTER ")
        }
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"ERROR: dataset not found at {path}\n"
            "Check that databasecsv.csv is inside the /data/ folder."
        )

    except pd.errors.EmptyDataError:
        raise RuntimeError(f"ERROR: The file at {path} exists but is empty.")

    except Exception as e:
        raise RuntimeError(f"Unexpected error when loading dataset at {path}: {e}")


# 2. Feature selection and preparation to match the documented model specification


def prepare_dataframe(df: pd.DataFrame):

    # Encode cantons so fixed effects can capture unobserved regional heterogeneity.
    df["canton_id"] = df["canton"].astype("category").cat.codes

    # Lagged migration captures persistence in mobility flows.
    df["migration_lag1"] = df.groupby("canton_id")["migration_rate"].shift(1)

    # One-hot canton effects allow structural differences to inform predictions.
    FE = pd.get_dummies(df["canton_id"], prefix="FE", drop_first=True)
    df = pd.concat([df, FE], axis=1)

    # Use the report's core housing, labor, and shock features to limit overfitting.
    feature_cols = [
        "log_rent_avg",
        "log_avg_income",
        "log_unemployment",
        "log_schockexposure",
        "CLUSTER0",
        "CLUSTER1",
        "CLUSTER2",
        "migration_lag1",
    ]

    target_col = "migration_rate"

    # Fail fast if upstream preprocessing changed expected columns.
    cols_needed = feature_cols + [target_col]
    missing = [c for c in cols_needed if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Columns available in dataset: {list(df.columns)}"
        )

    # Keep a consistent sample across features and target.
    df = df.dropna(subset=cols_needed).copy()

    # Guard against mixed types in the merged CSV exports.
    try:
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").values

    except Exception as e:
        raise ValueError(f"Failed to convert data to numeric: {e}")

    # Avoid silent NaNs that would bias training or evaluation.
    if df[feature_cols].isna().any().any() or df[target_col].isna().any():
        raise ValueError(
            "NaN values detected in features or target after numeric conversion."
        )

    return df, feature_cols, target_col


# 3. Temporal train-test split to mirror real forecasting and prevent leakage


def time_split(df: pd.DataFrame, feature_cols, target_col):

    # Preserve temporal ordering so test years are strictly out-of-sample.
    df = df.sort_values("year").reset_index(drop=True)

    years = df["year"].unique()
    cut = int(0.8 * len(years))  # Keep the split close to the report's 80/20 setup.

    train_years = set(years[:cut])
    test_years = set(years[cut:])

    X_train = df.loc[df["year"].isin(train_years), feature_cols].to_numpy()
    y_train = df.loc[df["year"].isin(train_years), target_col].to_numpy()

    X_test = df.loc[df["year"].isin(test_years), feature_cols].to_numpy()
    y_test = df.loc[df["year"].isin(test_years), target_col].to_numpy()

    return X_train, X_test, y_train, y_test


# 4. Random Forest model to capture nonlinearities and interactions


def run_random_forest(X_train, y_train, X_test, y_test):

    # Light tuning keeps the model flexible without overfitting a small panel.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=0,
        n_jobs=-1,
    )

    # Fit once so results remain comparable across models.
    try:
        rf.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Random Forest training failed: {e}")

    # Track in-sample fit to diagnose potential overfitting.
    y_train_pred = rf.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)

    # Evaluate generalization on later years, consistent with the time split.
    y_pred = rf.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rf, y_pred, r2, rmse, r2_train


# 5. Feature importance extraction for interpretation


def get_feature_importance(rf, feature_cols):
    # Impurity-based importances provide a readable ranking of predictors.
    importances = rf.feature_importances_

    # Sort for quick inspection in the CLI output.
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    return feat_imp


# CLI entry point to reproduce the report's Random Forest results

if __name__ == "__main__":
    # Keep a one-command run for reproducible reporting.
    df = load_data()
    df, feature_cols, target = prepare_dataframe(df)

    X_train, X_test, y_train, y_test = time_split(df, feature_cols, target)

    rf, y_pred, r2, rmse, r2_train = run_random_forest(X_train, y_train, X_test, y_test)

    # Print a compact summary for quick checks and report tables.
    print("\n=== RANDOM FOREST RESULTS ===")
    print(f"Train R² : {r2_train:.4f}")
    print(f"Test  R² : {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    print("\n=== FEATURE IMPORTANCES (Random Forest) ===")
    feat_imp = get_feature_importance(rf, feature_cols)

    for name, imp in feat_imp:
        print(f"{name:25s} -> {imp:.4f}")
