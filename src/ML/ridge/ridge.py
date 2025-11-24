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

# Load and trying to prepare the data
ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "databasecsv.csv"
#file loading with error handling
try:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()

except FileNotFoundError:
    raise FileNotFoundError(
        f"ERROR: dataset not found at {DATA_PATH}\n"
    )
except pd.errors.EmptyDataError:
    raise RuntimeError(
        f"ERROR: The file at {DATA_PATH} exists but is empty."
    )
except Exception as e:
    raise RuntimeError(
        f"Unexpected error when loading dataset at {DATA_PATH}: {e}"
    )
print("Columns :", df.columns.tolist())

# Base variables
base_vars = [ 
    "Z_score_rent",
    "avg_income_zscore",
    "z-score_unemployment",
    "shockexposure_zscore",
    "CLUSTER1",
    "CLUSTER2",
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

df_model = df.dropna(subset=required_cols).copy()
print(f"After initial cleaning: {len(df_model)} rows")

# Canton dummies 
df_model = pd.get_dummies(df_model, columns=["canton"], drop_first=True)

# Ensure year is numeric for chronological split
df_model["year"] = pd.to_numeric(df_model["year"], errors="coerce")

# Final feature list
feature_cols = base_vars + interaction_vars + [c for c in df_model.columns if c.startswith("canton_")]

# safe numeric conversion
try:
    X_df = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_ser = pd.to_numeric(df_model["migration_rate"], errors="coerce")
except Exception as e:
    raise ValueError(f"Failed to convert features to numeric: {e}")

# NaN check after coercion
if X_df.isna().any().any() or y_ser.isna().any():
    raise ValueError(
        "Some features or target contain non-numeric or missing values "
    )

# Drop invalid rows
mask_valid = (~X_df.isna().any(axis=1)) & (~y_ser.isna()) & (~df_model["year"].isna())
before = len(df_model)
X_df = X_df.loc[mask_valid]
y_ser = y_ser.loc[mask_valid]
df_model = df_model.loc[mask_valid].copy()
after = len(df_model)
print(f"Rows kept after numeric coercion: {after} (dropped {before-after})")

# times based train-test split
df_sorted = df_model.sort_values(["year"]).reset_index(drop=True)
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

#standardized scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# loop over different lambdas
lambdas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

print("\n=== Ridge results (time-based test) ===")
best_alpha = None
best_r2 = -1e9
results = []

for alpha in lambdas:
    model = Ridge(alpha=alpha)  # alpha = lambda
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((alpha, mse, r2))
    print(f"lambda={alpha:>7.4f}  MSE={mse:8.4f}  R^2={r2:8.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_alpha = alpha

print(f"\nBest lambda based on test R^2: {best_alpha} (R^2 = {best_r2:.4f})")

# refit best model and show some predictions
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

print("\nFirst 5 predictions vs true (best Ridge):")
for yp, yt in list(zip(y_pred_best[:5], y_test[:5])):
    print(f"pred={yp:.3f} | true={yt:.3f}")
