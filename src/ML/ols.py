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

#charge and clean data
DATA_PATH = Path("/Users/maximeducotterd/Desktop/finalcapstone/data/databasecsv.csv")
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()
print(f"Loaded {len(df)} rows")

# Variables de base (comme dans essairegression.py)
base_vars = [
    "log_rent_avg",
    "log_avg_income",
    "log_unemployment",
    "log_schockexposure",
    "CLUSTER1",
    "CLUSTER2",
]

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
df_model = df.dropna(subset=required_cols).copy()
print(f"After initial cleaning: {len(df_model)} rows")

# Dummies canton 
df_model = pd.get_dummies(df_model, columns=["canton"], drop_first=True)

# We have to ensure "year" is numeric
df_model["year"] = pd.to_numeric(df_model["year"], errors="coerce")

# Finales feature
feature_cols = base_vars + interaction_vars + [c for c in df_model.columns if c.startswith("canton_")]

# Conversion in numeric
X_df = df_model[feature_cols].apply(pd.to_numeric, errors="coerce")
y_ser = pd.to_numeric(df_model["migration_rate"], errors="coerce")

# Drop invalids lines
mask_valid = (~X_df.isna().any(axis=1)) & (~y_ser.isna()) & (~df_model["year"].isna())
before = len(df_model)
X_df = X_df.loc[mask_valid]
y_ser = y_ser.loc[mask_valid]
df_model = df_model.loc[mask_valid].copy()
after = len(df_model)

print(f"Rows kept after numeric coercion: {after} (dropped {before-after})")

# temporal train-test split based on years
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

# model creation
model = LinearRegression()

# training the model
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Sklearn OLS (LinearRegression) ===")
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# Aperçu debug
print("\nFirst 5 predictions vs true:")
for yp, yt in list(zip(y_pred[:5], y_test[:5])):
    print(f"pred={yp:.3f} | true={yt:.3f}")
