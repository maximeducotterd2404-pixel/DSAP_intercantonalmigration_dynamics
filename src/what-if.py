#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHAT-IF SCENARIO TOOL
Compatible with the boosting script
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


# =====================================================================
# 1. LOAD DATA 
# =====================================================================

def load_data(path=None):
    if path is None:
        ROOT = Path(__file__).resolve().parents[1]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


# =====================================================================
# 2. FEATURE ENGINEERING 
# =====================================================================

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


# =====================================================================
# 3. PREPARE DATAFRAME 
# =====================================================================

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


# =====================================================================
# 4. TIME SPLIT (same as boosting script)
# =====================================================================

def time_split(df, feature_cols, target_col):
    df = df.sort_values("year")

    years = df["year"].unique()
    cut = int(0.8 * len(years))

    train_years = years[:cut]
    test_years = years[cut:]

    df_train = df[df["year"].isin(train_years)]
    df_test = df[df["year"].isin(test_years)]

    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train[target_col].to_numpy()

    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test[target_col].to_numpy()

    return X_train, X_test, y_train, y_test


# =====================================================================
# 5. TRAIN BOOSTING 
# =====================================================================

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


# =====================================================================
# 6. WHAT-IF INTERACTIVE FUNCTION
# =====================================================================

def interactive_what_if(df_original, df_model, feature_cols, model):
    """
    Interactive mode to build what-if scenarios
    """
    print("\n" + "="*70)
    print("  WHAT-IF SCENARIO TOOL")
    print("="*70 + "\n")
    
    # --- 1) Select canton ---
    # Filter NaN and convert to list
    cantons = sorted([c for c in df_original["canton"].unique() if pd.notna(c)])
    print("Available cantons:")
    for i, c in enumerate(cantons, 1):
        print(f"  {i:2d}. {c}")
    
    canton_choice = input("\nSelect a canton (exact name): ").strip()
    if canton_choice not in cantons:
        raise ValueError(f"Invalid canton '{canton_choice}'.")
    
    df_canton = df_model[df_model["canton"] == canton_choice].sort_values("year")
    
    # --- 2) Select year ---
    years_available = sorted(df_canton["year"].unique())
    print(f"\nAvailable years for {canton_choice}: {int(years_available[0])} - {int(years_available[-1])}")
    
    year_str = input("Scenario year (e.g., 2025): ").strip()
    year = int(year_str)
    
    # Check that t-1 exists for lags/diffs
    if year - 1 not in df_canton["year"].values:
        raise ValueError(f"No data for {year-1}. Cannot compute lags/diffs.")
    
    row_prev = df_canton[df_canton["year"] == year - 1].iloc[0]
    
    # --- 3) Input variables ---
    print(f"\n{'='*70}")
    print(f"  INPUT VALUES FOR {canton_choice} IN {year}")
    print(f"{'='*70}\n")
    print("(Press ENTER to keep the default value)\n")
    print("⚠️  IMPORTANT: Enter normal values (not logs)\n")
    
    # Retrieve normal (non-log) values from previous year
    prev_rent = np.exp(row_prev["log_rent_avg"]) - 1
    prev_income = np.exp(row_prev["log_avg_income"]) - 1
    prev_unemp = np.exp(row_prev["log_unemployment"]) - 1
    prev_shock = np.exp(row_prev["log_schockexposure"]) - 1
    
    def ask_value(var_name, default_val):
        prompt = f"{var_name:25s} (default={default_val:8.2f}) : "
        user_input = input(prompt).strip()
        return default_val if user_input == "" else float(user_input)
    
    # Ask for normal values (except shockexposure, kept constant)
    rent_t = ask_value("rent_avg", prev_rent)
    income_t = ask_value("avg_income", prev_income)
    unemp_t = ask_value("unemployment", prev_unemp)
    
    # shockexposure remains constant (keep previous)
    shock_t = prev_shock
    
    print(f"\nℹ️  shockexposure (constant): {shock_t:.2f}")
    
    # Compute logs automatically
    log_rent_t = np.log(rent_t + 1)
    log_inc_t = np.log(income_t + 1)
    log_un_t = np.log(unemp_t + 1)
    log_sh_t = np.log(shock_t + 1)
    
    # --- 4) Compute derived features ---
    
    # LAG
    migration_lag1 = row_prev["migration_rate"]
    
    # DIFFS
    d_log_rent = log_rent_t - row_prev["log_rent_avg"]
    d_log_inc = log_inc_t - row_prev["log_avg_income"]
    d_log_un = log_un_t - row_prev["log_unemployment"]
    d_log_sh = log_sh_t - row_prev["log_schockexposure"]
    
    # TREND
    t_value = year - df_original["year"].min()
    
    # --- 5) Build scenario DataFrame ---
    scenario = pd.DataFrame([{col: 0.0 for col in feature_cols}])
    
    # Base variables
    scenario["log_rent_avg"] = log_rent_t
    scenario["log_avg_income"] = log_inc_t
    scenario["log_unemployment"] = log_un_t
    scenario["log_schockexposure"] = log_sh_t
    
    # Lag
    scenario["migration_lag1"] = migration_lag1
    
    # Differences
    scenario["d_log_rent_avg"] = d_log_rent
    scenario["d_log_avg_income"] = d_log_inc
    scenario["d_log_unemployment"] = d_log_un
    scenario["d_log_schockexposure"] = d_log_sh
    
    # Trend
    scenario["t"] = t_value
    
    # Fixed Effects (FE)
    canton_id = int(row_prev["canton_id"])
    for col in feature_cols:
        if col.startswith("FE_"):
            fe_id = int(col.split("_")[1])
            scenario[col] = 1.0 if fe_id == canton_id else 0.0
    
    # --- 6) Prediction ---
    X_scenario = scenario[feature_cols].to_numpy()
    y_pred = model.predict(X_scenario)[0]
    
    # --- 7) Display results ---
    print(f"\n{'='*70}")
    print("  PREDICTION RESULTS")
    print(f"{'='*70}\n")
    print(f"Canton               : {canton_choice}")
    print(f"Year                 : {year}")
    print(f"Predicted migration  : {y_pred:.5f}")
    
    print(f"\nEntered values (normal scale):")
    print(f"  - rent_avg              : {rent_t:.2f}")
    print(f"  - avg_income            : {income_t:.2f}")
    print(f"  - unemployment          : {unemp_t:.2f}")
    print(f"  - shockexposure         : {shock_t:.2f}")
    
    print(f"\nTransformed values (log):")
    print(f"  - log_rent_avg          : {log_rent_t:.5f}")
    print(f"  - log_avg_income        : {log_inc_t:.5f}")
    print(f"  - log_unemployment      : {log_un_t:.5f}")
    print(f"  - log_schockexposure    : {log_sh_t:.5f}")
    
    print(f"\nFeatures computed automatically:")
    print(f"  - migration_lag1        : {migration_lag1:.5f}")
    print(f"  - d_log_rent_avg        : {d_log_rent:+.5f}")
    print(f"  - d_log_avg_income      : {d_log_inc:+.5f}")
    print(f"  - d_log_unemployment    : {d_log_un:+.5f}")
    print(f"  - d_log_schockexposure  : {d_log_sh:+.5f}")
    print(f"  - t (trend)             : {t_value}")
    print(f"  - canton_id (FE)        : {canton_id}")
    print("="*70 + "\n")
    
    return scenario, y_pred


# =====================================================================
# 7. MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("  LOADING AND TRAINING MODEL")
    print("="*70 + "\n")
    
    # Load
    print("=== LOADING DATA ===")
    df = load_data()
    
    # Remove NaN cantons
    df = df[df["canton"].notna()].copy()
    
    print(f"✓ Dataset loaded : {df.shape}")
    
    # Engineer
    print("\n=== ENGINEERING FEATURES (LAGS, DIFFS, FE) ===")
    df = engineer_features(df)
    print(f"✓ Features engineered : {df.shape}")
    
    # Prepare
    df_clean, feature_cols, target_col = prepare_dataframe(df)
    print(f"✓ Cleaned dataset : {df_clean.shape}")
    print(f"✓ Number of features : {len(feature_cols)}")
    
    # Split
    print("\n=== TIME SPLIT ===")
    X_train, X_test, y_train, y_test = time_split(df_clean, feature_cols, target_col)
    print(f"✓ Train: {X_train.shape} | Test: {X_test.shape}")
    
    # Train
    print("\n=== TRAINING BOOSTING ===")
    model = train_boosting(X_train, y_train)
    
    # Evaluate
    print("\n=== RESULTS ===")
    y_pred_test = model.predict(X_test)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Train R² : {train_r2:.4f}")
    print(f"Test  R² : {test_r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Launch What-If
    try:
        scenario, y_pred = interactive_what_if(df, df_clean, feature_cols, model)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
