import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# load and cleand data
def load_data (path=None):
    if path is None:
        ROOT = Path(__file__).resolve().parents[2]
        path = ROOT / "data" / "databasecsv.csv"
    try:
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()

    except FileNotFoundError:
        raise FileNotFoundError(
            f"ERROR: dataset not found at {path}\n"
            "Check that databasecsv.csv is inside the /data/ folder."
        )

    except pd.errors.EmptyDataError:
        raise RuntimeError(
            f"ERROR: The file at {path} exists but is empty."
        )

    except Exception as e:
        raise RuntimeError(
            f"Unexpected error when loading dataset at {path}: {e}"
        )
    return df

# features choice
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    # Encode canton as categorical codes
    df["canton_id"] = df["canton"].astype("category").cat.codes
    df["migration_lag1"] = df.groupby("canton_id")["migration_rate"].shift(1)

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

    # verification of required columns
    cols_needed = feature_cols + [target_col]
    missing = [c for c in cols_needed if c not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Columns available in dataset: {list(df.columns)}"
        )

    df = df.dropna(subset=cols_needed).copy()
    

    # safe conversion to numeric
    try:
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").values
    except Exception as e:
        raise ValueError(f"Failed to convert data to numeric: {e}")

    if df[feature_cols].isna().any().any() or df[target_col].isna().any():
        raise ValueError("NaN values detected in features or target after numeric conversion.")

    return df, feature_cols, target_col
# temporal train-test split
def time_split(df: pd.DataFrame, feature_cols, target_col):
    df = df.sort_values("year").reset_index(drop=True)

    years = df["year"].unique()
    cut = int(0.8 * len(years))    # 80% train / 20% test

    train_years = set(years[:cut])
    test_years  = set(years[cut:])

    X_train = df.loc[df["year"].isin(train_years), feature_cols].to_numpy()
    y_train = df.loc[df["year"].isin(train_years), target_col].to_numpy()

    X_test  = df.loc[df["year"].isin(test_years),  feature_cols].to_numpy()
    y_test  = df.loc[df["year"].isin(test_years),  target_col].to_numpy()

    return X_train, X_test, y_train, y_test

def run_random_forest(X_train, y_train, X_test, y_test):
# Random Forest model
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,            
        min_samples_leaf=5,     
        max_features="sqrt",    
        random_state=0,
        n_jobs=-1
    )

    # trainig with error handling
    try:
        rf.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Random Forest training failed: {e}")

    y_train_pred = rf.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)


    # prediction on test set
    y_pred = rf.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rf, y_pred, r2, rmse, r2_train

# importation of features
def get_feature_importance(rf, feature_cols):
    importances = rf.feature_importances_
    feat_imp = sorted(
    zip(feature_cols, importances), 
    key=lambda x: x[1], 
    reverse=True
    )
    return feat_imp

if __name__ == "__main__":
    df = load_data()
    df, feature_cols, target = prepare_dataframe(df)

    X_train, X_test, y_train, y_test = time_split(df, feature_cols, target)

    rf, y_pred, r2, rmse, r2_train = run_random_forest(
        X_train, y_train, X_test, y_test
    )

    print("\n=== RANDOM FOREST RESULTS ===")
    print(f"Train R² : {r2_train:.4f}")
    print(f"Test  R² : {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    print("\n=== FEATURE IMPORTANCES (Random Forest) ===")
    feat_imp = get_feature_importance(rf, feature_cols)
    for name, imp in feat_imp:
        print(f"{name:25s} -> {imp:.4f}")
