import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# load and cleand data
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "databasecsv.csv"
try:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()

except FileNotFoundError:
    raise FileNotFoundError(
        f"ERROR: dataset not found at {DATA_PATH}\n"
        "Check that databasecsv.csv is inside the /data/ folder."
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

# features choice
feature_cols = [
    "log_rent_avg",
    "log_avg_income",
    "log_unemployment",
    "log_schockexposure",
    "CLUSTER0",
    "CLUSTER1",
    "CLUSTER2",
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
X = df[feature_cols].values
y = df[target_col].values

# safe conversion to numeric
try:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    y = pd.to_numeric(df[target_col], errors="coerce").values
except Exception as e:
    raise ValueError(f"Failed to convert data to numeric: {e}")

if np.isnan(X).any() or np.isnan(y).any():
    raise ValueError(
        "NaN values detected in features or target after numeric conversion.\n"
    )

# temporal train-test split

df = df.sort_values("year").reset_index(drop=True)

years = df["year"].unique()
cut = int(0.8 * len(years))    # 80% train / 20% test

train_years = set(years[:cut])
test_years  = set(years[cut:])

X_train = df.loc[df["year"].isin(train_years), feature_cols].to_numpy()
y_train = df.loc[df["year"].isin(train_years), target_col].to_numpy()

X_test  = df.loc[df["year"].isin(test_years),  feature_cols].to_numpy()
y_test  = df.loc[df["year"].isin(test_years),  target_col].to_numpy()

print(f"Train : {X_train.shape[0]} obs, Test : {X_test.shape[0]} obs")


print(f"Train : {X_train.shape[0]} obs, Test : {X_test.shape[0]} obs")

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
print(f"Random Forest — R² train = {r2_train:.4f}")

# prediction on test set
y_pred = rf.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest — RMSE = {rmse:.4f}")
print(f"Random Forest — R²   = {r2:.4f}")

# importation of features
importances = rf.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\nImportance des features (Random Forest) :")
for name, imp in feat_imp:
    print(f"  {name:20s} -> {imp:.3f}")
