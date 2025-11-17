import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# load and cleand data
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "databasecsv.csv"
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip() 

print("Colonnes dispo :", df.columns.tolist())

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

cols_needed = feature_cols + [target_col]
df = df.dropna(subset=cols_needed).copy()

X = df[feature_cols].values
y = df[target_col].values

# split data into train and tests sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
)

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

rf.fit(X_train, y_train)

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
