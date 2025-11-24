#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# charge the data
ROOT = Path(__file__).resolve().parents[2]
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

# preparation of te data
df = df.sort_values(["canton", "year"])

# target variable : direction of migration_rate change
df["migration_rate_diff"] = df.groupby("canton")["migration_rate"].diff()
df = df.dropna(subset=["migration_rate_diff"])
df["y_cls"] = (df["migration_rate_diff"] > 0).astype(int)

# explicative variables
X_cols = [
    "log_rent_avg",
    "log_avg_income",
    "log_unemployment",
    "log_schockexposure",
    "housing_construction_pc",
    "CLUSTER0",
    "CLUSTER1",
    "CLUSTER2"
]

#verification of required columns
required_cols = ["canton", "year", "migration_rate"] + X_cols
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise KeyError(
        f"Missing required columns in dataset: {missing}\n"
        f"Columns available: {list(df.columns)}"
    )

X = df[X_cols]
y = df["y_cls"]

# temporal train test split
train = df["year"] <= 2021
test  = df["year"] >= 2022

# validation of the split
if not np.issubdtype(df["year"].dtype, np.number):
    raise TypeError("Column 'year' must be numeric for time-based split.")

if (train.sum() == 0) or (test.sum() == 0):
    raise ValueError(
        "Train/test split invalid: no data in either train or test set."
    )


X_train, y_train = X[train], y[train]
X_test, y_test   = X[test], y[test]

# decision tree model
tree = DecisionTreeClassifier(max_depth=3, random_state=0)

# training with error handling
try:
    tree.fit(X_train, y_train)
except ValueError as e:
    raise ValueError(f"DecisionTree training failed: {e}")
except Exception as e:
    raise RuntimeError(f"Unexpected error during tree training: {e}")


# graphical plot
plt.figure(figsize=(16, 10))
plot_tree(
    tree,
    feature_names=X_cols,
    class_names=["baisse", "hausse"],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Arbre de décision (max_depth = 3) — Prévision direction migration")
plt.show()

