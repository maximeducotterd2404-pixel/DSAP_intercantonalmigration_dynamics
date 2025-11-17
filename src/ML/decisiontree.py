#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# charge the data
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "databasecsv.csv"
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()

# target variable : direction of migration_rate change
df = df.sort_values(["canton", "year"])
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

X = df[X_cols]
y = df["y_cls"]

# temporail train test split
train = df["year"] <= 2021
test  = df["year"] >= 2022

X_train, y_train = X[train], y[train]
X_test, y_test   = X[test], y[test]

# decision tree model
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

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

