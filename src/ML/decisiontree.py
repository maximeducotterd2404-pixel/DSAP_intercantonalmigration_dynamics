#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def load_data(path=None):
    if path is None:
        # get default path when not given
        ROOT = Path(__file__).resolve().parents[2]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        # read csv with ; separator
        df = pd.read_csv(path, sep=";")
        # clean column names
        df.columns = df.columns.str.strip()
        # fix cluster column names if they have spaces
        rename_map = {c: c.replace("CLUSTER ", "CLUSTER") for c in df.columns if c.startswith("CLUSTER ")}
        if rename_map:
            df = df.rename(columns=rename_map)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Dataset exists but is empty at {path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading dataset: {e}")


# explicative variables (features for the classifier)
FEATURE_COLS = [
    "log_rent_avg",
    "log_avg_income",
    "log_unemployment",
    "log_schockexposure",
    "housing_construction_pc",
    "CLUSTER0",
    "CLUSTER1",
    "CLUSTER2"
]

# preparation of the data
def prepare_dataset(df):
    # sort by canton and year to make diffs correct
    df = df.sort_values(["canton", "year"])

    # target variable: direction of migration_rate change
    df["migration_rate_diff"] = df.groupby("canton")["migration_rate"].diff()
    df = df.dropna(subset=["migration_rate_diff"])
    df["y_cls"] = (df["migration_rate_diff"] > 0).astype(int)

    # verification of required columns
    required_cols = ["canton", "year", "migration_rate"] + FEATURE_COLS
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns in dataset: {missing}\n"
            f"Columns available: {list(df.columns)}"
        )
    
    # validation of the split
    if not np.issubdtype(df["year"].dtype, np.number):
        raise TypeError("Column 'year' must be numeric for time-based split.")
    
    # temporal train test split
    train = df["year"] <= 2021
    test  = df["year"] >= 2022


    if (train.sum() == 0) or (test.sum() == 0):
        raise ValueError(
            "Train/test split invalid: no data in either train or test set."
        )

    X_train = df.loc[train, FEATURE_COLS].to_numpy()
    y_train = df.loc[train, "y_cls"].to_numpy()

    X_test = df.loc[test, FEATURE_COLS].to_numpy()
    y_test = df.loc[test, "y_cls"].to_numpy()

    return X_train, y_train, X_test, y_test

# decision tree model
def train_decision_tree(X_train, y_train, max_depth=3):
    # shallow tree to avoid big overfitting
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    # training with error handling
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        raise ValueError(f"DecisionTree training failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during tree training: {e}")
    
    return model

# prediction and evaluation

def evaluate_model(model, X_test, y_test):
    """Return accuracy of the decision tree."""
    # just do predictions and compare
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

def main():
    # load data
    df = load_data()
    X_train, y_train, X_test, y_test = prepare_dataset(df)

    # train model
    model = train_decision_tree(X_train, y_train)

    # evaluation
    acc = evaluate_model(model, X_test, y_test)
    print(f"Decision Tree Accuracy = {acc:.3f}")

    # graphical plot
    plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=FEATURE_COLS,
        class_names=["baisse", "hausse"],
        filled=True,
        rounded=True,
        fontsize=12
    )
    plt.title("Decision tree (max_depth = 3) — migration rate prediction")
    plt.show()


if __name__ == "__main__":
    main()
