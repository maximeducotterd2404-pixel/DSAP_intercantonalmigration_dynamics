#!/usr/bin/env python3
""""
This code hasn't been presented in the report, as it was a test of decision tree modeling
in order to me to understand what it represents and how it works"
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def load_data(path=None):
    if path is None:
        # Use a repo-relative path so CLI runs are reproducible.
        ROOT = Path(__file__).resolve().parents[2]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        # Raw data uses ";" as a delimiter, so avoid mis-parsing.
        df = pd.read_csv(path, sep=";")
        # Trim whitespace to keep column name matching stable.
        df.columns = df.columns.str.strip()
        # Normalize cluster labels to keep feature names consistent.
        rename_map = {
            c: c.replace("CLUSTER ", "CLUSTER")
            for c in df.columns
            if c.startswith("CLUSTER ")
        }
        if rename_map:
            df = df.rename(columns=rename_map)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Dataset exists but is empty at {path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading dataset: {e}")


# Predictors used to classify the direction of migration change.
FEATURE_COLS = [
    "log_rent_avg",
    "log_avg_income",
    "log_unemployment",
    "log_schockexposure",
    "housing_construction_pc",
    "CLUSTER0",
    "CLUSTER1",
    "CLUSTER2",
]


# Prepare data to turn migration dynamics into a classification target.
def prepare_dataset(df):
    # Sort to compute within-canton changes reliably.
    df = df.sort_values(["canton", "year"])

    # Directional target focuses on sign of change, not magnitude.
    df["migration_rate_diff"] = df.groupby("canton")["migration_rate"].diff()
    df = df.dropna(subset=["migration_rate_diff"])
    df["y_cls"] = (df["migration_rate_diff"] > 0).astype(int)

    # Fail fast if required columns are missing.
    required_cols = ["canton", "year", "migration_rate"] + FEATURE_COLS
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(
            f"Missing required columns in dataset: {missing}\n"
            f"Columns available: {list(df.columns)}"
        )

    # Ensure time-splitting logic works as intended.
    if not np.issubdtype(df["year"].dtype, np.number):
        raise TypeError("Column 'year' must be numeric for time-based split.")

    # Time split avoids peeking into future years.
    train = df["year"] <= 2021
    test = df["year"] >= 2022

    if (train.sum() == 0) or (test.sum() == 0):
        raise ValueError(
            "Train/test split invalid: no data in either train or test set."
        )

    X_train = df.loc[train, FEATURE_COLS].to_numpy()
    y_train = df.loc[train, "y_cls"].to_numpy()

    X_test = df.loc[test, FEATURE_COLS].to_numpy()
    y_test = df.loc[test, "y_cls"].to_numpy()

    return X_train, y_train, X_test, y_test


# Decision tree offers an interpretable nonlinear classifier.
def train_decision_tree(X_train, y_train, max_depth=3):
    # Shallow depth keeps the tree readable and reduces overfitting.
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    # Fit with guardrails so failures are explicit.
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        raise ValueError(f"DecisionTree training failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during tree training: {e}")

    return model


# Evaluate directional accuracy on held-out years.


def evaluate_model(model, X_test, y_test):
    """Return accuracy of the decision tree."""
    # Accuracy is adequate for the binary direction target.
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


def main():
    # Single load for reproducible inputs.
    df = load_data()
    X_train, y_train, X_test, y_test = prepare_dataset(df)

    # Fit an interpretable classifier.
    model = train_decision_tree(X_train, y_train)

    # Report out-of-sample accuracy.
    acc = evaluate_model(model, X_test, y_test)
    print(f"Decision Tree Accuracy = {acc:.3f}")

    # Visualize the tree to support interpretation.
    plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=FEATURE_COLS,
        class_names=["baisse", "hausse"],
        filled=True,
        rounded=True,
        fontsize=12,
    )
    plt.title("Decision tree (max_depth = 3) — migration rate prediction")
    plt.show()


if __name__ == "__main__":
    main()
