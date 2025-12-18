#!/usr/bin/env python3
"""
Unified entry point for running the different modeling pipelines.
Supported models:
  - ols
  - ridge
  - randomforest
  - gradientboosting
  - decisiontree
Use --model all to run them sequentially.
"""

from pathlib import Path
import argparse
import sys

from src import data_loader as dl
from src import models as mdl
from src import evaluation as evl
# We still import the underlying modules for time-based split helpers and feature utilities
from src.ML.ols import ols as ols_module
from src.ML.ridge import ridge as ridge_module
from src.ML.randomforest import randomforest as rf_module
from src.ML.Gradientboosting import gradientboosting as gb_module
from src.ML import decisiontree as dt_module


DATA_DEFAULT = Path(__file__).resolve().parent / "data" / "databasecsv.csv"


def run_ols(data_path: Path) -> None:
    # Use the data_loader facade (wraps existing loader/prep)
    df_model, X_df, y_ser, feature_cols = dl.load_for_ols(data_path)
    X_train, X_test, y_train, y_test = ols_module.time_split(df_model, feature_cols)
    model, y_pred, mse, r2, _, _ = mdl.train_ols(X_train, y_train, X_test, y_test)
    from sklearn.metrics import r2_score
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)

    print("=== OLS ===")
    print(f"Train R2: {r2_train:.4f} | Test R2: {r2:.4f} | Test RMSE: {mse**0.5:.4f}")
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)}")


def run_ridge(data_path: Path) -> None:
    df_model = dl.load_for_ridge(data_path)

    base_vars = [
        "Z_score_rent",
        "avg_income_zscore",
        "z-score_unemployment",
        "shockexposure_zscore",
        "CLUSTER1",
        "CLUSTER2",
        "migration_lag1",
    ]
    interaction_vars = [
        "avg_income_zscore_x_Z_score_rent",
        "z-score_unemployment_x_avg_income_zscore",
        "schockexposure_x_CLUSTER1",
        "schockexposure_x_CLUSTER2",
    ]
    feature_cols = base_vars + interaction_vars + [c for c in df_model.columns if c.startswith("canton_")]

    X_train, X_test, y_train, y_test = ridge_module.time_split(df_model, feature_cols)
    best_model, best_alpha, best_r2_test, results, scaler = mdl.train_ridge(
        X_train, y_train, X_test, y_test, alphas=[0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    )

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    from sklearn.metrics import r2_score, mean_squared_error  # local import to avoid polluting namespace

    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Alpha is selected on an inner validation split inside run_ridge (no test-set leakage)
    val_row = next((row for row in results if row[0] == best_alpha), None)
    r2_val = val_row[3] if val_row is not None else float("nan")

    print("=== Ridge Regression ===")
    print(f"Best alpha (selected on validation): {best_alpha}")
    print(f"Train R2: {r2_train:.4f} | Val R2: {r2_val:.4f} | Test R2: {best_r2_test:.4f} | Test MSE: {mse_test:.4f}")
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)}")


def run_randomforest(data_path: Path) -> None:
    df, feature_cols, target_col = dl.load_for_randomforest(data_path)
    X_train, X_test, y_train, y_test = rf_module.time_split(df, feature_cols, target_col)
    rf, _, r2_test, rmse, r2_train = mdl.train_randomforest(X_train, y_train, X_test, y_test)
    feat_imp = rf_module.get_feature_importance(rf, feature_cols)[:5]
    metrics = evl.eval_randomforest(rf, X_train, y_train, X_test, y_test)

    print("=== Random Forest ===")
    print(f"Train R2: {metrics['r2_train']:.4f} | Test R2: {metrics['r2_test']:.4f} | Test RMSE: {metrics['rmse_test']:.4f}")
    print("Top feature importances (name, importance):")
    for name, score in feat_imp:
        print(f"  {name:25s} {score:.4f}")


def run_gradientboosting(data_path: Path) -> None:
    df, feature_cols, target_col = dl.load_for_gradientboosting(data_path)
    X_train, X_test, y_train, y_test = gb_module.time_split(df, feature_cols, target_col)
    model = mdl.train_gradientboosting(X_train, y_train)
    metrics = evl.eval_gradientboosting(model, X_train, y_train, X_test, y_test)

    importance = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:5]

    print("=== Gradient Boosting ===")
    print(f"Train R2: {metrics['r2_train']:.4f} | Test R2: {metrics['r2_test']:.4f} | Test RMSE: {metrics['rmse_test']:.4f}")
    print("Top feature importances (name, importance):")
    for name, score in sorted_imp:
        print(f"  {name:25s} {score:.4f}")


def run_decisiontree(data_path: Path) -> None:
    df = dl.load_for_decisiontree(data_path)
    X_train, y_train, X_test, y_test = dt_module.prepare_dataset(df)
    model = mdl.train_decisiontree(X_train, y_train)
    metrics = evl.eval_decisiontree(model, X_test, y_test)

    print("=== Decision Tree (classification on migration direction) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run migration models from a single entry point."
    )
    parser.add_argument(
        "--model",
        choices=["ols", "ridge", "randomforest", "gradientboosting", "decisiontree", "all"],
        default="ols",
        help="Which model to run (use 'all' to run every model in sequence).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DEFAULT,
        help=f"Path to the data CSV (default: {DATA_DEFAULT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data

    if not data_path.exists():
        print(f"Dataset not found at {data_path}", file=sys.stderr)
        return 1

    runners = {
        "ols": run_ols,
        "ridge": run_ridge,
        "randomforest": run_randomforest,
        "gradientboosting": run_gradientboosting,
        "decisiontree": run_decisiontree,
    }

    to_run = list(runners.keys()) if args.model == "all" else [args.model]

    for name in to_run:
        print("\n" + "=" * 80)
        print(f"Running {name}...")
        print("=" * 80)
        runners[name](data_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
