#!/usr/bin/env python3
"""
Evaluation/visualization facade required by the rulebook.
Provides common metrics helpers and wraps existing evaluation utilities.
You can keep using your model-specific plots; this simply centralizes access.
"""

from typing import Dict, Any

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Generic regression metrics (MSE, RMSE, R2)."""
    # simple metrics in one dict
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def eval_randomforest(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Evaluate Random Forest using train/test R2 and test RMSE."""
    # predict on train and test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
    }


def eval_gradientboosting(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Wraps src.ML.Gradientboosting.gradientboosting.evaluate."""
    # call eval from GB module
    from src.ML.Gradientboosting import gradientboosting as gb_mod

    train_r2, test_r2, rmse = gb_mod.evaluate(model, X_train, y_train, X_test, y_test)
    return {"r2_train": train_r2, "r2_test": test_r2, "rmse_test": rmse}


def eval_decisiontree(model, X_test, y_test) -> Dict[str, float]:
    """Wraps src.ML.decisiontree.evaluate_model (returns accuracy)."""
    # just return accuracy
    from src.ML import decisiontree as dt_mod

    acc = dt_mod.evaluate_model(model, X_test, y_test)
    return {"accuracy": acc}


# Plot wrappers (reuse existing plotting utilities). Each simply delegates to the model-specific module.


def plot_ols_true_vs_pred(y_true, y_pred, title="OLS – True vs Predicted"):
    """Scatter plot for OLS predictions (uses src.ML.ols.plotols)."""
    # delegate to OLS plot
    from src.ML.ols import plotols

    return plotols.plot_true_vs_pred_rf(y_true, y_pred, title=title)


def plot_ridge_true_vs_pred(y_true, y_pred, title="Ridge – True vs Predicted"):
    """Scatter plot for Ridge predictions (uses src.ML.ridge.plotridge)."""
    # delegate to Ridge plot
    from src.ML.ridge import plotridge

    return plotridge.plot_true_vs_pred(y_true, y_pred, title=title)


def plot_randomforest_true_vs_pred(
    y_true, y_pred, title="Random Forest – True vs Predicted"
):
    """Scatter plot for RF predictions (uses src.ML.randomforest.plotrandomforest)."""
    # delegate to RF plot
    from src.ML.randomforest import plotrandomforest

    return plotrandomforest.plot_true_vs_pred_rf(y_true, y_pred, title=title)


def plot_gradientboosting_true_vs_pred(
    y_true, y_pred, title="Boosting – True vs Predicted"
):
    """Scatter plot for Gradient Boosting predictions (uses src.ML.Gradientboosting.plotgradientboosting)."""
    # delegate to GB plot
    from src.ML.Gradientboosting import plotgradientboosting

    return plotgradientboosting.plot_true_vs_pred_boost(y_true, y_pred, title=title)


def plot_kmeans_radar(df, title="Cluster profiles (K-means)"):
    """Radar plot for k-means clusters (uses src.ML.kmeans.radarplot)."""
    # delegate to KMeans plot
    from src.ML.kmeans import radarplot

    return radarplot.plot_cluster_radar(df, title=title)
