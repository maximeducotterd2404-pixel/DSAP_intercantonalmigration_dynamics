#!/usr/bin/env python3
"""
Models facade required by the rulebook.
These thin wrappers simply re-expose your existing train/run functions.
No changes to your current model implementations are needed.
"""

from typing import Any, Iterable


def train_ols(X_train, y_train, X_test, y_test):
    """Wraps src.ML.ols.ols.run_ols."""
    # call into OLS module
    from src.ML.ols import ols as ols_mod

    return ols_mod.run_ols(X_train, y_train, X_test, y_test)


def train_ridge(X_train, y_train, X_test, y_test, alphas: Iterable[float]):
    """Wraps src.ML.ridge.ridge.run_ridge."""
    # call into Ridge module
    from src.ML.ridge import ridge as ridge_mod

    return ridge_mod.run_ridge(X_train, y_train, X_test, y_test, alphas)


def train_randomforest(X_train, y_train, X_test, y_test):
    """Wraps src.ML.randomforest.randomforest.run_random_forest."""
    # call into RF module
    from src.ML.randomforest import randomforest as rf_mod

    return rf_mod.run_random_forest(X_train, y_train, X_test, y_test)


def train_gradientboosting(X_train, y_train):
    """Wraps src.ML.Gradientboosting.gradientboosting.train_boosting."""
    # call into GB module
    from src.ML.Gradientboosting import gradientboosting as gb_mod

    return gb_mod.train_boosting(X_train, y_train)


def train_decisiontree(X_train, y_train, max_depth: int = 3):
    """Wraps src.ML.decisiontree.train_decision_tree."""
    # call into decision tree module
    from src.ML import decisiontree as dt_mod

    return dt_mod.train_decision_tree(X_train, y_train, max_depth=max_depth)
