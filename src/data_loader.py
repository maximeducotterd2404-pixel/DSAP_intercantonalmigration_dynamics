#!/usr/bin/env python3
"""
Data loading / preprocessing facade required by the rulebook.
Each function is a thin wrapper around the existing module-specific loaders.
This keeps your current pipelines intact while providing the expected entry points.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd

# Canonical data path used across all wrappers
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "databasecsv.csv"


def load_raw(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    """Base loader (reuses OLS loader)."""
    # just call the OLS loader to read csv
    from src.ML.ols import ols as ols_mod
    return ols_mod.load_data(path)


def load_for_ols(path: Path = DEFAULT_DATA):
    """Loader + preprocessing for the OLS pipeline."""
    # load and prep for OLS
    from src.ML.ols import ols as ols_mod
    df = ols_mod.load_data(path)
    return ols_mod.prepare_dataframe(df)


def load_for_ridge(path: Path = DEFAULT_DATA):
    """Loader + preprocessing for the Ridge pipeline."""
    # load and prep for Ridge
    from src.ML.ridge import ridge as ridge_mod
    df = ridge_mod.load_data(path)
    return ridge_mod.prepare_dataframe(df)


def load_for_randomforest(path: Path = DEFAULT_DATA):
    """Loader + preprocessing for the Random Forest pipeline."""
    # load and prep for RF
    from src.ML.randomforest import randomforest as rf_mod
    df = rf_mod.load_data(path)
    return rf_mod.prepare_dataframe(df)


def load_for_gradientboosting(path: Path = DEFAULT_DATA):
    """Loader + preprocessing for the Gradient Boosting pipeline."""
    # load and prep for GB
    from src.ML.Gradientboosting import gradientboosting as gb_mod
    df = gb_mod.load_data(path)
    df = gb_mod.engineer_features(df)
    return gb_mod.prepare_dataframe(df)


def load_for_decisiontree(path: Path = DEFAULT_DATA):
    """Loader for the Decision Tree classifier (preprocessing is inside the module)."""
    # decision tree has its own prep
    from src.ML import decisiontree as dt_mod
    return dt_mod.load_data(path)
