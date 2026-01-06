import pytest
import pandas as pd
import numpy as np

from pathlib import Path

from src.ML.ridge.ridge import (
    load_data,
    prepare_dataframe,
    time_split,
    run_ridge,
)

# 1. Test load_data

def test_load_data_success(tmp_path):
    """load_data should load a valid CSV."""
    file = tmp_path / "fake.csv"
    df_test = pd.DataFrame({"A": [1, 2]})
    df_test.to_csv(file, sep=";", index=False)

    df_loaded = load_data(path=file)
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (2, 1)


def test_load_data_missing_file():
    """load_data should raise FileNotFoundError when the file is missing."""
    with pytest.raises(FileNotFoundError):
        load_data("FILE_DOES_NOT_EXIST.csv")



# 2. Test prepare_dataframe


def test_prepare_dataframe_valid():
    """prepare_dataframe should return a clean df with all features."""
    
    df = pd.DataFrame({
        # Need at least 2 years per canton to compute migration_lag1
        "migration_rate": [0.1, 0.2, 0.15, 0.25],
        "canton": ["VD", "VD", "GE", "GE"],
        "year": [2015, 2016, 2015, 2016],
        "Z_score_rent": [1, 2, 1, 2],
        "avg_income_zscore": [0.5, 0.7, 0.4, 0.6],
        "z-score_unemployment": [0.1, -0.2, 0.2, -0.1],
        "shockexposure_zscore": [0.3, 0.4, 0.35, 0.45],
        "CLUSTER1": [0, 0, 1, 1],
        "CLUSTER2": [1, 1, 0, 0],
    })

    df_prepared = prepare_dataframe(df)

    # Checks
    assert isinstance(df_prepared, pd.DataFrame)
    assert "canton_GE" in df_prepared.columns or "canton_VD" in df_prepared.columns
    # With migration_lag1, one first-year observation is dropped per canton -> 2 rows kept
    assert df_prepared.shape[0] == 2  

def test_prepare_dataframe_missing_cols():
    """prepare_dataframe should raise KeyError when columns are missing."""
    df = pd.DataFrame({"irrelevant": [1, 2]})
    with pytest.raises(KeyError):
        prepare_dataframe(df)



# 3. Test time_split


def test_time_split_basic():
    """time_split should return 4 well-separated numpy arrays."""
    
    df = pd.DataFrame({
        "migration_rate": [1,2,3,4],
        "year": [2010, 2011, 2012, 2013],
        "f1": [10,11,12,13],
        "f2": [20,21,22,23],
    })

    X_train, X_test, y_train, y_test = time_split(df, feature_cols=["f1", "f2"])

    # Checks
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert len(X_train) > 0
    assert len(y_test) > 0


# 4. Test run_ridge


def test_run_ridge_basic():
    """run_ridge should fit a Ridge model and return consistent results."""
    
    X_train = np.array([[0],[1],[2]])
    y_train = np.array([0,1,2])
    X_test  = np.array([[1],[2]])
    y_test  = np.array([1,2])

    model, best_alpha, best_r2, results, scaler = run_ridge(
        X_train, y_train, X_test, y_test,
        alphas=[0, 0.1, 1.0]
    )

    assert hasattr(model, "coef_")
    assert isinstance(best_alpha, (float, int))
    assert isinstance(results, list)
    assert len(results) == 3
    assert best_r2 <= 1 and best_r2 >= -10
    assert hasattr(scaler, "transform")
