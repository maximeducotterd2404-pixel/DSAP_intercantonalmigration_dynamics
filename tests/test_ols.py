import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import your module
from src.ML.ols import (
    load_data,
    prepare_dataframe,
    time_split,
    run_ols
)

# 1. Test load_data()

def test_load_data_success(tmp_path):
    """load_data should load a valid CSV file."""
    fake_file = tmp_path / "fake.csv"
    df_test = pd.DataFrame({"migration_rate": [1, 2], "year": [2014, 2015], "canton": ["VD", "GE"]})
    df_test.to_csv(fake_file, sep=";", index=False)

    df_loaded = load_data(path=fake_file)

    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == df_test.shape


def test_load_data_missing_file():
    """load_data should raise FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        load_data(path="DOES_NOT_EXIST.csv")

# 2. Test prepare_dataframe()

def test_prepare_dataframe_valid():
    """prepare_dataframe should return cleaned df + X + y + feature list."""

    df = pd.DataFrame({
        "migration_rate": [0.1, 0.2, 0.3],
        "canton": ["VD", "GE", "VS"],
        "year": [2014, 2015, 2016],
        "log_rent_avg": [1, 2, 3],
        "log_avg_income": [10, 20, 30],
        "log_unemployment": [0.5, 0.6, 0.7],
        "log_schockexposure": [1.1, 1.2, 1.3],
        "CLUSTER1": [1, 0, 0],
        "CLUSTER2": [0, 1, 0],
    })

    df_model, X_df, y_ser, feature_cols = prepare_dataframe(df)

    assert isinstance(df_model, pd.DataFrame)
    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_ser, pd.Series)
    assert isinstance(feature_cols, list)

    assert len(df_model) == 3   
    assert X_df.shape[0] == 3
    assert y_ser.shape[0] == 3
    assert len(feature_cols) > 6


def test_prepare_dataframe_missing_columns():
    """prepare_dataframe should raise KeyError when columns are missing."""
    df = pd.DataFrame({"wrong": [1, 2, 3]})

    with pytest.raises(KeyError):
        prepare_dataframe(df)


# 3. Test time_split()

def test_time_split_basic():
    """time_split should output correct shapes."""

    df = pd.DataFrame({
        "year": [2014, 2015, 2016, 2017],
        "migration_rate": [1, 2, 3, 4],
        "feat1": [10, 20, 30, 40],
        "feat2": [5, 6, 7, 8],
    })

    feature_cols = ["feat1", "feat2"]

    X_train, X_test, y_train, y_test = time_split(df, feature_cols)

    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert y_train.ndim == 1
    assert y_test.ndim == 1

    # 80% of years = first 3 years = train
    assert len(y_train) == 3
    assert len(y_test) == 1



# 4. Test run_ols()
def test_run_ols_basic():
    """run_ols should fit and return predictions, mse, r2, coefs, intercept."""

    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])
    X_test  = np.array([[4], [5]])
    y_test  = np.array([4, 5])

    model, y_pred, mse, r2, coefs, intercept = run_ols(X_train, y_train, X_test, y_test)

    assert y_pred.shape == (2,)
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert isinstance(coefs, np.ndarray)
    assert isinstance(intercept, float)
    assert r2 > 0.99
