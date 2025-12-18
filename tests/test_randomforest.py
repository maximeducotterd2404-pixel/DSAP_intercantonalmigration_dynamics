import pytest
import pandas as pd
import numpy as np

from pathlib import Path

# Import your module
from src.ML.randomforest.randomforest import (
    load_data,
    prepare_dataframe,
    time_split,
    run_random_forest,
    get_feature_importance
)

# 1. load_data
def test_load_data_success(tmp_path):
    """load_data should load a CSV successfully."""
    fake_file = tmp_path / "fake.csv"
    df_test = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_test.to_csv(fake_file, sep=";", index=False)

    df_loaded = load_data(path=fake_file)

    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (2, 2)


def test_load_data_missing():
    """load_data should raise FileNotFoundError when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_data(path="this_file_does_not_exist.csv")


# 2. prepare_dataframe

def test_prepare_dataframe_valid():
    """prepare_dataframe should return cleaned df + feature list + target name."""
    df = pd.DataFrame({
        "canton": ["VD", "VD", "VD"],
        "year": [2010, 2011, 2012],
        "log_rent_avg": [1, 2, 3],
        "log_avg_income": [4, 5, 6],
        "log_unemployment": [0.1, 0.2, 0.3],
        "log_schockexposure": [0.5, 0.6, 0.7],
        "CLUSTER0": [1, 0, 1],
        "CLUSTER1": [0, 1, 0],
        "CLUSTER2": [0, 0, 1],
        "migration_rate": [10, 20, 30],
    })

    df_clean, feature_cols, target = prepare_dataframe(df)

    assert isinstance(df_clean, pd.DataFrame)
    # migration_lag1 is undefined for the first year -> 2 rows kept
    assert len(df_clean) == 2
    assert target == "migration_rate"
    assert set(feature_cols) == {
        "log_rent_avg", "log_avg_income", "log_unemployment",
        "log_schockexposure", "CLUSTER0", "CLUSTER1", "CLUSTER2", "migration_lag1"
    }


def test_prepare_dataframe_missing_cols():
    """prepare_dataframe should raise KeyError when required columns are missing."""
    df = pd.DataFrame({"wrongcol": [1, 2]})

    with pytest.raises(KeyError):
        prepare_dataframe(df)



# 3. time_split

def test_time_split_basic():
    """time_split should return four numpy arrays with correct shapes."""
    df = pd.DataFrame({
        "canton": ["VD", "VD", "VD", "VD"],
        "log_rent_avg": [1, 2, 3, 4],
        "log_avg_income": [5, 6, 7, 8],
        "log_unemployment": [0.1, 0.2, 0.3, 0.4],
        "log_schockexposure": [0.5, 0.6, 0.7, 0.8],
        "CLUSTER0": [1, 0, 1, 1],
        "CLUSTER1": [0, 1, 0, 0],
        "CLUSTER2": [0, 0, 1, 0],
        "migration_rate": [10, 20, 30, 40],
        "year": [2010, 2011, 2012, 2013]
    })

    df_clean, feature_cols, target = prepare_dataframe(df)
    X_train, X_test, y_train, y_test = time_split(df_clean, feature_cols, target)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert X_train.shape[1] == len(feature_cols)
    assert y_train.ndim == 1



# 4. Random Forest

def test_run_random_forest():
    """run_random_forest should fit a model and return predictions + scores."""
    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 3)
    y_test = np.random.rand(5)

    rf, y_pred, r2, rmse, r2_train = run_random_forest(
        X_train, y_train, X_test, y_test
    )

    assert hasattr(rf, "predict")
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape[0] == X_test.shape[0]
    assert isinstance(r2, float)
    assert isinstance(rmse, float)


# 5. Feature importances


def test_feature_importance():
    """get_feature_importance should return non-empty sorted list."""
    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    # Use a plain RandomForestRegressor for this test
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X, y)

    feature_cols = ["a", "b", "c"]
    feat_imp = get_feature_importance(rf, feature_cols)

    assert isinstance(feat_imp, list)
    assert len(feat_imp) == 3
    assert isinstance(feat_imp[0][0], str)
    assert isinstance(feat_imp[0][1], float)
