import pytest
import pandas as pd
import numpy as np

from pathlib import Path

# Import the functions from your module
from src.ML.kmeans.kmeans import (
    load_data,
    prepare_matrix,
    run_kmeans,
    assign_clusters,
    FEATURE_COLS,
)

# Test kmeans.py


def test_load_data_success(tmp_path):
    """load_data should load a valid CSV file. Why: confirms the loader
    handles the expected delimiter and returns a DataFrame."""
    # Create fake csv
    file = tmp_path / "fake.csv"
    df_test = pd.DataFrame({"A": [1, 2]})
    df_test.to_csv(file, sep=";", index=False)

    df_loaded = load_data(path=file)
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (2, 1)


def test_load_data_missing_file():
    """load_data should raise an error if the file does not exist. Why:
    data access failures should be explicit and actionable."""
    with pytest.raises(FileNotFoundError):
        load_data(path="THIS_FILE_DOES_NOT_EXIST.csv")


# test matrix preparation
def test_prepare_matrix_valid():
    """prepare_matrix should return df_clean and X as a numpy matrix. Why:
    clustering relies on a consistent numeric feature matrix."""
    df = pd.DataFrame(
        {
            "Z_score_rent": [0.1, 0.2],
            "avg_income_zscore": [1.0, 0.5],
            "z-score_unemployment": [-0.1, 0.0],
            "Z-score-ownrrate": [0.3, 0.4],
            "Z-score-debt": [1.2, 1.0],
            "shockexposure_zscore": [0.7, 0.8],
        }
    )

    df_clean, X = prepare_matrix(df, FEATURE_COLS)

    assert isinstance(df_clean, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, len(FEATURE_COLS))


def test_prepare_matrix_missing_columns():
    """prepare_matrix should raise an error when columns are missing. Why:
    prevents silent dropping of required features."""
    df = pd.DataFrame({"wrong_col": [1, 2, 3]})

    with pytest.raises(KeyError):
        prepare_matrix(df, FEATURE_COLS)


# test run_kmeans


def test_run_kmeans_basic():
    """run_kmeans should fit a model with k clusters. Why: confirms the
    wrapper returns a fitted model and respects k."""
    # simple dataset
    X = np.array([[0], [1], [2], [10], [11], [12]])

    model = run_kmeans(X, k=2, random_state=0)

    assert hasattr(model, "labels_")
    assert len(model.labels_) == X.shape[0]
    assert model.n_clusters == 2


# test assign_cluster


def test_assign_clusters():
    """assign_clusters should add a cluster column. Why: downstream analysis
    expects a labeled DataFrame for grouping and plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    X = np.array([[0], [1], [2]])

    model = run_kmeans(X, k=3, random_state=0)
    df_clustered = assign_clusters(df, model)

    assert "cluster" in df_clustered.columns
    assert len(df_clustered) == 3
    assert set(df_clustered["cluster"].unique()).issubset({0, 1, 2})
