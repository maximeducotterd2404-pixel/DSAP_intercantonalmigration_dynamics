import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from .radarplot import plot_cluster_radar

# Use standardized variables to cluster on structural canton profiles.
FEATURE_COLS = [
    "Z_score_rent",
    "avg_income_zscore",
    "z-score_unemployment",
    "Z-score-ownrrate",
    "Z-score-debt",
    "shockexposure_zscore",
]


# Centralize loading so clustering uses the same cleaned inputs.
def load_data(path=None):
    if path is None:
        # Use a repo-relative path so CLI runs are reproducible.
        ROOT = Path(__file__).resolve().parents[3]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        # Raw data uses ";" as a delimiter, so avoid mis-parsing.
        df = pd.read_csv(path, sep=";")
        # Trim whitespace to keep column name matching stable.
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Dataset exists but is empty at {path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading dataset: {e}")


# Prepare the matrix used for clustering.


def prepare_matrix(df, feature_cols=FEATURE_COLS):
    # Fail fast if required features are missing.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Avoid NaNs so K-means distances are well-defined.
    if "canton" in df.columns:
        df_clean = df.dropna(subset=feature_cols + ["canton"]).copy()
        df_clean["canton"] = df_clean["canton"].astype(str).str.strip()
    else:
        df_clean = df.dropna(subset=feature_cols).copy()

    # Ensure numeric types before clustering.
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    if df_clean[feature_cols].isna().any().any():
        raise ValueError("NaN detected after conversion to numeric.")

    if "canton" in df_clean.columns:
        # Use canton averages to capture structural differences over time.
        df_canton = df_clean.groupby("canton", as_index=False)[feature_cols].mean()
        X = df_canton[feature_cols].to_numpy()
        return df_canton, X

    X = df_clean[feature_cols].to_numpy()
    return df_clean, X


# Run K-means to summarize cantonal profiles.
def run_kmeans(X, k=3, random_state=0):
    # Fixed seed and n_init improve stability for small samples.
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(X)
    return model


def assign_clusters(df_clean, model):
    # Attach cluster labels for downstream interpretation and interactions.
    df_out = df_clean.copy()
    df_out["cluster"] = model.labels_
    return df_out


def main():
    # End-to-end run for reproducible cluster assignment.
    df = load_data()
    df_clean, X = prepare_matrix(df)
    model = run_kmeans(X, k=3)
    df_clustered = assign_clusters(df_clean, model)

    print("\n=== TOTAL INERTIA ===")
    print(model.inertia_)

    print("\n=== CLUSTER PROFILES ===")
    profiles = df_clustered.groupby("cluster")[FEATURE_COLS].mean().round(3)
    print(profiles)

    print("\n=== FIRST ROWS WITH CLUSTERS ===")
    cols_show = ["canton", "cluster"] + FEATURE_COLS
    print(df_clustered[cols_show].head())

    print("\n=== CLUSTER PER CANTON ===")
    canton_main = df_clustered[["canton", "cluster"]].sort_values("cluster")
    print(canton_main)

    # Plot a radar chart to interpret cluster profiles.
    print("\n=== RADAR PLOT ===")
    plot_cluster_radar(df_clustered)

    return df_clustered


if __name__ == "__main__":
    main()
