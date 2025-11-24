import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# Load and trying to prepare the data
ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "databasecsv.csv"
#file loading with error handling
try:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()

except FileNotFoundError:
    raise FileNotFoundError(
        f"ERROR: dataset not found at {DATA_PATH}\n"
    )
except pd.errors.EmptyDataError:
    raise RuntimeError(
        f"ERROR: The file at {DATA_PATH} exists but is empty."
    )
except Exception as e:
    raise RuntimeError(
        f"Unexpected error when loading dataset at {DATA_PATH}: {e}"
    )
print("Columns :", df.columns.tolist())

# vars for kmeans
feature_cols = [
    "Z_score_rent",
    "avg_income_zscore",
    "z-score_unemployment",
    "Z-score-ownrrate",
    "Z-score-debt",
    "shockexposure_zscore"
]

# drop NaNs only for these cols
df_clean = df.dropna(subset=feature_cols).copy()

# matrix for kmeans
X = df_clean[feature_cols].values

# run kmeans
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans.fit(X)

# add cluster label
df_clean["cluster"] = kmeans.labels_

# quick cluster profiles
cluster_profiles = (
    df_clean.groupby("cluster")[feature_cols]
    .mean()
    .round(3)
)

print("total inertia:", kmeans.inertia_)
print("\ncluster profiles:")
print(cluster_profiles)

print("\npreview:")
print(df_clean[["canton", "year", "cluster"] + feature_cols].head())

# main cluster per canton
canton_main_cluster = (
    df_clean.groupby("canton")["cluster"]
    .agg(lambda s: s.value_counts().idxmax())
    .reset_index()
    .sort_values("cluster")
)

print("\nmain cluster per canton:")
print(canton_main_cluster)

__all__ = ["df_clean"]
