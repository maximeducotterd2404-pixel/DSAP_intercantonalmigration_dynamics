import matplotlib.pyplot as plt
import numpy as np
from math import pi
import textwrap
import sys
from pathlib import Path
from datetime import datetime

# used columna and their labels for radar plot
FEATURE_MAP = [
    ("Z_score_rent",        "Rent"),
    ("avg_income_zscore",   "Income"),
    ("z-score_unemployment","Unempl."),
    ("Z-score-ownrrate",    "Homeown"),
    ("Z-score-debt",        "Debt"),
    ("shockexposure_zscore","Shock"),
]

def _build_profiles(df):
    
    # constructing of the profiles DataFrame
    df = df.copy()

    # Verifying required feature colums
    missing = [col for col, _ in FEATURE_MAP if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df : {missing}")

    df["canton"] = df["canton"].astype(str).str.strip()

    feature_cols = [col for col, _ in FEATURE_MAP]

    # average profiles per cluster
    profiles = df.groupby("cluster")[feature_cols].mean()

    # list of cantons per cluster
    memberships = (
        df.groupby("cluster")["canton"]
        .apply(lambda x: ", ".join(sorted(set(x))))
    )
    profiles["canton_list"] = memberships

    return profiles


def plot_cluster_radar(df, title="Cluster profiles (K-means)"):
    
    "plot radart chart of cluster profiles"

    profiles = _build_profiles(df)

    # axes labels
    categories = [label for _, label in FEATURE_MAP]
    n_vars = len(categories)

    # axes angles
    angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
    angles += angles[:1]  

    n_clusters = profiles.shape[0]

    fig, axes = plt.subplots(
        1,
        n_clusters,
        figsize=(4 * n_clusters, 5),
        subplot_kw=dict(projection="polar"),
    )
    if n_clusters == 1:
        axes = [axes]

    # colors for each cluster
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    feature_cols = [col for col, _ in FEATURE_MAP]

    # set y-axis limit
    ymin, ymax = -3, 3


    for idx, (cluster_id, row) in enumerate(profiles.iterrows()):
        ax = axes[idx]

        # value for each feature
        values = row[feature_cols].tolist()
        values += values[:1]  

        # plot data
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

        # axes setting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)

        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        # reference line = global mean
        global_mean = profiles[feature_cols].values.mean()
        ax.plot(angles, [global_mean] * len(angles), "k--", linewidth=0.5, alpha=0.5)

        # title with cluster id and members
        members = row["canton_list"]

        # wrap en lignes de max 25–30 caractères
        wrapped = "\n".join(textwrap.wrap(members, width=25))

        ax.set_title(f"Cluster {cluster_id}\n{wrapped}", fontsize=10, fontweight="bold", pad=20)


    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    results_dir = Path(__file__).resolve().parents[3] / "results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = results_dir / f"kmeans_radar_{timestamp}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    return outfile


if __name__ == "__main__":
    # Ensure project root is on sys.path when running directly
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Quick CLI entry to produce the radar plot directly.
    from src.ML.kmeans.kmeans import load_data, prepare_matrix, run_kmeans, assign_clusters

    df = load_data()
    df_clean, X = prepare_matrix(df)
    model = run_kmeans(X, k=3, random_state=0)
    df_clustered = assign_clusters(df_clean, model)

    plot_cluster_radar(df_clustered)
