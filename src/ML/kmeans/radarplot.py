import matplotlib.pyplot as plt
import numpy as np
from math import pi
import textwrap
import sys
from pathlib import Path
from datetime import datetime

# Map standardized variables to readable labels for interpretation.
FEATURE_MAP = [
    ("Z_score_rent", "Rent"),
    ("avg_income_zscore", "Income"),
    ("z-score_unemployment", "Unempl."),
    ("Z-score-ownrrate", "Homeown"),
    ("Z-score-debt", "Debt"),
    ("shockexposure_zscore", "Shock"),
]


def _build_profiles(df):
    # Build per-cluster profiles for interpretation.
    df = df.copy()

    # Validate required inputs to avoid misleading plots.
    missing = [col for col, _ in FEATURE_MAP if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df : {missing}")
    if "canton" not in df.columns:
        raise ValueError("Missing column in df: canton")
    if "cluster" not in df.columns:
        raise ValueError("Missing column in df: cluster")

    # Normalize canton names for stable grouping.
    df["canton"] = df["canton"].astype(str).str.strip()

    feature_cols = [col for col, _ in FEATURE_MAP]

    if df["canton"].duplicated().any():
        # Aggregate to canton averages so profiles reflect structure, not noise.
        agg_map = {col: "mean" for col in feature_cols}
        agg_map["cluster"] = lambda s: s.value_counts().idxmax()
        df = df.groupby("canton", as_index=False).agg(agg_map)

    # Mean profiles summarize each cluster's typical structure.
    profiles = df.groupby("cluster")[feature_cols].mean()

    # Keep membership lists for interpretability.
    memberships = df.groupby("cluster")["canton"].apply(
        lambda x: ", ".join(sorted(set(x)))
    )
    profiles["canton_list"] = memberships

    return profiles


def plot_cluster_radar(df, title="Cluster profiles (K-means, canton averages)"):
    """Plot cluster profiles to compare structural patterns across cantons."""

    profiles = _build_profiles(df)

    # Use readable labels for the radar axes.
    categories = [label for _, label in FEATURE_MAP]
    n_vars = len(categories)

    # Evenly space variables around the circle.
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

    # Distinct colors keep clusters visually separable.
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    feature_cols = [col for col, _ in FEATURE_MAP]

    # Fix limits so clusters are comparable across panels.
    ymin, ymax = -3, 3

    for idx, (cluster_id, row) in enumerate(profiles.iterrows()):
        ax = axes[idx]

        # Close the polygon by repeating the first value.
        values = row[feature_cols].tolist()
        values += values[:1]

        # Outline and fill for readability in small panels.
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

        # Label axes for interpretation.
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)

        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        # Global mean gives a baseline for relative deviations.
        global_mean = profiles[feature_cols].values.mean()
        ax.plot(angles, [global_mean] * len(angles), "k--", linewidth=0.5, alpha=0.5)

        # Show cluster membership to make profiles tangible.
        members = row["canton_list"]

        # Wrap long membership lists for legibility.
        wrapped = "\n".join(textwrap.wrap(members, width=25))

        ax.set_title(
            f"Cluster {cluster_id}\n{wrapped}", fontsize=10, fontweight="bold", pad=20
        )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    results_dir = Path(__file__).resolve().parents[3] / "results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = results_dir / f"kmeans_radar_{timestamp}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    return outfile


if __name__ == "__main__":
    # Allow direct execution without installing the package.
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Convenience entry to reproduce the cluster plot.
    from src.ML.kmeans.kmeans import (
        load_data,
        prepare_matrix,
        run_kmeans,
        assign_clusters,
    )

    # Compute clusters before plotting.
    df = load_data()
    df_clean, X = prepare_matrix(df)
    model = run_kmeans(X, k=3, random_state=0)
    df_clustered = assign_clusters(df_clean, model)

    # Generate the radar plot for reporting.
    plot_cluster_radar(df_clustered)
