import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Colonnes utilisées pour le radar + label lisible
FEATURE_MAP = [
    ("Z_score_rent",        "Rent"),
    ("avg_income_zscore",   "Income"),
    ("z-score_unemployment","Unempl."),
    ("Z-score-ownrrate",    "Homeown"),
    ("Z-score-debt",        "Debt"),
    ("shockexposure_zscore","Shock"),
]

def _build_profiles(df):
    """
    Construit le tableau des moyennes par cluster
    + la liste des cantons dans chaque cluster.
    """
    df = df.copy()
    if "cluster" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'cluster'.")
    if "canton" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'canton'.")

    # Vérifier que toutes les features sont présentes
    missing = [col for col, _ in FEATURE_MAP if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans df : {missing}")

    df["canton"] = df["canton"].astype(str).str.strip()

    feature_cols = [col for col, _ in FEATURE_MAP]

    # Moyenne des features par cluster
    profiles = df.groupby("cluster")[feature_cols].mean()

    # Liste des cantons par cluster (pour les titres)
    memberships = (
        df.groupby("cluster")["canton"]
        .apply(lambda x: ", ".join(sorted(set(x))))
    )
    profiles["canton_list"] = memberships

    return profiles


def plot_cluster_radar(df, title="Cluster profiles (K-means)"):
    """
    Trace un radar plot pour chaque cluster.
    df : DataFrame contenant au moins les colonnes canton, cluster et les features de FEATURE_MAP.
    """
    profiles = _build_profiles(df)

    # Labels des axes
    categories = [label for _, label in FEATURE_MAP]
    n_vars = len(categories)

    # Angles des axes (radial)
    angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
    angles += angles[:1]  # fermer le cercle

    n_clusters = profiles.shape[0]

    fig, axes = plt.subplots(
        1,
        n_clusters,
        figsize=(4 * n_clusters, 5),
        subplot_kw=dict(projection="polar"),
    )
    if n_clusters == 1:
        axes = [axes]

    # Palette de couleurs
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    feature_cols = [col for col, _ in FEATURE_MAP]

    # Pour fixer une échelle cohérente, on regarde min/max des moyennes
    ymin, ymax = -3, 3


    for idx, (cluster_id, row) in enumerate(profiles.iterrows()):
        ax = axes[idx]

        # Valeurs dans l'ordre des features
        values = row[feature_cols].tolist()
        values += values[:1]  # fermer le polygone

        # Tracé
        ax.plot(angles, values, "o-", linewidth=2, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

        # Axes et ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)

        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        # Ligne de référence horizontale (~moyenne globale)
        global_mean = profiles[feature_cols].values.mean()
        ax.plot(angles, [global_mean] * len(angles), "k--", linewidth=0.5, alpha=0.5)

        # Titre = cluster + cantons
        members = row["canton_list"]
        ax.set_title(f"Cluster {cluster_id}\n({members})", fontsize=10, fontweight="bold", pad=20)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
