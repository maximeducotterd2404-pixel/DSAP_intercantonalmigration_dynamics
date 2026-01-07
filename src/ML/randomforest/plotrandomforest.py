# Plot helper to persist RF diagnostics for the report.
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_true_vs_pred_rf(y_true, y_pred, title="Random Forest – True vs Predicted"):
    # Scatter helps diagnose calibration and bias at a glance.

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Observations")

    # Diagonal reference highlights perfect predictions.
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("True migration_rate")
    plt.ylabel("Predicted migration_rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Persist the plot for reporting and comparisons.
    results_dir = Path(__file__).resolve().parents[3] / "results"
    outfile = results_dir / "randomforest_true_vs_pred.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    return outfile
