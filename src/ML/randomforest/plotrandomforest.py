# rf_plots.py
import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_pred_rf(y_true, y_pred, title="Random Forest – True vs Predicted"):


    plt.figure(figsize=(7,7))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Observations")

    # Perfect prediction reference line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle="--", linewidth=2, label="Perfect prediction")

    plt.xlabel("True migration_rate")
    plt.ylabel("Predicted migration_rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
