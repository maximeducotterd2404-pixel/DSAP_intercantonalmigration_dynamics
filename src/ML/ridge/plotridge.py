# ridge_plots.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_true_vs_pred(y_true, y_pred, title="Ridge – True vs Predicted"):
    """Scatter plot True vs Predicted with a 45° line."""
    plt.figure(figsize=(7,7))
    plt.scatter(y_true, y_pred, alpha=0.7, color="blue", label="Observations")

    # line of perfect predictions
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect prediction')

    plt.xlabel("True migration_rate")
    plt.ylabel("Predicted migration_rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
