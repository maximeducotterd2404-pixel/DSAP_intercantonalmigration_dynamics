#!/usr/bin/env python3
"""Fixed-effects regression using pyfixest with clustered standard errors."""

from pathlib import Path
import numpy as np
import pandas as pd
from pyfixest.estimation import feols


# Load and trying to prepare the data
def load_data(path=None):
    if path is None:
        ROOT = Path(__file__).resolve().parents[2]
        path = ROOT / "data" / "databasecsv.csv"

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Dataset exists but is empty at {path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading dataset: {e}")
    
#Formula for the fixed effect regression
FORMULA = (
    "migration_rate ~ log_rent_avg + log_avg_income + log_unemployment + "
    "shock_exposure + "
    "log_avg_income_x_log_rent_avg + log_unemployment_rate_x_log_avg_income + "
    "shockexposure_x_CLUSTER1 + shockexposure_x_CLUSTER2 | "
    "canton"
)
# Error standard clusering.
CLUSTER_SPEC = "canton"

#Build a summary of the regression rsults
def format_summary(title, tidy_df, r2, mse, rmse, n_obs, cluster_spec):
    """Build a regression summary."""
# control the formatting numbers
    def fmt(value, width=10, precision=4):
        if pd.notna(value):
            return f"{value:>{width}.{precision}f}"
        return f"{'nan':>{width}}"

    lines = [
        title,
        "=" * 60,
        f"Observations: {n_obs}",
        f"R² (within FE): {r2:.4f} | RMSE: {rmse:.4f} | MSE: {mse:.4f}",
        f"Clustered SEs (CRV1): {cluster_spec.replace('+', ', ')}",
        "",
        "-" * 60,
        f"{'Coefficient':<40}{'Estimate':>10}{'Std.Err.':>12}{'t':>8}{'p>|t|':>12}",
    ]

    for name, row in tidy_df.iterrows():
        lines.append(
            f"{name:<40}"
            f"{fmt(row['Estimate']):>10}"
            f"{fmt(row['Std. Error']):>12}"
            f"{fmt(row['t value'], width=8, precision=2)}"
            f"{fmt(row['Pr(>|t|)'], width=12, precision=4)}"
        )

    lines.append("-" * 60)
    return "\n".join(lines)

# determine the function to run the reression
def main():
    """charge data, construct the model."""

    ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = ROOT / "data" / "databasecsv.csv"

    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows\n")

    # variable used in the regression
    base_vars = [
        "log_rent_avg",
        "log_avg_income",
        "log_unemployment",
        "shock_exposure",
        "CLUSTER1",
        "CLUSTER2"
    ]
    # ===== CHECK REQUIRED COLUMNS BEFORE INTERACTIONS =====
    required_before_interactions = [
    "migration_rate", "canton", "year",
    "log_rent_avg", "log_avg_income",
    "log_unemployment", "shock_exposure",
    "CLUSTER0", "CLUSTER1", "CLUSTER2"
    ]

    missing = [c for c in required_before_interactions if c not in df.columns]
    if missing:
        raise KeyError(
        f"Missing required columns before interactions: {missing}\n"
        f"Columns available: {list(df.columns)}"
    )


    # creation of interactions terms
    df["log_avg_income_x_log_rent_avg"] = df["log_avg_income"] * df["log_rent_avg"]
    df["log_unemployment_rate_x_log_avg_income"] = (
        df["log_unemployment"] * df["log_avg_income"]
    )
    df["shockexposure_x_CLUSTER1"] = (
        df["shock_exposure"] * df["CLUSTER1"]
    )
    df["shockexposure_x_CLUSTER2"] = (
        df["shock_exposure"] * df["CLUSTER2"]
    )

    interaction_vars = [
        "log_avg_income_x_log_rent_avg",
        "log_unemployment_rate_x_log_avg_income",
        "shockexposure_x_CLUSTER1",
        "shockexposure_x_CLUSTER2",
    ]

    # cleaning data: drops rows with missing value (should be 0, but we never know)
    required_cols = ["migration_rate", "canton", "year"] + base_vars + interaction_vars
    df_model = df.dropna(subset=required_cols).copy()
    missing_after = [c for c in required_cols if c not in df.columns]
    if missing_after:
        raise KeyError(
            f"Missing interaction columns: {missing_after}\n"
            f"Columns available: {list(df.columns)}"
        )

    print(f"After cleaning: {len(df_model)} observations\n")

    if df_model.empty:
        print("No valid data remaining!")
        return

    # FE estimation with clustered sdandard errors.
    try:
        result = feols(FORMULA, data=df_model, vcov={"CRV1": CLUSTER_SPEC})
    except Exception as e:
        raise RuntimeError(f"Fixed-effects regression failed: {e}")


    # construct of residuals and metrics to construct mse, rmse, r2 within
    y_true = df_model["migration_rate"].to_numpy()
    try:
        y_pred = result.predict()
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
    residuals = y_true - y_pred

    mse = float((residuals**2).mean())
    rmse = mse**0.5
    mu_canton = df_model.groupby("canton")["migration_rate"].transform("mean").to_numpy()
    mu_year = df_model.groupby("year")["migration_rate"].transform("mean").to_numpy()
    mu_all = y_true.mean()
    y_within = y_true - mu_canton - mu_year + mu_all
    ss_resid = float((residuals ** 2).sum())
    ss_total = float((y_within ** 2).sum())
    r2_within = 1 - ss_resid / ss_total


    tidy_df = result.tidy()

    # marginal effects calculation
    coefs = result.coef() #get the coefficent from the regression

    b0 = coefs.get("shock_exposure", np.nan)
    b1 = coefs.get("shockexposure_x_CLUSTER1", 0)
    b2 = coefs.get("shockexposure_x_CLUSTER2", 0)


    # calulate the marginal effects for each cluster
    eff_cluster0 = b0
    eff_cluster1 = b0 + b1
    eff_cluster2 = b0 + b2

    # Create dataframe
    marginal_effects = pd.DataFrame({
        "Cluster": ["CLUSTER0 (baseline)", "CLUSTER1", "CLUSTER2"],
        "Effet marginal du shock": [eff_cluster0, eff_cluster1, eff_cluster2]
    })

    print("\n=== Effets marginaux du shock par cluster ===")
    print(marginal_effects.to_string(index=False))
    print("\n")

# building the summary
    summary = format_summary(
        "Fixed-effects regression (Canton & Year FE) with clustered standard errors",
        tidy_df,
        r2_within,
        mse,
        rmse,
        len(df_model),
        CLUSTER_SPEC,
    )

    print( "Regression finished\n")
    print(summary)

if __name__ == "__main__":
    main()
