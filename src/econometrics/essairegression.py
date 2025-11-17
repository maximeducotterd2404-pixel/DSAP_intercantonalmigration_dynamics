#!/usr/bin/env python3
"""Fixed-effects regression using pyfixest with clustered standard errors."""

from pathlib import Path
import numpy as np
import pandas as pd
from pyfixest.estimation import feols


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "databasecsv.csv"
OUTPUT_DIR = ROOT / "results"

df = pd.read_csv(DATA_PATH, sep=";")
print(f"Loaded {len(df)} rows")

#Formula for the fixed effect regression
FORMULA = (
    "migration_rate ~ log_rent_avg + log_avg_income + log_unemployment + "
    "log_schockexposure + "
    "log_avg_income_x_log_rent_avg + log_unemployment_rate_x_log_avg_income + "
    "log_schockexposure_x_CLUSTER1 + log_schockexposure_x_CLUSTER2 | "
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

    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows\n")

    # variable used in the regression
    base_vars = [
        "log_rent_avg",
        "log_avg_income",
        "log_unemployment",
        "log_schockexposure",
        "CLUSTER1",
        "CLUSTER2"
    ]

    # creation of interactions terms
    df["log_avg_income_x_log_rent_avg"] = df["log_avg_income"] * df["log_rent_avg"]
    df["log_unemployment_rate_x_log_avg_income"] = (
        df["log_unemployment"] * df["log_avg_income"]
    )
    df["log_schockexposure_x_CLUSTER0"] = (
        df["log_schockexposure"] * df["CLUSTER0"]
    )
    df["log_schockexposure_x_CLUSTER1"] = (
        df["log_schockexposure"] * df["CLUSTER1"]
    )
    df["log_schockexposure_x_CLUSTER2"] = (
        df["log_schockexposure"] * df["CLUSTER2"]
    )

    interaction_vars = [
        "log_avg_income_x_log_rent_avg",
        "log_unemployment_rate_x_log_avg_income",
        "log_schockexposure_x_CLUSTER1",
        "log_schockexposure_x_CLUSTER2",
    ]

    # cleaning data: drops rows with missing value (should be 0, but we never know)
    required_cols = ["migration_rate", "canton", "year"] + base_vars + interaction_vars
    df_model = df.dropna(subset=required_cols).copy()

    print(f"After cleaning: {len(df_model)} observations\n")

    if df_model.empty:
        print("❌ No valid data remaining!")
        return

    # FE estimation with clustered sdandard errors.
    result = feols(FORMULA, data=df_model, vcov={"CRV1": CLUSTER_SPEC})


    # construct of residuals and metrics to construct mse, rmse, r2 within
    y_true = df_model["migration_rate"].to_numpy()
    y_pred = result.predict()
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

    b0 = coefs.get("log_schockexposure", np.nan)
    b1 = coefs.get("log_schockexposure_x_CLUSTER1", 0)
    b2 = coefs.get("log_schockexposure_x_CLUSTER2", 0)


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

    output_file = OUTPUT_DIR / "lin_log.txt"
    output_file.write_text(summary + "\n", encoding="utf-8")

    print( "Regression finished\n")
    print(summary)
    print(f"\nSaved to: {output_file}")



if __name__ == "__main__":
    main()
