"""Empirical calibration helpers for the RCK model."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import ensure_directory, fetch_bcb_sgs_series, fetch_world_bank_panel, metadata_entry, write_metadata
from params import BRASIL


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_directory(MODULE_DIR / "empirical_outputs")
RBCB_REFERENCE = "https://wilsonfreitas.github.io/rbcb/"


def annualize_daily_rate(daily_percent_rate: pd.Series):
    return np.power(1.0 + daily_percent_rate.astype(float) / 100.0, 252.0) - 1.0


def annualize_monthly_inflation(monthly_percent_inflation: pd.Series):
    return np.power(1.0 + monthly_percent_inflation.astype(float) / 100.0, 12.0) - 1.0


def compute_real_rate(nominal_annual_rate: pd.Series, inflation_annual_rate: pd.Series):
    aligned = pd.concat([nominal_annual_rate, inflation_annual_rate], axis=1, join="inner").dropna()
    aligned.columns = ["nominal_rate", "inflation_rate"]
    aligned["real_rate"] = (1.0 + aligned["nominal_rate"]) / (1.0 + aligned["inflation_rate"]) - 1.0
    return aligned


def fetch_brazil_real_rate(start_date="2000-01-01", end_date=None, selic_series=11, ipca_series=433):
    final_date = end_date or date.today().isoformat()
    selic = fetch_bcb_sgs_series(selic_series, start_date=start_date, end_date=final_date)
    ipca = fetch_bcb_sgs_series(ipca_series, start_date=start_date, end_date=final_date)

    selic = selic.set_index("date")
    selic["nominal_rate"] = annualize_daily_rate(selic["value"])
    nominal_monthly = selic["nominal_rate"].resample("ME").mean()

    ipca = ipca.set_index("date")
    inflation_monthly = annualize_monthly_inflation(ipca["value"].resample("ME").last())
    real_rate = compute_real_rate(nominal_monthly, inflation_monthly)
    real_rate.index = real_rate.index.year
    annual_real_rate = real_rate.groupby(level=0).mean()
    annual_real_rate.index.name = "year"
    return annual_real_rate


def fetch_brazil_consumption_per_capita(start_year=2000, end_year=2023):
    indicators = {
        "NE.CON.PRVT.KD": "consumption_total",
        "SP.POP.TOTL": "population",
    }
    panel = fetch_world_bank_panel(["BRA"], indicators, start_year, end_year)
    brazil = panel.loc[panel["countryiso3code"] == "BRA"].sort_values("date").set_index("date")
    series = (brazil["consumption_total"] / brazil["population"]).dropna()
    note = "World Bank total household consumption divided by population."
    series.name = "consumption_per_capita"
    return series, note


def calibrate_brazil_rho(theta=None, estimate_theta=False, start_year=2000, end_year=2023):
    theta_default = BRASIL["theta"] if theta is None else float(theta)
    real_rate = fetch_brazil_real_rate(start_date=f"{start_year}-01-01", end_date=f"{end_year}-12-31")
    consumption_per_capita, consumption_note = fetch_brazil_consumption_per_capita(start_year=start_year, end_year=end_year)

    panel = pd.concat(
        [
            real_rate["real_rate"],
            np.log(consumption_per_capita).diff().rename("consumption_growth"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    theta_used = theta_default
    estimation_method = "fixed-theta moment condition"
    if estimate_theta and len(panel) >= 8:
        design = np.column_stack([np.ones(len(panel)), panel["real_rate"].to_numpy()])
        coefficients, _, _, _ = np.linalg.lstsq(design, panel["consumption_growth"].to_numpy(), rcond=None)
        slope = float(coefficients[1])
        if slope > 0:
            theta_used = 1.0 / slope
            rho_hat = -float(coefficients[0]) * theta_used
            estimation_method = "OLS Euler equation"
        else:
            rho_hat = float((panel["real_rate"] - theta_used * panel["consumption_growth"]).mean())
    else:
        rho_hat = float((panel["real_rate"] - theta_used * panel["consumption_growth"]).mean())

    summary = {
        "rho_hat": rho_hat,
        "theta_used": theta_used,
        "sample_start": int(panel.index.min()),
        "sample_end": int(panel.index.max()),
        "mean_real_rate": float(panel["real_rate"].mean()),
        "mean_consumption_growth": float(panel["consumption_growth"].mean()),
        "estimation_method": estimation_method,
    }
    metadata = {
        "title": "Brazil calibration inputs for the RCK model",
        "sources": [
            metadata_entry(
                source="BCB SGS direct API",
                frequency="daily to annual aggregation",
                unit="annual real interest rate",
                fallback="rbcb reference package for future R bridge",
                note="Immediate implementation uses Python and the BCB endpoint directly. rbcb reference: "
                + RBCB_REFERENCE,
            ),
            metadata_entry(
                source="World Bank API",
                frequency="annual",
                unit="household final consumption per capita, constant prices",
                fallback="IBGE or IPEA future bridge",
                note=consumption_note,
            ),
        ],
    }
    return panel, summary, metadata


def plot_real_rate_vs_consumption_growth(panel: pd.DataFrame, summary: dict, output_dir=OUTPUT_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(100 * panel["real_rate"], 100 * panel["consumption_growth"], color="steelblue", alpha=0.75)
    if len(panel) >= 2:
        slope, intercept = np.polyfit(100 * panel["real_rate"], 100 * panel["consumption_growth"], deg=1)
        x_line = np.linspace(100 * panel["real_rate"].min(), 100 * panel["real_rate"].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color="darkorange", lw=2.0)
    ax.set_xlabel("Real interest rate, percent")
    ax.set_ylabel("Consumption growth, percent")
    ax.set_title(f"Brazil Euler-equation calibration, rho={summary['rho_hat']:.3f}")
    fig.tight_layout()
    path = Path(output_dir) / "rck_brazil_calibration.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    panel, summary, metadata = calibrate_brazil_rho(estimate_theta=True)
    figure_path = plot_real_rate_vs_consumption_growth(panel, summary)
    panel.to_csv(OUTPUT_DIR / "rck_brazil_calibration_panel.csv")
    write_metadata(metadata, OUTPUT_DIR / "rck_empirics_metadata.json")
    write_metadata(summary, OUTPUT_DIR / "rck_brazil_calibration_summary.json")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
