"""Empirical calibration helpers for the RCK model with official Brazil data."""

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

from data_utils import (
    BRAZIL_SOURCE_MAP,
    aggregate_quarterly_to_annual,
    compute_validation_residuals,
    ensure_directory,
    fetch_bcb_sgs_series,
    fetch_brazil_cna_6784_annual,
    fetch_brazil_cnt_1620_quarterly,
    fetch_brazil_cnt_1846_quarterly,
    fetch_brazil_scn_annual_current,
    filter_tidy_series,
    metadata_entry,
    write_metadata,
)
from params import BRASIL


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_directory(MODULE_DIR / "empirical_outputs")
RBCB_REFERENCE = "https://wilsonfreitas.github.io/rbcb/"

BRAZIL_CURRENT_CODES = ["90707", "93404", "93406"]


def annualize_daily_rate(daily_percent_rate: pd.Series):
    return np.power(1.0 + daily_percent_rate.astype(float) / 100.0, 252.0) - 1.0


def annualize_monthly_inflation(monthly_percent_inflation: pd.Series):
    return np.power(1.0 + monthly_percent_inflation.astype(float) / 100.0, 12.0) - 1.0


def compute_real_rate(nominal_annual_rate: pd.Series, inflation_annual_rate: pd.Series):
    aligned = pd.concat([nominal_annual_rate, inflation_annual_rate], axis=1, join="inner").dropna()
    aligned.columns = ["nominal_rate", "inflation_rate"]
    aligned["real_rate"] = (1.0 + aligned["nominal_rate"]) / (1.0 + aligned["inflation_rate"]) - 1.0
    return aligned


def annotate_series(frame: pd.DataFrame, model_use: str, concept_note: str, proxy_used=None, aggregation=None):
    annotated = frame.copy()
    annotated["model_use"] = model_use
    annotated["concept_note"] = concept_note
    if proxy_used is not None:
        annotated["proxy_used"] = proxy_used
    if aggregation is not None:
        annotated["aggregation"] = aggregation
    return annotated


def annual_series_to_tidy(series: pd.Series, series_id: str, series_name: str, unit: str, source: str, dataset_id: str, series_code, model_use: str, concept_note: str, proxy_used=None, aggregation=None):
    frame = pd.DataFrame(
        {
            "period": series.index.astype(int).astype(str),
            "period_label": series.index.astype(int).astype(str),
            "series_id": series_id,
            "series_name": series_name,
            "value": series.to_numpy(dtype=float),
            "unit": unit,
            "source": source,
            "dataset_id": dataset_id,
            "frequency": "annual",
            "series_code": str(series_code),
        }
    )
    return annotate_series(frame, model_use=model_use, concept_note=concept_note, proxy_used=proxy_used, aggregation=aggregation)


def build_rck_brazil_metadata():
    return {
        "title": "Brazil calibration inputs for the RCK model with official SIDRA and BCB data",
        "brazil": {
            "real_rate_input": "selic_ipca",
            "consumption_input": "household_consumption_volume_index_per_capita",
            "future_extensions": BRAZIL_SOURCE_MAP["future_extensions"],
            "warnings": [
                "The real rate remains an operational proxy based on Selic deflated by IPCA, not a structural long-run natural rate.",
                "A future upgrade can swap selic_ipca for NTN-B or term-structure based real yields.",
            ],
            "sources": [
                metadata_entry(
                    source="BCB SGS",
                    frequency="daily to annual aggregation",
                    unit="annualized nominal policy rate",
                    dataset_id="BCB SGS",
                    series_code="11",
                    aggregation="daily -> monthly mean -> annual mean",
                    concept_note="Selic is used as the immediate operational rate proxy. rbcb remains the reference R package for a future bridge: " + RBCB_REFERENCE,
                ),
                metadata_entry(
                    source="BCB SGS",
                    frequency="monthly to annual aggregation",
                    unit="annual inflation rate",
                    dataset_id="BCB SGS",
                    series_code="433",
                    aggregation="monthly compounding within calendar year",
                    concept_note="IPCA is compounded within each calendar year to produce annual inflation.",
                ),
                metadata_entry(
                    source="IBGE SIDRA",
                    frequency="annual",
                    unit="thousand persons",
                    dataset_id="CNA 6784",
                    series_code="93",
                    proxy_used="population",
                    aggregation="annual official series",
                    concept_note="Population is used to move from aggregate household consumption to a per-capita real proxy.",
                ),
                metadata_entry(
                    source="IBGE SIDRA",
                    frequency="quarterly to annual aggregation",
                    unit="index, base mean 1995=100",
                    dataset_id="CNT 1620",
                    series_code="583 / 11255(93404)",
                    aggregation="quarterly_to_annual_mean",
                    concept_note="The household consumption volume index is annualized by mean and then divided by population to build a real per-capita consumption proxy.",
                ),
                metadata_entry(
                    source="IBGE SCN annual workbook",
                    frequency="annual",
                    unit="1_000_000 BRL",
                    dataset_id="SCN 2023 tab05",
                    series_code="tab05:B:consumo_familias:valor_corrente",
                    aggregation="annual official reference",
                    revision_reference="Sistema de Contas Nacionais 2023",
                    concept_note="Annual household consumption current values are included as the official SCN level reference.",
                ),
            ],
        },
    }


def fetch_brazil_real_rate(start_date="2000-01-01", end_date=None, selic_series=11, ipca_series=433):
    final_date = end_date or date.today().isoformat()
    selic = fetch_bcb_sgs_series(selic_series, start_date=start_date, end_date=final_date).set_index("date")
    ipca = fetch_bcb_sgs_series(ipca_series, start_date=start_date, end_date=final_date).set_index("date")

    nominal_monthly = annualize_daily_rate(selic["value"]).resample("ME").mean()
    nominal_annual = nominal_monthly.groupby(nominal_monthly.index.year).mean().rename("nominal_rate")

    monthly_ipca = ipca["value"].resample("ME").last()
    inflation_annual = monthly_ipca.groupby(monthly_ipca.index.year).apply(
        lambda series: float(np.prod(1.0 + series.astype(float) / 100.0) - 1.0)
    )
    inflation_annual = inflation_annual.rename("inflation_rate")

    real_rate = compute_real_rate(nominal_annual, inflation_annual)
    real_rate.index.name = "year"
    return real_rate


def build_brazil_consumption_inputs(start_year=1996, end_year=2023):
    population = filter_tidy_series(fetch_brazil_cna_6784_annual(), ["93"])
    population = population.loc[population["period"].astype(int).between(start_year, end_year)].copy()

    consumption_volume_quarterly = filter_tidy_series(fetch_brazil_cnt_1620_quarterly(), ["93404"])
    consumption_volume_annual = aggregate_quarterly_to_annual(consumption_volume_quarterly, aggregation="mean")
    consumption_volume_annual = consumption_volume_annual.loc[
        consumption_volume_annual["period"].astype(int).between(start_year, end_year)
    ].copy()

    scn_annual_current = filter_tidy_series(fetch_brazil_scn_annual_current(), ["93404"])
    scn_annual_current = scn_annual_current.loc[
        scn_annual_current["period"].astype(int).between(max(start_year, 2000), end_year)
    ].copy()

    population_series = (
        population.loc[:, ["period", "value"]]
        .rename(columns={"value": "population"})
        .assign(period=lambda frame: frame["period"].astype(int))
    )
    consumption_volume_series = (
        consumption_volume_annual.loc[:, ["period", "value"]]
        .rename(columns={"value": "consumption_volume_index"})
        .assign(period=lambda frame: frame["period"].astype(int))
    )

    panel = (
        consumption_volume_series.merge(population_series, on="period", how="inner")
        .sort_values("period")
        .set_index("period")
    )
    panel["consumption_per_capita_real_proxy"] = panel["consumption_volume_index"] / panel["population"]

    official_series = pd.concat(
        [
            annotate_series(
                population,
                model_use="rck_per_capita_denominator",
                concept_note="Annual resident population proxy used to convert aggregate consumption into per-capita terms.",
                proxy_used="population",
                aggregation="annual_official",
            ),
            annotate_series(
                consumption_volume_annual,
                model_use="rck_real_consumption_proxy",
                concept_note="Quarterly household consumption volume index annualized by mean.",
                aggregation="quarterly_to_annual_mean",
            ),
            annotate_series(
                scn_annual_current,
                model_use="rck_official_current_reference",
                concept_note="Annual SCN current-price household consumption reference.",
                aggregation="annual_official",
            ),
        ],
        ignore_index=True,
    ).sort_values(["dataset_id", "series_id", "period"])

    return panel, official_series


def build_validation_residuals(start_year=2000, end_year=2023):
    cnt_current_quarterly = filter_tidy_series(fetch_brazil_cnt_1846_quarterly(), BRAZIL_CURRENT_CODES)
    cnt_current_annual = aggregate_quarterly_to_annual(cnt_current_quarterly, aggregation="sum")
    cnt_current_annual = cnt_current_annual.loc[cnt_current_annual["period"].astype(int).between(start_year, end_year)]
    scn_annual_current = filter_tidy_series(fetch_brazil_scn_annual_current(), BRAZIL_CURRENT_CODES)
    scn_annual_current = scn_annual_current.loc[scn_annual_current["period"].astype(int).between(start_year, end_year)]
    return compute_validation_residuals(cnt_current_annual, scn_annual_current)


def calibrate_brazil_rho(theta=None, estimate_theta=False, start_year=2000, end_year=2023):
    theta_default = BRASIL["theta"] if theta is None else float(theta)
    real_rate = fetch_brazil_real_rate(start_date=f"{start_year}-01-01", end_date=f"{end_year}-12-31")
    consumption_panel, official_series = build_brazil_consumption_inputs(start_year=1996, end_year=end_year)

    panel = pd.concat(
        [
            real_rate["real_rate"],
            np.log(consumption_panel["consumption_per_capita_real_proxy"]).diff().rename("consumption_growth"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    panel = panel.loc[(panel.index >= start_year) & (panel.index <= end_year)].copy()

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
        "operational_proxy": "selic_deflated_ipca",
    }

    real_rate_series = pd.concat(
        [
            annual_series_to_tidy(
                real_rate["nominal_rate"],
                series_id="11",
                series_name="Selic nominal annualized",
                unit="annual rate",
                source="BCB SGS",
                dataset_id="BCB SGS",
                series_code=11,
                model_use="rck_real_rate_proxy",
                concept_note="Daily Selic annualized, averaged by month, then averaged within year.",
                aggregation="daily -> monthly mean -> annual mean",
            ),
            annual_series_to_tidy(
                real_rate["inflation_rate"],
                series_id="433",
                series_name="IPCA annual inflation",
                unit="annual rate",
                source="BCB SGS",
                dataset_id="BCB SGS",
                series_code=433,
                model_use="rck_real_rate_proxy",
                concept_note="Monthly IPCA compounded within each calendar year.",
                aggregation="monthly compounding within calendar year",
            ),
            annual_series_to_tidy(
                real_rate["real_rate"],
                series_id="selic_ipca_real_rate",
                series_name="Real rate from Selic and IPCA",
                unit="annual rate",
                source="BCB SGS",
                dataset_id="BCB SGS",
                series_code="11-433",
                model_use="rck_real_rate_proxy",
                concept_note="Operational real-rate proxy used in the RCK calibration.",
                proxy_used="selic_ipca",
                aggregation="annual Fisher transformation",
            ),
        ],
        ignore_index=True,
    )

    official_series = pd.concat([official_series, real_rate_series], ignore_index=True).sort_values(["dataset_id", "series_id", "period"])
    metadata = build_rck_brazil_metadata()
    return panel, summary, metadata, official_series


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
    panel, summary, metadata, official_series = calibrate_brazil_rho(estimate_theta=True)
    validation = build_validation_residuals()
    figure_path = plot_real_rate_vs_consumption_growth(panel, summary)
    panel.to_csv(OUTPUT_DIR / "rck_brazil_calibration_panel.csv")
    official_series.to_csv(OUTPUT_DIR / "brazil_official_series.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "brazil_validation_residuals.csv", index=False)
    write_metadata(metadata, OUTPUT_DIR / "rck_empirics_metadata.json")
    write_metadata(summary, OUTPUT_DIR / "rck_brazil_calibration_summary.json")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
