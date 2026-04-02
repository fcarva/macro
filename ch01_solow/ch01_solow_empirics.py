"""Empirical extension for the Solow model with Brazil-first outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow import SolowModel
from data_utils import ensure_directory, fetch_world_bank_panel, metadata_entry, write_metadata
from params import BRASIL, OECD_COUNTRIES, SOLOW, clone_params


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_directory(MODULE_DIR / "empirical_outputs")


def perpetual_inventory(investment: pd.Series, depreciation_rate: float):
    investment = investment.dropna().astype(float)
    if investment.empty:
        raise ValueError("Investment series is empty.")

    investment_growth = np.log(investment).diff().dropna()
    average_growth = float(max(investment_growth.head(10).mean(), 0.01)) if not investment_growth.empty else 0.03
    denominator = max(np.exp(average_growth) - 1.0 + depreciation_rate, 0.03)
    capital = np.zeros(len(investment))
    capital[0] = investment.iloc[0] / denominator
    for index in range(1, len(investment)):
        capital[index] = (1.0 - depreciation_rate) * capital[index - 1] + investment.iloc[index]
    return pd.Series(capital, index=investment.index, name="capital")


def fetch_cross_country_solow_panel(countries, start_year=1990, end_year=2023):
    indicators = {
        "NY.GDP.MKTP.PP.KD": "gdp_ppp",
        "SL.TLF.TOTL.IN": "labor_force",
        "NE.GDI.TOTL.ZS": "investment_share",
    }
    panel = fetch_world_bank_panel(countries, indicators, start_year, end_year)
    panel["output_per_worker"] = panel["gdp_ppp"] / panel["labor_force"]
    return panel.dropna(subset=["output_per_worker"])


def build_convergence_panel(panel: pd.DataFrame, initial_year=2000, final_year=2023):
    subset = panel.loc[panel["date"].isin([initial_year, final_year]), ["country", "countryiso3code", "date", "output_per_worker"]]
    pivot = subset.pivot_table(index=["country", "countryiso3code"], columns="date", values="output_per_worker", aggfunc="first").dropna()
    pivot = pivot.rename(columns={initial_year: "initial_output_per_worker", final_year: "final_output_per_worker"}).reset_index()
    pivot["annualized_growth"] = (
        np.log(pivot["final_output_per_worker"]) - np.log(pivot["initial_output_per_worker"])
    ) / (final_year - initial_year)
    pivot["oecd"] = pivot["countryiso3code"].isin(OECD_COUNTRIES)
    return pivot


def build_brazil_growth_accounting(start_year=1960, end_year=2023):
    indicators = {
        "NY.GDP.MKTP.KN": "output",
        "NE.GDI.TOTL.KN": "investment",
        "SL.TLF.TOTL.IN": "labor_force",
        "SP.POP.TOTL": "population",
    }
    panel = fetch_world_bank_panel(["BRA"], indicators, start_year, end_year)
    brazil = panel.loc[panel["countryiso3code"] == "BRA"].sort_values("date").set_index("date")

    labor_column = "labor_force" if brazil["labor_force"].notna().sum() >= 0.8 * len(brazil) else "population"
    growth_data = brazil[["output", "investment", labor_column]].rename(columns={labor_column: "labor"}).dropna()
    growth_data["capital"] = perpetual_inventory(growth_data["investment"], depreciation_rate=BRASIL["delta"])

    model = SolowModel(clone_params(SOLOW, BRASIL))
    accounting = model.growth_accounting(growth_data[["output", "capital", "labor"]], alpha=BRASIL["alpha"])
    metadata = {
        "title": "Brazil growth accounting in Solow empirics",
        "sources": [
            metadata_entry(
                source="World Bank API",
                frequency="annual",
                unit="constant local currency units / persons",
                fallback="IBGE or IPEA local-national accounts bridge planned for future revisions",
                note=f"Labor input used: {labor_column}",
            )
        ],
    }
    return accounting, metadata


def plot_output_per_worker(panel: pd.DataFrame, output_dir=OUTPUT_DIR):
    sample = panel.loc[panel["countryiso3code"].isin(["BRA", "USA", "KOR", "CHN", "MEX"])].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for code, group in sample.groupby("countryiso3code"):
        ordered = group.sort_values("date")
        ax.plot(ordered["date"], ordered["output_per_worker"], lw=2.0, label=code)
    ax.set_title("GDP per worker, selected countries")
    ax.set_xlabel("Year")
    ax.set_ylabel("PPP-adjusted output per worker")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "solow_output_per_worker.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_investment_vs_income(panel: pd.DataFrame, output_dir=OUTPUT_DIR):
    latest = panel.sort_values("date").groupby("countryiso3code").tail(1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(latest["investment_share"], np.log(latest["output_per_worker"]), alpha=0.7, color="steelblue")
    for _, row in latest.loc[latest["countryiso3code"].isin(["BRA", "USA", "KOR", "CHN"])].iterrows():
        ax.annotate(row["countryiso3code"], (row["investment_share"], np.log(row["output_per_worker"])), fontsize=9)
    ax.set_xlabel("Investment share of GDP")
    ax.set_ylabel("log GDP per worker")
    ax.set_title("Investment and income in the Solow cross-section")
    fig.tight_layout()
    path = Path(output_dir) / "solow_investment_vs_income.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_convergence(convergence_panel: pd.DataFrame, output_dir=OUTPUT_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = convergence_panel["oecd"].map({True: "darkorange", False: "steelblue"})
    ax.scatter(np.log(convergence_panel["initial_output_per_worker"]), 100 * convergence_panel["annualized_growth"], c=colors, alpha=0.75)
    ax.set_xlabel("log initial GDP per worker")
    ax.set_ylabel("Annualized growth, percent")
    ax.set_title("Conditional convergence: OECD vs world")
    fig.tight_layout()
    path = Path(output_dir) / "solow_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_brazil_growth_accounting(accounting: pd.DataFrame, output_dir=OUTPUT_DIR):
    sample = accounting.dropna(subset=["output_growth"]).tail(25)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample.index, sample["output_growth_pct"], color="black", lw=2.0, label="Output growth")
    ax.bar(sample.index, sample["capital_contribution_pct"], alpha=0.5, label="Capital")
    ax.bar(sample.index, sample["labor_contribution_pct"], bottom=sample["capital_contribution_pct"], alpha=0.5, label="Labor")
    ax.bar(
        sample.index,
        sample["tfp_contribution_pct"],
        bottom=sample["capital_contribution_pct"] + sample["labor_contribution_pct"],
        alpha=0.5,
        label="TFP",
    )
    ax.set_title("Brazil growth accounting contributions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent")
    ax.legend(ncol=4)
    fig.tight_layout()
    path = Path(output_dir) / "solow_brazil_growth_accounting.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    countries = sorted(set(OECD_COUNTRIES + ["BRA", "CHN", "IND", "MEX", "ZAF"]))
    cross_country = fetch_cross_country_solow_panel(countries)
    convergence = build_convergence_panel(cross_country)
    accounting, metadata = build_brazil_growth_accounting()

    output_paths = [
        plot_output_per_worker(cross_country),
        plot_investment_vs_income(cross_country),
        plot_convergence(convergence),
        plot_brazil_growth_accounting(accounting),
    ]
    metadata["sources"].append(
        metadata_entry(
            source="World Bank API",
            frequency="annual",
            unit="PPP output, labor force, percent of GDP",
            fallback="Penn World Tables",
            note="Cross-country comparison block.",
        )
    )
    write_metadata(metadata, OUTPUT_DIR / "solow_empirics_metadata.json")
    accounting.to_csv(OUTPUT_DIR / "brazil_growth_accounting.csv")
    convergence.to_csv(OUTPUT_DIR / "convergence_panel.csv", index=False)
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
