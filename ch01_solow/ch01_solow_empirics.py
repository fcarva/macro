"""Empirical extension for the Solow model with official Brazil data."""

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
from data_utils import (
    BRAZIL_SOURCE_MAP,
    aggregate_quarterly_to_annual,
    compute_validation_residuals,
    ensure_directory,
    fetch_brazil_cna_6784_annual,
    fetch_brazil_cnt_1620_quarterly,
    fetch_brazil_cnt_1846_quarterly,
    fetch_brazil_scn_annual_current,
    fetch_world_bank_panel,
    filter_tidy_series,
    metadata_entry,
    write_metadata,
)
from params import BRASIL, OECD_COUNTRIES, SOLOW, clone_params
from plotting_style import (
    COLORS,
    SERIES_COLORS,
    add_country_badge,
    finalize_figure,
    place_country_end_labels,
    percent_formatter,
    plain_number_formatter,
    style_axis,
    style_legend,
    year_formatter,
)


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_directory(MODULE_DIR / "empirical_outputs")

BRAZIL_CURRENT_CODES = ["90707", "93404", "93406"]
BRAZIL_CNA_CODES = ["9808", "9810", "9811", "9812", "93"]


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


def annotate_series(frame: pd.DataFrame, model_use: str, concept_note: str, proxy_used=None, aggregation=None):
    annotated = frame.copy()
    annotated["model_use"] = model_use
    annotated["concept_note"] = concept_note
    if proxy_used is not None:
        annotated["proxy_used"] = proxy_used
    if aggregation is not None:
        annotated["aggregation"] = aggregation
    return annotated


def build_brazil_solow_metadata():
    return {
        "title": "Brazil empirical layer for Solow with official IBGE and BCB references",
        "brazil": {
            "labor_input": "population",
            "capital_input": "perpetual inventory on annualized FBCF volume index",
            "future_extensions": BRAZIL_SOURCE_MAP["future_extensions"],
            "warnings": [
                "Labor uses resident population as a long-window proxy. Migration to PEA or occupied persons via PNAD remains planned.",
                "Annual SCN validation is available from 2000 onward because the pinned annual workbook reference starts in 2000.",
            ],
            "sources": [
                metadata_entry(
                    source="IBGE SIDRA",
                    frequency="annual",
                    unit="1_000_000 BRL, percent, BRL, thousand persons",
                    dataset_id="CNA 6784",
                    series_code="9808,9810,9811,9812,93",
                    proxy_used="labor_input=population",
                    aggregation="annual official series",
                    concept_note="Population is used as a long-window labor proxy for Solow until PNAD-based labor inputs are added.",
                ),
                metadata_entry(
                    source="IBGE SIDRA",
                    frequency="quarterly to annual aggregation",
                    unit="1_000_000 BRL",
                    dataset_id="CNT 1846",
                    series_code="585 / 11255(90707,93404,93406)",
                    aggregation="quarterly_to_annual_sum",
                    concept_note="Quarterly current-price flows are annualized for SCN-CNT validation and for the pre-2000 FBCF fallback window.",
                ),
                metadata_entry(
                    source="IBGE SIDRA",
                    frequency="quarterly to annual aggregation",
                    unit="index, base mean 1995=100",
                    dataset_id="CNT 1620",
                    series_code="583 / 11255(90707,93404,93406)",
                    aggregation="quarterly_to_annual_mean",
                    concept_note="Quarterly volume indices are annualized to keep growth accounting in real terms.",
                ),
                metadata_entry(
                    source="IBGE SCN annual workbook",
                    frequency="annual",
                    unit="1_000_000 BRL",
                    dataset_id="SCN 2023 tab05",
                    series_code="tab05:B:total,consumo_familias,fbcf:valor_corrente",
                    aggregation="annual official reference",
                    revision_reference="Sistema de Contas Nacionais 2023",
                    concept_note="Annual SCN tab05 values are used as the official reference for validation against aggregated quarterly CNT flows.",
                ),
            ],
        },
        "cross_country": {
            "sources": [
                metadata_entry(
                    source="World Bank API",
                    frequency="annual",
                    unit="PPP output, labor force, percent of GDP",
                    fallback="Penn World Tables",
                    note="Cross-country comparison block.",
                )
            ]
        },
    }


def build_brazil_solow_inputs(start_year=1996, end_year=2023):
    cna_annual = filter_tidy_series(fetch_brazil_cna_6784_annual(), BRAZIL_CNA_CODES)
    cnt_current_quarterly = filter_tidy_series(fetch_brazil_cnt_1846_quarterly(), BRAZIL_CURRENT_CODES)
    cnt_volume_quarterly = filter_tidy_series(fetch_brazil_cnt_1620_quarterly(), BRAZIL_CURRENT_CODES)
    scn_annual_current = filter_tidy_series(fetch_brazil_scn_annual_current(), BRAZIL_CURRENT_CODES)

    cna_annual = cna_annual.loc[cna_annual["period"].astype(int).between(start_year, end_year)].copy()
    cnt_current_annual = aggregate_quarterly_to_annual(cnt_current_quarterly, aggregation="sum")
    cnt_current_annual = cnt_current_annual.loc[cnt_current_annual["period"].astype(int).between(start_year, end_year)].copy()
    cnt_volume_annual = aggregate_quarterly_to_annual(cnt_volume_quarterly, aggregation="mean")
    cnt_volume_annual = cnt_volume_annual.loc[cnt_volume_annual["period"].astype(int).between(start_year, end_year)].copy()
    scn_annual_current = scn_annual_current.loc[
        scn_annual_current["period"].astype(int).between(max(start_year, 2000), end_year)
    ].copy()

    population = (
        cna_annual.loc[cna_annual["series_id"] == "93", ["period", "value"]]
        .rename(columns={"value": "labor"})
        .assign(period=lambda frame: frame["period"].astype(int))
    )
    output_real = (
        cnt_volume_annual.loc[cnt_volume_annual["series_id"] == "90707", ["period", "value"]]
        .rename(columns={"value": "output"})
        .assign(period=lambda frame: frame["period"].astype(int))
    )
    investment_real = (
        cnt_volume_annual.loc[cnt_volume_annual["series_id"] == "93406", ["period", "value"]]
        .rename(columns={"value": "investment_real"})
        .assign(period=lambda frame: frame["period"].astype(int))
    )

    model_panel = (
        output_real.merge(investment_real, on="period", how="inner")
        .merge(population, on="period", how="inner")
        .set_index("period")
        .sort_index()
    )
    model_panel["capital"] = perpetual_inventory(model_panel["investment_real"], depreciation_rate=BRASIL["delta"])

    model = SolowModel(clone_params(SOLOW, BRASIL))
    accounting = model.growth_accounting(model_panel[["output", "capital", "labor"]], alpha=BRASIL["alpha"])
    accounting.index = accounting.index.astype(int)
    accounting.index.name = "year"

    validation = compute_validation_residuals(
        annualized_quarterly=cnt_current_annual.loc[cnt_current_annual["period"].astype(int) >= 2000],
        annual_official=scn_annual_current,
    )

    official_series = pd.concat(
        [
            annotate_series(
                cna_annual,
                model_use="brazil_annual_context",
                concept_note="Annual official Brazil macro context from CNA 6784.",
                aggregation="annual_official",
            ),
            annotate_series(
                cnt_current_annual,
                model_use="scn_cnt_validation",
                concept_note="Quarterly CNT current-price flows annualized by sum.",
                aggregation="quarterly_to_annual_sum",
            ),
            annotate_series(
                cnt_volume_annual,
                model_use="solow_real_quantities",
                concept_note="Quarterly CNT volume indices annualized by mean for real-series proxies.",
                aggregation="quarterly_to_annual_mean",
            ),
            annotate_series(
                scn_annual_current,
                model_use="scn_annual_reference",
                concept_note="Annual official SCN tab05 current-price reference for GDP, household consumption, and FBCF.",
                aggregation="annual_official",
            ),
        ],
        ignore_index=True,
    ).sort_values(["dataset_id", "series_id", "period"])

    return accounting, official_series, validation, build_brazil_solow_metadata()


def plot_output_per_worker(panel: pd.DataFrame, output_dir=OUTPUT_DIR):
    sample = panel.loc[panel["countryiso3code"].isin(["BRA", "USA", "KOR", "CHN", "MEX"])].copy()
    figure_path = Path(output_dir) / "solow_output_per_worker.png"

    fig, ax = plt.subplots(figsize=(9.6, 5.9))
    series_for_labels = []
    for code, group in sample.groupby("countryiso3code"):
        ordered = group.sort_values("date")
        values = ordered["output_per_worker"] / 1_000.0
        color = SERIES_COLORS.get(code, COLORS["line_neutral"])
        ax.plot(ordered["date"], values, color=color)
        series_for_labels.append(
            {
                "code": code,
                "x": float(ordered["date"].iloc[-1]),
                "y": float(values.iloc[-1]),
                "color": color,
            }
        )

    style_axis(
        ax,
        xlabel="Ano",
        ylabel="PIB por trabalhador (US$ mil PPC)",
        y_grid=True,
        integer_x=True,
    )
    ax.yaxis.set_major_formatter(plain_number_formatter(0))
    ax.xaxis.set_major_formatter(year_formatter())
    ax.margins(x=0.04)
    place_country_end_labels(ax, series_for_labels, min_gap_frac=0.1, x_pad_frac=0.1)
    return finalize_figure(
        fig,
        figure_path,
        title="PIB por trabalhador em países selecionados",
        subtitle="Série anual desde 1990, em milhares de dólares PPC por trabalhador.",
        source="World Bank API; cálculos do projeto.",
        note="PPC = paridade do poder de compra.",
    )


def plot_investment_vs_income(panel: pd.DataFrame, output_dir=OUTPUT_DIR):
    latest = panel.sort_values("date").groupby("countryiso3code").tail(1)
    highlight_codes = {"BRA", "USA", "KOR", "CHN", "MEX"}
    figure_path = Path(output_dir) / "solow_investment_vs_income.png"
    badge_offsets = {
        "BRA": (10, 10),
        "USA": (10, 12),
        "KOR": (10, 12),
        "CHN": (12, 12),
        "MEX": (12, -14),
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.scatter(
        latest["investment_share"],
        latest["output_per_worker"] / 1_000.0,
        s=38,
        color=COLORS["muted"],
        alpha=0.55,
        edgecolors="none",
    )

    highlighted = latest.loc[latest["countryiso3code"].isin(highlight_codes)].copy()
    for _, row in highlighted.iterrows():
        color = SERIES_COLORS.get(row["countryiso3code"], COLORS["highlight"])
        x_value = float(row["investment_share"])
        y_value = float(row["output_per_worker"] / 1_000.0)
        ax.scatter(
            x_value,
            y_value,
            s=62,
            color=color,
            zorder=4,
        )
        dx, dy = badge_offsets.get(row["countryiso3code"], (8, 8))
        add_country_badge(
            ax,
            x=x_value,
            y=y_value,
            code=row["countryiso3code"],
            color=color,
            dx=dx,
            dy=dy,
            with_connector=True,
            connector_color=color,
        )

    style_axis(
        ax,
        xlabel="Taxa de investimento (% do PIB)",
        ylabel="PIB por trabalhador (US$ mil PPC, escala log)",
    )
    ax.xaxis.set_major_formatter(percent_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(0))
    ax.set_yscale("log")
    return finalize_figure(
        fig,
        figure_path,
        title="Investimento e renda no corte internacional do Solow",
        subtitle="Último ano disponível por país; o eixo vertical está em escala logarítmica para acomodar a heterogeneidade entre economias.",
        source="World Bank API; cálculos do projeto.",
    )


def plot_convergence(convergence_panel: pd.DataFrame, output_dir=OUTPUT_DIR):
    figure_path = Path(output_dir) / "solow_convergence.png"
    colors = convergence_panel["oecd"].map({True: COLORS["line_compare"], False: COLORS["line_main"]})
    x_values = convergence_panel["initial_output_per_worker"].to_numpy()
    y_values = 100 * convergence_panel["annualized_growth"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.scatter(
        convergence_panel["initial_output_per_worker"] / 1_000.0,
        y_values,
        c=colors,
        s=42,
        alpha=0.72,
        edgecolors="none",
    )
    if len(convergence_panel) >= 3:
        slope, intercept = np.polyfit(np.log(x_values), y_values, deg=1)
        x_line = np.geomspace(x_values.min(), x_values.max(), 120)
        ax.plot(
            x_line / 1_000.0,
            intercept + slope * np.log(x_line),
            color=COLORS["axis"],
            linestyle=":",
            linewidth=1.6,
            label="Tendência",
        )

    brazil_row = convergence_panel.loc[convergence_panel["countryiso3code"] == "BRA"]
    if not brazil_row.empty:
        row = brazil_row.iloc[0]
        add_country_badge(
            ax,
            x=float(row["initial_output_per_worker"] / 1_000.0),
            y=float(100 * row["annualized_growth"]),
            code="BRA",
            color=SERIES_COLORS["BRA"],
            dx=10,
            dy=10,
            with_connector=True,
            connector_color=COLORS["axis"],
        )

    style_axis(
        ax,
        xlabel="PIB inicial por trabalhador em 2000 (US$ mil PPC, escala log)",
        ylabel="Crescimento anualizado 2000-2023 (%)",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(percent_formatter(0))
    ax.scatter([], [], color=COLORS["line_main"], label="Não OCDE")
    ax.scatter([], [], color=COLORS["line_compare"], label="OCDE")
    style_legend(ax, loc="upper right", ncol=1)
    return finalize_figure(
        fig,
        figure_path,
        title="Convergência condicional: OCDE versus resto do mundo",
        subtitle="Países com observações válidas em 2000 e 2023; a relação negativa sugere maior crescimento entre economias inicialmente mais pobres.",
        source="World Bank API; cálculos do projeto.",
        note="A linha pontilhada resume a tendência média da amostra.",
    )


def plot_brazil_growth_accounting(accounting: pd.DataFrame, output_dir=OUTPUT_DIR):
    sample = accounting.dropna(subset=["output_growth"]).tail(25)
    years = sample.index.to_numpy()
    figure_path = Path(output_dir) / "solow_brazil_growth_accounting.png"
    components = [
        ("capital_contribution_pct", "Capital", COLORS["line_neutral"]),
        ("labor_contribution_pct", "Trabalho", COLORS["line_compare"]),
        ("tfp_contribution_pct", "PTF", COLORS["positive"]),
    ]

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    positive_bottom = np.zeros(len(sample))
    negative_bottom = np.zeros(len(sample))

    for column, label, color in components:
        values = sample[column].to_numpy(dtype=float)
        positive = np.where(values > 0, values, 0)
        negative = np.where(values < 0, values, 0)
        ax.bar(years, positive, bottom=positive_bottom, width=0.78, color=color, alpha=0.88, label=label)
        ax.bar(years, negative, bottom=negative_bottom, width=0.78, color=color, alpha=0.88)
        positive_bottom += positive
        negative_bottom += negative

    ax.plot(years, sample["output_growth_pct"], color=COLORS["black"], linewidth=2.5, label="Crescimento do PIB", zorder=4)
    style_axis(
        ax,
        xlabel="Ano",
        ylabel="Crescimento do PIB e contribuicoes (%, p.p.)",
        zero_line=True,
        integer_x=True,
    )
    tick_years = [year for year in years if year % 2 == 0 or year == years[-1]]
    ax.set_xticks(tick_years)
    ax.xaxis.set_major_formatter(year_formatter())
    ax.yaxis.set_major_formatter(percent_formatter(0))
    style_legend(ax, loc="upper left", ncol=4)
    return finalize_figure(
        fig,
        figure_path,
        title="Brasil: decomposicao do crescimento do PIB",
        subtitle="Contribuicoes aproximadas de capital, trabalho e produtividade total dos fatores nos ultimos 25 anos com dados anuais oficiais.",
        source="IBGE SIDRA/SCN; calculos do projeto.",
        note="O trabalho usa populacao residente como proxy de janela longa; a PTF e o residuo de Solow.",
    )


def main():
    countries = sorted(set(OECD_COUNTRIES + ["BRA", "CHN", "IND", "MEX", "ZAF"]))
    cross_country = fetch_cross_country_solow_panel(countries)
    convergence = build_convergence_panel(cross_country)
    accounting, official_series, validation, metadata = build_brazil_solow_inputs()

    output_paths = [
        plot_output_per_worker(cross_country),
        plot_investment_vs_income(cross_country),
        plot_convergence(convergence),
        plot_brazil_growth_accounting(accounting),
    ]

    write_metadata(metadata, OUTPUT_DIR / "solow_empirics_metadata.json")
    accounting.to_csv(OUTPUT_DIR / "brazil_growth_accounting.csv", index=True)
    official_series.to_csv(OUTPUT_DIR / "brazil_official_series.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "brazil_validation_residuals.csv", index=False)
    convergence.to_csv(OUTPUT_DIR / "convergence_panel.csv", index=False)
    for path in output_paths:
        print(f"Saved {path}")
        print(f"Saved {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
