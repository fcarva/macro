"""Empirical calibration and moments comparison for the RBC model (Chapter 5).

Workflow:
1. Fetch quarterly volume indices (GDP, Consumption, Investment) from IBGE SIDRA CNT-1620.
2. Apply the Hodrick-Prescott filter (λ=1600 for quarterly data) to extract cycles.
3. Compute empirical second moments (std devs, relative volatilities, correlations).
4. Compare against model-simulated moments from RBCModel.
5. Export panel CSV, metadata JSON, and publication-quality figures.

Reference: Romer Ch. 5; Hodrick & Prescott (1997).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import diags as sparse_diags
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch05_rbc.ch05_rbc import RBCModel
from data_utils import (
    ensure_directory,
    fetch_brazil_cnt_1620_quarterly,
    filter_tidy_series,
    write_metadata,
    aggregate_quarterly_to_annual,
)
from params import BRASIL, RBC, clone_params
from plotting_style import (
    COLORS,
    add_callout,
    finalize_figure,
    format_number_ptbr,
    percent_formatter,
    plain_number_formatter,
    style_axis,
    style_legend,
)


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_directory(MODULE_DIR / "empirical_outputs")

CNT_SERIES = {
    "90707": "pib",
    "93404": "consumo",
    "93406": "investimento",
}


# ---------------------------------------------------------------------------
# Hodrick-Prescott filter
# ---------------------------------------------------------------------------


def hp_filter(series: np.ndarray, lam: float = 1600.0) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter.

    Args:
        series: 1-D array of the raw series (levels or logs).
        lam: Smoothing parameter (1600 for quarterly, 100 for annual).

    Returns:
        (trend, cycle) where cycle = series - trend.
    """
    n = len(series)
    if n < 4:
        raise ValueError("HP filter requires at least 4 observations.")

    # Second-difference matrix
    diag_vals = [
        np.ones(n - 2),
        -2.0 * np.ones(n - 2),
        np.ones(n - 2),
    ]
    D = sparse_diags(diag_vals, offsets=[0, 1, 2], shape=(n - 2, n)).toarray()
    A = np.eye(n) + lam * D.T @ D
    trend = np.linalg.solve(A, series)
    cycle = series - trend
    return trend, cycle


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def fetch_rbc_brazil_panel(start_year: int = 1996, end_year: int = 2024) -> pd.DataFrame:
    """Download and clean CNT-1620 quarterly volume indices for Y, C, I."""
    raw = fetch_brazil_cnt_1620_quarterly()
    series_ids = list(CNT_SERIES.keys())
    filtered = filter_tidy_series(raw, series_ids)

    # Parse quarter-period strings (e.g. "2010 1. trimestre") to date
    def parse_quarter(period_str: str) -> pd.Period | None:
        try:
            parts = str(period_str).split()
            year = int(parts[0])
            quarter = int(parts[1].replace(".", ""))
            return pd.Period(f"{year}Q{quarter}", freq="Q")
        except Exception:
            return None

    filtered["period_q"] = filtered["period"].apply(parse_quarter)
    filtered = filtered.dropna(subset=["period_q"])
    filtered["year"] = filtered["period_q"].apply(lambda p: p.year)
    filtered = filtered.loc[
        filtered["year"].between(start_year, end_year)
    ].copy()

    # Pivot to wide format
    rename_map = {sid: name for sid, name in CNT_SERIES.items()}
    wide = (
        filtered.pivot_table(
            index="period_q", columns="series_id", values="value", aggfunc="first"
        )
        .rename(columns=rename_map)
        .sort_index()
    )
    wide.index.name = "period"
    return wide.dropna()


def compute_cycles(panel: pd.DataFrame, lam: float = 1600.0) -> pd.DataFrame:
    """Apply HP filter to log of each series; return cycle components."""
    cycles = {}
    trends = {}
    for col in panel.columns:
        log_series = np.log(panel[col].to_numpy(dtype=float))
        trend, cycle = hp_filter(log_series, lam=lam)
        cycles[f"{col}_cycle"] = cycle
        trends[f"{col}_trend"] = trend

    result = pd.DataFrame({**cycles, **trends}, index=panel.index)
    return result


# ---------------------------------------------------------------------------
# Empirical moments
# ---------------------------------------------------------------------------


def compute_empirical_moments(cycles: pd.DataFrame) -> dict:
    """Compute standard RBC moments from HP-filtered cycle components."""
    y = cycles["pib_cycle"].to_numpy()
    c = cycles["consumo_cycle"].to_numpy()
    i = cycles["investimento_cycle"].to_numpy()

    std_y = float(np.std(y, ddof=1))
    std_c = float(np.std(c, ddof=1))
    std_i = float(np.std(i, ddof=1))

    corr_cy = float(np.corrcoef(c, y)[0, 1]) if std_c > 0 else np.nan
    corr_iy = float(np.corrcoef(i, y)[0, 1]) if std_i > 0 else np.nan
    autocorr_y = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if len(y) > 1 else np.nan

    return {
        "std_y": std_y,
        "std_c": std_c,
        "std_i": std_i,
        "rel_std_c": std_c / std_y if std_y > 0 else np.nan,
        "rel_std_i": std_i / std_y if std_y > 0 else np.nan,
        "corr_cy": corr_cy,
        "corr_iy": corr_iy,
        "autocorr_y": autocorr_y,
        "n_obs": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_rbc_brazil(model: RBCModel, empirical_moments: dict) -> dict:
    """Simple calibration summary comparing model and data moments."""
    model_moments = model.moments(T_sim=5000, n_draws=40, seed=0)
    return {
        "alpha": model.alpha,
        "beta": model.beta,
        "delta": model.delta,
        "rho_z": model.rho_z,
        "sigma_z": model.sigma_z,
        "model_rel_std_c": model_moments["rel_std_c"],
        "model_rel_std_i": model_moments["rel_std_i"],
        "model_corr_cy": model_moments["corr_cy"],
        "model_corr_iy": model_moments["corr_iy"],
        "model_autocorr_y": model_moments["autocorr_y"],
        "data_rel_std_c": empirical_moments["rel_std_c"],
        "data_rel_std_i": empirical_moments["rel_std_i"],
        "data_corr_cy": empirical_moments["corr_cy"],
        "data_corr_iy": empirical_moments["corr_iy"],
        "data_autocorr_y": empirical_moments["autocorr_y"],
        "data_n_obs": empirical_moments["n_obs"],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_hp_cycles(panel: pd.DataFrame, cycles: pd.DataFrame,
                   output_dir=OUTPUT_DIR):
    """Plot HP trend and cycle for Brazilian GDP, Consumption, Investment."""
    figure_path = Path(output_dir) / "rbc_brazil_hp_cycles.png"
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True)

    series_info = [
        ("pib", "PIB", COLORS["line_main"]),
        ("consumo", "Consumo", COLORS["line_compare"]),
        ("investimento", "Investimento", COLORS["line_neutral"]),
    ]

    x_dates = np.arange(len(cycles))

    for ax, (key, label, color) in zip(axes, series_info):
        cycle_vals = 100.0 * cycles[f"{key}_cycle"].to_numpy()
        ax.fill_between(x_dates, 0.0, cycle_vals,
                        where=cycle_vals >= 0,
                        color=COLORS["positive"], alpha=0.50, linewidth=0)
        ax.fill_between(x_dates, 0.0, cycle_vals,
                        where=cycle_vals < 0,
                        color=COLORS["negative"], alpha=0.50, linewidth=0)
        ax.plot(x_dates, cycle_vals, color=color, linewidth=1.5, alpha=0.9)
        ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.9, linestyle=":")
        style_axis(ax, ylabel=f"{label} — ciclo HP (%)", y_grid=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

        # Quarter tick labels (every 4 years)
        period_index = cycles.index
        tick_positions = [i for i, p in enumerate(period_index)
                          if p.quarter == 1 and p.year % 4 == 0]
        tick_labels = [str(period_index[i].year) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)

    axes[-1].set_xlabel("Trimestre", labelpad=8)

    return finalize_figure(
        fig,
        figure_path,
        title="Brasil: componentes cíclicos HP — PIB, Consumo e Investimento",
        subtitle=(
            "Filtro Hodrick-Prescott (λ=1.600) aplicado ao log dos índices de volume "
            "das Contas Nacionais Trimestrais (IBGE CNT-1620)."
        ),
        source="IBGE SIDRA CNT-1620; cálculos do projeto.",
        note=(
            "Verde = ciclo positivo (acima da tendência); "
            "vermelho = ciclo negativo. Série em log-desvio percentual."
        ),
        top=0.88,
        bottom=0.07,
    )


def plot_moments_comparison(calibration: dict, output_dir=OUTPUT_DIR):
    """Bar chart comparing model-simulated vs Brazilian data moments."""
    figure_path = Path(output_dir) / "rbc_brazil_moments_comparison.png"

    moments_labels = [
        ("rel_std_c", "σ(C)/σ(Y)"),
        ("rel_std_i", "σ(I)/σ(Y)"),
        ("corr_cy", "Corr(C,Y)"),
        ("corr_iy", "Corr(I,Y)"),
        ("autocorr_y", "Autocorr(Y)"),
    ]

    model_vals = [calibration.get(f"model_{k}", np.nan) for k, _ in moments_labels]
    data_vals = [calibration.get(f"data_{k}", np.nan) for k, _ in moments_labels]
    labels = [lab for _, lab in moments_labels]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bars_m = ax.bar(x - width / 2, model_vals, width,
                    color=COLORS["line_main"], alpha=0.88, label="Modelo RBC")
    bars_d = ax.bar(x + width / 2, data_vals, width,
                    color=COLORS["line_compare"], alpha=0.88, label="Brasil (HP)")

    ax.axhline(0.0, color=COLORS["axis"], linewidth=0.8)
    ax.axhline(1.0, color=COLORS["axis_light"], linewidth=1.0,
               linestyle="--", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    style_axis(ax, ylabel="Momento", y_grid=True)
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    style_legend(ax, loc="upper right")

    for bars, vals in [(bars_m, model_vals), (bars_d, data_vals)]:
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * np.sign(val),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8.8,
                    color=COLORS["text"],
                )

    n_obs = calibration.get("data_n_obs", "?")
    return finalize_figure(
        fig,
        figure_path,
        title="RBC: comparação de momentos — modelo vs. Brasil",
        subtitle=(
            "Modelo com choques de PTF calibrado para α=0,40, β=0,99, δ=0,025, "
            "ρ_z=0,95. Dados: CNT-1620 (IBGE), filtro HP, 1996–2024."
        ),
        source="IBGE SIDRA CNT-1620; cálculos do projeto.",
        note=f"N = {n_obs} trimestres. λ_HP = 1.600.",
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Use Brazil-calibrated parameters
    brazil_params = clone_params(RBC, {
        "alpha": BRASIL["alpha"],   # 0.40
        "delta": BRASIL["delta"],   # 0.05
    })
    model = RBCModel(brazil_params)

    print("Fetching Brazilian national accounts data...")
    try:
        panel = fetch_rbc_brazil_panel()
        print(f"  Loaded {len(panel)} quarterly observations.")
        cycles = compute_cycles(panel)
        empirical_moments = compute_empirical_moments(cycles)
        calibration = calibrate_rbc_brazil(model, empirical_moments)
        data_available = True
    except Exception as exc:
        print(f"  Data fetch failed ({exc}); saving model-only outputs.")
        panel = pd.DataFrame()
        cycles = pd.DataFrame()
        empirical_moments = {}
        calibration = {"model_" + k: v for k, v in model.moments().items()}
        data_available = False

    # Save panel
    if data_available:
        panel.to_csv(OUTPUT_DIR / "rbc_brazil_panel.csv")
        cycles.to_csv(OUTPUT_DIR / "rbc_brazil_hp_cycles.csv")
        fig_path = plot_hp_cycles(panel, cycles)
        print(f"Saved {fig_path}")
        print(f"Saved {fig_path.with_suffix('.svg')}")

    # Moments comparison
    fig_path_m = plot_moments_comparison(calibration)
    print(f"Saved {fig_path_m}")
    print(f"Saved {fig_path_m.with_suffix('.svg')}")

    # Metadata
    metadata = {
        "title": "RBC empirical calibration for Brazil — IBGE CNT-1620 and HP filter",
        "hp_lambda": 1600,
        "series": CNT_SERIES,
        "data_available": data_available,
        "calibration": calibration,
        "source": "IBGE SIDRA CNT-1620; Hodrick-Prescott filter (lambda=1600)",
    }
    write_metadata(metadata, OUTPUT_DIR / "rbc_brazil_empirics_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
