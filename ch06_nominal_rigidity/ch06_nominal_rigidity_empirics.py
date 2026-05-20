"""Empirical calibration for Nominal Rigidity models using Brazilian data (Chapter 6).

Workflow:
1. Fetch Brazilian IPCA (inflation) from BCB SGS series 433.
2. Fetch Selic (nominal interest rate) from BCB SGS series 4189.
3. Compute realized vs. Calvo-model-implied inflation.
4. Plot inflation time series and compare to NKPC predictions.
5. Export metadata JSON.

Reference: Romer Ch. 6; Calvo (1983); Ball & Romer (1990).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch06_nominal_rigidity.ch06_nominal_rigidity import CalvoModel
from data_utils import ensure_directory, write_metadata
from params import NK, NR, clone_params
from plotting_style import (
    COLORS,
    add_callout,
    finalize_figure,
    plain_number_formatter,
    style_axis,
    style_legend,
)


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_directory(MODULE_DIR / "empirical_outputs")

# BCB SGS series codes
BCB_IPCA = 433       # IPCA — índice geral de inflação (% a.m.)
BCB_SELIC = 4189     # Selic acumulada no mês (% a.m.)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_bcb_series(series_id: int, start: str = "2000-01-01",
                     end: str = "2024-12-31") -> pd.Series:
    """Fetch a BCB SGS series by code.

    Args:
        series_id: BCB SGS series identifier.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        Monthly pd.Series indexed by date.
    """
    from bcb import sgs
    data = sgs.get({series_id: series_id}, start=start, end=end)
    return data[series_id].dropna()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_calvo_implied_inflation(
    model: CalvoModel,
    output_gap_proxy: np.ndarray,
    pi_expect: float = 0.0,
) -> np.ndarray:
    """Compute Calvo-NKPC implied inflation given output gap series.

    Uses backward induction:
        pi_t = beta * E[pi_{t+1}] + kappa * x_t

    Args:
        model: Calibrated CalvoModel.
        output_gap_proxy: Array of output gap estimates.
        pi_expect: Terminal inflation expectation.

    Returns:
        Array of NKPC-implied inflation.
    """
    return model.pi_dynamics(output_gap_proxy, pi0=pi_expect)


def compute_taylor_implied_rate(
    pi_series: pd.Series,
    r_bar: float = 0.0075,
    phi_pi: float = 1.5,
) -> pd.Series:
    """Simplified Taylor rule: i_t = r̄ + phi_pi * pi_t.

    Approximates a Taylor rule with no output gap term (data limitation).

    Args:
        pi_series: Monthly inflation rate (fraction, not percent).
        r_bar: Real neutral rate (monthly, default ≈ 0.75% p.m.).
        phi_pi: Inflation response coefficient.

    Returns:
        Taylor-implied nominal rate series.
    """
    return r_bar + phi_pi * pi_series


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_inflation_series(
    ipca: pd.Series,
    selic: pd.Series,
    output_dir=OUTPUT_DIR,
):
    """Plot IPCA and Selic time series for Brazil."""
    figure_path = Path(output_dir) / "ch06_brazil_inflation.png"
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True)
    fig.subplots_adjust(hspace=0.10)

    dates = ipca.index
    ipca_vals = ipca.values
    selic_vals = selic.reindex(dates).interpolate().values

    axes[0].fill_between(dates, 0, ipca_vals,
                         where=(ipca_vals >= 0),
                         color=COLORS["negative"], alpha=0.35, linewidth=0)
    axes[0].plot(dates, ipca_vals, color=COLORS["line_main"], linewidth=1.8,
                 alpha=0.92, label="IPCA (% a.m.)")
    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(axes[0], ylabel="IPCA  (% ao mês)", y_grid=True)
    axes[0].yaxis.set_major_formatter(plain_number_formatter(2))
    style_legend(axes[0], loc="upper right")

    axes[1].plot(dates, selic_vals, color=COLORS["line_compare"], linewidth=1.8,
                 alpha=0.92, label="Selic (% a.m.)")
    axes[1].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(axes[1], xlabel="Data", ylabel="Selic  (% ao mês)", y_grid=True)
    axes[1].yaxis.set_major_formatter(plain_number_formatter(2))
    style_legend(axes[1], loc="upper right")

    return finalize_figure(
        fig,
        figure_path,
        title="Brasil: IPCA e Selic — séries temporais",
        subtitle=(
            "IPCA: índice geral de preços (variação mensal). "
            "Selic: taxa nominal acumulada no mês."
        ),
        source="BCB SGS (séries 433 e 4189); cálculos do projeto.",
        note="Período: 2000–2024 (dados mensais).",
        top=0.87,
        bottom=0.08,
    )


def plot_taylor_comparison(
    selic: pd.Series,
    taylor_rate: pd.Series,
    output_dir=OUTPUT_DIR,
):
    """Plot actual Selic vs Taylor-implied rate."""
    figure_path = Path(output_dir) / "ch06_taylor_comparison.png"

    common_idx = selic.index.intersection(taylor_rate.index)
    selic_c = selic.loc[common_idx]
    taylor_c = taylor_rate.loc[common_idx]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(common_idx, selic_c.values,
            color=COLORS["line_main"], linewidth=1.9,
            label="Selic efetiva")
    ax.plot(common_idx, taylor_c.values,
            color=COLORS["line_compare"], linewidth=1.7, linestyle="--",
            label=r"Taxa Taylor implícita ($\bar{r} + \phi_\pi \pi_t$)")
    ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(ax, xlabel="Data", ylabel="Taxa de juros  (% ao mês)", y_grid=True)
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    style_legend(ax, loc="upper right")

    return finalize_figure(
        fig,
        figure_path,
        title="Brasil: Selic efetiva vs taxa de Taylor implícita",
        subtitle=(
            r"Regra de Taylor simplificada: $i_t = \bar{r} + \phi_\pi\,\pi_t$ "
            r"com $\phi_\pi = 1{,}5$ e $\bar{r} = 0{,}75\%$ a.m."
        ),
        source="BCB SGS (séries 433 e 4189); cálculos do projeto.",
        note="Desvios persistentes indicam resposta ao hiato do produto e outras variáveis.",
        top=0.87,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = CalvoModel()
    print("Fetching Brazilian IPCA and Selic from BCB SGS...")

    try:
        ipca = fetch_bcb_series(BCB_IPCA, start="2000-01-01", end="2024-12-31")
        selic = fetch_bcb_series(BCB_SELIC, start="2000-01-01", end="2024-12-31")
        print(f"  IPCA: {len(ipca)} observations  |  Selic: {len(selic)} observations.")
        data_available = True
    except Exception as exc:
        print(f"  Data fetch failed ({exc}); generating model-only metadata.")
        ipca = pd.Series(dtype=float)
        selic = pd.Series(dtype=float)
        data_available = False

    if data_available:
        # Convert from % to fraction for model comparison
        ipca_frac = ipca / 100.0
        selic_frac = selic / 100.0

        # Taylor rule comparison
        taylor_rate = compute_taylor_implied_rate(ipca_frac, r_bar=0.0075, phi_pi=1.5)

        fig_path = plot_inflation_series(ipca, selic)
        print(f"Saved {fig_path}")
        print(f"Saved {fig_path.with_suffix('.svg')}")

        fig_path2 = plot_taylor_comparison(selic_frac, taylor_rate)
        print(f"Saved {fig_path2}")
        print(f"Saved {fig_path2.with_suffix('.svg')}")

        # NKPC calibration summary
        kappa = model.nkpc_slope()
        mean_ipca = float(ipca_frac.mean())
        std_ipca = float(ipca_frac.std())

        # Export data
        panel = pd.DataFrame({"ipca_pct": ipca, "selic_pct": selic}).dropna()
        panel.to_csv(OUTPUT_DIR / "ch06_brazil_inflation_panel.csv")

        calibration = {
            "alpha_calvo": model.alpha,
            "beta": model.beta,
            "sigma": model.sigma,
            "omega": model.omega,
            "nkpc_slope_kappa": float(kappa),
            "mean_ipca_monthly": mean_ipca,
            "std_ipca_monthly": std_ipca,
            "mean_selic_monthly": float(selic_frac.mean()),
            "n_obs": len(panel),
        }
    else:
        calibration = {
            "alpha_calvo": model.alpha,
            "beta": model.beta,
            "sigma": model.sigma,
            "omega": model.omega,
            "nkpc_slope_kappa": float(model.nkpc_slope()),
        }

    metadata = {
        "title": "Chapter 6 — Nominal Rigidity empirical calibration for Brazil",
        "bcb_series": {"IPCA": BCB_IPCA, "Selic": BCB_SELIC},
        "data_available": data_available,
        "model": "CalvoModel",
        "calibration": calibration,
        "source": "BCB SGS series 433 (IPCA) and 4189 (Selic)",
    }
    write_metadata(metadata, OUTPUT_DIR / "ch06_nominal_rigidity_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
