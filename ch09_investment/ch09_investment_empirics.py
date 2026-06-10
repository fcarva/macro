"""Empirical illustration -- Tobin's q vs investment rate (Chapter 9).

Workflow:
1. Try to fetch firm-level (or aggregate) data on Tobin's q and the
   investment rate I/K from FRED (US nonfinancial corporate sector:
   market value of equity + liabilities over net worth, vs
   gross investment / capital stock).
2. If the fetch fails (no network access), fall back to a synthetic
   panel generated from the calibrated `AdjustmentCostFirm` model with
   added noise -- clearly labeled as synthetic.
3. Plot the scatter of I/K against q with the model-implied linear line
   I/K = (q-1)/a.

Reference: Romer Ch. 9; Hayashi (1982); FRED Z.1 Financial Accounts.
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

from ch09_investment.ch09_investment import AdjustmentCostFirm
from data_utils import ensure_directory, write_metadata
from params import INVESTMENT
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

# FRED series for the US nonfinancial corporate sector (Z.1 Financial Accounts)
FRED_MARKET_VALUE = "MVEONWMVBSNNCB"   # Market value of equities, % of net worth
FRED_INVESTMENT_RATE = "NCBIRGRQ027S"  # Gross private nonresidential fixed investment / capital


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def fetch_q_investment_panel(start: str = "1990-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """Fetch Tobin's q proxy and I/K from FRED via pandas-datareader."""
    import pandas_datareader.data as web

    q_series = web.DataReader(FRED_MARKET_VALUE, "fred", start, end)
    ik_series = web.DataReader(FRED_INVESTMENT_RATE, "fred", start, end)

    panel = q_series.join(ik_series, how="inner").dropna()
    panel.columns = ["q", "i_k"]
    panel["q"] = panel["q"] / 100.0
    panel["i_k"] = panel["i_k"] / 100.0
    return panel


def synthetic_q_investment_panel(firm: AdjustmentCostFirm, n: int = 150,
                                   seed: int = 0) -> pd.DataFrame:
    """Synthetic firm-level panel from the calibrated adjustment-cost model.

    q is drawn around 1 + a*delta with idiosyncratic noise; I/K follows the
    model's linear schedule plus measurement noise. Clearly labeled as
    synthetic (not observed data).
    """
    rng = np.random.default_rng(seed)
    q_star = 1.0 + firm.a * firm.delta
    q = q_star + rng.normal(0.0, 0.15, size=n)
    q = np.clip(q, 0.5, 2.0)
    _, i_k_model = firm.i_k_schedule(q)
    i_k = i_k_model + rng.normal(0.0, 0.01, size=n)
    return pd.DataFrame({"q": q, "i_k": i_k})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_q_investment_scatter(panel: pd.DataFrame, firm: AdjustmentCostFirm,
                               is_synthetic: bool, output_dir=OUTPUT_DIR):
    """Scatter of I/K vs q with the model-implied linear relationship."""
    figure_path = Path(output_dir) / "investment_q_scatter.png"

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.scatter(panel["q"], panel["i_k"], color=COLORS["line_main"], alpha=0.55,
               s=30, edgecolors=COLORS["paper"], linewidths=0.5, zorder=4)

    q_grid = np.linspace(panel["q"].min(), panel["q"].max(), 50)
    _, i_k_line = firm.i_k_schedule(q_grid)
    ax.plot(q_grid, i_k_line, color=COLORS["highlight"], linewidth=2.4,
            label=f"Modelo: $I/K=(q-1)/a$,  $a={firm.a}$")

    style_axis(ax, xlabel="$q$ de Tobin", ylabel="Taxa de investimento  $I/K$")
    ax.xaxis.set_major_formatter(plain_number_formatter(2))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    style_legend(ax, loc="upper left")

    if is_synthetic:
        add_callout(
            ax,
            text="DADOS SINTÉTICOS\n(gerados a partir do modelo calibrado)",
            xy=(float(panel["q"].max()), float(panel["i_k"].max())),
            dx=-160, dy=10,
            color=COLORS["negative"], text_color=COLORS["negative"],
            with_connector=False,
        )

    title = "Q de Tobin vs taxa de investimento" + (" (dados sintéticos)" if is_synthetic else " — EUA, setor não-financeiro")
    return finalize_figure(
        fig,
        figure_path,
        title=title,
        subtitle=(
            "Sob o teorema de Hayashi, a relação entre $I/K$ e $q$ é linear "
            "com inclinação $1/a$, onde $a$ governa o custo de ajustamento."
        ),
        source=("Painel sintético calibrado em ch09_investment.AdjustmentCostFirm."
                if is_synthetic else
                "FRED: MVEONWMVBSNNCB, NCBIRGRQ027S (Federal Reserve Z.1)."),
        note="Cada ponto representa uma observação (q, I/K).",
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    firm = AdjustmentCostFirm(INVESTMENT)
    print("Fetching Tobin's q / investment-rate data from FRED...")
    try:
        panel = fetch_q_investment_panel()
        if panel.empty:
            raise ValueError("Empty panel returned.")
        print(f"  Loaded {len(panel)} observations.")
        is_synthetic = False
    except Exception as exc:
        print(f"  Data fetch failed ({exc}); using synthetic panel.")
        panel = synthetic_q_investment_panel(firm)
        is_synthetic = True

    fig_path = plot_q_investment_scatter(panel, firm, is_synthetic)
    print(f"Saved {fig_path}")
    print(f"Saved {fig_path.with_suffix('.svg')}")

    metadata = {
        "title": "Tobin's q vs investment rate",
        "model_params": {"a": firm.a, "delta": firm.delta},
        "is_synthetic": is_synthetic,
        "n_obs": int(len(panel)),
        "source": ("synthetic" if is_synthetic
                    else "FRED MVEONWMVBSNNCB, NCBIRGRQ027S"),
    }
    write_metadata(metadata, OUTPUT_DIR / "investment_q_empirics_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
