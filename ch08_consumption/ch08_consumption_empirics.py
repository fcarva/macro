"""Empirical test — Campbell-Mankiw "rule of thumb" for Brazil (Chapter 8).

Workflow:
1. Fetch quarterly real consumption ("Despesa de consumo das familias") and
   GDP ("PIB a precos de mercado") volume indices from IBGE SIDRA (CNT 1846).
2. Compute log growth rates Delta c_t and Delta y_t.
3. Estimate the Campbell-Mankiw regression Delta c_t = (1-lambda)*const +
   lambda*Delta y_t + eps_t via OLS.
4. Compare the estimated lambda to the model's calibrated value and plot
   the scatter with the fitted line.

Reference: Romer Ch. 8; Campbell & Mankiw (1989); IBGE SIDRA table 1846.
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

from ch08_consumption.ch08_consumption import CampbellMankiw
from data_utils import (
    ensure_directory,
    fetch_brazil_cnt_1846_quarterly,
    filter_tidy_series,
    write_metadata,
)
from params import CONSUMPTION
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

SERIES_CONSUMPTION = "93404"   # Despesa de consumo das familias
SERIES_GDP = "90707"           # PIB a precos de mercado


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def fetch_consumption_income_panel() -> pd.DataFrame:
    """Quarterly log growth of household consumption and GDP (Brazil)."""
    tidy = fetch_brazil_cnt_1846_quarterly()
    sub = filter_tidy_series(tidy, [SERIES_CONSUMPTION, SERIES_GDP])
    if sub.empty:
        return pd.DataFrame()

    pivot = sub.pivot_table(index="period", columns="series_id", values="value")
    pivot = pivot.sort_index()
    pivot = pivot.rename(columns={SERIES_CONSUMPTION: "consumption", SERIES_GDP: "gdp"})
    pivot = pivot.dropna()

    panel = pd.DataFrame(index=pivot.index[1:])
    panel["dc"] = np.diff(np.log(pivot["consumption"].to_numpy()))
    panel["dy"] = np.diff(np.log(pivot["gdp"].to_numpy()))
    return panel


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_campbell_mankiw(panel: pd.DataFrame, est: dict, model: CampbellMankiw,
                          output_dir=OUTPUT_DIR):
    """Scatter Delta c vs Delta y with the Campbell-Mankiw fitted line."""
    figure_path = Path(output_dir) / "consumption_campbell_mankiw_brazil.png"

    if panel.empty or not est:
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        ax.text(0.5, 0.5, "Dados não disponíveis", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color=COLORS["axis"])
        style_axis(ax)
        return finalize_figure(
            fig, figure_path,
            title="Campbell-Mankiw para o Brasil — dados indisponíveis",
            top=0.88, bottom=0.10,
        )

    dy = panel["dy"].to_numpy() * 100.0
    dc = panel["dc"].to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.scatter(dy, dc, color=COLORS["line_main"], alpha=0.70, s=40,
               edgecolors=COLORS["paper"], linewidths=0.6, zorder=4)

    x_grid = np.linspace(dy.min(), dy.max(), 50)
    y_fit = est["intercept"] * 100.0 + est["lambda_hat"] * x_grid
    ax.plot(x_grid, y_fit, color=COLORS["highlight"], linewidth=2.4,
            label=f"Ajuste OLS: $\\hat\\lambda$ = {est['lambda_hat']:.2f}")

    ax.plot(x_grid, x_grid, color=COLORS["axis_light"], linewidth=1.2,
            linestyle="--", alpha=0.75, label="$\\Delta c = \\Delta y$ (45°)")

    style_axis(ax, xlabel="Crescimento do PIB  $\\Delta y_t$  (%, t/t-1)",
               ylabel="Crescimento do consumo  $\\Delta c_t$  (%, t/t-1)")
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    style_legend(ax, loc="upper left")

    add_callout(
        ax,
        text=(
            f"$\\hat\\lambda$ = {est['lambda_hat']:.2f} (R²={est['r_squared']:.2f})\n"
            f"Calibração do modelo: λ = {model.lam:.2f}\n"
            f"N = {est['n_obs']} trimestres"
        ),
        xy=(float(dy[np.argmax(dy)]), float(dc[np.argmax(dy)])),
        dx=-150, dy=20,
        color=COLORS["highlight"], text_color=COLORS["highlight"],
        with_connector=False,
    )

    return finalize_figure(
        fig,
        figure_path,
        title="Brasil: teste de Campbell-Mankiw (sensibilidade excessiva)",
        subtitle=(
            "Se PIH/Hall valesse, $\\hat\\lambda \\approx 0$. "
            "$\\hat\\lambda > 0$ indica famílias 'rule-of-thumb' que consomem "
            "a renda corrente."
        ),
        source="IBGE SIDRA, tabela 1846 (Contas Nacionais Trimestrais).",
        note="Estimação OLS: Delta c_t = (1-lambda) const + lambda * Delta y_t + eps_t.",
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = CampbellMankiw(CONSUMPTION)
    print("Fetching Brazilian consumption/GDP data (IBGE SIDRA 1846)...")
    try:
        panel = fetch_consumption_income_panel()
        if panel.empty:
            raise ValueError("Empty panel returned.")
        print(f"  Loaded {len(panel)} quarterly observations.")
        est = model.estimate_lambda(panel["dc"].to_numpy(), panel["dy"].to_numpy())
        data_available = True
    except Exception as exc:
        print(f"  Data fetch failed ({exc}); saving model-only outputs.")
        panel = pd.DataFrame()
        est = {}
        data_available = False

    fig_path = plot_campbell_mankiw(panel, est, model)
    print(f"Saved {fig_path}")
    print(f"Saved {fig_path.with_suffix('.svg')}")

    metadata = {
        "title": "Campbell-Mankiw empirical test — Brazil IBGE data",
        "model_params": {
            "lambda_calibrated": model.lam,
            "sigma_y": CONSUMPTION["sigma_y"],
        },
        "data_available": data_available,
        "estimated": {k: v for k, v in est.items() if not hasattr(v, "__len__")},
        "source": "IBGE SIDRA tabela 1846 (series 93404, 90707)",
    }
    write_metadata(metadata, OUTPUT_DIR / "consumption_brazil_empirics_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
