"""Empirical calibration — Taylor rule estimation for Brazil (Chapter 7).

Workflow:
1. Fetch IPCA (inflation) from BCB SGS 433 and Selic from BCB SGS 4189.
2. Fetch output gap proxy from BCB SGS 4380 (IBC-Br).
3. Estimate Taylor rule: i_t = r̄ + phi_pi*pi_t + phi_x*x_t via OLS.
4. Compare estimated coefficients to the calibrated NK model.
5. Plot actual Selic vs Taylor-implied rate.

Reference: Romer Ch. 7; Taylor (1993); BCB Focus Report.
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

from ch07_dsge_nk.ch07_nk import NKModel
from data_utils import ensure_directory, write_metadata
from params import NK, clone_params
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

# BCB SGS codes
SGS_IPCA = 433        # IPCA monthly (% a.m.)
SGS_SELIC = 4189      # Selic daily rate (% a.a.) — end of period
SGS_IBC_BR = 24364    # IBC-Br volume index (monthly activity proxy)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _fetch_bcb_series(codes: dict[str, int], start: str, end: str) -> pd.DataFrame:
    """Fetch BCB SGS series via python-bcb."""
    from bcb import sgs
    frames = []
    for name, code in codes.items():
        try:
            s = sgs.get({name: code}, start=start, end=end)
            frames.append(s)
        except Exception as exc:
            print(f"  Warning: could not fetch {name} (code {code}): {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_brazil_taylor_panel(
    start: str = "2003-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Download Selic, IPCA, and IBC-Br; resample to quarterly."""
    raw = _fetch_bcb_series(
        {"selic": SGS_SELIC, "ipca": SGS_IPCA, "ibc_br": SGS_IBC_BR},
        start=start, end=end,
    )
    if raw.empty:
        return pd.DataFrame()

    # Annualise IPCA (monthly → % a.a. equivalent)
    if "ipca" in raw.columns:
        raw["ipca_ann"] = (np.power(1.0 + raw["ipca"] / 100.0, 12) - 1.0) * 100.0

    # IBC-Br output gap: HP-detrend log series
    if "ibc_br" in raw.columns:
        log_ibc = np.log(raw["ibc_br"].dropna())
        if len(log_ibc) >= 4:
            from scipy.sparse import diags as sparse_diags
            n = len(log_ibc)
            lam = 1600.0
            diag_vals = [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)]
            D = sparse_diags(diag_vals, offsets=[0, 1, 2], shape=(n - 2, n)).toarray()
            A_mat = np.eye(n) + lam * D.T @ D
            trend = np.linalg.solve(A_mat, log_ibc.values)
            gap = pd.Series(
                log_ibc.values - trend,
                index=log_ibc.index,
                name="output_gap",
            )
            raw["output_gap"] = gap * 100.0   # percent

    # Resample to quarterly
    quarterly = raw.resample("QE").mean()
    quarterly.index = quarterly.index.to_period("Q")
    quarterly = quarterly.dropna(subset=["selic"])
    return quarterly


# ---------------------------------------------------------------------------
# OLS Taylor rule estimation
# ---------------------------------------------------------------------------


def estimate_taylor_rule(panel: pd.DataFrame) -> dict:
    """Estimate i_t = a0 + phi_pi*pi_t + phi_x*x_t via OLS (numpy).

    Returns:
        dict with intercept, phi_pi, phi_x, r_squared, n_obs.
    """
    cols_needed = {"selic", "ipca_ann", "output_gap"}
    available = cols_needed.intersection(panel.columns)
    if len(available) < 2:
        return {}

    sub = panel[list(available)].dropna()
    if len(sub) < 10:
        return {}

    y = sub["selic"].to_numpy()
    regs = [np.ones(len(sub))]
    col_names = ["intercept"]
    if "ipca_ann" in sub.columns:
        regs.append(sub["ipca_ann"].to_numpy())
        col_names.append("phi_pi")
    if "output_gap" in sub.columns:
        regs.append(sub["output_gap"].to_numpy())
        col_names.append("phi_x")

    X = np.column_stack(regs)
    coeffs, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result = dict(zip(col_names, coeffs.tolist()))
    result["r_squared"] = r2
    result["n_obs"] = len(sub)
    result["y_hat"] = y_hat
    result["dates"] = sub.index
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_taylor_rule(panel: pd.DataFrame, taylor: dict,
                     output_dir=OUTPUT_DIR):
    """Actual Selic vs Taylor-implied rate."""
    figure_path = Path(output_dir) / "nk_brazil_taylor_rule.png"

    if panel.empty or not taylor:
        fig, ax = plt.subplots(figsize=(12.0, 5.0))
        ax.text(0.5, 0.5, "Dados não disponíveis", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color=COLORS["axis"])
        style_axis(ax)
        return finalize_figure(fig, figure_path,
                               title="Regra de Taylor para o Brasil — dados indisponíveis",
                               top=0.88, bottom=0.10)

    dates = taylor["dates"]
    x_plot = np.arange(len(dates))

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Upper: Selic actual vs Taylor-implied
    selic_actual = panel.loc[dates, "selic"].to_numpy()
    selic_taylor = taylor["y_hat"]

    axes[0].plot(x_plot, selic_actual, color=COLORS["line_main"],
                 linewidth=2.0, label="Selic (realizada)")
    axes[0].plot(x_plot, selic_taylor, color=COLORS["line_compare"],
                 linewidth=1.8, linestyle="--", alpha=0.90,
                 label="Selic Taylor (estimada)")
    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(axes[0], ylabel="Taxa Selic  (% a.a.)", y_grid=True)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    style_legend(axes[0], loc="upper right")

    # Lower: inflation
    if "ipca_ann" in panel.columns:
        ipca = panel.loc[dates, "ipca_ann"].to_numpy()
        axes[1].plot(x_plot, ipca, color=COLORS["negative"],
                     linewidth=1.8, alpha=0.90)
        axes[1].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
        axes[1].axhline(4.5, color=COLORS["axis_light"], linewidth=1.0,
                        linestyle="--", alpha=0.70)
        add_callout(axes[1], text="Meta 4,5%", xy=(x_plot[-1], 4.5),
                    dx=-40, dy=6,
                    color=COLORS["axis_light"], text_color=COLORS["axis"],
                    with_connector=False)
        style_axis(axes[1], xlabel="Trimestre",
                   ylabel="IPCA  (% a.a., equiv.)", y_grid=True)
        axes[1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

        tick_step = max(1, len(x_plot) // 8)
        tick_pos = x_plot[::tick_step]
        tick_lab = [str(dates[i]) for i in tick_pos if i < len(dates)]
        axes[1].set_xticks(tick_pos[:len(tick_lab)])
        axes[1].set_xticklabels(tick_lab, rotation=30, ha="right", fontsize=8.5)

    phi_pi = taylor.get("phi_pi", float("nan"))
    phi_x  = taylor.get("phi_x", float("nan"))
    r2     = taylor.get("r_squared", float("nan"))
    n_obs  = taylor.get("n_obs", "?")

    return finalize_figure(
        fig,
        figure_path,
        title="Brasil: estimação da Regra de Taylor (2003–2024)",
        subtitle=(
            f"φπ estimado = {phi_pi:.2f}; φx estimado = {phi_x:.2f}  "
            f"(R² = {r2:.2f}, N = {n_obs} trimestres). "
            "Linha tracejada = taxa implícita pela regra estimada."
        ),
        source="BCB SGS 4189 (Selic), 433 (IPCA), 24364 (IBC-Br); cálculos do projeto.",
        note=(
            "Estimação OLS. Taxa Selic = r̄ + φπ·π + φx·x. "
            "IBC-Br dessazonalizado pelo filtro HP (λ=1.600)."
        ),
        top=0.87,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = NKModel(NK)
    print("Fetching Brazilian macro data for Taylor rule estimation...")
    try:
        panel = fetch_brazil_taylor_panel()
        if panel.empty:
            raise ValueError("Empty panel returned.")
        print(f"  Loaded {len(panel)} quarterly observations.")
        taylor = estimate_taylor_rule(panel)
        data_available = bool(taylor)
    except Exception as exc:
        print(f"  Data fetch failed ({exc}); saving model-only outputs.")
        panel = pd.DataFrame()
        taylor = {}
        data_available = False

    fig_path = plot_taylor_rule(panel, taylor)
    print(f"Saved {fig_path}")
    print(f"Saved {fig_path.with_suffix('.svg')}")

    # Metadata
    metadata = {
        "title": "NK empirical Taylor rule — Brazil BCB data",
        "model_params": {
            "beta": model.beta,
            "sigma": model.sigma,
            "kappa": model.kappa,
            "phi_pi": model.phi_pi,
            "phi_x": model.phi_x,
            "rho_v": model.rho_v,
            "rho_u": model.rho_u,
        },
        "data_available": data_available,
        "estimated_taylor": {k: v for k, v in taylor.items()
                              if not hasattr(v, "__len__")},
        "source": "BCB SGS 4189 (Selic), 433 (IPCA), 24364 (IBC-Br)",
    }
    write_metadata(metadata, OUTPUT_DIR / "nk_brazil_empirics_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
