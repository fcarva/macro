"""Editorial plot scripts for the New Keynesian DSGE model (Chapter 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch07_dsge_nk.ch07_nk import NKModel
from data_utils import ensure_directory
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
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def _pct_formatter(decimals: int = 2) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")


def _pp_formatter(decimals: int = 2) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:.{decimals}f} p.p.")


# ---------------------------------------------------------------------------
# 1. IRF — monetary policy (demand) shock
# ---------------------------------------------------------------------------


def plot_irf_demand(model: NKModel, shock_size: float = 0.01,
                    T: int = 40, output_dir=FIGURES_DIR):
    """IRF of x, pi, i after a 1% contractionary monetary shock."""
    irf = model.irf("demand", shock_size=shock_size, T=T)
    figure_path = Path(output_dir) / "nk_irf_demand.png"
    scale = 100.0

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2))
    time = irf["time"]

    panels = [
        ("x",  "Hiato do produto  $x_t$",   COLORS["line_main"]),
        ("pi", "Inflação  $\\pi_t$",          COLORS["line_compare"]),
        ("i",  "Taxa nominal  $i_t$",          COLORS["line_neutral"]),
    ]

    for ax, (key, label, color) in zip(axes, panels):
        vals = scale * irf[key]
        ax.plot(time, vals, color=color, linewidth=2.4)
        ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.9,
                   linestyle=":", alpha=0.85)
        style_axis(ax, xlabel="Período", ylabel="Desvio (p.p.)")
        ax.xaxis.set_major_formatter(plain_number_formatter(0))
        ax.yaxis.set_major_formatter(_pp_formatter(3))
        ax.margins(x=0.05)
        ax.set_title(label, fontsize=10.5, pad=6, color=COLORS["text"])

        peak_idx = int(np.argmax(np.abs(vals)))
        ax.scatter([time[peak_idx]], [float(vals[peak_idx])],
                   color=COLORS["highlight"], s=46,
                   edgecolors=COLORS["paper"], linewidths=0.8, zorder=5)
        add_callout(
            ax,
            text=f"t=0: {float(vals[0]):.3f} p.p.",
            xy=(time[0], float(vals[0])),
            dx=14, dy=10 if vals[0] >= 0 else -14,
            color=color, text_color=color, with_connector=False,
        )

    return finalize_figure(
        fig,
        figure_path,
        title="NK: IRF — choque de política monetária contracionista",
        subtitle=(
            f"Choque de +{shock_size*100:.0f}% na regra de Taylor (v₀ = {shock_size:.2f}). "
            "Aperto monetário contrai x e π; taxa nominal sobe no impacto e reverte."
        ),
        note=(
            f"β={model.beta} · σ={model.sigma} · κ={model.kappa} · "
            f"φπ={model.phi_pi} · φx={model.phi_x} · ρv={model.rho_v}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 2. IRF — cost-push (supply) shock
# ---------------------------------------------------------------------------


def plot_irf_supply(model: NKModel, shock_size: float = 0.01,
                    T: int = 40, output_dir=FIGURES_DIR):
    """IRF of x, pi, i after a 1% cost-push shock."""
    irf = model.irf("supply", shock_size=shock_size, T=T)
    figure_path = Path(output_dir) / "nk_irf_supply.png"
    scale = 100.0

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2))
    time = irf["time"]

    panels = [
        ("x",  "Hiato do produto  $x_t$",   COLORS["line_main"]),
        ("pi", "Inflação  $\\pi_t$",          COLORS["negative"]),
        ("i",  "Taxa nominal  $i_t$",          COLORS["line_neutral"]),
    ]

    for ax, (key, label, color) in zip(axes, panels):
        vals = scale * irf[key]
        ax.plot(time, vals, color=color, linewidth=2.4)
        ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.9,
                   linestyle=":", alpha=0.85)
        style_axis(ax, xlabel="Período", ylabel="Desvio (p.p.)")
        ax.xaxis.set_major_formatter(plain_number_formatter(0))
        ax.yaxis.set_major_formatter(_pp_formatter(3))
        ax.margins(x=0.05)
        ax.set_title(label, fontsize=10.5, pad=6, color=COLORS["text"])
        add_callout(
            ax,
            text=f"t=0: {float(vals[0]):.3f} p.p.",
            xy=(time[0], float(vals[0])),
            dx=14, dy=10 if vals[0] >= 0 else -14,
            color=color, text_color=color, with_connector=False,
        )

    return finalize_figure(
        fig,
        figure_path,
        title="NK: IRF — choque de custo-push (oferta adversa)",
        subtitle=(
            f"Choque de +{shock_size*100:.0f}% no custo (u₀ = {shock_size:.2f}). "
            "Inflação sobe, produto cai (estagflação). "
            "BC reage subindo a taxa nominal."
        ),
        note=(
            f"β={model.beta} · σ={model.sigma} · κ={model.kappa} · "
            f"φπ={model.phi_pi} · φx={model.phi_x} · ρu={model.rho_u}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 3. Blanchard-Kahn determinacy region
# ---------------------------------------------------------------------------


def plot_determinacy_region(model: NKModel, output_dir=FIGURES_DIR):
    """Heatmap of (phi_pi, phi_x) determinacy region."""
    figure_path = Path(output_dir) / "nk_determinacy.png"

    phi_pi_grid = np.linspace(0.0, 3.5, 120)
    phi_x_grid = np.linspace(0.0, 2.5, 100)
    det_map = model.blanchard_kahn(phi_pi_grid, phi_x_grid)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    ax.contourf(phi_pi_grid, phi_x_grid, det_map.T.astype(float),
                levels=[0.5, 1.5], colors=[COLORS["positive"]], alpha=0.30)
    ax.contourf(phi_pi_grid, phi_x_grid, (~det_map).T.astype(float),
                levels=[0.5, 1.5], colors=[COLORS["negative"]], alpha=0.20)

    # Taylor principle boundary: phi_pi = 1 (approximate)
    ax.axvline(1.0, color=COLORS["axis"], linewidth=1.8, linestyle="--", alpha=0.80)
    add_callout(ax, text="Princípio de Taylor\n$\\phi_\\pi = 1$",
                xy=(1.0, 1.5), dx=14, dy=0,
                color=COLORS["axis"], text_color=COLORS["axis"])

    # Mark default parameters
    ax.scatter([model.phi_pi], [model.phi_x],
               color=COLORS["highlight"], s=80,
               edgecolors=COLORS["paper"], linewidths=0.9, zorder=6)
    add_callout(ax,
                text=f"Calibração base\n(φπ={model.phi_pi}, φx={model.phi_x})",
                xy=(model.phi_pi, model.phi_x),
                dx=18, dy=16,
                color=COLORS["highlight"], text_color=COLORS["highlight"])

    style_axis(ax, xlabel=r"Peso da inflação  $\phi_\pi$",
               ylabel=r"Peso do hiato  $\phi_x$")
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))

    legend_handles = [
        Line2D([0], [0], color=COLORS["positive"], linewidth=8,
               alpha=0.40, label="Determinado (REE único)"),
        Line2D([0], [0], color=COLORS["negative"], linewidth=8,
               alpha=0.30, label="Indeterminado (múltiplos REE)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9.5)

    return finalize_figure(
        fig,
        figure_path,
        title="NK: região de determinação de Blanchard-Kahn",
        subtitle=(
            "Verde = determinado (ambos |λ| > 1). Vermelho = indeterminado. "
            "O princípio de Taylor (φπ > 1) é necessário mas não suficiente."
        ),
        note=(
            f"β={model.beta} · σ={model.sigma} · κ={model.kappa}. "
            "Condição de determinação: ambos os valores próprios fora do círculo unitário."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 4. Stochastic simulation
# ---------------------------------------------------------------------------


def plot_simulation(model: NKModel, T: int = 120, seed: int = 42,
                    output_dir=FIGURES_DIR):
    """Stochastic simulation of the NK model."""
    sim = model.simulate(T=T, seed=seed, sigma_v=0.01, sigma_u=0.005)
    figure_path = Path(output_dir) / "nk_simulation.png"
    scale = 100.0
    time = sim["time"]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    series = [
        ("x",  "Hiato do produto  $x_t$",  COLORS["line_main"]),
        ("pi", "Inflação  $\\pi_t$",         COLORS["line_compare"]),
        ("i",  "Taxa nominal  $i_t$",         COLORS["line_neutral"]),
    ]

    for ax, (key, label, color) in zip(axes, series):
        vals = scale * sim[key]
        ax.plot(time, vals, color=color, linewidth=1.6, alpha=0.90)
        ax.fill_between(time, 0.0, vals, where=(vals >= 0),
                        color=color, alpha=0.12, linewidth=0)
        ax.fill_between(time, 0.0, vals, where=(vals < 0),
                        color=COLORS["negative"], alpha=0.10, linewidth=0)
        ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.9, linestyle=":")
        style_axis(ax, ylabel=f"{label}  (p.p.)", y_grid=True)
        ax.yaxis.set_major_formatter(_pp_formatter(2))
        ax.margins(x=0.02)

    axes[-1].set_xlabel("Período", labelpad=8)
    axes[-1].xaxis.set_major_formatter(plain_number_formatter(0))

    return finalize_figure(
        fig,
        figure_path,
        title="NK: simulação estocástica — hiato, inflação e taxa nominal",
        subtitle=(
            "Choques de demanda (σv=1%) e de custo-push (σu=0,5%) propagam "
            "flutuações correlacionadas. Regra de Taylor estabiliza o sistema."
        ),
        note=(
            f"φπ={model.phi_pi} · φx={model.phi_x} · "
            f"ρv={model.rho_v} · ρu={model.rho_u} · semente={seed}."
        ),
        top=0.87,
        bottom=0.07,
    )


# ---------------------------------------------------------------------------
# 5. Policy frontier  var(pi) × var(x)
# ---------------------------------------------------------------------------


def plot_policy_frontier(model: NKModel, output_dir=FIGURES_DIR):
    """Inflation-output variance trade-off frontier."""
    figure_path = Path(output_dir) / "nk_policy_frontier.png"

    phi_pi_range = np.linspace(1.05, 5.0, 25)
    frontier = model.policy_frontier(
        phi_pi_range=phi_pi_range,
        sigma_v=0.01, sigma_u=0.005,
        T_sim=3000, n_draws=20, seed=0,
    )

    var_pi = frontier["var_pi"] * 1e4   # in basis-points^2
    var_x  = frontier["var_x"]  * 1e4
    phi_pi = frontier["phi_pi"]
    mask = np.isfinite(var_pi) & np.isfinite(var_x)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    sc = ax.scatter(var_x[mask], var_pi[mask],
                    c=phi_pi[mask], cmap="RdYlGn_r",
                    s=60, zorder=4,
                    edgecolors=COLORS["paper"], linewidths=0.6)
    ax.plot(var_x[mask], var_pi[mask],
            color=COLORS["axis"], linewidth=1.6, alpha=0.60, zorder=3)

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"Peso da inflação  $\phi_\pi$", color=COLORS["axis"], fontsize=9.5)
    cb.ax.tick_params(colors=COLORS["axis"], labelsize=9)

    # Mark baseline
    idx_base = int(np.argmin(np.abs(phi_pi - model.phi_pi)))
    if mask[idx_base]:
        ax.scatter([var_x[idx_base]], [var_pi[idx_base]],
                   color=COLORS["highlight"], s=90,
                   edgecolors=COLORS["paper"], linewidths=0.9, zorder=6)
        add_callout(ax,
                    text=f"Calibração\nφπ={model.phi_pi}",
                    xy=(float(var_x[idx_base]), float(var_pi[idx_base])),
                    dx=14, dy=14,
                    color=COLORS["highlight"], text_color=COLORS["highlight"])

    style_axis(ax,
               xlabel=r"Var($x_t$)  ×10⁻⁴",
               ylabel=r"Var($\pi_t$)  ×10⁻⁴")
    ax.xaxis.set_major_formatter(plain_number_formatter(2))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))

    return finalize_figure(
        fig,
        figure_path,
        title="NK: fronteira de política — trade-off var(π) × var(x)",
        subtitle=(
            "Maior φπ reduz var(π) mas eleva var(x). "
            "A fronteira traça o conjunto de resultados ótimos de segundo momento."
        ),
        note=(
            f"σv={0.01} · σu={0.005} · T=3.000 · 20 replicações. "
            "φx fixo = {:.2f}.".format(model.phi_x)
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = NKModel(NK)
    plots = [
        ("IRF demanda",   plot_irf_demand,         {}),
        ("IRF oferta",    plot_irf_supply,          {}),
        ("Determinação",  plot_determinacy_region,  {}),
        ("Simulação",     plot_simulation,          {}),
        ("Fronteira",     plot_policy_frontier,     {}),
    ]
    for name, func, kwargs in plots:
        path = func(model, **kwargs)
        print(f"[{name}] {path}")
        print(f"[{name}] {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
