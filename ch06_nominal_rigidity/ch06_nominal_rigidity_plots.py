"""Editorial plot scripts for the Nominal Rigidity models (Chapter 6)."""

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

from ch06_nominal_rigidity.ch06_nominal_rigidity import (
    AggregateSupplyDemand,
    CalvoModel,
    MenuCostModel,
)
from data_utils import ensure_directory
from params import NK, NR, clone_params
from plotting_style import (
    COLORS,
    add_callout,
    direct_label_last,
    finalize_figure,
    plain_number_formatter,
    style_axis,
    style_legend,
)


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def _pct_formatter(decimals: int = 2) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:.{decimals}f}%")


# ---------------------------------------------------------------------------
# 1. Menu-cost diagram
# ---------------------------------------------------------------------------


def plot_menu_cost_diagram(model: MenuCostModel, output_dir=FIGURES_DIR):
    """2-panel: profit gain parabola vs menu cost; price rigidity region."""
    figure_path = Path(output_dir) / "ch06_menu_cost.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ---- Left: profit gain parabola vs menu cost threshold ----
    p_dev = np.linspace(-0.15, 0.15, 400)
    gains = model.profit_gain(p_dev)

    menu_costs = [0.002, 0.005, 0.010]
    mc_colors = [COLORS["line_compare"], COLORS["line_neutral"], COLORS["negative"]]

    axes[0].plot(p_dev, gains, color=COLORS["line_main"], linewidth=2.4,
                 label=r"$G(p_{dev}) = \frac{\eta-1}{2}\,p_{dev}^2$")
    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")

    for z, col in zip(menu_costs, mc_colors):
        thr = model.adjustment_threshold(z)
        axes[0].axhline(z, color=col, linewidth=1.6, linestyle="--",
                        label=f"$z = {z:.3f}$  (thr = {thr:.3f})")
        axes[0].axvline(thr, color=col, linewidth=1.0, linestyle=":", alpha=0.7)
        axes[0].axvline(-thr, color=col, linewidth=1.0, linestyle=":", alpha=0.7)

    axes[0].fill_between(
        p_dev, 0.0, gains,
        where=(np.abs(p_dev) < model.adjustment_threshold(menu_costs[1])),
        color=COLORS["negative"], alpha=0.15,
        label="Região de inércia (z médio)",
    )

    style_axis(axes[0],
               xlabel=r"Desvio de preço  $p_{dev} = \log p - \log p^*$",
               ylabel=r"Ganho de lucro  $G(p_{dev})$")
    axes[0].xaxis.set_major_formatter(plain_number_formatter(2))
    axes[0].yaxis.set_major_formatter(plain_number_formatter(4))
    axes[0].set_title(r"Modelo de custo de menu: $G$ vs $p_{dev}$",
                      fontsize=10.2, pad=7, color=COLORS["text"])
    style_legend(axes[0], loc="upper center")

    add_callout(
        axes[0],
        text="Não ajusta\n(rigidez)",
        xy=(0.0, model.profit_gain(0.0) + 0.0003),
        dx=14, dy=16,
        color=COLORS["negative"], text_color=COLORS["negative"],
        with_connector=False,
    )

    # ---- Right: price rigidity region in (menu_cost, demand_shock) space ----
    z_grid = np.linspace(0.0001, 0.02, 200)
    d_grid = np.linspace(-0.15, 0.15, 200)
    adjusts = model.price_rigidity_region(z_grid, d_grid)

    ZZ, DD = np.meshgrid(d_grid, z_grid)
    axes[1].contourf(ZZ, DD, adjusts.astype(float),
                     levels=[0.5, 1.5],
                     colors=[COLORS["positive"]], alpha=0.45)
    axes[1].contourf(ZZ, DD, (1 - adjusts).astype(float),
                     levels=[0.5, 1.5],
                     colors=[COLORS["negative"]], alpha=0.25)

    # Threshold boundary curve
    d_boundary = np.sqrt(2.0 * z_grid / (model.eta - 1.0))
    axes[1].plot(d_boundary, z_grid, color=COLORS["line_main"], linewidth=2.2,
                 label="Fronteira de ajuste")
    axes[1].plot(-d_boundary, z_grid, color=COLORS["line_main"], linewidth=2.2)

    style_axis(axes[1],
               xlabel=r"Choque de demanda  $\Delta m$",
               ylabel=r"Custo de menu  $z$")
    axes[1].xaxis.set_major_formatter(plain_number_formatter(2))
    axes[1].yaxis.set_major_formatter(plain_number_formatter(3))
    axes[1].set_title("Região de rigidez de preços",
                      fontsize=10.2, pad=7, color=COLORS["text"])

    legend_handles = [
        Line2D([0], [0], color=COLORS["line_main"], linewidth=2.2,
               label="Fronteira de ajuste"),
        Line2D([0], [0], color=COLORS["positive"], linewidth=8, alpha=0.6,
               label="Ajusta (G ≥ z)"),
        Line2D([0], [0], color=COLORS["negative"], linewidth=8, alpha=0.4,
               label="Inércia (G < z)"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right",
                   frameon=False, fontsize=9.0)

    add_callout(
        axes[1],
        text="Externalidade\nde demanda\nagregada",
        xy=(0.0, float(z_grid[100])),
        dx=10, dy=20,
        color=COLORS["axis"], text_color=COLORS["axis"],
        with_connector=False,
    )

    return finalize_figure(
        fig,
        figure_path,
        title="Cap. 6: Modelo de Custo de Menu (Mankiw 1985)",
        subtitle=(
            "Esquerda: ganho de lucro $G(p_{dev})$ vs limiares de ajuste para diferentes $z$. "
            "Direita: região de rigidez de preço no espaço (choque de demanda, custo de menu)."
        ),
        note=(
            f"Elasticidade de demanda η = {model.eta:.1f}; "
            "markup = η/(η-1) = "
            f"{model.markup:.3f}. "
            "Verde = firma ajusta preço; vermelho = inércia ótima."
        ),
        top=0.84,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# 2. Calvo pricing and NKPC
# ---------------------------------------------------------------------------


def plot_calvo_nkpc(model: CalvoModel, output_dir=FIGURES_DIR):
    """2-panel: Calvo weight decay; NKPC for different alpha values."""
    figure_path = Path(output_dir) / "ch06_calvo_nkpc.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ---- Left: Calvo weight alpha^s vs time since last adjustment ----
    s_vals = np.arange(0, 25)
    alpha_values = [0.60, 0.75, 0.85, 0.90]
    alpha_colors = [
        COLORS["line_main"],
        COLORS["line_compare"],
        COLORS["line_neutral"],
        COLORS["positive"],
    ]

    for alpha, col in zip(alpha_values, alpha_colors):
        m_temp = CalvoModel({"alpha_calvo": alpha, "beta": model.beta,
                              "sigma": model.sigma, "omega": model.omega})
        weights = m_temp.calvo_weight(s_vals)
        kappa_temp = m_temp.nkpc_slope()
        axes[0].plot(s_vals, weights, color=col, linewidth=2.2,
                     label=fr"$\alpha={alpha}$ (κ={kappa_temp:.3f})")

    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(axes[0],
               xlabel=r"Períodos desde último ajuste  $s$",
               ylabel=r"Peso de Calvo  $\alpha^s$")
    axes[0].yaxis.set_major_formatter(plain_number_formatter(2))
    axes[0].xaxis.set_major_formatter(plain_number_formatter(0))
    axes[0].set_title(r"Probabilidade de preço ainda vigente: $\alpha^s$",
                      fontsize=10.2, pad=7, color=COLORS["text"])
    style_legend(axes[0], loc="upper right")

    add_callout(
        axes[0],
        text="Maior α → preços\nmais rígidos",
        xy=(12, float(CalvoModel({"alpha_calvo": 0.90, "beta": model.beta,
                                   "sigma": model.sigma, "omega": model.omega})
                      .calvo_weight(12))),
        dx=10, dy=16,
        color=COLORS["positive"], text_color=COLORS["positive"],
    )

    # ---- Right: NKPC (pi vs x) for different alpha values ----
    x_range = np.linspace(-0.04, 0.04, 300)
    pi_expect = 0.0

    for alpha, col in zip(alpha_values, alpha_colors):
        m_temp = CalvoModel({"alpha_calvo": alpha, "beta": model.beta,
                              "sigma": model.sigma, "omega": model.omega})
        kappa_temp = m_temp.nkpc_slope()
        pi_vals = model.beta * pi_expect + kappa_temp * x_range
        axes[1].plot(x_range * 100, pi_vals * 100, color=col, linewidth=2.2,
                     label=fr"$\alpha={alpha}$ (κ={kappa_temp:.3f})")

    axes[1].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    axes[1].axvline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")

    style_axis(axes[1],
               xlabel="Hiato do produto  $x_t$  (%)",
               ylabel=r"Inflação  $\pi_t$  (%)")
    axes[1].xaxis.set_major_formatter(_pct_formatter(1))
    axes[1].yaxis.set_major_formatter(_pct_formatter(2))
    axes[1].set_title(r"CPNK: $\pi_t = \beta\,\mathbb{E}[\pi_{t+1}] + \kappa\,x_t$",
                      fontsize=10.2, pad=7, color=COLORS["text"])
    style_legend(axes[1], loc="upper left")

    add_callout(
        axes[1],
        text=r"Inclinação $= \kappa$" "\n(maior α → menor κ)",
        xy=(float(x_range[250] * 100),
            float(model.beta * 0.0 + model.nkpc_slope() * x_range[250] * 100)),
        dx=-60, dy=16,
        color=COLORS["line_compare"], text_color=COLORS["line_compare"],
        ha="right",
    )

    return finalize_figure(
        fig,
        figure_path,
        title="Cap. 6: Precificação de Calvo e Curva de Phillips Novo-Keynesiana",
        subtitle=(
            "Esquerda: probabilidade de preço ainda vigente após $s$ períodos. "
            "Direita: inclinação da CPNK para diferentes graus de rigidez nominal."
        ),
        note=(
            f"β = {model.beta:.2f}, σ = {model.sigma:.1f}, ω = {model.omega:.1f}. "
            "κ = (1-α)(1-αβ)/α · (ω + 1/σ). "
            "Maior α (mais rigidez) → CPNK mais plana (menor κ)."
        ),
        top=0.84,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# 3. AD-AS diagram
# ---------------------------------------------------------------------------


def plot_ad_as(model: AggregateSupplyDemand, output_dir=FIGURES_DIR):
    """2-panel: AD-AS with demand shock; AD-AS with supply shock."""
    figure_path = Path(output_dir) / "ch06_ad_as.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    x_range = np.linspace(-0.06, 0.06, 300)
    pi_range = np.linspace(-0.04, 0.04, 300)

    def _draw_baseline(ax, pi_expect=0.0, demand=0.0):
        as_pi = model.as_curve(x_range, pi_expect=pi_expect)
        ad_x = model.ad_curve(pi_range, demand=demand)
        ax.plot(x_range * 100, as_pi * 100,
                color=COLORS["line_main"], linewidth=2.2, label="OA (CPNK)")
        ax.plot(ad_x * 100, pi_range * 100,
                color=COLORS["line_compare"], linewidth=2.2, label="DA")

    # ---- Left panel: demand shock ----
    _draw_baseline(axes[0])
    x0, pi0 = model.equilibrium()

    demand_shock = 0.03
    x1, pi1 = model.equilibrium(demand_shock=demand_shock)
    ad_x_shock = model.ad_curve(pi_range, demand=demand_shock)
    axes[0].plot(ad_x_shock * 100, pi_range * 100,
                 color=COLORS["line_compare"], linewidth=2.0, linestyle="--",
                 label=f"DA' (choque +{demand_shock*100:.0f}%)")

    axes[0].scatter([x0 * 100], [pi0 * 100],
                    color=COLORS["highlight"], s=70,
                    edgecolors=COLORS["paper"], linewidths=0.9, zorder=5)
    axes[0].scatter([x1 * 100], [pi1 * 100],
                    color=COLORS["positive"], s=70,
                    edgecolors=COLORS["paper"], linewidths=0.9, zorder=5)

    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    axes[0].axvline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(axes[0],
               xlabel="Hiato do produto  $x_t$  (%)",
               ylabel=r"Inflação  $\pi_t$  (%)")
    axes[0].xaxis.set_major_formatter(_pct_formatter(1))
    axes[0].yaxis.set_major_formatter(_pct_formatter(2))
    axes[0].set_title("OA-DA: choque de demanda",
                      fontsize=10.2, pad=7, color=COLORS["text"])
    style_legend(axes[0], loc="upper right")
    add_callout(axes[0], text="E₀",
                xy=(x0 * 100, pi0 * 100), dx=10, dy=10,
                color=COLORS["highlight"], text_color=COLORS["highlight"])
    add_callout(axes[0], text="E₁",
                xy=(x1 * 100, pi1 * 100), dx=10, dy=10,
                color=COLORS["positive"], text_color=COLORS["positive"])

    # ---- Right panel: supply shock ----
    _draw_baseline(axes[1])

    supply_shock = 0.015
    x2, pi2 = model.equilibrium(supply_shock=supply_shock)
    as_pi_shock = model.as_curve(x_range, supply_shock=supply_shock)
    axes[1].plot(x_range * 100, as_pi_shock * 100,
                 color=COLORS["line_main"], linewidth=2.0, linestyle="--",
                 label=f"OA' (choque cost-push)")

    axes[1].scatter([x0 * 100], [pi0 * 100],
                    color=COLORS["highlight"], s=70,
                    edgecolors=COLORS["paper"], linewidths=0.9, zorder=5)
    axes[1].scatter([x2 * 100], [pi2 * 100],
                    color=COLORS["negative"], s=70,
                    edgecolors=COLORS["paper"], linewidths=0.9, zorder=5)

    axes[1].axhline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    axes[1].axvline(0.0, color=COLORS["axis_light"], linewidth=0.8, linestyle=":")
    style_axis(axes[1],
               xlabel="Hiato do produto  $x_t$  (%)",
               ylabel=r"Inflação  $\pi_t$  (%)")
    axes[1].xaxis.set_major_formatter(_pct_formatter(1))
    axes[1].yaxis.set_major_formatter(_pct_formatter(2))
    axes[1].set_title("OA-DA: choque de oferta (cost-push)",
                      fontsize=10.2, pad=7, color=COLORS["text"])
    style_legend(axes[1], loc="upper right")
    add_callout(axes[1], text="E₀",
                xy=(x0 * 100, pi0 * 100), dx=10, dy=10,
                color=COLORS["highlight"], text_color=COLORS["highlight"])
    add_callout(axes[1], text="E₁",
                xy=(x2 * 100, pi2 * 100), dx=10, dy=-14,
                color=COLORS["negative"], text_color=COLORS["negative"])

    return finalize_figure(
        fig,
        figure_path,
        title="Cap. 6: Diagrama Oferta Agregada – Demanda Agregada",
        subtitle=(
            "Esquerda: choque positivo de demanda eleva produto e inflação. "
            "Direita: choque de custo (cost-push) eleva inflação e reduz produto."
        ),
        note=(
            f"σ = {model.sigma:.1f} (inclinação da DA); "
            f"κ = {model.kappa:.2f} (inclinação da OA); "
            f"β = {model.beta:.2f}."
        ),
        top=0.84,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    mc_model = MenuCostModel()
    calvo_model = CalvoModel()
    ads_model = AggregateSupplyDemand()

    plots = [
        ("Menu Cost",   plot_menu_cost_diagram, mc_model),
        ("Calvo NKPC",  plot_calvo_nkpc,        calvo_model),
        ("AD-AS",       plot_ad_as,              ads_model),
    ]
    for name, func, model in plots:
        path = func(model)
        print(f"[{name}] {path}")
        print(f"[{name}] {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
