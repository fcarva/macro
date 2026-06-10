"""Editorial plot scripts for the investment / Tobin's q model (Chapter 9)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch09_investment.ch09_investment import AdjustmentCostFirm, TobinQModel
from data_utils import ensure_directory
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
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


# ---------------------------------------------------------------------------
# 1. (K, q) phase diagram with isoclines, vector field and saddle path
# ---------------------------------------------------------------------------


def plot_phase_diagram(model: TobinQModel | None = None, output_dir=FIGURES_DIR):
    """Phase diagram in (K, q) space: isoclines, quiver field, saddle path."""
    model = model or TobinQModel()
    data = model.phase_diagram_data()
    figure_path = Path(output_dir) / "investment_phase_diagram.png"
    ss = data["steady_state"]

    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    ax.quiver(data["K"], data["Q"], data["dK"], data["dQ"],
              color=COLORS["axis_light"], alpha=0.55, width=0.0028,
              pivot="mid")

    ax.plot(data["k_grid"], data["q_locus"], color=COLORS["line_compare"],
            linewidth=2.4, label=r"$\dot{q}=0$")
    ax.axhline(data["k_locus"], color=COLORS["line_main"], linewidth=2.4,
               label=r"$\dot{K}=0$  ($q=1+a\delta$)")

    saddle = data["saddle_path"]
    if not saddle.empty:
        ax.plot(saddle["K"], saddle["q"], color=COLORS["highlight"],
                linewidth=2.6, linestyle="--", label="Trajetória de sela")

    ax.scatter([ss["K_star"]], [ss["q_star"]], color=COLORS["text"], s=70,
               zorder=6, edgecolors=COLORS["paper"], linewidths=1.0)
    add_callout(
        ax,
        text=f"$(K^*, q^*)$ = ({ss['K_star']:.2f}, {ss['q_star']:.2f})",
        xy=(ss["K_star"], ss["q_star"]), dx=16, dy=18,
        color=COLORS["text"], text_color=COLORS["text"],
    )

    style_axis(ax, xlabel="Estoque de capital  $K$", ylabel="$q$ de Tobin")
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    ax.set_xlim(left=0)
    style_legend(ax, loc="upper right")

    return finalize_figure(
        fig,
        figure_path,
        title="Investimento: diagrama de fases $(K, q)$",
        subtitle=(
            r"$\dot{K}=0$ é horizontal em $q=1+a\delta$; "
            r"$\dot{q}=0$ é decrescente em $K$. "
            "O ponto de sela é o único equilíbrio estável."
        ),
        note=(
            f"r={model.r} · δ={model.delta} · a={model.a} · "
            f"α={model.alpha} · A={model.tfp}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 2. IRF — productivity (TFP) shock
# ---------------------------------------------------------------------------


def plot_irf_productivity(model: TobinQModel | None = None,
                           shock_size: float = 0.05, T: float = 40.0,
                           output_dir=FIGURES_DIR):
    """IRF of K, q and I/K after a permanent positive TFP shock."""
    model = model or TobinQModel()
    irf = model.irf("productivity", shock_size=shock_size, T=T)
    figure_path = Path(output_dir) / "investment_irf_productivity.png"

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2))
    time = irf["time"]

    panels = [
        ("K",   "Capital  $K_t$",          COLORS["line_main"], irf["old_steady_state"]["K_star"]),
        ("q",   "$q$ de Tobin",            COLORS["line_compare"], irf["old_steady_state"]["q_star"]),
        ("I_K", "Taxa de investimento  $I/K$", COLORS["line_neutral"], model.delta),
    ]

    for ax, (key, label, color, ref) in zip(axes, panels):
        ax.plot(time, irf[key], color=color, linewidth=2.4)
        ax.axhline(ref, color=COLORS["axis_light"], linewidth=0.9,
                   linestyle=":", alpha=0.85)
        style_axis(ax, xlabel="Tempo", ylabel=label)
        ax.xaxis.set_major_formatter(plain_number_formatter(0))
        ax.yaxis.set_major_formatter(plain_number_formatter(3))
        ax.margins(x=0.02)
        ax.set_title(label, fontsize=10.5, pad=6, color=COLORS["text"])

    return finalize_figure(
        fig,
        figure_path,
        title="Investimento: IRF — choque positivo de produtividade (TFP)",
        subtitle=(
            f"Choque permanente de +{shock_size*100:.0f}% em A. "
            "$q$ salta para cima no impacto, e $K$ converge gradualmente "
            "para o novo $K^*$, mais alto."
        ),
        note=(
            f"K*: {irf['old_steady_state']['K_star']:.3f} -> "
            f"{irf['new_steady_state']['K_star']:.3f}. "
            f"q*: {irf['old_steady_state']['q_star']:.3f} -> "
            f"{irf['new_steady_state']['q_star']:.3f} (q* não muda com A)."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 3. IRF — interest-rate shock
# ---------------------------------------------------------------------------


def plot_irf_interest(model: TobinQModel | None = None,
                       shock_size: float = 0.25, T: float = 40.0,
                       output_dir=FIGURES_DIR):
    """IRF of K, q and I/K after a permanent increase in r."""
    model = model or TobinQModel()
    irf = model.irf("interest", shock_size=shock_size, T=T)
    figure_path = Path(output_dir) / "investment_irf_interest.png"

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2))
    time = irf["time"]

    panels = [
        ("K",   "Capital  $K_t$",          COLORS["line_main"]),
        ("q",   "$q$ de Tobin",            COLORS["negative"]),
        ("I_K", "Taxa de investimento  $I/K$", COLORS["line_neutral"]),
    ]

    for ax, (key, label, color) in zip(axes, panels):
        ax.plot(time, irf[key], color=color, linewidth=2.4)
        style_axis(ax, xlabel="Tempo", ylabel=label)
        ax.xaxis.set_major_formatter(plain_number_formatter(0))
        ax.yaxis.set_major_formatter(plain_number_formatter(3))
        ax.margins(x=0.02)
        ax.set_title(label, fontsize=10.5, pad=6, color=COLORS["text"])

    return finalize_figure(
        fig,
        figure_path,
        title="Investimento: IRF — aumento permanente da taxa de juros $r$",
        subtitle=(
            f"Choque permanente de +{shock_size*100:.0f}% em $r$. "
            "$q$ cai no impacto (custo de oportunidade do capital sobe), "
            "e $K$ converge para um novo $K^*$ menor."
        ),
        note=(
            f"r: {model.r:.3f} -> {irf['new_model'].r:.3f}. "
            f"K*: {irf['old_steady_state']['K_star']:.3f} -> "
            f"{irf['new_steady_state']['K_star']:.3f}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 4. I/K vs q — linear relationship (firm-level)
# ---------------------------------------------------------------------------


def plot_investment_q_relation(firm: AdjustmentCostFirm | None = None,
                                output_dir=FIGURES_DIR):
    """The linear I/K = (q-1)/a relationship."""
    firm = firm or AdjustmentCostFirm()
    figure_path = Path(output_dir) / "investment_ik_vs_q.png"

    q_grid = np.linspace(0.8, 1.6, 100)
    _, i_k = firm.i_k_schedule(q_grid)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.plot(q_grid, i_k, color=COLORS["line_main"], linewidth=2.6)
    ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.9, linestyle=":")
    ax.axvline(1.0, color=COLORS["axis"], linewidth=1.2, linestyle="--", alpha=0.75)

    add_callout(
        ax,
        text="$q=1$: $I/K=0$\n(sem investimento líquido)",
        xy=(1.0, 0.0), dx=14, dy=-50,
        color=COLORS["axis"], text_color=COLORS["axis"],
    )
    add_callout(
        ax,
        text=f"Inclinação = $1/a = {1/firm.a:.2f}$",
        xy=(float(q_grid[-10]), float(i_k[-10])), dx=-110, dy=-10,
        color=COLORS["highlight"], text_color=COLORS["highlight"],
        with_connector=False,
    )

    style_axis(ax, xlabel="$q$ de Tobin (marginal)", ylabel="Taxa de investimento  $I/K$")
    ax.xaxis.set_major_formatter(plain_number_formatter(2))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))

    return finalize_figure(
        fig,
        figure_path,
        title="Investimento: relação linear $I/K = (q-1)/a$",
        subtitle=(
            "Custos de ajustamento convexos implicam que o investimento "
            "responde linearmente a $q$, com inclinação $1/a$."
        ),
        note=f"a={firm.a} · δ={firm.delta}. Pelo teorema de Hayashi, q marginal = q médio (Q de Tobin observável).",
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = TobinQModel(INVESTMENT)
    firm = AdjustmentCostFirm(INVESTMENT)

    plots = [
        ("Diagrama de fases", plot_phase_diagram, model, {}),
        ("IRF produtividade", plot_irf_productivity, model, {}),
        ("IRF juros", plot_irf_interest, model, {}),
        ("I/K vs q", plot_investment_q_relation, firm, {}),
    ]
    for name, func, obj, kwargs in plots:
        path = func(obj, **kwargs)
        print(f"[{name}] {path}")
        print(f"[{name}] {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
