"""Editorial plot scripts for the Real Business Cycle model (Chapter 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch05_rbc.ch05_rbc import LaborLeisureConditions, RBCModel
from data_utils import ensure_directory
from params import RBC, clone_params
from plotting_style import (
    COLORS,
    add_callout,
    direct_label_last,
    finalize_figure,
    plain_number_formatter,
    percent_formatter,
    style_axis,
    style_legend,
)


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


# ---------------------------------------------------------------------------
# 1. Four-panel IRF (Lista I Q8)
# ---------------------------------------------------------------------------


def plot_irf(model: RBCModel, shock_size: float = 0.01,
             T: int = 40, output_dir=FIGURES_DIR):
    """Impulse-response functions for Y, C, I, K after a 1% TFP shock."""
    irf = model.irf(shock_size=shock_size, T=T)
    ss = model.steady_state()
    figure_path = Path(output_dir) / "rbc_irf.png"

    scale = 100.0   # log-deviations → percentage points

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    axes = axes.flatten()

    series = [
        ("y", r"Produto $Y$"),
        ("c", r"Consumo $C$"),
        ("i", r"Investimento $I$"),
        ("k", r"Capital $K$"),
    ]

    for ax, (key, label) in zip(axes, series):
        values = scale * irf[key]
        ax.plot(irf["time"], values, color=COLORS["line_main"], linewidth=2.4)
        ax.axhline(0.0, color=COLORS["axis_light"], linewidth=1.0,
                   linestyle=":", alpha=0.9)
        style_axis(ax, xlabel="Período", ylabel="Desvio do estado estacionário (%)")
        ax.xaxis.set_major_formatter(plain_number_formatter(0))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2f}%")
        )
        add_callout(
            ax,
            text=label,
            xy=(irf["time"][3], values[3]),
            dx=20, dy=10,
            color=COLORS["line_main"],
            text_color=COLORS["line_main"],
            with_connector=False,
        )
        # Peak annotation
        peak_idx = int(np.argmax(np.abs(values)))
        ax.scatter([irf["time"][peak_idx]], [values[peak_idx]],
                   color=COLORS["highlight"], s=42,
                   edgecolors=COLORS["paper"], linewidths=0.7, zorder=5)

    # Add z path to first panel as dashed overlay
    z_vals = scale * irf["z"]
    axes[0].plot(irf["time"], z_vals, color=COLORS["line_compare"],
                 linestyle="--", linewidth=1.6, alpha=0.85)
    add_callout(
        axes[0],
        text=r"TFP $z$",
        xy=(irf["time"][6], z_vals[6]),
        dx=14, dy=-14,
        color=COLORS["line_compare"],
        text_color=COLORS["line_compare"],
        with_connector=False,
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.84))
    return finalize_figure(
        fig,
        figure_path,
        title="RBC: funções de impulso-resposta a um choque de produtividade",
        subtitle=(
            f"Choque inicial de +{shock_size*100:.0f}% na PTF (z); "
            "desvios percentuais em relação ao estado estacionário."
        ),
        note=(
            "Modelo RBC com utilidade logarítmica e trabalho inelástico. "
            "Parâmetros: α=0,33; β=0,99; δ=0,025; ρ_z=0,95."
        ),
        top=0.84,
        bottom=0.09,
    )


# ---------------------------------------------------------------------------
# 2. Phase diagram in (k̂, ẑ) space
# ---------------------------------------------------------------------------


def plot_phase_diagram(model: RBCModel, output_dir=FIGURES_DIR):
    """Phase diagram showing the linear saddle-path policy manifold."""
    ll = model.log_linearize()
    a_k, a_z = ll["a_k"], ll["a_z"]
    A, B = ll["A"], ll["B"]

    k_range = np.linspace(-0.15, 0.15, 300)
    z_range = np.linspace(-0.05, 0.05, 300)
    KK, ZZ = np.meshgrid(k_range, z_range)

    # Policy surface: ĉ = a_k·k̂ + a_z·ẑ
    CC = a_k * KK + a_z * ZZ

    # Capital law for the vector field
    dK = A * KK + B * ZZ - KK   # K_{t+1} - K_t
    dZ = model.rho_z * ZZ - ZZ  # z_{t+1} - z_t
    norm = np.sqrt(dK ** 2 + dZ ** 2) + 1e-12

    figure_path = Path(output_dir) / "rbc_phase_diagram.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # Left panel: consumption policy surface (heat map)
    im = axes[0].contourf(KK, ZZ, CC, levels=20, cmap="coolwarm", alpha=0.85)
    axes[0].contour(KK, ZZ, CC, levels=[0.0], colors=[COLORS["black"]], linewidths=1.6)
    plt.colorbar(im, ax=axes[0], label="ĉ (desvio do consumo)", pad=0.02)
    axes[0].scatter([0], [0], color=COLORS["highlight"], s=62,
                    edgecolors=COLORS["paper"], linewidths=0.8, zorder=5)
    style_axis(axes[0],
               xlabel=r"Capital log-desvio $\hat{k}$",
               ylabel=r"PTF log-desvio $\hat{z}$")
    add_callout(axes[0], text="Estado\nestacionário", xy=(0, 0),
                dx=14, dy=14, color=COLORS["highlight"],
                text_color=COLORS["highlight"])
    axes[0].set_title("Política de consumo  ĉ = aₖ·k̂ + aᵤ·ẑ",
                      fontsize=10.5, pad=6, color=COLORS["text"])

    # Right panel: vector field + sample IRF trajectories
    axes[1].quiver(KK[::12, ::12], ZZ[::12, ::12],
                   dK[::12, ::12] / norm[::12, ::12],
                   dZ[::12, ::12] / norm[::12, ::12],
                   color=COLORS["muted"], alpha=0.55, scale=22,
                   headwidth=3, headlength=4)

    for shock in [0.03, 0.02, 0.01]:
        irf = model.irf(shock_size=shock, T=30)
        axes[1].plot(irf["k"], irf["z"], color=COLORS["line_main"],
                     linewidth=1.6, alpha=0.7 + 0.1 * (shock / 0.03))
        axes[1].scatter([irf["k"][0]], [irf["z"][0]],
                        color=COLORS["line_compare"], s=28,
                        edgecolors=COLORS["paper"], linewidths=0.6, zorder=5)

    axes[1].scatter([0], [0], color=COLORS["highlight"], s=62,
                    edgecolors=COLORS["paper"], linewidths=0.8, zorder=6)
    style_axis(axes[1],
               xlabel=r"Capital log-desvio $\hat{k}$",
               ylabel=r"PTF log-desvio $\hat{z}$")
    axes[1].set_title("Campo vetorial e trajetórias de convergência",
                      fontsize=10.5, pad=6, color=COLORS["text"])

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: diagrama de fase no espaço (k̂, ẑ)",
        subtitle=(
            "O coeficiente de transição A ≈ "
            f"{ll['A']:.3f} garante estabilidade local; "
            "ρ_z determina a persistência do choque."
        ),
        note="As trajetórias partem de choques de PTF de 1%, 2% e 3%.",
        top=0.85,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 3. Stochastic simulation (Lista I Q8)
# ---------------------------------------------------------------------------


def plot_stochastic_simulation(model: RBCModel, T: int = 120,
                                seed: int = 42, output_dir=FIGURES_DIR):
    """Time series from stochastic simulation: Y, C, I versus TFP path."""
    sim = model.simulate(T=T, seed=seed)
    figure_path = Path(output_dir) / "rbc_simulation.png"

    scale = 100.0
    times = sim["time"]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)

    # Upper: Y, C, I
    for key, label, color in [
        ("y", "Produto", COLORS["line_main"]),
        ("c", "Consumo", COLORS["line_compare"]),
        ("i", "Investimento", COLORS["line_neutral"]),
    ]:
        axes[0].plot(times, scale * sim[key], color=color,
                     linewidth=1.8 if key != "i" else 1.4,
                     linestyle="-" if key != "i" else "--", alpha=0.92)

    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.9,
                    linestyle=":", alpha=0.85)
    style_axis(axes[0], ylabel="Desvio do estado est. (%)", y_grid=True)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%")
    )
    direct_label_last(axes[0], times, scale * sim["y"],
                      label="Y", color=COLORS["line_main"], dx=8)
    direct_label_last(axes[0], times, scale * sim["c"],
                      label="C", color=COLORS["line_compare"], dx=8, dy=-8)
    direct_label_last(axes[0], times, scale * sim["i"],
                      label="I", color=COLORS["line_neutral"], dx=8, dy=8)

    # Lower: TFP z
    axes[1].fill_between(times, 0.0, scale * sim["z"],
                          where=sim["z"] >= 0,
                          color=COLORS["positive"], alpha=0.55, linewidth=0)
    axes[1].fill_between(times, 0.0, scale * sim["z"],
                          where=sim["z"] < 0,
                          color=COLORS["negative"], alpha=0.55, linewidth=0)
    axes[1].plot(times, scale * sim["z"],
                 color=COLORS["axis"], linewidth=1.3, alpha=0.9)
    axes[1].axhline(0.0, color=COLORS["axis_light"], linewidth=0.9, linestyle=":")
    style_axis(axes[1], xlabel="Período",
               ylabel=r"PTF $z$ — desvio (%)", y_grid=True)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.2f}%")
    )

    fig.tight_layout(rect=(0, 0.09, 1, 0.84))
    return finalize_figure(
        fig,
        figure_path,
        title="RBC: simulação estocástica",
        subtitle=(
            "Choques de PTF (AR(1), ρ_z=0,95) geram flutuações correlacionadas "
            "em Y, C e I, com investimento mais volátil."
        ),
        note=(
            f"σ_z = {model.sigma_z:.3f}; semente={seed}. "
            "Verde = PTF acima da tendência; vermelho = abaixo."
        ),
        top=0.84,
        bottom=0.09,
    )


# ---------------------------------------------------------------------------
# 4. Second-moment comparison (model vs calibration targets)
# ---------------------------------------------------------------------------


def plot_moments(model: RBCModel, data_moments: dict | None = None,
                 output_dir=FIGURES_DIR):
    """Bar chart comparing simulated and (optionally) empirical moments."""
    model_mom = model.moments(T_sim=5000, n_draws=40, seed=0)
    figure_path = Path(output_dir) / "rbc_moments.png"

    variables = ["C", "I", "K"]
    model_vals = [model_mom["rel_std_c"],
                  model_mom["rel_std_i"],
                  model_mom["rel_std_k"]]

    x = np.arange(len(variables))
    width = 0.35 if data_moments else 0.55

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars_m = ax.bar(x - width / 2 if data_moments else x,
                    model_vals, width,
                    color=COLORS["line_main"], alpha=0.88, label="Modelo RBC")

    if data_moments:
        data_vals = [
            data_moments.get("rel_std_c", np.nan),
            data_moments.get("rel_std_i", np.nan),
            data_moments.get("rel_std_k", np.nan),
        ]
        ax.bar(x + width / 2, data_vals, width,
               color=COLORS["line_compare"], alpha=0.88, label="Dados BR")

    ax.axhline(1.0, color=COLORS["axis_light"], linewidth=1.2,
               linestyle="--", alpha=0.8)
    add_callout(ax, text="σ(X)/σ(Y) = 1", xy=(x[-1], 1.0),
                dx=14, dy=8, color=COLORS["axis_light"],
                text_color=COLORS["axis"], with_connector=False)

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=11)
    style_axis(ax, ylabel="Volatilidade relativa ao PIB  [σ(X)/σ(Y)]")
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    style_legend(ax, loc="upper right")

    # Annotate bars
    bars_to_annotate = bars_m
    for bar, val in zip(bars_to_annotate, model_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=9.2, color=COLORS["text"])

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: momentos simulados — volatilidade relativa ao PIB",
        subtitle=(
            "O investimento é substancialmente mais volátil que o produto; "
            "consumo é menos volátil. Consistente com a teoria RBC."
        ),
        note=(
            "Simulação: T=5.000 períodos, 40 replicações. "
            "σ_z=0,007; ρ_z=0,95."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 5. Labour-leisure diagram (Lista I Q5–7)
# ---------------------------------------------------------------------------


def plot_labor_leisure(model: RBCModel, output_dir=FIGURES_DIR):
    """Diagram showing the consumption-leisure tradeoff and optimal condition."""
    ss = model.steady_state()
    llc = LaborLeisureConditions(b=llc_b := LaborLeisureConditions(b=2.0).calibrate_b(model))

    wage_grid = np.linspace(0.5 * ss["w_star"], 2.0 * ss["w_star"], 200)
    leisure_schedule = llc.leisure_grid(wage_grid, consumption=ss["c_star"])

    # Shock: higher b (more leisure preference)
    b_high = llc_b * 1.5
    llc_high = LaborLeisureConditions(b=b_high)
    leisure_high = llc_high.optimal_leisure(ss["c_star"], wage_grid)

    figure_path = Path(output_dir) / "rbc_labor_leisure.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # Left: labour supply schedule (w vs N)
    axes[0].plot(leisure_schedule["labor"], wage_grid / ss["w_star"],
                 color=COLORS["line_main"], linewidth=2.4, label=f"b = {llc_b:.2f}")
    labor_high = 1.0 - leisure_high
    axes[0].plot(np.clip(labor_high, 0, 1), wage_grid / ss["w_star"],
                 color=COLORS["line_compare"], linewidth=2.0, linestyle="--",
                 label=f"b = {b_high:.2f} (maior lazer)")
    axes[0].axhline(1.0, color=COLORS["axis_light"], linewidth=1.1,
                    linestyle=":", alpha=0.9)
    axes[0].axvline(1.0 - ss["c_star"] * llc_b / ss["w_star"],
                    color=COLORS["axis_light"], linewidth=1.0, linestyle=":",
                    alpha=0.8)
    style_axis(axes[0],
               xlabel=r"Oferta de trabalho $N_t$",
               ylabel=r"Salário real $w_t / w^*$")
    axes[0].xaxis.set_major_formatter(plain_number_formatter(2))
    axes[0].yaxis.set_major_formatter(plain_number_formatter(2))
    add_callout(axes[0], text="Maior b → menor N\n(mais lazer)", xy=(0.1, 1.3),
                dx=20, dy=10, color=COLORS["line_compare"],
                text_color=COLORS["line_compare"], with_connector=False)
    style_legend(axes[0], loc="lower right")

    # Right: marginal disutility of labour
    leisure_vals = np.linspace(0.05, 0.98, 300)
    mud_base = llc.marginal_disutility_labor(leisure_vals)
    mud_high = llc_high.marginal_disutility_labor(leisure_vals)
    axes[1].plot(1.0 - leisure_vals, mud_base, color=COLORS["line_main"],
                 linewidth=2.4, label=f"b = {llc_b:.2f}")
    axes[1].plot(1.0 - leisure_vals, mud_high, color=COLORS["line_compare"],
                 linewidth=2.0, linestyle="--", label=f"b = {b_high:.2f}")
    axes[1].set_ylim(0, min(30, mud_base.max() * 1.15))
    style_axis(axes[1],
               xlabel=r"Oferta de trabalho $N = 1 - l$",
               ylabel=r"Desutilidade marginal $b / l$")
    axes[1].xaxis.set_major_formatter(plain_number_formatter(2))
    axes[1].yaxis.set_major_formatter(plain_number_formatter(1))
    add_callout(axes[1],
                text=r"$\frac{\partial(-u)}{\partial N} = \frac{b}{1-N}$",
                xy=(0.6, float(llc.marginal_disutility_labor(0.4))),
                dx=20, dy=20, color=COLORS["line_main"],
                text_color=COLORS["line_main"])
    style_legend(axes[1], loc="upper left")

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: oferta de trabalho e desutilidade marginal",
        subtitle=(
            "Condição ótima: b·C/l = w  (MRS consumo-lazer = salário real). "
            "Maior peso do lazer (b) desloca oferta de trabalho para a esquerda."
        ),
        note=(
            r"Utilidade: u(C,l) = log C + b·log l,  l = 1 − N.  "
            f"Calibrado com N* ≈ 0,33 (b = {llc_b:.2f})."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = RBCModel(RBC)
    output_paths = [
        plot_irf(model),
        plot_phase_diagram(model),
        plot_stochastic_simulation(model),
        plot_moments(model),
        plot_labor_leisure(model),
    ]
    for path in output_paths:
        print(f"Saved {path}")
        print(f"Saved {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
