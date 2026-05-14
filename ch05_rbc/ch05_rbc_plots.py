"""Editorial plot scripts for the Real Business Cycle model (Chapter 5)."""

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

from ch05_rbc.ch05_rbc import LaborLeisureConditions, RBCModel
from data_utils import ensure_directory
from params import RBC, clone_params
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
# 1. Four-panel IRF  (Lista I Q8 — três canais do choque de PTF)
# ---------------------------------------------------------------------------


def plot_irf(model: RBCModel, shock_size: float = 0.01,
             T: int = 40, output_dir=FIGURES_DIR):
    """Impulse-response functions for Y, C, I, K after a 1 % TFP shock."""
    irf_data = model.irf(shock_size=shock_size, T=T)
    ll = irf_data["ll"]
    figure_path = Path(output_dir) / "rbc_irf.png"
    scale = 100.0

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    axes_flat = axes.flatten()

    panels = [
        ("y", r"Produto  $Y$",      COLORS["line_main"]),
        ("c", r"Consumo  $C$",      COLORS["line_compare"]),
        ("i", r"Investimento  $I$", COLORS["line_neutral"]),
        ("k", r"Capital  $K$",      COLORS["positive"]),
    ]

    for ax, (key, label, color) in zip(axes_flat, panels):
        vals = scale * irf_data[key]
        ax.plot(irf_data["time"], vals, color=color, linewidth=2.4)
        ax.axhline(0.0, color=COLORS["axis_light"], linewidth=0.9,
                   linestyle=":", alpha=0.85)
        style_axis(ax, xlabel="Período",
                   ylabel="Desvio percentual do est. est.")
        ax.xaxis.set_major_formatter(plain_number_formatter(0))
        ax.yaxis.set_major_formatter(_pct_formatter(2))
        ax.margins(x=0.05)

        # Direct label on the line
        add_callout(
            ax,
            text=label,
            xy=(irf_data["time"][2], float(vals[2])),
            dx=18, dy=12,
            color=color, text_color=color,
            with_connector=False,
        )
        # Mark peak
        peak_idx = int(np.argmax(np.abs(vals)))
        ax.scatter([irf_data["time"][peak_idx]], [float(vals[peak_idx])],
                   color=COLORS["highlight"], s=44,
                   edgecolors=COLORS["paper"], linewidths=0.7, zorder=5)

    # Overlay TFP path on the Y panel
    z_vals = scale * irf_data["z"]
    axes_flat[0].plot(irf_data["time"], z_vals,
                      color=COLORS["axis_light"], linestyle="--",
                      linewidth=1.5, alpha=0.85, zorder=1)
    add_callout(
        axes_flat[0],
        text=r"PTF $z$",
        xy=(irf_data["time"][8], float(z_vals[8])),
        dx=12, dy=-14,
        color=COLORS["axis_light"], text_color=COLORS["axis"],
        with_connector=False,
    )

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: funções de impulso-resposta a um choque de PTF",
        subtitle=(
            f"Choque inicial de +{shock_size * 100:.0f}% na PTF; "
            "desvios percentuais em relação ao estado estacionário. "
            f"Coeficiente de transição A = {ll['A']:.3f}."
        ),
        note=(
            "Utilidade logarítmica, trabalho inelástico. "
            "α=0,33 · β=0,99 · δ=0,025 · ρ_z=0,95 · σ_z=0,007."
        ),
        top=0.84,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# 2. Phase diagram in (k̂, ẑ) space
# ---------------------------------------------------------------------------


def plot_phase_diagram(model: RBCModel, output_dir=FIGURES_DIR):
    """Consumption policy surface and convergence trajectories in (k̂, ẑ)."""
    ll = model.log_linearize()
    a_k, a_z = ll["a_k"], ll["a_z"]
    A, B = ll["A"], ll["B"]

    k_range = np.linspace(-0.15, 0.15, 260)
    z_range = np.linspace(-0.05, 0.05, 260)
    KK, ZZ = np.meshgrid(k_range, z_range)
    CC = a_k * KK + a_z * ZZ

    dK = A * KK + B * ZZ - KK
    dZ = model.rho_z * ZZ - ZZ
    norm = np.sqrt(dK ** 2 + dZ ** 2) + 1e-12

    figure_path = Path(output_dir) / "rbc_phase_diagram.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ---- Left: policy surface heat-map ----
    cf = axes[0].contourf(KK, ZZ, CC, levels=22,
                          cmap="RdYlGn", alpha=0.82)
    axes[0].contour(KK, ZZ, CC, levels=[0.0],
                    colors=[COLORS["black"]], linewidths=1.8)
    cb = plt.colorbar(cf, ax=axes[0], pad=0.02)
    cb.set_label("ĉ  (log-desvio do consumo)", color=COLORS["axis"],
                 fontsize=9.5)
    cb.ax.tick_params(colors=COLORS["axis"], labelsize=9)

    axes[0].scatter([0], [0], color=COLORS["highlight"], s=70,
                    edgecolors=COLORS["paper"], linewidths=0.9, zorder=5)
    style_axis(axes[0],
               xlabel=r"Capital log-desvio  $\hat{k}_t$",
               ylabel=r"PTF log-desvio  $\hat{z}_t$")
    add_callout(axes[0], text="Estado\nestacionário", xy=(0.0, 0.0),
                dx=14, dy=16, color=COLORS["highlight"],
                text_color=COLORS["highlight"])
    axes[0].set_title(r"Política de consumo:  $\hat{c}_t = a_k\hat{k}_t + a_z\hat{z}_t$",
                      fontsize=10.2, pad=7, color=COLORS["text"])
    axes[0].text(0.03, 0.97,
                 f"$a_k={a_k:.3f}$     $a_z={a_z:.3f}$",
                 transform=axes[0].transAxes,
                 fontsize=9.5, color=COLORS["text"],
                 va="top", ha="left",
                 bbox={"facecolor": COLORS["paper"], "edgecolor": "none",
                       "alpha": 0.9, "pad": 3})

    # ---- Right: vector field + IRF trajectories ----
    step = 14
    axes[1].quiver(
        KK[::step, ::step], ZZ[::step, ::step],
        dK[::step, ::step] / norm[::step, ::step],
        dZ[::step, ::step] / norm[::step, ::step],
        color=COLORS["muted"], alpha=0.50, scale=24,
        headwidth=3.2, headlength=4.5,
    )

    shocks = [0.01, 0.02, 0.03]
    alphas = [0.72, 0.85, 1.0]
    for shock, alpha in zip(shocks, alphas):
        irf_data = model.irf(shock_size=shock, T=35)
        axes[1].plot(irf_data["k"], irf_data["z"],
                     color=COLORS["line_main"], linewidth=1.8, alpha=alpha)
        axes[1].scatter([float(irf_data["k"][0])], [float(irf_data["z"][0])],
                        color=COLORS["line_compare"], s=32,
                        edgecolors=COLORS["paper"], linewidths=0.6, zorder=5)

    axes[1].scatter([0], [0], color=COLORS["highlight"], s=70,
                    edgecolors=COLORS["paper"], linewidths=0.9, zorder=6)
    style_axis(axes[1],
               xlabel=r"Capital log-desvio  $\hat{k}_t$",
               ylabel=r"PTF log-desvio  $\hat{z}_t$")
    axes[1].set_title(f"Campo vetorial e convergência  (A = {A:.3f})",
                      fontsize=10.2, pad=7, color=COLORS["text"])

    legend_handles = [
        Line2D([0], [0], color=COLORS["line_main"], linewidth=1.8,
               label="Trajetória pós-choque"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=COLORS["line_compare"],
               markeredgecolor=COLORS["paper"], markersize=6,
               label="Ponto de impacto"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=COLORS["highlight"],
               markeredgecolor=COLORS["paper"], markersize=8,
               label="Estado estacionário"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right",
                   frameon=False, fontsize=9.0)

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: diagrama de fase no espaço  (k̂, ẑ)",
        subtitle=(
            "Esquerda: intensidade da política de consumo ótimo. "
            "Direita: trajetórias após choques de PTF de 1 %, 2 % e 3 %."
        ),
        note=(
            f"Persistência da PTF ρ_z = {model.rho_z:.2f}; "
            "convergência garantida por |A| < 1."
        ),
        top=0.86,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 3. Stochastic simulation
# ---------------------------------------------------------------------------


def plot_stochastic_simulation(model: RBCModel, T: int = 120,
                                seed: int = 42, output_dir=FIGURES_DIR):
    """Y, C, I time series from one stochastic simulation draw."""
    sim = model.simulate(T=T, seed=seed)
    figure_path = Path(output_dir) / "rbc_simulation.png"
    scale = 100.0
    times = sim["time"]

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # Upper panel: Y, C, I
    series_top = [
        ("y", "Produto (Y)",      COLORS["line_main"],    "-",  2.0),
        ("c", "Consumo (C)",      COLORS["line_compare"], "-",  1.8),
        ("i", "Investimento (I)", COLORS["line_neutral"],  "--", 1.5),
    ]
    for key, label, color, ls, lw in series_top:
        axes[0].plot(times, scale * sim[key],
                     color=color, linestyle=ls, linewidth=lw, alpha=0.92)

    axes[0].axhline(0.0, color=COLORS["axis_light"], linewidth=0.9,
                    linestyle=":", alpha=0.80)
    style_axis(axes[0], ylabel="Desvio do estado est. (%)", y_grid=True)
    axes[0].yaxis.set_major_formatter(_pct_formatter(1))
    axes[0].margins(x=0.02)

    for key, label, color, _, _ in series_top:
        direct_label_last(axes[0], times, scale * sim[key],
                          label=label, color=color, dx=6)

    # Lower panel: TFP
    z = scale * sim["z"]
    axes[1].fill_between(times, 0.0, z, where=(z >= 0),
                          color=COLORS["positive"], alpha=0.50, linewidth=0)
    axes[1].fill_between(times, 0.0, z, where=(z < 0),
                          color=COLORS["negative"], alpha=0.50, linewidth=0)
    axes[1].plot(times, z, color=COLORS["axis"], linewidth=1.4, alpha=0.90)
    axes[1].axhline(0.0, color=COLORS["axis_light"], linewidth=0.9, linestyle=":")
    style_axis(axes[1], xlabel="Período",
               ylabel=r"PTF $z$  —  desvio (%)", y_grid=True)
    axes[1].yaxis.set_major_formatter(_pct_formatter(2))
    axes[1].margins(x=0.02)

    # Shade recessions on upper panel too
    axes[0].fill_between(times, axes[0].get_ylim()[0],
                          axes[0].get_ylim()[1],
                          where=(z < 0),
                          color=COLORS["negative"], alpha=0.06, linewidth=0)

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: simulação estocástica — Y, C, I e PTF",
        subtitle=(
            "Choques AR(1) de PTF propagam flutuações correlacionadas em produto, "
            "consumo e investimento. Investimento amplifica; consumo suaviza."
        ),
        note=(
            f"σ_z={model.sigma_z:.3f} · ρ_z={model.rho_z:.2f} · "
            f"semente={seed}. Fundo vermelho = PTF abaixo da tendência."
        ),
        top=0.85,
        bottom=0.08,
    )


# ---------------------------------------------------------------------------
# 4. Second-moment bar chart
# ---------------------------------------------------------------------------


def plot_moments(model: RBCModel, data_moments: dict | None = None,
                 output_dir=FIGURES_DIR):
    """Simulated volatility moments vs. optional empirical benchmarks."""
    model_mom = model.moments(T_sim=5000, n_draws=40, seed=0)
    figure_path = Path(output_dir) / "rbc_moments.png"

    variables = ["C", "I", "K"]
    model_vals = [
        model_mom["rel_std_c"],
        model_mom["rel_std_i"],
        model_mom["rel_std_k"],
    ]
    has_data = data_moments is not None
    x = np.arange(len(variables))
    width = 0.38 if has_data else 0.55

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    x_model = x - width / 2 if has_data else x
    bars_m = ax.bar(x_model, model_vals, width,
                    color=COLORS["line_main"], alpha=0.88,
                    label="Modelo RBC", zorder=3)

    if has_data:
        data_vals = [
            data_moments.get("rel_std_c", np.nan),
            data_moments.get("rel_std_i", np.nan),
            data_moments.get("rel_std_k", np.nan),
        ]
        ax.bar(x + width / 2, data_vals, width,
               color=COLORS["line_compare"], alpha=0.88,
               label="Dados BR (HP)", zorder=3)

    ax.axhline(1.0, color=COLORS["axis_light"], linewidth=1.3,
               linestyle="--", alpha=0.85, zorder=2)
    add_callout(ax, text="Referência: σ(X)/σ(Y) = 1",
                xy=(float(x[-1]) + 0.1, 1.0),
                dx=10, dy=10,
                color=COLORS["axis_light"], text_color=COLORS["axis"],
                with_connector=False)

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=12)
    style_axis(ax, ylabel="Volatilidade relativa ao PIB   σ(X) / σ(Y)")
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    ax.set_xlim(-0.6, len(variables) - 0.4 + (0.5 if has_data else 0))
    style_legend(ax, loc="upper right")

    for bar, val in zip(bars_m, model_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=9.5, color=COLORS["text"], fontweight="bold")

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: volatilidade relativa dos agregados",
        subtitle=(
            "Investimento amplifica os ciclos (I mais volátil que Y); "
            "consumo é relativamente estável (C menos volátil que Y)."
        ),
        note=(
            "T = 5.000 períodos · 40 replicações · "
            "σ_z = 0,007 · ρ_z = 0,95."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 5. Labour-leisure tradeoff  (Lista I Q5–7)
# ---------------------------------------------------------------------------


def plot_labor_leisure(model: RBCModel, output_dir=FIGURES_DIR):
    """Labour supply schedule and marginal disutility (Lista I Q5–7)."""
    ss = model.steady_state()

    # Calibrate b so N* ≈ 1/3
    llc_b = LaborLeisureConditions(b=2.0).calibrate_b(model, n_star=0.33)
    llc = LaborLeisureConditions(b=llc_b)
    b_high = llc_b * 1.5
    llc_high = LaborLeisureConditions(b=b_high)

    wage_grid = np.linspace(0.4 * ss["w_star"], 2.2 * ss["w_star"], 300)
    df_base = llc.leisure_grid(wage_grid, consumption=ss["c_star"])
    leisure_high = llc_high.optimal_leisure(ss["c_star"], wage_grid)
    labor_high = np.clip(1.0 - leisure_high, 0.0, 1.0)

    # Optimal labour at SS wage
    n_star_base = float(1.0 - llc.optimal_leisure(ss["c_star"], ss["w_star"]))
    w_norm = wage_grid / ss["w_star"]

    figure_path = Path(output_dir) / "rbc_labor_leisure.png"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ---- Left: labour supply schedule (N vs w/w*) ----
    axes[0].plot(df_base["labor"], w_norm,
                 color=COLORS["line_main"], linewidth=2.4,
                 label=f"$b = {llc_b:.2f}$")
    axes[0].plot(labor_high, w_norm,
                 color=COLORS["line_compare"], linewidth=2.0, linestyle="--",
                 label=f"$b = {b_high:.2f}$  (↑ lazer)")

    # Mark equilibrium point
    axes[0].scatter([n_star_base], [1.0],
                    color=COLORS["highlight"], s=64,
                    edgecolors=COLORS["paper"], linewidths=0.8, zorder=5)
    axes[0].axhline(1.0, color=COLORS["axis_light"], linewidth=0.9,
                    linestyle=":", alpha=0.85)
    axes[0].axvline(n_star_base, color=COLORS["axis_light"], linewidth=0.9,
                    linestyle=":", alpha=0.85)

    style_axis(axes[0],
               xlabel=r"Oferta de trabalho  $N_t$",
               ylabel=r"Salário real  $w_t\,/\,w^*$")
    axes[0].xaxis.set_major_formatter(plain_number_formatter(2))
    axes[0].yaxis.set_major_formatter(plain_number_formatter(2))
    axes[0].set_xlim(-0.02, 1.0)
    style_legend(axes[0], loc="lower right")
    add_callout(axes[0],
                text=f"Equilíbrio\n$N^*={n_star_base:.2f}$",
                xy=(n_star_base, 1.0),
                dx=16, dy=18,
                color=COLORS["highlight"], text_color=COLORS["highlight"])
    add_callout(axes[0],
                text="↑b  →  curva\ndesloca à esq.",
                xy=(float(labor_high[180]), float(w_norm[180])),
                dx=-60, dy=-16,
                color=COLORS["line_compare"], text_color=COLORS["line_compare"],
                ha="right")

    axes[0].set_title(r"Oferta de trabalho:  $l^* = b\,C / w$   (Q6–7)",
                      fontsize=10.2, pad=7, color=COLORS["text"])

    # ---- Right: marginal disutility of labour ----
    n_vals = np.linspace(0.01, 0.98, 400)
    l_vals = 1.0 - n_vals
    mud_base = llc.marginal_disutility_labor(l_vals)
    mud_high = llc_high.marginal_disutility_labor(l_vals)

    cap = float(llc.marginal_disutility_labor(np.array([0.05])))[0] if False else 25.0
    mask = mud_base <= cap

    axes[1].plot(n_vals[mask], mud_base[mask],
                 color=COLORS["line_main"], linewidth=2.4,
                 label=f"$b = {llc_b:.2f}$")
    axes[1].plot(n_vals[mask], mud_high[mask],
                 color=COLORS["line_compare"], linewidth=2.0, linestyle="--",
                 label=f"$b = {b_high:.2f}$")

    # Mark equilibrium disutility
    n_eq = n_star_base
    mud_eq = float(llc.marginal_disutility_labor(1.0 - n_eq))
    if mud_eq <= cap:
        axes[1].scatter([n_eq], [mud_eq],
                        color=COLORS["highlight"], s=64,
                        edgecolors=COLORS["paper"], linewidths=0.8, zorder=5)

    style_axis(axes[1],
               xlabel=r"Oferta de trabalho  $N_t = 1 - l_t$",
               ylabel=r"Desutilidade marginal  $b\,/\,(1 - N)$")
    axes[1].xaxis.set_major_formatter(plain_number_formatter(2))
    axes[1].yaxis.set_major_formatter(plain_number_formatter(1))
    axes[1].set_ylim(0.0, cap * 1.05)
    style_legend(axes[1], loc="upper left")
    add_callout(axes[1],
                text=r"$\frac{\partial(-u)}{\partial N}=\frac{b}{1-N}$",
                xy=(0.55, float(llc.marginal_disutility_labor(0.45))),
                dx=22, dy=18,
                color=COLORS["line_main"], text_color=COLORS["line_main"])
    axes[1].set_title("Desutilidade marginal do trabalho  (Q5)",
                      fontsize=10.2, pad=7, color=COLORS["text"])

    return finalize_figure(
        fig,
        figure_path,
        title="RBC: lazer, trabalho e desutilidade marginal",
        subtitle=(
            r"Condição ótima: $b\,C_t / l_t = w_t$  (MRS = salário real). "
            "Maior $b$ reduz oferta de trabalho e eleva o custo marginal."
        ),
        note=(
            r"$u(C,l)=\log C + b\,\log l$,  $l=1-N$. "
            f"Calibrado com $N^*\\approx 0{{,}}33$  ($b={llc_b:.2f}$)."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    model = RBCModel(RBC)
    plots = [
        ("IRF",                plot_irf,                     {}),
        ("Diagrama de fase",   plot_phase_diagram,           {}),
        ("Simulação",          plot_stochastic_simulation,   {}),
        ("Momentos",           plot_moments,                 {}),
        ("Trabalho-lazer",     plot_labor_leisure,           {}),
    ]
    for name, func, kwargs in plots:
        path = func(model, **kwargs)
        print(f"[{name}] {path}")
        print(f"[{name}] {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
