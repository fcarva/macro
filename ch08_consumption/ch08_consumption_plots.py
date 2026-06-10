"""Editorial plot scripts for the consumption models (Chapter 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch08_consumption.ch08_consumption import (
    BufferStockModel,
    CampbellMankiw,
    HallRandomWalk,
    PermanentIncomeModel,
)
from data_utils import ensure_directory
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
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


# ---------------------------------------------------------------------------
# 1. PIH — consumption smoothing vs income
# ---------------------------------------------------------------------------


def plot_pih_smoothing(model: PermanentIncomeModel | None = None,
                        T: int = 80, seed: int = 7,
                        output_dir=FIGURES_DIR):
    """Simulated income (volatile) vs PIH consumption (smooth)."""
    model = model or PermanentIncomeModel()
    sim = model.simulate(T=T, a0=0.0, seed=seed)
    figure_path = Path(output_dir) / "consumption_pih_smoothing.png"

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    time = sim["time"]

    ax.plot(time, sim["y"], color=COLORS["line_compare"], linewidth=1.6,
            alpha=0.85, label="Renda corrente $y_t$")
    ax.plot(time, sim["c"], color=COLORS["line_main"], linewidth=2.4,
            label="Consumo $c_t$ (PIH)")
    ax.axhline(model.y_bar, color=COLORS["axis_light"], linewidth=0.9,
               linestyle=":", alpha=0.85)

    style_axis(ax, xlabel="Período", ylabel="Nível")
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    ax.margins(x=0.02)
    style_legend(ax, loc="upper right")

    add_callout(
        ax,
        text=(
            f"Var($y_t$) = {np.var(sim['y']):.4f}\n"
            f"Var($c_t$) = {np.var(sim['c']):.4f}"
        ),
        xy=(time[-1], float(sim["c"][-1])),
        dx=-90, dy=30,
        color=COLORS["highlight"], text_color=COLORS["highlight"],
        with_connector=False,
    )

    return finalize_figure(
        fig,
        figure_path,
        title="PIH: consumo suaviza flutuações transitórias da renda",
        subtitle=(
            "Renda segue um AR(1); o consumo responde apenas à riqueza "
            "humana (valor presente da renda), variando muito menos."
        ),
        note=(
            f"r={model.r} · ρ_y={model.rho_y} · σ_y={model.sigma_y} · "
            f"ȳ={model.y_bar} · semente={seed}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 2. Hall (1978) — martingale paths
# ---------------------------------------------------------------------------


def plot_hall_martingale(model: HallRandomWalk | None = None,
                          n_paths: int = 6, T: int = 100, sigma: float = 0.05,
                          output_dir=FIGURES_DIR):
    """Several simulated consumption paths c_{t+1} = c_t + eps_{t+1}."""
    model = model or HallRandomWalk({"beta": 1.0 / 1.03, "r": 0.03})
    figure_path = Path(output_dir) / "consumption_hall_martingale.png"

    fig, ax = plt.subplots(figsize=(11.0, 6.0))

    palette = [COLORS["line_main"], COLORS["line_compare"], COLORS["line_neutral"],
               COLORS["positive"], COLORS["negative"], COLORS["highlight"]]
    for i in range(n_paths):
        sim = model.simulate_martingale(c0=1.0, T=T, sigma=sigma, seed=i)
        ax.plot(sim["time"], sim["c"], color=palette[i % len(palette)],
                linewidth=1.6, alpha=0.80)

    ax.axhline(1.0, color=COLORS["axis"], linewidth=1.2, linestyle="--",
               alpha=0.75)
    add_callout(
        ax,
        text="$c_0 = 1$: melhor previsor\nde $c_{t+1}$ é $c_t$",
        xy=(0, 1.0), dx=14, dy=18,
        color=COLORS["axis"], text_color=COLORS["axis"],
    )

    style_axis(ax, xlabel="Período", ylabel="Consumo $c_t$")
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    ax.margins(x=0.02)

    return finalize_figure(
        fig,
        figure_path,
        title="Hall (1978): consumo segue um passeio aleatório",
        subtitle=(
            "Sob $\\beta(1+r)=1$ e utilidade quadrática, "
            "$c_t = E_t[c_{t+1}]$: as trajetórias divergem sem tendência."
        ),
        note=(
            f"β={model.beta:.4f} · r={model.r} · σ_ε={sigma} · "
            f"{n_paths} trajetórias simuladas."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 3. Buffer-stock policy function vs PIH 45-degree rule
# ---------------------------------------------------------------------------


def plot_buffer_stock_policy(model: BufferStockModel | None = None,
                              output_dir=FIGURES_DIR):
    """Consumption policy c*(a, y) vs the 45-degree "consume everything" line."""
    model = model or BufferStockModel(CONSUMPTION)
    pf = model.policy_function()
    figure_path = Path(output_dir) / "consumption_buffer_stock_policy.png"

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    a_grid = pf["a_grid"]

    labels = [
        (f"$y_{{baixo}} = {model.y_low:.2f}$", COLORS["negative"]),
        (f"$y_{{alto}} = {model.y_high:.2f}$", COLORS["positive"]),
    ]
    for iy, (label, color) in enumerate(labels):
        ax.plot(a_grid, pf["policy_c"][:, iy], color=color, linewidth=2.4,
                label=f"$c^*(a, y)$, {label}")

    cash_low = (1.0 + model.r) * a_grid + model.y_low
    ax.plot(a_grid, cash_low, color=COLORS["axis"], linewidth=1.2,
            linestyle="--", alpha=0.75,
            label="Restrição: $c = (1+r)a + y_{baixo}$")

    style_axis(ax, xlabel="Ativos no início do período $a$",
               ylabel="Consumo $c^*(a, y)$")
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(2))
    ax.margins(x=0.02)
    style_legend(ax, loc="upper left")

    add_callout(
        ax,
        text=(
            "Restrição de crédito ativa\n($a'\\geq 0$): poupança\n"
            "precaucional move $c^*$\nabaixo do recurso total"
        ),
        xy=(a_grid[2], float(pf["policy_c"][2, 0])),
        dx=40, dy=-50,
        color=COLORS["highlight"], text_color=COLORS["highlight"],
    )

    return finalize_figure(
        fig,
        figure_path,
        title="Buffer-stock: função-política de consumo $c^*(a, y)$",
        subtitle=(
            "Com restrição de crédito e renda incerta, o consumo ótimo "
            "fica abaixo dos recursos correntes — poupança precaucional."
        ),
        note=(
            f"β={model.beta} · r={model.r} · θ={model.theta} · "
            f"P(permanece)={model.prob_stay}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# 4. Simulated wealth distribution
# ---------------------------------------------------------------------------


def plot_wealth_distribution(model: BufferStockModel | None = None,
                              N: int = 800, T: int = 300, seed: int = 0,
                              output_dir=FIGURES_DIR):
    """Cross-sectional distribution of simulated assets (long-run buffer stock)."""
    model = model or BufferStockModel(CONSUMPTION)
    panel = model.simulate_panel(N=N, T=T, seed=seed, burn_in=100)
    figure_path = Path(output_dir) / "consumption_wealth_distribution.png"

    a_final = panel["a"][-1, :]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.hist(a_final, bins=24, color=COLORS["line_main"], alpha=0.75,
            edgecolor=COLORS["paper"])
    ax.axvline(float(np.mean(a_final)), color=COLORS["highlight"],
               linewidth=2.0, linestyle="--")

    add_callout(
        ax,
        text=f"Média = {np.mean(a_final):.2f}\nMediana = {np.median(a_final):.2f}",
        xy=(float(np.mean(a_final)), 0.0),
        dx=14, dy=120,
        color=COLORS["highlight"], text_color=COLORS["highlight"],
    )

    style_axis(ax, xlabel="Ativos $a$ (estado estacionário ergódico)",
               ylabel="Número de famílias")
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(0))

    return finalize_figure(
        fig,
        figure_path,
        title="Buffer-stock: distribuição estacionária de ativos",
        subtitle=(
            "Famílias acumulam um 'estoque-tampão' de ativos para se "
            "autossegurar contra choques de renda — não acumulação ilimitada."
        ),
        note=(
            f"N={N} famílias · T={T} períodos (descartando 100 de burn-in) · "
            f"semente={seed}."
        ),
        top=0.87,
        bottom=0.10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    pih = PermanentIncomeModel()
    hall = HallRandomWalk({"beta": 1.0 / 1.03, "r": 0.03})
    buf = BufferStockModel(CONSUMPTION)

    plots = [
        ("PIH suavização",        plot_pih_smoothing,        pih,  {}),
        ("Hall martingale",       plot_hall_martingale,      hall, {}),
        ("Buffer-stock política", plot_buffer_stock_policy,  buf,  {}),
        ("Distribuição riqueza",  plot_wealth_distribution,  buf,  {}),
    ]
    for name, func, model, kwargs in plots:
        path = func(model, **kwargs)
        print(f"[{name}] {path}")
        print(f"[{name}] {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
