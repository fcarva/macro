"""Editorial plot scripts for the Solow growth model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow import SolowModel
from data_utils import ensure_directory
from params import SOLOW, clone_params
from plotting_style import (
    COLORS,
    finalize_figure,
    percent_formatter,
    plain_number_formatter,
    style_axis,
    style_legend,
)


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def plot_solow_diagram(model: SolowModel, output_dir=FIGURES_DIR):
    curves = model.solow_curves()
    steady = model.steady_state()
    figure_path = Path(output_dir) / "solow_diagram.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(curves["k"], curves["savings_curve"], color=COLORS["line_main"], label=r"$s f(k)$")
    ax.plot(
        curves["k"],
        curves["break_even_curve"],
        color=COLORS["line_compare"],
        linestyle="--",
        label=r"$(n+g+\delta)k$",
    )
    ax.scatter(
        [steady["k_star"]],
        [model.s * model.f(steady["k_star"])],
        s=38,
        color=COLORS["black"],
        zorder=4,
    )
    ax.axvline(steady["k_star"], color=COLORS["axis"], linestyle=":", linewidth=1.3)
    ax.annotate(
        r"$k^*$",
        xy=(steady["k_star"], 0),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        color=COLORS["axis"],
    )
    style_axis(
        ax,
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel="Produção e investimento de reposição",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    style_legend(ax, loc="upper left")
    return finalize_figure(
        fig,
        figure_path,
        title="Modelo de Solow: poupança e investimento de reposição",
        subtitle="O estado estacionário surge quando a poupança cobre depreciação, crescimento populacional e progresso técnico.",
        note="Todas as curvas estão em unidades por trabalhador efetivo.",
    )


def plot_phase_diagram(model: SolowModel, output_dir=FIGURES_DIR):
    curves = model.solow_curves()
    steady = model.steady_state()
    figure_path = Path(output_dir) / "solow_phase_diagram.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.plot(curves["k"], curves["k_dot"], color=COLORS["line_neutral"])
    ax.fill_between(
        curves["k"],
        0,
        curves["k_dot"],
        where=curves["k_dot"] >= 0,
        color=COLORS["positive"],
        alpha=0.2,
    )
    ax.fill_between(
        curves["k"],
        0,
        curves["k_dot"],
        where=curves["k_dot"] < 0,
        color=COLORS["negative"],
        alpha=0.16,
    )
    ax.axvline(steady["k_star"], color=COLORS["axis"], linestyle=":", linewidth=1.3)

    sample = curves.iloc[::25].copy()
    directions = np.sign(sample["k_dot"]).replace(0, 1)
    ax.quiver(
        sample["k"].to_numpy(),
        np.zeros(len(sample)),
        directions.to_numpy(),
        np.zeros(len(sample)),
        angles="xy",
        scale_units="xy",
        scale=9.5,
        width=0.003,
        color=COLORS["muted"],
        alpha=0.85,
    )
    ax.annotate(
        r"$k^*$",
        xy=(steady["k_star"], 0),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        color=COLORS["axis"],
    )
    style_axis(
        ax,
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Variação do capital, $\dot{k}$",
        zero_line=True,
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    return finalize_figure(
        fig,
        figure_path,
        title="Diagrama de fase do Solow",
        subtitle="À esquerda de k* o capital cresce; à direita de k* ele recua em direção ao estado estacionário.",
        note="As áreas em verde indicam crescimento de k; as áreas em vermelho indicam contração.",
    )


def plot_transition(model: SolowModel, k0: float, output_dir=FIGURES_DIR):
    path_data = model.transition_path(k0, T=60.0, dt=0.2)
    steady = model.steady_state()
    figure_path = Path(output_dir) / "solow_transition.png"

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharex=True)
    series_info = [
        ("k", "k_star", "Capital", COLORS["line_main"]),
        ("y", "y_star", "Produto", COLORS["line_neutral"]),
        ("c", "c_star", "Consumo", COLORS["line_compare"]),
    ]

    for axis, (key, steady_key, title, color) in zip(axes, series_info):
        axis.plot(path_data["time"], path_data[key], color=color)
        axis.axhline(steady[steady_key], color=COLORS["axis"], linestyle=":", linewidth=1.2)
        axis.set_title(title, fontsize=11.5, loc="left", color=COLORS["text"])
        style_axis(axis, xlabel="Tempo", y_grid=True)
        axis.xaxis.set_major_formatter(plain_number_formatter(0))
        axis.yaxis.set_major_formatter(plain_number_formatter(1))

    axes[0].set_ylabel("Nível")
    return finalize_figure(
        fig,
        figure_path,
        title="Trajetória de transição no modelo de Solow",
        subtitle="Partindo de um nível inicial de capital abaixo do estado estacionário, produto e consumo convergem gradualmente.",
        note="Linhas pontilhadas marcam os níveis de estado estacionário de cada variável.",
        top=0.83,
        bottom=0.12,
    )


def plot_savings_shock(model: SolowModel, k0: float, new_s: float, output_dir=FIGURES_DIR):
    before = model.transition_path(k0, T=25.0, dt=0.2)
    shock_capital = float(before["k"][-1])
    shocked_model = SolowModel(clone_params(model.params, {"s": new_s}))
    after = shocked_model.transition_path(shock_capital, T=35.0, dt=0.2)
    figure_path = Path(output_dir) / "solow_savings_shock.png"

    time_after = after["time"] + before["time"][-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharex=False)
    series_map = [
        ("k", "k_star", "Capital"),
        ("y", "y_star", "Produto"),
        ("c", "c_star", "Consumo"),
    ]

    baseline_steady = model.steady_state()
    shocked_steady = shocked_model.steady_state()

    for axis, (series_name, steady_name, panel_title) in zip(axes, series_map):
        axis.plot(before["time"], before[series_name], color=COLORS["line_main"], label="Antes do choque")
        axis.plot(time_after, after[series_name], color=COLORS["line_compare"], label="Depois do choque")
        axis.axvline(before["time"][-1], color=COLORS["axis"], linestyle=":", linewidth=1.1)
        axis.axhline(baseline_steady[steady_name], color=COLORS["line_main"], linestyle=":", linewidth=1.0)
        axis.axhline(shocked_steady[steady_name], color=COLORS["line_compare"], linestyle=":", linewidth=1.0)
        axis.set_title(panel_title, fontsize=11.5, loc="left", color=COLORS["text"])
        style_axis(axis, xlabel="Tempo", y_grid=True)
        axis.xaxis.set_major_formatter(plain_number_formatter(0))
        axis.yaxis.set_major_formatter(plain_number_formatter(1))

    axes[0].set_ylabel("Nível")
    style_legend(axes[0], loc="upper left")
    return finalize_figure(
        fig,
        figure_path,
        title="Choque de poupança no modelo de Solow",
        subtitle="Um aumento da taxa de poupança desloca o estado estacionário para cima e altera a trajetória de convergência.",
        note=f"A taxa de poupança sobe de {model.s:.0%} para {new_s:.0%}.",
        top=0.83,
        bottom=0.12,
    )


def plot_golden_rule(model: SolowModel, output_dir=FIGURES_DIR):
    savings_grid = np.linspace(0.05, 0.95, 181)
    steady_consumption = np.array([model.steady_state(savings)["c_star"] for savings in savings_grid])
    golden = model.golden_rule()
    current = model.steady_state()
    figure_path = Path(output_dir) / "solow_golden_rule.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.plot(savings_grid, steady_consumption, color=COLORS["positive"])
    ax.axvline(model.s, color=COLORS["line_main"], linestyle="--", linewidth=1.2)
    ax.axvline(golden["s_gold"], color=COLORS["line_compare"], linestyle="--", linewidth=1.2)
    ax.scatter([model.s], [current["c_star"]], color=COLORS["line_main"], s=36, zorder=4)
    ax.scatter([golden["s_gold"]], [golden["c_gold"]], color=COLORS["line_compare"], s=36, zorder=4)
    ax.annotate("Poupança atual", xy=(model.s, current["c_star"]), xytext=(8, -16), textcoords="offset points", color=COLORS["line_main"])
    ax.annotate("Regra de Ouro", xy=(golden["s_gold"], golden["c_gold"]), xytext=(8, 8), textcoords="offset points", color=COLORS["line_compare"])

    style_axis(
        ax,
        xlabel="Taxa de poupança, s",
        ylabel="Consumo de estado estacionário",
    )
    ax.xaxis.set_major_formatter(percent_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    return finalize_figure(
        fig,
        figure_path,
        title="Curva da Regra de Ouro",
        subtitle="No caso Cobb-Douglas, a taxa de poupança que maximiza o consumo de longo prazo coincide com alpha.",
        note="O pico da curva identifica a combinação de poupança e capital que maximiza o consumo de estado estacionário.",
    )


def main():
    model = SolowModel(SOLOW)
    steady = model.steady_state()
    output_paths = [
        plot_solow_diagram(model),
        plot_phase_diagram(model),
        plot_transition(model, k0=0.45 * steady["k_star"]),
        plot_savings_shock(model, k0=0.45 * steady["k_star"], new_s=0.28),
        plot_golden_rule(model),
    ]
    for path in output_paths:
        print(f"Saved {path}")
        print(f"Saved {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
