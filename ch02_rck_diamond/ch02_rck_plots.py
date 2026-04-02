"""Editorial plot scripts for the Ramsey-Cass-Koopmans model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow import SolowModel
from ch02_rck_diamond.ch02_rck import RCKModel
from data_utils import ensure_directory
from params import RCK, SOLOW, clone_params
from plotting_style import (
    COLORS,
    direct_label_last,
    finalize_figure,
    plain_number_formatter,
    style_axis,
    style_legend,
)


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def plot_phase_diagram(model: RCKModel, output_dir=FIGURES_DIR):
    data = model.phase_diagram_data(arrow_points=18, saddle_points=8, saddle_T=45.0, include_saddle_path=True)
    saddle = data["saddle_path"]
    steady = data["steady_state"]
    figure_path = Path(output_dir) / "rck_phase_diagram.png"

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.streamplot(
        data["K"][0],
        data["C"][:, 0],
        data["dK"],
        data["dC"],
        color=COLORS["muted"],
        density=1.0,
        linewidth=0.8,
        arrowsize=0.8,
        zorder=1,
    )
    ax.plot(data["k_grid"], data["k_dot_zero"], color=COLORS["line_main"], zorder=3)
    ax.axvline(data["c_dot_zero"], color=COLORS["line_compare"], linestyle="--", zorder=2)
    if not saddle.empty:
        ax.plot(saddle["k"], saddle["c"], color=COLORS["black"], linewidth=2.6, zorder=4)
    ax.scatter([steady["k_star"]], [steady["c_star"]], color=COLORS["highlight"], s=40, zorder=5)
    ax.annotate(
        "Estado estacionario",
        xy=(steady["k_star"], steady["c_star"]),
        xytext=(12, 8),
        textcoords="offset points",
        color=COLORS["axis"],
    )
    ax.text(data["k_grid"][-55], data["k_dot_zero"][-55] + 0.03, r"$\dot{k}=0$", color=COLORS["line_main"])
    ax.text(data["c_dot_zero"] + 0.1, ax.get_ylim()[1] * 0.9, r"$\dot{c}=0$", color=COLORS["line_compare"])
    if not saddle.empty:
        ax.text(saddle["k"].iloc[-2] + 0.05, saddle["c"].iloc[-2] + 0.03, "Saddle path", color=COLORS["black"])

    style_axis(
        ax,
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Consumo por trabalhador efetivo, $c$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    return finalize_figure(
        fig,
        figure_path,
        title="RCK: diagrama de fase",
        subtitle="A trajetória de sela identifica a combinação inicial de capital e consumo compatível com o equilíbrio ótimo.",
        note="O campo vetorial mostra a direção do sistema dinâmico em torno das isóclinas.",
    )


def plot_compare_rck_solow(model: RCKModel, output_dir=FIGURES_DIR):
    steady = model.steady_state()
    k0 = 0.8 * steady["k_star"]
    saddle = model.find_saddle_path(k0, T=80.0)
    rck_path = saddle["simulation"]

    solow_model = SolowModel(
        clone_params(
            SOLOW,
            {"alpha": model.alpha, "n": model.n, "g": model.g, "delta": model.delta},
        )
    )
    solow_path = solow_model.transition_path(k0=k0, T=80.0, dt=0.1)
    figure_path = Path(output_dir) / "rck_vs_solow_capital.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(rck_path["time"], rck_path["k"], color=COLORS["line_main"])
    ax.plot(solow_path["time"], solow_path["k"], color=COLORS["line_compare"], linestyle="--")
    ax.axhline(steady["k_star"], color=COLORS["axis"], linestyle=":", linewidth=1.2)
    direct_label_last(ax, rck_path["time"], rck_path["k"], label="RCK", color=COLORS["line_main"])
    direct_label_last(ax, solow_path["time"], solow_path["k"], label="Solow", color=COLORS["line_compare"], dy=-10)

    style_axis(
        ax,
        xlabel="Tempo",
        ylabel=r"Capital por trabalhador efetivo, $k(t)$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    ax.margins(x=0.08)
    return finalize_figure(
        fig,
        figure_path,
        title="RCK versus Solow: transição do capital",
        subtitle="Com o mesmo capital inicial, o RCK ajusta o consumo de forma ótima e produz uma dinâmica distinta da economia de Solow.",
        note="A linha pontilhada marca o estado estacionário do RCK.",
    )


def plot_rho_shock(model: RCKModel, output_dir=FIGURES_DIR):
    old_steady = model.steady_state()
    shocked_model = RCKModel(clone_params(model.params, {"rho": 0.75 * model.rho}))
    shocked_steady = shocked_model.steady_state()
    shocked_saddle = shocked_model.find_saddle_path(old_steady["k_star"], T=80.0)
    transition = shocked_saddle["simulation"]

    old_phase = model.phase_diagram_data(arrow_points=18, saddle_points=6, saddle_T=35.0, include_saddle_path=True)
    new_phase = shocked_model.phase_diagram_data(arrow_points=18, saddle_points=6, saddle_T=35.0, include_saddle_path=True)
    figure_path = Path(output_dir) / "rck_rho_shock.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))

    axes[0].plot(old_phase["k_grid"], old_phase["k_dot_zero"], color=COLORS["line_neutral"], label=r"$\dot{k}=0$ inicial")
    axes[0].plot(new_phase["k_grid"], new_phase["k_dot_zero"], color=COLORS["line_compare"], label=r"$\dot{k}=0$ apos choque")
    axes[0].axvline(old_phase["c_dot_zero"], color=COLORS["line_neutral"], linestyle="--", linewidth=1.2)
    axes[0].axvline(new_phase["c_dot_zero"], color=COLORS["line_compare"], linestyle="--", linewidth=1.2)
    if not old_phase["saddle_path"].empty:
        axes[0].plot(old_phase["saddle_path"]["k"], old_phase["saddle_path"]["c"], color=COLORS["line_neutral"], linewidth=2.0, label="Trajetoria inicial")
    if not new_phase["saddle_path"].empty:
        axes[0].plot(new_phase["saddle_path"]["k"], new_phase["saddle_path"]["c"], color=COLORS["black"], linewidth=2.2, label="Nova trajetoria")
    axes[0].scatter([old_steady["k_star"], shocked_steady["k_star"]], [old_steady["c_star"], shocked_steady["c_star"]], color=[COLORS["line_neutral"], COLORS["line_compare"]], s=32)
    style_axis(
        axes[0],
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Consumo por trabalhador efetivo, $c$",
    )
    axes[0].xaxis.set_major_formatter(plain_number_formatter(1))
    axes[0].yaxis.set_major_formatter(plain_number_formatter(1))
    style_legend(axes[0], loc="lower right")

    axes[1].plot(transition["time"], transition["k"], color=COLORS["line_main"], label="Capital")
    axes[1].plot(transition["time"], transition["c"], color=COLORS["line_compare"], label="Consumo")
    axes[1].axhline(shocked_steady["k_star"], color=COLORS["line_main"], linestyle=":", linewidth=1.0)
    axes[1].axhline(shocked_steady["c_star"], color=COLORS["line_compare"], linestyle=":", linewidth=1.0)
    style_axis(axes[1], xlabel="Tempo", ylabel="Nível")
    axes[1].xaxis.set_major_formatter(plain_number_formatter(0))
    axes[1].yaxis.set_major_formatter(plain_number_formatter(1))
    style_legend(axes[1], loc="upper right")

    return finalize_figure(
        fig,
        figure_path,
        title="Queda em rho no modelo RCK",
        subtitle="Menor impaciência desloca a trajetória de sela e aumenta o capital de longo prazo compatível com o equilíbrio ótimo.",
        note="O painel da direita mostra a transição a partir do capital inicial do antigo estado estacionário.",
        top=0.83,
        bottom=0.12,
    )


def plot_government_spending_effect(model: RCKModel, output_dir=FIGURES_DIR):
    steady = model.steady_state()
    higher_g_model = RCKModel(clone_params(model.params, {"G": 0.05 * model.f(steady["k_star"])}))
    baseline = model.phase_diagram_data(arrow_points=18)
    with_government = higher_g_model.phase_diagram_data(arrow_points=18)
    figure_path = Path(output_dir) / "rck_government_spending.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(baseline["k_grid"], baseline["k_dot_zero"], color=COLORS["line_main"], label="Linha base")
    ax.plot(with_government["k_grid"], with_government["k_dot_zero"], color=COLORS["line_compare"], label="G maior")
    ax.axvline(baseline["c_dot_zero"], color=COLORS["axis"], linestyle="--", linewidth=1.2)
    style_axis(
        ax,
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Consumo por trabalhador efetivo, $c$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    style_legend(ax, loc="upper right")
    return finalize_figure(
        fig,
        figure_path,
        title="Gastos do governo e a isóclina de k",
        subtitle="Um aumento de G desloca a condição de dot{k}=0 para baixo, reduzindo o espaço de consumo compatível com a acumulação de capital.",
        note="A isóclina de dot{c}=0 permanece fixa no exercício.",
    )


def plot_consumption_comparison(model: RCKModel, output_dir=FIGURES_DIR):
    steady = model.steady_state()
    k0 = 1.2 * steady["k_star"]
    saddle = model.find_saddle_path(k0, T=80.0)
    rck_path = saddle["simulation"]
    solow_model = SolowModel(
        clone_params(
            SOLOW,
            {"alpha": model.alpha, "n": model.n, "g": model.g, "delta": model.delta},
        )
    )
    solow_path = solow_model.transition_path(k0=k0, T=80.0, dt=0.1)
    figure_path = Path(output_dir) / "rck_vs_solow_consumption.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(rck_path["time"], rck_path["c"], color=COLORS["line_main"])
    ax.plot(solow_path["time"], solow_path["c"], color=COLORS["line_compare"], linestyle="--")
    ax.axhline(steady["c_star"], color=COLORS["axis"], linestyle=":", linewidth=1.1)
    direct_label_last(ax, rck_path["time"], rck_path["c"], label="RCK", color=COLORS["line_main"])
    direct_label_last(ax, solow_path["time"], solow_path["c"], label="Solow", color=COLORS["line_compare"], dy=-10)
    style_axis(
        ax,
        xlabel="Tempo",
        ylabel=r"Consumo por trabalhador efetivo, $c(t)$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    ax.margins(x=0.08)
    return finalize_figure(
        fig,
        figure_path,
        title="Consumo ótimo no RCK versus consumo no Solow",
        subtitle="Ao endogeneizar a poupança, o RCK escolhe uma trajetória de consumo diferente daquela implícita por uma taxa fixa de poupança.",
        note="A linha pontilhada marca o consumo de estado estacionário do RCK.",
    )


def main():
    model = RCKModel(RCK)
    output_paths = [
        plot_phase_diagram(model),
        plot_compare_rck_solow(model),
        plot_rho_shock(model),
        plot_government_spending_effect(model),
        plot_consumption_comparison(model),
    ]
    for path in output_paths:
        print(f"Saved {path}")
        print(f"Saved {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
