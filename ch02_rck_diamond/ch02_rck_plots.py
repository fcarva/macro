"""Editorial plot scripts for the Ramsey-Cass-Koopmans model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow import SolowModel
from ch02_rck_diamond.ch02_rck import RCKModel
from data_utils import ensure_directory
from params import RCK, SOLOW, clone_params
from plotting_style import (
    COLORS,
    add_callout,
    direct_label_last,
    finalize_figure,
    plain_number_formatter,
    style_axis,
)


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def plot_phase_diagram(model: RCKModel, output_dir=FIGURES_DIR):
    data = model.phase_diagram_data(arrow_points=18, saddle_points=8, saddle_T=45.0, include_saddle_path=True)
    saddle = data["saddle_path"]
    steady = data["steady_state"]
    figure_path = Path(output_dir) / "rck_phase_diagram.png"

    fig, ax = plt.subplots(figsize=(9.25, 6.1))
    stream = ax.streamplot(
        data["K"][0],
        data["C"][:, 0],
        data["dK"],
        data["dC"],
        color=COLORS["muted_light"],
        density=0.95,
        linewidth=0.7,
        arrowsize=0.72,
        zorder=1,
    )
    stream.lines.set_alpha(0.85)
    stream.arrows.set_alpha(0.7)

    ax.plot(data["k_grid"], data["k_dot_zero"], color=COLORS["line_main"], linewidth=2.4, zorder=3)
    ax.axvline(
        data["c_dot_zero"],
        color=COLORS["line_compare"],
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    if not saddle.empty:
        ax.plot(saddle["k"], saddle["c"], color=COLORS["black"], linewidth=2.65, zorder=4)
    ax.scatter(
        [steady["k_star"]],
        [steady["c_star"]],
        color=COLORS["highlight"],
        s=58,
        edgecolors=COLORS["paper"],
        linewidths=0.8,
        zorder=5,
    )

    style_axis(
        ax,
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Consumo por trabalhador efetivo, $c$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))

    add_callout(
        ax,
        text=r"$\dot{k}=0$",
        xy=(data["k_grid"][-56], data["k_dot_zero"][-56]),
        dx=18,
        dy=14,
        color=COLORS["line_main"],
        text_color=COLORS["line_main"],
        with_connector=False,
    )
    add_callout(
        ax,
        text=r"$\dot{c}=0$",
        xy=(data["c_dot_zero"], ax.get_ylim()[1] * 0.83),
        dx=12,
        dy=28,
        color=COLORS["line_compare"],
        text_color=COLORS["line_compare"],
        with_connector=False,
    )
    if not saddle.empty:
        saddle_target = saddle.iloc[-3]
        add_callout(
            ax,
            text="trajetória de sela",
            xy=(saddle_target["k"], saddle_target["c"]),
            dx=26,
            dy=24,
            color=COLORS["black"],
            text_color=COLORS["text"],
        )

    legend_handles = [
        Line2D([0], [0], color=COLORS["line_main"], linewidth=2.4, label=r"$\dot{k}=0$"),
        Line2D([0], [0], color=COLORS["line_compare"], linewidth=1.4, linestyle="--", label=r"$\dot{c}=0$"),
        Line2D([0], [0], color=COLORS["black"], linewidth=2.65, label="Trajetória de sela"),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=7,
            linestyle="None",
            markerfacecolor=COLORS["highlight"],
            markeredgecolor=COLORS["paper"],
            markeredgewidth=0.8,
            label="Estado estacionário",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    for text in ax.get_legend().get_texts():
        text.set_color(COLORS["text"])

    return finalize_figure(
        fig,
        figure_path,
        title="RCK: diagrama de fase",
        subtitle="A trajetória de sela identifica a combinação inicial de capital e consumo compatível com o equilíbrio ótimo.",
        note="O campo vetorial mostra a direção do sistema dinâmico em torno das isóclinas.",
        bottom=0.16,
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

    fig, ax = plt.subplots(figsize=(9.25, 5.55))
    ax.plot(rck_path["time"], rck_path["k"], color=COLORS["line_main"])
    ax.plot(solow_path["time"], solow_path["k"], color=COLORS["line_compare"], linestyle="--")
    ax.axhline(steady["k_star"], color=COLORS["axis_light"], linestyle=":", linewidth=1.15, alpha=0.95)

    style_axis(
        ax,
        xlabel="Tempo",
        ylabel=r"Capital por trabalhador efetivo, $k(t)$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    ax.margins(x=0.12)

    direct_label_last(ax, rck_path["time"], rck_path["k"], label="RCK", color=COLORS["line_main"], dx=10)
    direct_label_last(ax, solow_path["time"], solow_path["k"], label="Solow", color=COLORS["line_compare"], dx=10, dy=-10)
    add_callout(
        ax,
        text=r"$k^*$ do RCK",
        xy=(rck_path["time"][6], steady["k_star"]),
        dx=0,
        dy=10,
        color=COLORS["axis_light"],
        text_color=COLORS["axis"],
        with_connector=False,
    )

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

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7))

    axes[0].plot(old_phase["k_grid"], old_phase["k_dot_zero"], color=COLORS["line_neutral"], linewidth=2.2)
    axes[0].plot(new_phase["k_grid"], new_phase["k_dot_zero"], color=COLORS["line_compare"], linewidth=2.2)
    axes[0].axvline(
        old_phase["c_dot_zero"],
        color=COLORS["line_neutral"],
        linestyle="--",
        linewidth=1.25,
        alpha=0.85,
    )
    axes[0].axvline(
        new_phase["c_dot_zero"],
        color=COLORS["line_compare"],
        linestyle="--",
        linewidth=1.25,
        alpha=0.85,
    )
    if not old_phase["saddle_path"].empty:
        axes[0].plot(old_phase["saddle_path"]["k"], old_phase["saddle_path"]["c"], color=COLORS["line_neutral"], linewidth=2.0)
    if not new_phase["saddle_path"].empty:
        axes[0].plot(new_phase["saddle_path"]["k"], new_phase["saddle_path"]["c"], color=COLORS["black"], linewidth=2.25)
    axes[0].scatter(
        [old_steady["k_star"], shocked_steady["k_star"]],
        [old_steady["c_star"], shocked_steady["c_star"]],
        color=[COLORS["line_neutral"], COLORS["line_compare"]],
        s=42,
        edgecolors=COLORS["paper"],
        linewidths=0.8,
        zorder=5,
    )

    style_axis(
        axes[0],
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Consumo por trabalhador efetivo, $c$",
    )
    axes[0].xaxis.set_major_formatter(plain_number_formatter(1))
    axes[0].yaxis.set_major_formatter(plain_number_formatter(1))

    add_callout(
        axes[0],
        text=r"$\dot{k}=0$ inicial",
        xy=(old_phase["k_grid"][-44], old_phase["k_dot_zero"][-44]),
        dx=10,
        dy=16,
        color=COLORS["line_neutral"],
        text_color=COLORS["line_neutral"],
        with_connector=False,
    )
    add_callout(
        axes[0],
        text=r"$\dot{k}=0$ após queda de $\rho$",
        xy=(new_phase["k_grid"][-48], new_phase["k_dot_zero"][-48]),
        dx=10,
        dy=-18,
        color=COLORS["line_compare"],
        text_color=COLORS["line_compare"],
        with_connector=False,
    )
    if not old_phase["saddle_path"].empty:
        old_target = old_phase["saddle_path"].iloc[2]
        add_callout(
            axes[0],
            text="trajetória inicial",
            xy=(old_target["k"], old_target["c"]),
            dx=-56,
            dy=-10,
            color=COLORS["line_neutral"],
            text_color=COLORS["line_neutral"],
            fontsize=9.0,
            ha="right",
        )
    if not new_phase["saddle_path"].empty:
        new_target = new_phase["saddle_path"].iloc[-3]
        add_callout(
            axes[0],
            text="nova trajetória",
            xy=(new_target["k"], new_target["c"]),
            dx=18,
            dy=22,
            color=COLORS["black"],
            text_color=COLORS["text"],
        )
        add_callout(
            axes[0],
            text="estado\ninicial",
            xy=(old_steady["k_star"], old_steady["c_star"]),
            dx=-62,
            dy=18,
            color=COLORS["line_neutral"],
            text_color=COLORS["line_neutral"],
            fontsize=9.0,
            ha="right",
        )
        add_callout(
            axes[0],
            text="novo\nestado",
            xy=(shocked_steady["k_star"], shocked_steady["c_star"]),
            dx=16,
            dy=12,
            color=COLORS["line_compare"],
            text_color=COLORS["line_compare"],
            fontsize=9.0,
        )

    axes[1].plot(transition["time"], transition["k"], color=COLORS["line_main"])
    axes[1].plot(transition["time"], transition["c"], color=COLORS["line_compare"])
    axes[1].axhline(shocked_steady["k_star"], color=COLORS["axis_light"], linestyle=":", linewidth=1.0, alpha=0.95)
    axes[1].axhline(shocked_steady["c_star"], color=COLORS["axis_light"], linestyle=":", linewidth=1.0, alpha=0.95)

    style_axis(axes[1], xlabel="Tempo", ylabel="Nível")
    axes[1].xaxis.set_major_formatter(plain_number_formatter(0))
    axes[1].yaxis.set_major_formatter(plain_number_formatter(1))
    axes[1].margins(x=0.12)

    direct_label_last(axes[1], transition["time"], transition["k"], label="Capital", color=COLORS["line_main"], dx=10)
    direct_label_last(axes[1], transition["time"], transition["c"], label="Consumo", color=COLORS["line_compare"], dx=10)
    add_callout(
        axes[1],
        text=r"novo $k^*$",
        xy=(transition["time"][7], shocked_steady["k_star"]),
        dx=0,
        dy=10,
        color=COLORS["line_main"],
        text_color=COLORS["line_main"],
        with_connector=False,
    )
    add_callout(
        axes[1],
        text=r"novo $c^*$",
        xy=(transition["time"][7], shocked_steady["c_star"]),
        dx=0,
        dy=10,
        color=COLORS["line_compare"],
        text_color=COLORS["line_compare"],
        with_connector=False,
    )

    return finalize_figure(
        fig,
        figure_path,
        title="Menor impaciência no modelo RCK",
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

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.plot(baseline["k_grid"], baseline["k_dot_zero"], color=COLORS["line_main"])
    ax.plot(with_government["k_grid"], with_government["k_dot_zero"], color=COLORS["line_compare"])
    ax.axvline(
        baseline["c_dot_zero"],
        color=COLORS["axis_light"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.95,
    )

    style_axis(
        ax,
        xlabel=r"Capital por trabalhador efetivo, $k$",
        ylabel=r"Consumo por trabalhador efetivo, $c$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(1))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    ax.margins(x=0.12)

    direct_label_last(ax, baseline["k_grid"], baseline["k_dot_zero"], label="Linha base", color=COLORS["line_main"], dx=10)
    direct_label_last(ax, with_government["k_grid"], with_government["k_dot_zero"], label="G maior", color=COLORS["line_compare"], dx=10, dy=-8)
    add_callout(
        ax,
        text=r"$\dot{c}=0$",
        xy=(baseline["c_dot_zero"], ax.get_ylim()[1] * 0.84),
        dx=0,
        dy=0,
        color=COLORS["axis_light"],
        text_color=COLORS["axis"],
        ha="center",
        with_connector=False,
    )

    return finalize_figure(
        fig,
        figure_path,
        title="Gastos do governo e a isóclina de k",
        subtitle="Um aumento de G desloca a condição de acumulação nula para baixo, reduzindo o espaço de consumo compatível com a acumulação de capital.",
        note="A linha vertical indica a condição em que o consumo deixa de variar.",
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

    fig, ax = plt.subplots(figsize=(9.25, 5.55))
    ax.plot(rck_path["time"], rck_path["c"], color=COLORS["line_main"])
    ax.plot(solow_path["time"], solow_path["c"], color=COLORS["line_compare"], linestyle="--")
    ax.axhline(steady["c_star"], color=COLORS["axis_light"], linestyle=":", linewidth=1.1, alpha=0.95)

    style_axis(
        ax,
        xlabel="Tempo",
        ylabel=r"Consumo por trabalhador efetivo, $c(t)$",
    )
    ax.xaxis.set_major_formatter(plain_number_formatter(0))
    ax.yaxis.set_major_formatter(plain_number_formatter(1))
    ax.margins(x=0.12)

    direct_label_last(ax, rck_path["time"], rck_path["c"], label="RCK", color=COLORS["line_main"], dx=10)
    direct_label_last(ax, solow_path["time"], solow_path["c"], label="Solow", color=COLORS["line_compare"], dx=10, dy=-10)
    add_callout(
        ax,
        text=r"$c^*$ do RCK",
        xy=(rck_path["time"][6], steady["c_star"]),
        dx=0,
        dy=10,
        color=COLORS["axis_light"],
        text_color=COLORS["axis"],
        with_connector=False,
    )

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
