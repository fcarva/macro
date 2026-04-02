"""Plot scripts for the Ramsey-Cass-Koopmans model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow import SolowModel
from ch02_rck_diamond.ch02_rck import RCKModel
from data_utils import ensure_directory
from params import RCK, SOLOW, clone_params


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def plot_phase_diagram(model: RCKModel, output_dir=FIGURES_DIR):
    data = model.phase_diagram_data(arrow_points=14, saddle_points=4, saddle_T=30.0, include_saddle_path=True)
    saddle = data["saddle_path"]
    steady = data["steady_state"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["k_grid"], data["k_dot_zero"], lw=2.0, color="steelblue", label=r"$\dot{k}=0$")
    ax.axvline(data["c_dot_zero"], lw=2.0, color="darkorange", linestyle="--", label=r"$\dot{c}=0$")
    ax.quiver(data["K"], data["C"], data["dK"], data["dC"], color="gray", alpha=0.35)
    if not saddle.empty:
        ax.plot(saddle["k"], saddle["c"], color="black", lw=2.2, label="Saddle path")
    ax.scatter([steady["k_star"]], [steady["c_star"]], color="black", zorder=5)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$c$")
    ax.set_title("RCK phase diagram")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "rck_phase_diagram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_compare_rck_solow(model: RCKModel, output_dir=FIGURES_DIR):
    steady = model.steady_state()
    k0 = 0.8 * steady["k_star"]
    saddle = model.find_saddle_path(k0, T=80.0)
    rck_path = saddle["simulation"]

    solow_model = SolowModel(
        clone_params(
            SOLOW,
            {
                "alpha": model.alpha,
                "n": model.n,
                "g": model.g,
                "delta": model.delta,
            },
        )
    )
    solow_path = solow_model.transition_path(k0=k0, T=80.0, dt=0.1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rck_path["time"], rck_path["k"], label="RCK capital", lw=2.0)
    ax.plot(solow_path["time"], solow_path["k"], label="Solow capital", lw=2.0, linestyle="--")
    ax.axhline(steady["k_star"], color="black", linestyle=":", lw=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$k(t)$")
    ax.set_title("Capital transition: RCK vs Solow")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "rck_vs_solow_capital.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_rho_shock(model: RCKModel, output_dir=FIGURES_DIR):
    old_steady = model.steady_state()
    shocked_model = RCKModel(clone_params(model.params, {"rho": 0.75 * model.rho}))
    shocked_steady = shocked_model.steady_state()
    shocked_saddle = shocked_model.find_saddle_path(old_steady["k_star"], T=80.0)
    transition = shocked_saddle["simulation"]

    old_phase = model.phase_diagram_data(arrow_points=14, saddle_points=4, saddle_T=30.0, include_saddle_path=True)
    new_phase = shocked_model.phase_diagram_data(arrow_points=14, saddle_points=4, saddle_T=30.0, include_saddle_path=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(old_phase["k_grid"], old_phase["k_dot_zero"], color="steelblue", lw=2.0, label="Old " + r"$\dot{k}=0$")
    axes[0].plot(new_phase["k_grid"], new_phase["k_dot_zero"], color="darkorange", lw=2.0, label="New " + r"$\dot{k}=0$")
    axes[0].axvline(old_phase["c_dot_zero"], color="steelblue", linestyle="--", lw=1.4)
    axes[0].axvline(new_phase["c_dot_zero"], color="darkorange", linestyle="--", lw=1.4)
    if not old_phase["saddle_path"].empty:
        axes[0].plot(old_phase["saddle_path"]["k"], old_phase["saddle_path"]["c"], color="steelblue", lw=1.8, label="Old saddle")
    if not new_phase["saddle_path"].empty:
        axes[0].plot(new_phase["saddle_path"]["k"], new_phase["saddle_path"]["c"], color="darkorange", lw=1.8, label="New saddle")
    axes[0].scatter([old_steady["k_star"], shocked_steady["k_star"]], [old_steady["c_star"], shocked_steady["c_star"]], color=["steelblue", "darkorange"])
    axes[0].set_xlabel(r"$k$")
    axes[0].set_ylabel(r"$c$")
    axes[0].set_title("Fall in rho and the saddle path")
    axes[0].legend(fontsize=8)

    axes[1].plot(transition["time"], transition["k"], lw=2.0, label="Capital")
    axes[1].plot(transition["time"], transition["c"], lw=2.0, label="Consumption")
    axes[1].axhline(shocked_steady["k_star"], color="steelblue", linestyle=":", lw=1.0)
    axes[1].axhline(shocked_steady["c_star"], color="darkorange", linestyle=":", lw=1.0)
    axes[1].set_xlabel("Time")
    axes[1].set_title("Transition after rho shock")
    axes[1].legend()

    fig.tight_layout()
    path = Path(output_dir) / "rck_rho_shock.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_government_spending_effect(model: RCKModel, output_dir=FIGURES_DIR):
    steady = model.steady_state()
    higher_g_model = RCKModel(clone_params(model.params, {"G": 0.05 * model.f(steady["k_star"])}))
    baseline = model.phase_diagram_data(arrow_points=14)
    with_government = higher_g_model.phase_diagram_data(arrow_points=14)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(baseline["k_grid"], baseline["k_dot_zero"], color="steelblue", lw=2.0, label="Baseline " + r"$\dot{k}=0$")
    ax.plot(with_government["k_grid"], with_government["k_dot_zero"], color="darkorange", lw=2.0, label="Higher G " + r"$\dot{k}=0$")
    ax.axvline(baseline["c_dot_zero"], color="black", linestyle="--", lw=1.2, label=r"$\dot{c}=0$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$c$")
    ax.set_title("Government spending shifts the k-dot locus")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "rck_government_spending.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_consumption_comparison(model: RCKModel, output_dir=FIGURES_DIR):
    steady = model.steady_state()
    k0 = 1.2 * steady["k_star"]
    saddle = model.find_saddle_path(k0, T=80.0)
    rck_path = saddle["simulation"]
    solow_model = SolowModel(
        clone_params(
            SOLOW,
            {
                "alpha": model.alpha,
                "n": model.n,
                "g": model.g,
                "delta": model.delta,
            },
        )
    )
    solow_path = solow_model.transition_path(k0=k0, T=80.0, dt=0.1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rck_path["time"], rck_path["c"], lw=2.0, label="RCK consumption")
    ax.plot(solow_path["time"], solow_path["c"], lw=2.0, linestyle="--", label="Solow consumption")
    ax.axhline(steady["c_star"], color="black", linestyle=":", lw=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$c(t)$")
    ax.set_title("Optimal consumption versus Solow consumption")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "rck_vs_solow_consumption.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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


if __name__ == "__main__":
    main()
