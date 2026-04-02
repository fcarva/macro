"""Plot scripts for the Solow growth model."""

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


MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ensure_directory(MODULE_DIR / "figures")


def plot_solow_diagram(model: SolowModel, output_dir=FIGURES_DIR):
    curves = model.solow_curves()
    steady = model.steady_state()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curves["k"], curves["savings_curve"], label=r"$sf(k)$", lw=2.0)
    ax.plot(curves["k"], curves["break_even_curve"], label=r"$(n+g+\delta)k$", lw=2.0, linestyle="--")
    ax.axvline(steady["k_star"], color="black", linestyle=":", lw=1.2)
    ax.scatter([steady["k_star"]], [model.s * model.f(steady["k_star"])], color="black", zorder=5)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Per effective worker")
    ax.set_title("Solow diagram")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "solow_diagram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_phase_diagram(model: SolowModel, output_dir=FIGURES_DIR):
    curves = model.solow_curves()
    steady = model.steady_state()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(curves["k"], curves["k_dot"], color="steelblue", lw=2.0)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axvline(steady["k_star"], color="black", linestyle=":", lw=1.2)
    sample = curves.iloc[::25].copy()
    directions = np.sign(sample["k_dot"]).replace(0, 1)
    ax.quiver(
        sample["k"].to_numpy(),
        np.zeros(len(sample)),
        directions.to_numpy(),
        np.zeros(len(sample)),
        angles="xy",
        scale_units="xy",
        scale=8,
        width=0.003,
        color="gray",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\dot{k}$")
    ax.set_title("Solow phase diagram")
    fig.tight_layout()
    path = Path(output_dir) / "solow_phase_diagram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_transition(model: SolowModel, k0: float, output_dir=FIGURES_DIR):
    path_data = model.transition_path(k0, T=60.0, dt=0.2)
    steady = model.steady_state()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for axis, key, steady_key, title in zip(
        axes,
        ["k", "y", "c"],
        ["k_star", "y_star", "c_star"],
        ["Capital", "Output", "Consumption"],
    ):
        axis.plot(path_data["time"], path_data[key], lw=2.0)
        axis.axhline(steady[steady_key], color="black", linestyle="--", lw=1.0)
        axis.set_title(title)
        axis.set_xlabel("Time")

    fig.suptitle("Transition path in the Solow model")
    fig.tight_layout()
    path = Path(output_dir) / "solow_transition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_savings_shock(model: SolowModel, k0: float, new_s: float, output_dir=FIGURES_DIR):
    before = model.transition_path(k0, T=25.0, dt=0.2)
    shock_capital = float(before["k"][-1])
    shocked_model = SolowModel(clone_params(model.params, {"s": new_s}))
    after = shocked_model.transition_path(shock_capital, T=35.0, dt=0.2)

    time_after = after["time"] + before["time"][-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    series_map = [("k", "k_star"), ("y", "y_star"), ("c", "c_star")]

    for axis, (series_name, steady_name) in zip(axes, series_map):
        axis.plot(before["time"], before[series_name], color="steelblue", lw=2.0, label="Before shock")
        axis.plot(time_after, after[series_name], color="darkorange", lw=2.0, label="After shock")
        axis.axvline(before["time"][-1], color="black", linestyle=":", lw=1.0)
        axis.axhline(model.steady_state()[steady_name], color="steelblue", linestyle="--", lw=1.0)
        axis.axhline(shocked_model.steady_state()[steady_name], color="darkorange", linestyle="--", lw=1.0)
        axis.set_xlabel("Time")
        axis.set_title(series_name.upper())

    axes[0].legend(loc="best")
    fig.suptitle("Savings-rate shock in the Solow model")
    fig.tight_layout()
    path = Path(output_dir) / "solow_savings_shock.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_golden_rule(model: SolowModel, output_dir=FIGURES_DIR):
    savings_grid = np.linspace(0.05, 0.95, 181)
    steady_consumption = np.array([model.steady_state(savings)["c_star"] for savings in savings_grid])
    golden = model.golden_rule()
    current = model.steady_state()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(savings_grid, steady_consumption, color="seagreen", lw=2.0)
    ax.axvline(model.s, color="steelblue", linestyle="--", lw=1.2, label="Current s")
    ax.axvline(golden["s_gold"], color="darkorange", linestyle="--", lw=1.2, label="Golden-rule s")
    ax.scatter([model.s, golden["s_gold"]], [current["c_star"], golden["c_gold"]], color=["steelblue", "darkorange"])
    ax.set_xlabel("Savings rate")
    ax.set_ylabel("Steady-state consumption")
    ax.set_title("Golden rule consumption curve")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / "solow_golden_rule.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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


if __name__ == "__main__":
    main()
