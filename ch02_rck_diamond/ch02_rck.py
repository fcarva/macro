"""Core Ramsey-Cass-Koopmans model utilities for Chapter 2A."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from params import RCK


class RCKModel:
    """Continuous-time Ramsey-Cass-Koopmans model."""

    def __init__(self, params: dict | None = None):
        self.params = dict(RCK if params is None else params)
        self.alpha = float(self.params["alpha"])
        self.rho = float(self.params["rho"])
        self.theta = float(self.params["theta"])
        self.n = float(self.params["n"])
        self.g = float(self.params["g"])
        self.delta = float(self.params["delta"])
        self.G = float(self.params.get("G", 0.0))
        self.dilution = self.n + self.g + self.delta

    def f(self, capital_per_effective_worker):
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        return np.power(np.maximum(capital, 0.0), self.alpha)

    def f_prime(self, capital_per_effective_worker):
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        capital_safe = np.maximum(capital, 1e-12)
        return self.alpha * np.power(capital_safe, self.alpha - 1.0)

    def k_locus(self, capital_per_effective_worker, government_spending: float | None = None):
        government = self.G if government_spending is None else float(government_spending)
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        return self.f(capital) - government - self.dilution * capital

    def k_dot(self, capital_per_effective_worker, consumption, government_spending: float | None = None):
        return self.k_locus(capital_per_effective_worker, government_spending) - consumption

    def c_dot(self, capital_per_effective_worker, consumption):
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        cons = np.asarray(consumption, dtype=float)
        return cons * (self.f_prime(capital) - self.delta - self.rho - self.theta * self.g) / self.theta

    def system(self, _time, state):
        capital, consumption = state
        if capital <= 0 or consumption <= 0:
            return np.array([0.0, 0.0])
        return np.array(
            [
                self.k_dot(capital, consumption),
                self.c_dot(capital, consumption),
            ]
        )

    def steady_state(self, numeric: bool = False):
        target_interest_rate = self.rho + self.theta * self.g + self.delta
        k_star_analytic = np.power(self.alpha / target_interest_rate, 1.0 / (1.0 - self.alpha))
        c_star_analytic = float(self.k_locus(k_star_analytic))
        if c_star_analytic <= 0:
            raise ValueError("Steady-state consumption is not positive under the current parameters.")

        if not numeric:
            return {
                "k_star": float(k_star_analytic),
                "c_star": c_star_analytic,
                "target_interest_rate": target_interest_rate,
                "G": self.G,
            }

        def euler_root(capital):
            return float(self.f_prime(capital) - target_interest_rate)

        upper_bound = max(10.0, k_star_analytic * 3.0)
        while euler_root(upper_bound) > 0:
            upper_bound *= 2.0

        k_star_numeric = brentq(euler_root, 1e-10, upper_bound)
        c_star_numeric = float(self.k_locus(k_star_numeric))
        return {
            "k_star": float(k_star_numeric),
            "c_star": c_star_numeric,
            "target_interest_rate": target_interest_rate,
            "G": self.G,
        }

    def simulate(
        self,
        k0: float,
        c0: float,
        T: float = 80.0,
        n_points: int = 1600,
        max_step: float = 0.1,
    ):
        if k0 <= 0 or c0 <= 0:
            raise ValueError("Initial conditions must be strictly positive.")
        if T <= 0:
            raise ValueError("T must be strictly positive.")

        t_eval = np.linspace(0.0, T, n_points)

        def infeasible_event(_time, state):
            capital, consumption = state
            return min(capital - 1e-8, consumption - 1e-8)

        infeasible_event.terminal = True
        infeasible_event.direction = -1

        solution = solve_ivp(
            self.system,
            (0.0, T),
            np.array([k0, c0], dtype=float),
            t_eval=t_eval,
            max_step=max_step,
            rtol=1e-8,
            atol=1e-8,
            events=infeasible_event,
        )

        terminal_reason = "completed"
        if solution.t_events and solution.t_events[0].size:
            terminal_reason = "infeasible"

        return {
            "time": solution.t,
            "k": solution.y[0],
            "c": solution.y[1],
            "success": solution.success,
            "status": solution.status,
            "message": solution.message,
            "terminal_reason": terminal_reason,
        }

    def _terminal_gap(self, k0: float, c0: float, T: float):
        steady = self.steady_state()
        simulation = self.simulate(k0, c0, T=T)
        return float(simulation["k"][-1] - steady["k_star"])

    def _find_consumption_bracket(self, k0: float, T: float, max_expansions: int = 20):
        steady = self.steady_state()
        c_star = steady["c_star"]
        lower = 1e-8
        upper = max(float(self.f(k0)) * 1.25 + c_star, c_star * 2.0, 1e-3)
        gap_lower = self._terminal_gap(k0, lower, T)
        gap_upper = self._terminal_gap(k0, upper, T)

        expansions = 0
        while gap_lower * gap_upper > 0 and expansions < max_expansions:
            upper *= 1.5
            gap_upper = self._terminal_gap(k0, upper, T)
            expansions += 1

        if gap_lower * gap_upper > 0:
            raise RuntimeError("Failed to bracket the saddle-path consumption value.")
        return lower, upper

    def find_saddle_path(self, k0: float, T: float = 120.0):
        steady = self.steady_state()
        if np.isclose(k0, steady["k_star"], rtol=1e-8, atol=1e-8):
            simulation = self.simulate(steady["k_star"], steady["c_star"], T=T)
            return {
                "k0": float(steady["k_star"]),
                "c0": float(steady["c_star"]),
                "bracket": (steady["c_star"], steady["c_star"]),
                "terminal_gap": 0.0,
                "simulation": simulation,
            }

        lower, upper = self._find_consumption_bracket(k0, T=T)
        root_function = lambda c0: self._terminal_gap(k0, c0, T)  # noqa: E731
        c0 = brentq(root_function, lower, upper, xtol=1e-8, rtol=1e-8, maxiter=200)
        simulation = self.simulate(k0, c0, T=T)
        terminal_gap = float(simulation["k"][-1] - steady["k_star"])
        return {
            "k0": float(k0),
            "c0": float(c0),
            "bracket": (float(lower), float(upper)),
            "terminal_gap": terminal_gap,
            "simulation": simulation,
        }

    def sample_saddle_path(
        self,
        n_points: int = 15,
        k_min: float | None = None,
        k_max: float | None = None,
        T: float = 80.0,
    ):
        steady = self.steady_state()
        lower = k_min or steady["k_star"] * 0.45
        upper = k_max or steady["k_star"] * 1.65
        k_values = np.linspace(lower, upper, n_points)
        records = []
        for capital in k_values:
            try:
                saddle = self.find_saddle_path(float(capital), T=T)
                records.append({"k": saddle["k0"], "c": saddle["c0"]})
            except RuntimeError:
                continue
        if not records:
            return pd.DataFrame(columns=["k", "c"])
        return pd.DataFrame(records).sort_values("k").reset_index(drop=True)

    def phase_diagram_data(
        self,
        k_max: float | None = None,
        c_max: float | None = None,
        grid_points: int = 250,
        arrow_points: int = 20,
        saddle_points: int = 15,
        saddle_T: float = 80.0,
        include_saddle_path: bool = False,
    ):
        steady = self.steady_state()
        k_upper = k_max or steady["k_star"] * 2.1
        c_upper = c_max or steady["c_star"] * 2.0

        k_grid = np.linspace(1e-4, k_upper, grid_points)
        c_grid = np.linspace(1e-4, c_upper, arrow_points)
        K, C = np.meshgrid(np.linspace(1e-4, k_upper, arrow_points), c_grid)
        dK = self.k_dot(K, C)
        dC = self.c_dot(K, C)
        norm = np.sqrt(dK ** 2 + dC ** 2) + 1e-12

        saddle_path = self.sample_saddle_path(n_points=saddle_points, T=saddle_T) if include_saddle_path else pd.DataFrame(columns=["k", "c"])

        return {
            "steady_state": steady,
            "k_grid": k_grid,
            "k_dot_zero": self.k_locus(k_grid),
            "c_dot_zero": steady["k_star"],
            "K": K,
            "C": C,
            "dK": dK / norm,
            "dC": dC / norm,
            "saddle_path": saddle_path,
        }
