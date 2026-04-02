"""Core Solow growth model utilities for Chapter 1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from params import SOLOW


class SolowModel:
    """Solow model with Cobb-Douglas technology in intensive form."""

    def __init__(self, params: dict | None = None):
        self.params = dict(SOLOW if params is None else params)
        self.alpha = float(self.params["alpha"])
        self.s = float(self.params["s"])
        self.n = float(self.params["n"])
        self.g = float(self.params["g"])
        self.delta = float(self.params["delta"])
        self.A0 = float(self.params.get("A0", 1.0))
        self.L0 = float(self.params.get("L0", 1.0))
        self.dilution = self.n + self.g + self.delta

    def f(self, capital_per_effective_worker):
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        return np.power(np.maximum(capital, 0.0), self.alpha)

    def f_prime(self, capital_per_effective_worker):
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        capital_safe = np.maximum(capital, 1e-12)
        return self.alpha * np.power(capital_safe, self.alpha - 1.0)

    def consumption(self, capital_per_effective_worker, savings_rate: float | None = None):
        savings = self.s if savings_rate is None else float(savings_rate)
        return (1.0 - savings) * self.f(capital_per_effective_worker)

    def steady_state(self, savings_rate: float | None = None):
        savings = self.s if savings_rate is None else float(savings_rate)
        k_star = np.power(savings / self.dilution, 1.0 / (1.0 - self.alpha))
        y_star = float(self.f(k_star))
        c_star = float(self.consumption(k_star, savings))
        return {
            "s": savings,
            "k_star": float(k_star),
            "y_star": y_star,
            "c_star": c_star,
        }

    def golden_rule(self):
        k_gold = np.power(self.alpha / self.dilution, 1.0 / (1.0 - self.alpha))
        y_gold = float(self.f(k_gold))
        s_gold = float(self.dilution * k_gold / y_gold)
        c_gold = float(y_gold - self.dilution * k_gold)
        return {
            "k_gold": float(k_gold),
            "y_gold": y_gold,
            "c_gold": c_gold,
            "s_gold": s_gold,
        }

    def k_dot(self, capital_per_effective_worker, savings_rate: float | None = None):
        savings = self.s if savings_rate is None else float(savings_rate)
        capital = np.asarray(capital_per_effective_worker, dtype=float)
        return savings * self.f(capital) - self.dilution * capital

    def transition_path(self, k0: float, T: float = 200.0, dt: float = 0.1, savings_rate: float | None = None):
        if k0 <= 0:
            raise ValueError("k0 must be strictly positive.")
        if T <= 0 or dt <= 0:
            raise ValueError("T and dt must be strictly positive.")

        savings = self.s if savings_rate is None else float(savings_rate)
        steps = int(np.ceil(T / dt))
        time = np.linspace(0.0, steps * dt, steps + 1)
        capital = np.empty(steps + 1)
        capital[0] = float(k0)

        for step in range(steps):
            capital[step + 1] = max(capital[step] + self.k_dot(capital[step], savings) * dt, 1e-10)

        output = self.f(capital)
        consumption = self.consumption(capital, savings)
        return {
            "time": time,
            "k": capital,
            "y": output,
            "c": consumption,
            "s": savings,
            "dt": dt,
        }

    def solow_curves(
        self,
        k_grid=None,
        savings_rate: float | None = None,
        num_points: int = 400,
        max_k: float | None = None,
    ):
        savings = self.s if savings_rate is None else float(savings_rate)
        steady = self.steady_state(savings)
        upper_bound = max_k or steady["k_star"] * 2.25
        grid = np.asarray(k_grid if k_grid is not None else np.linspace(1e-4, upper_bound, num_points), dtype=float)
        output = self.f(grid)
        savings_curve = savings * output
        break_even = self.dilution * grid
        return pd.DataFrame(
            {
                "k": grid,
                "output": output,
                "savings_curve": savings_curve,
                "break_even_curve": break_even,
                "k_dot": savings_curve - break_even,
            }
        )

    def growth_accounting(
        self,
        data: pd.DataFrame,
        alpha: float | None = None,
        output_col: str = "output",
        capital_col: str = "capital",
        labor_col: str = "labor",
        technology_col: str | None = None,
    ):
        required = [output_col, capital_col, labor_col]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise KeyError(f"Missing columns for growth accounting: {missing}")

        accounting_alpha = self.alpha if alpha is None else float(alpha)
        working = data.copy().sort_index()
        for column in required:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            if (working[column] <= 0).any():
                raise ValueError(f"Column '{column}' must be strictly positive for log growth accounting.")

        log_output = np.log(working[output_col])
        log_capital = np.log(working[capital_col])
        log_labor = np.log(working[labor_col])

        working["output_growth"] = log_output.diff()
        working["capital_contribution"] = accounting_alpha * log_capital.diff()
        working["labor_contribution"] = (1.0 - accounting_alpha) * log_labor.diff()

        if technology_col and technology_col in working.columns:
            working[technology_col] = pd.to_numeric(working[technology_col], errors="coerce")
            if (working[technology_col] <= 0).any():
                raise ValueError(f"Column '{technology_col}' must be strictly positive for log growth accounting.")
            working["tfp_contribution"] = np.log(working[technology_col]).diff()
            explained = (
                working["capital_contribution"]
                + working["labor_contribution"]
                + working["tfp_contribution"]
            )
            working["residual_gap"] = working["output_growth"] - explained
        else:
            working["tfp_contribution"] = (
                working["output_growth"]
                - working["capital_contribution"]
                - working["labor_contribution"]
            )
            working["residual_gap"] = 0.0

        working["explained_growth"] = (
            working["capital_contribution"]
            + working["labor_contribution"]
            + working["tfp_contribution"]
        )
        working["output_growth_pct"] = 100.0 * working["output_growth"]
        working["capital_contribution_pct"] = 100.0 * working["capital_contribution"]
        working["labor_contribution_pct"] = 100.0 * working["labor_contribution"]
        working["tfp_contribution_pct"] = 100.0 * working["tfp_contribution"]
        working["tfp_index"] = np.exp(working["tfp_contribution"].fillna(0.0).cumsum())
        return working
