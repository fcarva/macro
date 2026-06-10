"""Chapter 9 -- Investment: Tobin's q and convex adjustment costs.

Implements the continuous-time model of Romer Ch. 9 (pp. 420-456):

  * The representative firm chooses investment I to maximize the present
    value of profits net of investment and adjustment costs
    C(I, K) = (a/2)(I/K)^2 K, subject to K_dot = I - delta*K.
  * The first-order condition gives I/K = (q-1)/a, where q is the costate
    (shadow value of capital, "marginal q").
  * The (K, q) phase diagram has a unique steady state (K*, q*) and a
    saddle path; q* = 1 + a*delta is independent of K*.
  * Hayashi's theorem: under constant returns to scale and perfect
    competition, marginal q equals average q (V(K)/K).

References: Romer (2019), Advanced Macroeconomics, 5th ed., Ch. 9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from params import INVESTMENT


class TobinQModel:
    """Continuous-time (K, q) model with convex adjustment costs.

    Parameters
    ----------
    params : dict, optional
        Keys: r, delta, a, alpha, tfp.
    """

    def __init__(self, params: dict | None = None):
        p = dict(INVESTMENT if params is None else params)
        self.r = float(p["r"])
        self.delta = float(p["delta"])
        self.a = float(p["a"])
        self.alpha = float(p["alpha"])
        self.tfp = float(p["tfp"])

    # ------------------------------------------------------------------
    # Production / profit function
    # ------------------------------------------------------------------

    def profit_marginal(self, K):
        """Marginal profit of capital, pi'(K) = alpha * tfp * K^(alpha-1)."""
        K = np.maximum(np.asarray(K, dtype=float), 1e-12)
        return self.alpha * self.tfp * np.power(K, self.alpha - 1.0)

    def profit_marginal_prime(self, K):
        """Derivative pi''(K) = alpha*(alpha-1) * tfp * K^(alpha-2) < 0."""
        K = np.maximum(np.asarray(K, dtype=float), 1e-12)
        return self.alpha * (self.alpha - 1.0) * self.tfp * np.power(K, self.alpha - 2.0)

    # ------------------------------------------------------------------
    # Optimal investment (FOC)
    # ------------------------------------------------------------------

    def investment_rate(self, q):
        """Optimal I/K = (q-1)/a from C_I(I,K) = q-1."""
        return (np.asarray(q, dtype=float) - 1.0) / self.a

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def k_dot(self, K, q):
        K = np.asarray(K, dtype=float)
        return K * (self.investment_rate(q) - self.delta)

    def q_dot(self, K, q):
        q = np.asarray(q, dtype=float)
        return ((self.r + self.delta) * q
                - self.profit_marginal(K)
                - (q - 1.0) ** 2 / (2.0 * self.a))

    def system(self, _time, state):
        K, q = state
        return [float(self.k_dot(K, q)), float(self.q_dot(K, q))]

    # ------------------------------------------------------------------
    # Steady state and local dynamics
    # ------------------------------------------------------------------

    def steady_state(self) -> dict:
        """Steady state (K*, q*).

        q* = 1 + a*delta (from K_dot=0), and K* solves
        pi'(K*) = (r+delta)*q* - a*delta^2/2 (from q_dot=0).
        """
        q_star = 1.0 + self.a * self.delta
        rhs = (self.r + self.delta) * q_star - self.a * self.delta ** 2 / 2.0
        K_star = (self.alpha * self.tfp / rhs) ** (1.0 / (1.0 - self.alpha))
        return {
            "K_star": float(K_star),
            "q_star": float(q_star),
            "I_star": float(self.delta * K_star),
        }

    def jacobian(self, K: float | None = None, q: float | None = None) -> np.ndarray:
        """Jacobian of (K_dot, q_dot) at (K, q), default at the steady state."""
        ss = self.steady_state()
        K = ss["K_star"] if K is None else K
        q = ss["q_star"] if q is None else q
        j11 = self.investment_rate(q) - self.delta
        j12 = K / self.a
        j21 = -self.profit_marginal_prime(K)
        j22 = (self.r + self.delta) - (q - 1.0) / self.a
        return np.array([[float(j11), float(j12)], [float(j21), float(j22)]])

    def eigen(self):
        """Eigenvalues/eigenvectors of the Jacobian at the steady state."""
        return np.linalg.eig(self.jacobian())

    def saddle_path_slope(self) -> tuple[float, float]:
        """Local slope dq/dK of the stable (saddle) arm at the steady state.

        Returns (slope, stable_eigenvalue).
        """
        eigvals, eigvecs = self.eigen()
        idx = int(np.argmin(eigvals.real))
        vec = eigvecs[:, idx]
        slope = (vec[1] / vec[0]).real
        return float(slope), float(eigvals[idx].real)

    # ------------------------------------------------------------------
    # Isoclines
    # ------------------------------------------------------------------

    def k_locus(self) -> float:
        """K_dot=0 isocline: q = 1 + a*delta (horizontal line)."""
        return 1.0 + self.a * self.delta

    def q_locus(self, K):
        """q_dot=0 isocline: solve the quadratic in q for given K.

        (q-1)^2/(2a) - (r+delta)*q + pi'(K) = 0
        => q^2 - [2a(r+delta)+2] q + [2a*pi'(K) + 1] = 0

        Returns the lower root (the economically relevant branch, with
        q near 1 for K near K*).
        """
        K = np.asarray(K, dtype=float)
        pi_K = self.profit_marginal(K)
        b = -(2.0 * self.a * (self.r + self.delta) + 2.0)
        c = 2.0 * self.a * pi_K + 1.0
        disc = np.maximum(b ** 2 - 4.0 * c, 0.0)
        q_low = (-b - np.sqrt(disc)) / 2.0
        q_high = (-b + np.sqrt(disc)) / 2.0
        return q_low, q_high

    # ------------------------------------------------------------------
    # Simulation and saddle path
    # ------------------------------------------------------------------

    def simulate(self, K0: float, q0: float, T: float = 60.0,
                  n_points: int = 600, max_step: float = 0.05) -> dict:
        """Integrate the (K, q) system forward from (K0, q0)."""
        if K0 <= 0:
            raise ValueError("K0 must be strictly positive.")

        t_eval = np.linspace(0.0, T, n_points)

        def blow_up(_time, state):
            K, q = state
            return min(K - 1e-8, 50.0 - abs(q))

        blow_up.terminal = True
        blow_up.direction = -1

        solution = solve_ivp(
            self.system,
            (0.0, T),
            np.array([K0, q0], dtype=float),
            t_eval=t_eval,
            max_step=max_step,
            rtol=1e-9,
            atol=1e-9,
            events=blow_up,
        )
        return {
            "time": solution.t,
            "K": solution.y[0],
            "q": solution.y[1],
            "success": solution.success,
        }

    def _terminal_gap(self, K0: float, q0: float, T: float) -> float:
        ss = self.steady_state()
        sim = self.simulate(K0, q0, T=T)
        return float(sim["q"][-1] - ss["q_star"])

    def find_saddle_path(self, K0: float, T: float = 40.0) -> dict:
        """Find q0 such that (K0, q0) lies on the saddle path (shooting)."""
        ss = self.steady_state()
        if np.isclose(K0, ss["K_star"], rtol=1e-8, atol=1e-8):
            return {
                "K0": ss["K_star"],
                "q0": ss["q_star"],
                "simulation": self.simulate(ss["K_star"], ss["q_star"], T=T),
            }

        slope, _ = self.saddle_path_slope()
        q0_guess = ss["q_star"] + slope * (K0 - ss["K_star"])

        span = max(abs(q0_guess - ss["q_star"]), 0.05)
        lower, upper = q0_guess - span, q0_guess + span
        gap_lower, gap_upper = self._terminal_gap(K0, lower, T), self._terminal_gap(K0, upper, T)

        expansions = 0
        while gap_lower * gap_upper > 0 and expansions < 30:
            span *= 1.5
            lower, upper = q0_guess - span, q0_guess + span
            gap_lower, gap_upper = self._terminal_gap(K0, lower, T), self._terminal_gap(K0, upper, T)
            expansions += 1

        if gap_lower * gap_upper > 0:
            raise RuntimeError("Failed to bracket the saddle-path q0 value.")

        root = lambda q0: self._terminal_gap(K0, q0, T)  # noqa: E731
        q0 = brentq(root, lower, upper, xtol=1e-10, rtol=1e-10, maxiter=200)
        return {"K0": float(K0), "q0": float(q0), "simulation": self.simulate(K0, q0, T=T)}

    def sample_saddle_path(self, n_points: int = 15,
                            k_min: float | None = None,
                            k_max: float | None = None,
                            T: float = 40.0) -> pd.DataFrame:
        """Sample (K0, q0) points along the saddle path."""
        ss = self.steady_state()
        lower = k_min or ss["K_star"] * 0.5
        upper = k_max or ss["K_star"] * 1.5
        records = []
        for K0 in np.linspace(lower, upper, n_points):
            try:
                saddle = self.find_saddle_path(float(K0), T=T)
                records.append({"K": saddle["K0"], "q": saddle["q0"]})
            except RuntimeError:
                continue
        if not records:
            return pd.DataFrame(columns=["K", "q"])
        return pd.DataFrame(records).sort_values("K").reset_index(drop=True)

    def phase_diagram_data(self, k_max: float | None = None, q_max: float | None = None,
                            grid_points: int = 200, arrow_points: int = 18,
                            saddle_points: int = 15) -> dict:
        """Grids, isoclines, and a quiver field for the (K, q) phase diagram."""
        ss = self.steady_state()
        k_upper = k_max or ss["K_star"] * 2.0
        q_upper = q_max or ss["q_star"] * 1.6

        k_grid = np.linspace(1e-3, k_upper, grid_points)
        q_low, _ = self.q_locus(k_grid)

        K, Q = np.meshgrid(np.linspace(1e-3, k_upper, arrow_points),
                            np.linspace(0.0, q_upper, arrow_points))
        dK = self.k_dot(K, Q)
        dQ = self.q_dot(K, Q)
        norm = np.sqrt(dK ** 2 + dQ ** 2) + 1e-12

        saddle_path = self.sample_saddle_path(n_points=saddle_points)

        return {
            "steady_state": ss,
            "k_grid": k_grid,
            "q_locus": q_low,
            "k_locus": self.k_locus(),
            "K": K,
            "Q": Q,
            "dK": dK / norm,
            "dQ": dQ / norm,
            "saddle_path": saddle_path,
        }

    # ------------------------------------------------------------------
    # Impulse responses
    # ------------------------------------------------------------------

    def irf(self, shock_type: str, shock_size: float = 0.05, T: float = 40.0) -> dict:
        """IRF for a permanent shock to TFP or the interest rate.

        The economy starts at the old steady state K0 = K*_old. The shock
        permanently changes a parameter; q jumps onto the new saddle path
        (through K0) and K converges to the new K*.
        """
        if shock_type == "productivity":
            new_params = {"r": self.r, "delta": self.delta, "a": self.a,
                           "alpha": self.alpha, "tfp": self.tfp * (1.0 + shock_size)}
        elif shock_type == "interest":
            new_params = {"r": self.r * (1.0 + shock_size), "delta": self.delta,
                           "a": self.a, "alpha": self.alpha, "tfp": self.tfp}
        else:
            raise ValueError("shock_type must be 'productivity' or 'interest'.")

        old_ss = self.steady_state()
        new_model = TobinQModel(new_params)
        new_ss = new_model.steady_state()

        K0 = old_ss["K_star"]
        saddle = new_model.find_saddle_path(K0, T=T)
        sim = saddle["simulation"]

        return {
            "time": sim["time"],
            "K": sim["K"],
            "q": sim["q"],
            "I_K": new_model.investment_rate(sim["q"]),
            "old_steady_state": old_ss,
            "new_steady_state": new_ss,
            "new_model": new_model,
        }


class AdjustmentCostFirm:
    """Single-firm investment decision and Hayashi's theorem.

    Under constant returns to scale and perfect competition, the firm's
    value function is linear in capital, V(K) = q*K, so "average q"
    (V(K)/K) coincides with "marginal q" (the costate q).

    Parameters
    ----------
    params : dict, optional
        Keys: a, delta.
    """

    def __init__(self, params: dict | None = None):
        p = dict(INVESTMENT if params is None else params)
        self.a = float(p["a"])
        self.delta = float(p["delta"])

    def investment_rate(self, q):
        """Optimal I/K = (q-1)/a."""
        return (np.asarray(q, dtype=float) - 1.0) / self.a

    def investment(self, K, q):
        return np.asarray(K, dtype=float) * self.investment_rate(q)

    def adjustment_cost(self, K, q):
        """C(I, K) = (a/2)(I/K)^2 K, evaluated at the optimal I/K."""
        i_k = self.investment_rate(q)
        return 0.5 * self.a * i_k ** 2 * np.asarray(K, dtype=float)

    def firm_value(self, K, marginal_q):
        """V(K) = q*K under CRS (linear value function)."""
        return np.asarray(marginal_q, dtype=float) * np.asarray(K, dtype=float)

    def average_q(self, K, marginal_q):
        """Average q = V(K) / K."""
        return self.firm_value(K, marginal_q) / np.asarray(K, dtype=float)

    def hayashi_holds(self, K, marginal_q, atol: float = 1e-10) -> bool:
        """Check Hayashi's theorem: average q = marginal q."""
        return bool(np.allclose(self.average_q(K, marginal_q), marginal_q, atol=atol))

    def i_k_schedule(self, q_grid):
        """The linear I/K = (q-1)/a relationship for a grid of q values."""
        q_grid = np.asarray(q_grid, dtype=float)
        return q_grid, self.investment_rate(q_grid)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test():
    model = TobinQModel()
    ss = model.steady_state()
    assert ss["K_star"] > 0 and ss["q_star"] > 1.0
    assert np.isclose(model.k_dot(ss["K_star"], ss["q_star"]), 0.0, atol=1e-8)
    assert np.isclose(model.q_dot(ss["K_star"], ss["q_star"]), 0.0, atol=1e-8)
    print(f"Steady state: K*={ss['K_star']:.4f}, q*={ss['q_star']:.4f}")

    eigvals, _ = model.eigen()
    n_stable = int(np.sum(eigvals.real < 0))
    assert n_stable == 1, "The (K,q) system should have exactly one stable eigenvalue (saddle)."
    print(f"Eigenvalues: {eigvals.real}")

    saddle = model.find_saddle_path(ss["K_star"] * 0.8)
    sim = saddle["simulation"]
    assert abs(sim["K"][-1] - ss["K_star"]) < 1e-2
    assert abs(sim["q"][-1] - ss["q_star"]) < 1e-2
    print(f"Saddle path from K0={saddle['K0']:.4f} found q0={saddle['q0']:.4f}")

    irf_tfp = model.irf("productivity", shock_size=0.05)
    assert irf_tfp["new_steady_state"]["K_star"] > irf_tfp["old_steady_state"]["K_star"]
    print(f"Productivity IRF: K* {irf_tfp['old_steady_state']['K_star']:.4f} "
          f"-> {irf_tfp['new_steady_state']['K_star']:.4f}")

    irf_r = model.irf("interest", shock_size=0.25)
    assert irf_r["new_steady_state"]["K_star"] < irf_r["old_steady_state"]["K_star"]
    print(f"Interest-rate IRF: K* {irf_r['old_steady_state']['K_star']:.4f} "
          f"-> {irf_r['new_steady_state']['K_star']:.4f}")

    firm = AdjustmentCostFirm()
    q_test = 1.2
    K_test = 5.0
    assert firm.hayashi_holds(K_test, q_test)
    assert np.isclose(firm.investment_rate(q_test), (q_test - 1.0) / firm.a)
    print("Hayashi's theorem holds: average q = marginal q.")

    print("All ch09 smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
