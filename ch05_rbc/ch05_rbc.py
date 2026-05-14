"""Core Real Business Cycle model utilities for Chapter 5.

Implements two complementary analyses following Romer Chapter 5:

1. RBCModel — inelastic-labour version (N=1) solved by log-linearisation via
   undetermined coefficients. Used for impulse-response functions, stochastic
   simulations and second-moment comparisons.

2. LaborLeisureConditions — static analysis of the intratemporal optimality
   conditions when utility is u(C, l) = log C + b·log l (l = 1−N). Used to
   answer questions 5–7 of Lista I (marginal disutility of labour, optimal
   leisure ratio and effects of parameter changes).

Reference: Romer, D. (2019). Advanced Macroeconomics, 5th ed. Ch. 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from params import RBC


# ---------------------------------------------------------------------------
# Inelastic-labour RBC model
# ---------------------------------------------------------------------------


class RBCModel:
    """Discrete-time RBC model with inelastic labour supply (N=1) and log utility.

    Technology process:
        log z_{t+1} = rho_z log z_t + eps_{t+1},  eps ~ N(0, sigma_z^2)

    The model is solved by log-linearisation around the deterministic steady
    state (z*=1, k*=k_star), yielding exact linear policy functions:

        ĉ_t = a_k · k̂_t + a_z · ẑ_t
        k̂_{t+1} = A · k̂_t + B · ẑ_t

    where hatted variables denote log-deviations from steady state.

    Derivation of a_k via undetermined coefficients (Romer §5.5 generalised):

    Euler equation log-linearised:
        ĉ_t = ĉ_{t+1} − β r* [ẑ_{t+1} + (α−1) k̂_{t+1}]

    Using λ = a_k + β r*(1−α)  (> 0), matching the k̂-coefficient gives
    a quadratic with a unique stable root:
        −γ_c a_k² + (γ_k − 1 − m γ_c) a_k + m γ_k = 0
    where m = β r*(1−α), γ_k = (1−δ)+αη, γ_c = η·(C*/Y*), η = δ/(I*/Y*).
    """

    def __init__(self, params: dict | None = None):
        self.params = dict(RBC if params is None else params)
        self.alpha = float(self.params["alpha"])
        self.beta = float(self.params["beta"])
        self.delta = float(self.params["delta"])
        self.rho_z = float(self.params["rho_z"])
        self.sigma_z = float(self.params["sigma_z"])

    # ------------------------------------------------------------------
    # Steady state
    # ------------------------------------------------------------------

    def steady_state(self) -> dict:
        """Deterministic steady state with z*=1 and N*=1."""
        r_star = 1.0 / self.beta - (1.0 - self.delta)
        if r_star <= 0:
            raise ValueError("Steady-state interest rate is not positive; check beta and delta.")
        k_star = np.power(self.alpha / r_star, 1.0 / (1.0 - self.alpha))
        y_star = np.power(k_star, self.alpha)
        i_star = self.delta * k_star
        c_star = y_star - i_star
        if c_star <= 0:
            raise ValueError("Steady-state consumption is not positive; check parameters.")
        w_star = (1.0 - self.alpha) * y_star   # = (1-alpha) y* when N*=1
        return {
            "k_star": float(k_star),
            "y_star": float(y_star),
            "c_star": float(c_star),
            "i_star": float(i_star),
            "r_star": float(r_star),
            "w_star": float(w_star),
            "iy": float(i_star / y_star),
            "cy": float(c_star / y_star),
        }

    # ------------------------------------------------------------------
    # Log-linearisation via undetermined coefficients
    # ------------------------------------------------------------------

    def log_linearize(self) -> dict:
        """Solve the log-linearised model by undetermined coefficients.

        Returns policy coefficients for consumption (a_k, a_z) and transition
        coefficients for capital (A, B), plus derived investment coefficients
        (inv_k, inv_z).

        The quadratic for a_k always has two real roots; the stable one
        (|A| < 1) is selected.
        """
        ss = self.steady_state()
        r_star = ss["r_star"]
        cy = ss["cy"]
        iy = ss["iy"]

        # Auxiliary scalars ---------------------------------------------------
        eta = self.delta / iy          # = r_star / alpha (capital-output derivative)
        gamma_k = (1.0 - self.delta) + self.alpha * eta
        gamma_c = eta * cy
        m = self.beta * r_star * (1.0 - self.alpha)   # m > 0 always

        # Quadratic: -gamma_c a_k^2 + (gamma_k-1 - m*gamma_c) a_k + m*gamma_k = 0
        qa = -gamma_c
        qb = gamma_k - 1.0 - m * gamma_c
        qc = m * gamma_k
        discriminant = qb ** 2 - 4.0 * qa * qc   # = qb^2 + 4*gamma_c*m*gamma_k > 0
        if discriminant < 0:
            raise RuntimeError("Negative discriminant in log-linear solve; check parameters.")

        root1 = (-qb + np.sqrt(discriminant)) / (2.0 * qa)
        root2 = (-qb - np.sqrt(discriminant)) / (2.0 * qa)

        def transition_A(ak: float) -> float:
            return gamma_k - gamma_c * ak

        stable = [r for r in (root1, root2) if abs(transition_A(r)) < 1.0]
        if not stable:
            raise RuntimeError("No stable log-linear solution; check beta, delta, alpha.")
        a_k = min(stable, key=lambda r: abs(transition_A(r)))
        A = transition_A(a_k)

        # a_z solved analytically given a_k -----------------------------------
        lam = a_k + m       # λ = a_k + β r*(1-α)  > 0
        a_z = (lam * eta - self.beta * r_star * self.rho_z) / (
            1.0 - self.rho_z + lam * eta * cy
        )
        B = eta * (1.0 - cy * a_z)

        # Investment policy coefficients  î = (ŷ − cy·ĉ) / iy ---------------
        inv_k = (self.alpha - cy * a_k) / iy
        inv_z = (1.0 - cy * a_z) / iy

        return {
            "a_k": float(a_k),
            "a_z": float(a_z),
            "A": float(A),
            "B": float(B),
            "inv_k": float(inv_k),
            "inv_z": float(inv_z),
            "eta": float(eta),
            "gamma_k": float(gamma_k),
            "gamma_c": float(gamma_c),
            "m": float(m),
        }

    # ------------------------------------------------------------------
    # Impulse response functions
    # ------------------------------------------------------------------

    def irf(self, shock_size: float = 0.01, T: int = 40) -> dict:
        """Impulse response to a one-time positive TFP shock at t=0.

        Args:
            shock_size: Initial log-deviation of z (default 1%).
            T: Number of periods to simulate.

        Returns:
            Dict with arrays of log-deviations from steady state for
            k, z, y, c, i over periods 0..T.
        """
        ll = self.log_linearize()
        a_k, a_z = ll["a_k"], ll["a_z"]
        A, B = ll["A"], ll["B"]
        inv_k, inv_z = ll["inv_k"], ll["inv_z"]

        k_hat = np.zeros(T + 1)
        z_hat = np.zeros(T + 1)
        z_hat[0] = float(shock_size)

        for t in range(T):
            k_hat[t + 1] = A * k_hat[t] + B * z_hat[t]
            z_hat[t + 1] = self.rho_z * z_hat[t]

        y_hat = self.alpha * k_hat + z_hat
        c_hat = a_k * k_hat + a_z * z_hat
        i_hat = inv_k * k_hat + inv_z * z_hat

        return {
            "time": np.arange(T + 1),
            "k": k_hat,
            "z": z_hat,
            "y": y_hat,
            "c": c_hat,
            "i": i_hat,
            "ll": ll,
        }

    # ------------------------------------------------------------------
    # Stochastic simulation
    # ------------------------------------------------------------------

    def simulate(self, T: int = 200, seed: int | None = None) -> dict:
        """Stochastic simulation of the log-linearised RBC model.

        Args:
            T: Number of periods.
            seed: RNG seed for reproducibility.

        Returns:
            Dict with log-deviation time series k, z, y, c, i (length T+1).
        """
        rng = np.random.default_rng(seed)
        ll = self.log_linearize()
        a_k, a_z = ll["a_k"], ll["a_z"]
        A, B = ll["A"], ll["B"]
        inv_k, inv_z = ll["inv_k"], ll["inv_z"]

        eps = rng.normal(0.0, self.sigma_z, size=T)
        k_hat = np.zeros(T + 1)
        z_hat = np.zeros(T + 1)

        for t in range(T):
            z_hat[t + 1] = self.rho_z * z_hat[t] + eps[t]
            k_hat[t + 1] = A * k_hat[t] + B * z_hat[t]

        y_hat = self.alpha * k_hat + z_hat
        c_hat = a_k * k_hat + a_z * z_hat
        i_hat = inv_k * k_hat + inv_z * z_hat

        return {
            "time": np.arange(T + 1),
            "k": k_hat,
            "z": z_hat,
            "y": y_hat,
            "c": c_hat,
            "i": i_hat,
        }

    # ------------------------------------------------------------------
    # Simulated second moments
    # ------------------------------------------------------------------

    def moments(self, T_sim: int = 5000, n_draws: int = 50, seed: int = 0) -> dict:
        """Compute simulated second moments via Monte Carlo.

        Returns:
            Dict with std devs, relative std devs (to Y) and pairwise
            correlations of Y, C, I, K.
        """
        rng = np.random.default_rng(seed)
        collectors: dict[str, list[float]] = {
            key: [] for key in
            ["std_y", "std_c", "std_i", "std_k",
             "corr_cy", "corr_iy", "autocorr_y"]
        }

        for _ in range(n_draws):
            sim = self.simulate(T=T_sim, seed=int(rng.integers(int(1e9))))
            y = sim["y"][1:]
            c = sim["c"][1:]
            i = sim["i"][1:]
            k = sim["k"][1:]

            collectors["std_y"].append(float(np.std(y, ddof=1)))
            collectors["std_c"].append(float(np.std(c, ddof=1)))
            collectors["std_i"].append(float(np.std(i, ddof=1)))
            collectors["std_k"].append(float(np.std(k, ddof=1)))

            if np.std(y) > 0:
                if np.std(c) > 0:
                    collectors["corr_cy"].append(float(np.corrcoef(c, y)[0, 1]))
                if np.std(i) > 0:
                    collectors["corr_iy"].append(float(np.corrcoef(i, y)[0, 1]))
            if len(y) > 1:
                collectors["autocorr_y"].append(float(np.corrcoef(y[:-1], y[1:])[0, 1]))

        sy = float(np.mean(collectors["std_y"]))
        return {
            "std_y": sy,
            "std_c": float(np.mean(collectors["std_c"])),
            "std_i": float(np.mean(collectors["std_i"])),
            "std_k": float(np.mean(collectors["std_k"])),
            "rel_std_c": float(np.mean(collectors["std_c"])) / sy if sy > 0 else np.nan,
            "rel_std_i": float(np.mean(collectors["std_i"])) / sy if sy > 0 else np.nan,
            "rel_std_k": float(np.mean(collectors["std_k"])) / sy if sy > 0 else np.nan,
            "corr_cy": float(np.mean(collectors["corr_cy"])),
            "corr_iy": float(np.mean(collectors["corr_iy"])),
            "autocorr_y": float(np.mean(collectors["autocorr_y"])),
        }

    # ------------------------------------------------------------------
    # Capital stability analysis (Lista I Q4)
    # ------------------------------------------------------------------

    def capital_stability(self) -> dict:
        """Analyse capital-stock stability (Lista I, question 4).

        Equilibrium investment I*_t = δ K*_t ensures K̇ = 0.
        Effective investment I_t = K_{t+1} − (1−δ)K_t.
        Stability requires |∂K_{t+1}/∂K_t| < 1.

        From the log-linear transition:
            K̂_{t+1} = A · K̂_t + B · ẑ_t
        Stability is guaranteed if |A| < 1.
        """
        ss = self.steady_state()
        ll = self.log_linearize()
        A = ll["A"]
        return {
            "equilibrium_investment": float(ss["i_star"]),   # I* = delta * K*
            "transition_coefficient_A": float(A),
            "is_stable": abs(A) < 1.0,
            "convergence_rate": float(1.0 - A),             # speed of convergence
            "half_life_periods": float(np.log(0.5) / np.log(abs(A))),
        }


# ---------------------------------------------------------------------------
# Labour-leisure static conditions (Lista I Q5–7)
# ---------------------------------------------------------------------------


class LaborLeisureConditions:
    """Analytical labour-leisure optimality for Romer Ch. 5 RBC model.

    Period utility: u(C_t, l_t) = log(C_t) + b · log(l_t)
    where l_t = 1 − N_t is leisure and N_t ∈ [0,1] is labour supply.

    Intratemporal optimality (MRS = real wage):
        MU_leisure / MU_consumption = w_t
        (b / l_t) / (1 / C_t) = w_t
        b · C_t / l_t = w_t                              ...(*)

    This yields:
        l_t = b · C_t / w_t    (optimal leisure, Lista I Q6-7)
        C_t / l_t = w_t / b   (optimal consumption-leisure ratio)
    """

    def __init__(self, b: float = 2.0):
        if b <= 0:
            raise ValueError("Leisure weight b must be strictly positive.")
        self.b = float(b)

    # Q5 -------------------------------------------------------------------
    def marginal_disutility_labor(self, leisure: float | np.ndarray) -> float | np.ndarray:
        """Marginal disutility of labour (= MU of leisure) = b / l.

        Since utility rises with leisure and falls with labour,
        one additional unit of labour costs b/l in utility.
        """
        l = np.asarray(leisure, dtype=float)
        return self.b / np.maximum(l, 1e-12)

    # Q7 -------------------------------------------------------------------
    def optimal_leisure(self, consumption: float | np.ndarray,
                        wage: float | np.ndarray) -> float | np.ndarray:
        """Optimal leisure l* = b · C / w  from condition (*)."""
        c = np.asarray(consumption, dtype=float)
        w = np.asarray(wage, dtype=float)
        return self.b * c / np.maximum(w, 1e-12)

    def optimal_cn_ratio(self, wage: float | np.ndarray) -> float | np.ndarray:
        """Optimal C/l ratio = w / b."""
        w = np.asarray(wage, dtype=float)
        return w / self.b

    # Q6 -------------------------------------------------------------------
    def leisure_response_to_interest_rate(self) -> str:
        """Qualitative response of l_t to a rise in r_t (Lista I Q6a).

        Higher r_t → intertemporal substitution → work more today (N_t rises),
        so leisure l_t = 1 − N_t falls.  From the Euler equation a higher
        current r raises the opportunity cost of leisure today relative to
        future leisure, inducing households to substitute leisure
        intertemporally toward the future.
        """
        return "decrease"

    def leisure_response_to_future_wages(self) -> str:
        """Qualitative response of l_t to a rise in E_t[w_{t+1}] (Lista I Q6b).

        Higher future wages raise the return to working in t+1 relative to t.
        Households substitute work toward the future → more leisure today.
        l_t increases (intertemporal substitution of labour supply).
        """
        return "increase"

    # Q7 – effect of b on optimal leisure ----------------------------------
    def leisure_sensitivity_to_b(self, consumption: float,
                                 wage: float) -> float:
        """∂l*/∂b = C/w > 0: higher leisure weight → more leisure."""
        return float(consumption / max(wage, 1e-12))

    # Calibration: back out b from steady-state target n* ------------------
    def calibrate_b(self, rbc_model: RBCModel, n_star: float = 0.33) -> float:
        """Back out b so that N* = n_star at the RBC steady state (N=1 base).

        At SS with consistent endogenous labour (z*=1):
            w* = (1−α) y* / n*   (marginal product of labour)
            c* + i* = y*  with y* = k*^α · n*^(1−α)
            b · C* / l* = w*  →  b = w* · l* / C* = w*(1−n*)/C*

        This uses the inelastic-labour SS as an approximation.
        """
        if not (0.0 < n_star < 1.0):
            raise ValueError("n_star must be in (0, 1).")
        ss = rbc_model.steady_state()
        alpha = rbc_model.alpha
        # Adjust w* for n*: w*(n*) = (1-alpha) * y_base / n_base * (n*/1)^(-alpha)
        # Simpler: use w* = (1-alpha)*k_star^alpha (N=1 base model) as approximation
        w_star = ss["w_star"]
        c_star = ss["c_star"]
        l_star = 1.0 - n_star
        return float(w_star * l_star / c_star)

    def leisure_grid(self, wage_grid: np.ndarray,
                     consumption: float) -> pd.DataFrame:
        """Generate a labour-supply schedule for plotting."""
        l_values = self.optimal_leisure(consumption, wage_grid)
        n_values = 1.0 - l_values
        return pd.DataFrame({
            "wage": np.asarray(wage_grid, dtype=float),
            "leisure": l_values,
            "labor": n_values,
            "marginal_disutility": self.marginal_disutility_labor(l_values),
        })
