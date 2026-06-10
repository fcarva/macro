"""Core consumption models for Chapter 8.

Implements the canonical consumption-theory models from Romer Chapter 8
(pp. 368-418):

    PermanentIncomeModel  -- Friedman's permanent-income hypothesis (PIH).
                              Consumption is the annuity value of total
                              wealth (financial + human).
    HallRandomWalk        -- Hall (1978): with quadratic utility and
                              beta(1+r)=1, the Euler equation implies
                              c_t = E_t[c_{t+1}], i.e. consumption follows
                              a random walk.
    BufferStockModel      -- Precautionary saving with a borrowing
                              constraint, solved by value-function
                              iteration (VFI) on a discretized asset grid.
    CampbellMankiw        -- Campbell-Mankiw (1989) "lambda" model:
                              a fraction lambda of consumers are
                              rule-of-thumb (Delta c = Delta y) and the
                              rest are PIH consumers (Delta c = innovation
                              to permanent income).

Reference: Romer, D. (2019). Advanced Macroeconomics, 5th ed., Ch. 8.
"""

from __future__ import annotations

import numpy as np

from params import CONSUMPTION


class PermanentIncomeModel:
    """Friedman/Hall permanent-income model with AR(1) labor income.

    Income follows y_t = ybar + rho_y*(y_{t-1} - ybar) + eps_t. Human
    wealth h_t is the present value of the expected future income stream
    discounted at rate r. Consumption is the annuity (perpetuity) value
    of total wealth a_t + h_t:

        c_t = (r / (1+r)) * (a_t + h_t)

    Parameters
    ----------
    params : dict, optional
        Keys: r, rho_y, sigma_y, y_bar.
    """

    def __init__(self, params: dict | None = None):
        p = dict(CONSUMPTION if params is None else params)
        self.r = float(p.get("r", 0.03))
        self.rho_y = float(p.get("rho_y", 0.90))
        self.sigma_y = float(p.get("sigma_y", 0.10))
        self.y_bar = float(p.get("y_bar", 1.0))

    # ------------------------------------------------------------------
    # Human wealth and permanent income
    # ------------------------------------------------------------------

    def human_wealth(self, y: float) -> float:
        """Present value of the expected future income stream given y_t.

        h_t = sum_{s=0}^inf E_t[y_{t+s}] / (1+r)^s
            = ybar*(1+r)/r + (y_t - ybar) / (1 - rho_y/(1+r))
        """
        annuity_const = self.y_bar * (1.0 + self.r) / self.r
        denom = 1.0 - self.rho_y / (1.0 + self.r)
        deviation_term = (y - self.y_bar) / denom
        return float(annuity_const + deviation_term)

    def permanent_income(self, a: float, y: float) -> tuple[float, float]:
        """Optimal consumption c_t and human wealth h_t given (a_t, y_t).

        c_t = (r/(1+r)) * (a_t + h_t)
        """
        h = self.human_wealth(y)
        c = (self.r / (1.0 + self.r)) * (a + h)
        return float(c), float(h)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, T: int = 200, a0: float = 0.0, seed: int | None = None) -> dict:
        """Simulate the PIH economy for T periods.

        Income evolves as an AR(1) around y_bar; assets accumulate at
        rate r net of consumption: a_{t+1} = (1+r)(a_t + y_t - c_t).

        Returns
        -------
        dict with arrays time, y, c, a, h of length T+1.
        """
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, self.sigma_y, size=T)

        y = np.empty(T + 1)
        a = np.empty(T + 1)
        c = np.empty(T + 1)
        h = np.empty(T + 1)

        y[0] = self.y_bar
        a[0] = a0
        c[0], h[0] = self.permanent_income(a[0], y[0])

        for t in range(T):
            y[t + 1] = self.y_bar + self.rho_y * (y[t] - self.y_bar) + eps[t]
            a[t + 1] = (1.0 + self.r) * (a[t] + y[t] - c[t])
            c[t + 1], h[t + 1] = self.permanent_income(a[t + 1], y[t + 1])

        return {"time": np.arange(T + 1), "y": y, "c": c, "a": a, "h": h}

    def random_walk_test(self, T: int = 5000, seed: int | None = 0) -> dict:
        """Check that consumption changes are unpredictable (Hall 1978).

        Under the PIH, Delta c_t is proportional to the innovation in
        permanent income and therefore should be uncorrelated with its
        own lag. Returns the mean of Delta c and the slope of a
        regression of Delta c_t on Delta c_{t-1}.
        """
        sim = self.simulate(T=T, seed=seed)
        dc = np.diff(sim["c"])
        dc_t = dc[1:]
        dc_lag = dc[:-1]
        slope = float(np.cov(dc_t, dc_lag, ddof=1)[0, 1] / np.var(dc_lag, ddof=1))
        return {"mean_dc": float(np.mean(dc)), "autocorr_slope": slope}


class HallRandomWalk:
    """Hall (1978): quadratic utility -> consumption is a martingale.

    With u(c) = c - (b/2)c^2, marginal utility u'(c) = 1 - b*c is linear
    in c. The Euler equation u'(c_t) = beta(1+r)E_t[u'(c_{t+1})], combined
    with beta(1+r)=1, collapses to:

        c_t = E_t[c_{t+1}]

    i.e. consumption follows a random walk: c_{t+1} = c_t + eps_{t+1},
    with E_t[eps_{t+1}] = 0.

    Parameters
    ----------
    params : dict, optional
        Keys: beta, r.
    """

    def __init__(self, params: dict | None = None):
        p = dict(CONSUMPTION if params is None else params)
        self.beta = float(p.get("beta", 0.96))
        self.r = float(p.get("r", 0.03))

    def euler_holds(self, atol: float = 1e-6) -> bool:
        """True if beta(1+r) = 1 (the condition behind c_t = E_t[c_{t+1}])."""
        return bool(np.isclose(self.beta * (1.0 + self.r), 1.0, atol=atol))

    def simulate_martingale(self, c0: float = 1.0, T: int = 200,
                             sigma: float = 0.05, seed: int | None = None) -> dict:
        """Simulate c_{t+1} = c_t + eps_{t+1}, eps ~ N(0, sigma^2)."""
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, sigma, size=T)
        c = np.empty(T + 1)
        c[0] = c0
        for t in range(T):
            c[t + 1] = c[t] + eps[t]
        return {"time": np.arange(T + 1), "c": c, "eps": eps}

    def test_martingale_property(self, n_sims: int = 1000, T: int = 50,
                                  sigma: float = 0.05, seed: int | None = 0) -> dict:
        """Monte Carlo check that E[c_{t+1} - c_t] = 0 across simulations.

        Returns the average increment across simulations and time, which
        should be close to zero (the martingale property).
        """
        rng = np.random.default_rng(seed)
        increments = []
        for _ in range(n_sims):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            sim = self.simulate_martingale(c0=1.0, T=T, sigma=sigma, seed=sub_seed)
            increments.append(np.diff(sim["c"]))
        increments = np.concatenate(increments)
        return {
            "mean_increment": float(np.mean(increments)),
            "std_increment": float(np.std(increments, ddof=1)),
            "n_increments": int(increments.size),
        }


class BufferStockModel:
    """Precautionary saving with a borrowing constraint (buffer-stock).

    The household solves, by value-function iteration on a grid of
    end-of-period assets a:

        V(a, y) = max_{0 <= a' <= m}  u(m - a') + beta * E[V(a', y') | y]

    where m = (1+r)*a + y is cash-on-hand, u(c) = c^(1-theta)/(1-theta)
    (CRRA), and y follows a 2-state Markov chain {y_low, y_high} with
    persistence `prob_stay`. The borrowing constraint a' >= 0 (no
    borrowing) generates precautionary saving: u'''(c) > 0 implies that
    income uncertainty raises the marginal utility of an extra unit of
    assets, so the household holds a "buffer stock" of wealth.

    Parameters
    ----------
    params : dict, optional
        Keys: beta, r, theta, y_low, y_high, prob_stay, a_max, n_grid.
    """

    def __init__(self, params: dict | None = None):
        p = dict(CONSUMPTION if params is None else params)
        self.beta = float(p.get("beta", 0.96))
        self.r = float(p.get("r", 0.03))
        self.theta = float(p.get("theta", 2.0))
        self.y_low = float(p.get("y_low", 0.7))
        self.y_high = float(p.get("y_high", 1.3))
        self.prob_stay = float(p.get("prob_stay", 0.90))
        self.a_max = float(p.get("a_max", 10.0))
        self.n_grid = int(p.get("n_grid", 60))

        self.a_grid = np.linspace(0.0, self.a_max, self.n_grid)
        self.y_states = np.array([self.y_low, self.y_high])
        self.P = np.array([
            [self.prob_stay, 1.0 - self.prob_stay],
            [1.0 - self.prob_stay, self.prob_stay],
        ])

        self.V: np.ndarray | None = None
        self.policy_c: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def utility(self, c) -> np.ndarray:
        """CRRA utility u(c) = c^(1-theta)/(1-theta), or log(c) if theta=1."""
        c = np.maximum(np.asarray(c, dtype=float), 1e-10)
        if abs(self.theta - 1.0) < 1e-10:
            return np.log(c)
        return (c ** (1.0 - self.theta)) / (1.0 - self.theta)

    # ------------------------------------------------------------------
    # Value-function iteration
    # ------------------------------------------------------------------

    def solve(self, tol: float = 1e-7, max_iter: int = 2000) -> dict:
        """Solve the household's problem by VFI on the asset grid.

        Returns
        -------
        dict with V (value function), policy_c (consumption policy),
        a_grid, y_states, and the number of iterations to convergence.
        """
        n = self.n_grid
        V = np.zeros((n, 2))
        policy_c = np.zeros((n, 2))

        for it in range(1, max_iter + 1):
            V_new = np.empty_like(V)
            for iy, y in enumerate(self.y_states):
                m = (1.0 + self.r) * self.a_grid + y
                EV = self.P[iy, 0] * V[:, 0] + self.P[iy, 1] * V[:, 1]
                for ia, cash in enumerate(m):
                    feasible = self.a_grid <= cash
                    a_next = self.a_grid[feasible]
                    c = cash - a_next
                    obj = self.utility(c) + self.beta * EV[feasible]
                    best = int(np.argmax(obj))
                    V_new[ia, iy] = obj[best]
                    policy_c[ia, iy] = c[best]
            diff = float(np.max(np.abs(V_new - V)))
            V = V_new
            if diff < tol:
                break

        self.V = V
        self.policy_c = policy_c
        return {
            "V": V,
            "policy_c": policy_c,
            "a_grid": self.a_grid,
            "y_states": self.y_states,
            "iterations": it,
            "diff": diff,
        }

    def policy_function(self) -> dict:
        """Return the consumption policy c*(a, y), solving first if needed."""
        if self.policy_c is None:
            self.solve()
        return {
            "a_grid": self.a_grid,
            "y_states": self.y_states,
            "policy_c": self.policy_c,
        }

    # ------------------------------------------------------------------
    # Panel simulation
    # ------------------------------------------------------------------

    def simulate_panel(self, N: int = 500, T: int = 200,
                        seed: int | None = 0, burn_in: int = 50) -> dict:
        """Simulate N households for T periods using the optimal policy.

        Returns
        -------
        dict with arrays a (T+1, N), c (T, N), y (T, N) of asset, consumption
        and income paths (the first `burn_in` periods are included so the
        caller can discard the transient).
        """
        if self.policy_c is None:
            self.solve()

        rng = np.random.default_rng(seed)
        m_grid = {
            0: (1.0 + self.r) * self.a_grid + self.y_states[0],
            1: (1.0 + self.r) * self.a_grid + self.y_states[1],
        }

        a = np.zeros((T + 1, N))
        c = np.zeros((T, N))
        y_idx = rng.integers(0, 2, size=N)
        y_path = np.zeros((T, N), dtype=int)

        for t in range(T):
            y_path[t] = y_idx
            m = (1.0 + self.r) * a[t] + self.y_states[y_idx]

            c_low = np.interp(m, m_grid[0], self.policy_c[:, 0])
            c_high = np.interp(m, m_grid[1], self.policy_c[:, 1])
            c[t] = np.where(y_idx == 0, c_low, c_high)
            c[t] = np.minimum(c[t], m)  # respect cash-on-hand

            a[t + 1] = np.maximum(m - c[t], 0.0)

            stay_prob = np.where(y_idx == 0, self.P[0, 0], self.P[1, 1])
            draw = rng.random(N)
            stay = draw < stay_prob
            y_idx = np.where(stay, y_idx, 1 - y_idx)

        return {
            "a": a[burn_in:],
            "c": c[burn_in:],
            "y": self.y_states[y_path[burn_in:]],
            "n_households": N,
            "n_periods": T - burn_in,
        }


class CampbellMankiw:
    """Campbell-Mankiw (1989) "lambda" rule-of-thumb consumption model.

    A fraction lambda of consumers are rule-of-thumb ("hand-to-mouth"):
    they simply consume their current income, so Delta c = Delta y for
    them. The remaining (1-lambda) are PIH consumers whose consumption
    growth reflects only innovations to permanent income, eps_t:

        Delta c_t = lambda * Delta y_t + (1 - lambda) * eps_t

    Parameters
    ----------
    params : dict, optional
        Keys: lambda_rt, sigma_y.
    """

    def __init__(self, params: dict | None = None):
        p = dict(CONSUMPTION if params is None else params)
        self.lam = float(p.get("lambda_rt", 0.5))
        self.sigma_y = float(p.get("sigma_y", 0.10))

    def simulate(self, T: int = 200, seed: int | None = 0,
                  sigma_eps: float = 0.05) -> dict:
        """Simulate Delta y_t and Delta c_t for T periods."""
        rng = np.random.default_rng(seed)
        dy = rng.normal(0.0, self.sigma_y, size=T)
        eps = rng.normal(0.0, sigma_eps, size=T)
        dc = self.lam * dy + (1.0 - self.lam) * eps
        return {"time": np.arange(T), "dy": dy, "dc": dc, "eps": eps}

    def estimate_lambda(self, dc: np.ndarray, dy: np.ndarray) -> dict:
        """OLS estimate of lambda in Delta c_t = a + lambda * Delta y_t + u_t."""
        dc = np.asarray(dc, dtype=float)
        dy = np.asarray(dy, dtype=float)
        X = np.column_stack([np.ones_like(dy), dy])
        coeffs, _, _, _ = np.linalg.lstsq(X, dc, rcond=None)
        y_hat = X @ coeffs
        ss_res = float(np.sum((dc - y_hat) ** 2))
        ss_tot = float(np.sum((dc - np.mean(dc)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "intercept": float(coeffs[0]),
            "lambda_hat": float(coeffs[1]),
            "r_squared": r2,
            "n_obs": int(len(dc)),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test():
    pih = PermanentIncomeModel()
    c0, h0 = pih.permanent_income(a=0.0, y=pih.y_bar)
    assert np.isclose(c0, pih.y_bar, atol=1e-8), "c = ybar at the deterministic SS."
    sim = pih.simulate(T=2000, seed=0)
    assert np.all(np.isfinite(sim["c"]))
    rwt = pih.random_walk_test(T=5000, seed=0)
    assert abs(rwt["autocorr_slope"]) < 0.1, "Delta c should be (close to) unpredictable."
    print(f"PIH random-walk test: autocorr(Delta c) = {rwt['autocorr_slope']:.4f}")

    hall = HallRandomWalk({"beta": 1.0 / 1.03, "r": 0.03})
    assert hall.euler_holds(), "beta = 1/(1+r) should satisfy beta(1+r)=1."
    mart = hall.test_martingale_property(n_sims=500, T=50, seed=0)
    assert abs(mart["mean_increment"]) < 0.01
    print(f"Hall martingale test: mean(Delta c) = {mart['mean_increment']:.5f}")

    buf = BufferStockModel({**CONSUMPTION, "n_grid": 30})
    sol = buf.solve(max_iter=500)
    assert np.all(sol["policy_c"] >= 0.0)
    assert np.all(sol["policy_c"] <= (1.0 + buf.r) * buf.a_grid[:, None] + buf.y_states[None, :] + 1e-8)
    panel = buf.simulate_panel(N=50, T=100, seed=0)
    assert np.all(np.isfinite(panel["a"]))
    print(f"Buffer-stock VFI converged in {sol['iterations']} iterations.")

    cm = CampbellMankiw({"lambda_rt": 0.4, "sigma_y": 0.10})
    sim_cm = cm.simulate(T=20000, seed=0)
    est = cm.estimate_lambda(sim_cm["dc"], sim_cm["dy"])
    assert abs(est["lambda_hat"] - cm.lam) < 0.05
    print(f"Campbell-Mankiw: true lambda={cm.lam}, estimated={est['lambda_hat']:.3f}")

    print("All ch08 smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
