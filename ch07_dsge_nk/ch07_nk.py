"""Core New Keynesian DSGE model for Chapter 7.

Implements the canonical 3-equation New Keynesian model following
Romer Chapter 7 (pp. 306–366):

    IS  (Dynamic IS):  x_t = E[x_{t+1}] - (1/sigma)(i_t - E[pi_{t+1}] - r_n)
    NKPC:              pi_t = beta*E[pi_{t+1}] + kappa*x_t + u_t
    Taylor rule:       i_t = r_n + phi_pi*pi_t + phi_x*x_t + v_t

where x_t is the output gap, pi_t is inflation, i_t is the nominal rate,
r_n = -log(beta) is the natural real rate, v_t is a monetary policy shock
(AR1 with persistence rho_v), and u_t is a cost-push shock (AR1 with rho_u).

The model is solved analytically by undetermined coefficients. The
Blanchard-Kahn determinacy condition is analysed via eigenvalues of the
reduced-form system matrix.

Reference: Romer, D. (2019). Advanced Macroeconomics, 5th ed. Ch. 7.
"""

from __future__ import annotations

import numpy as np

from params import NK


class NKModel:
    """Canonical 3-equation New Keynesian model.

    Parameters
    ----------
    params : dict, optional
        Overrides for the NK parameter dictionary. Keys accepted:
        beta, sigma, kappa, phi_pi, phi_x, rho_v, rho_u.
    """

    def __init__(self, params: dict | None = None):
        p = dict(NK if params is None else params)
        self.beta = float(p.get("beta", 0.99))
        self.sigma = float(p.get("sigma", 1.0))
        self.kappa = float(p.get("kappa", 0.1))
        self.phi_pi = float(p.get("phi_pi", 1.5))
        self.phi_x = float(p.get("phi_x", 0.5))
        self.rho_v = float(p.get("rho_v", 0.5))
        self.rho_u = float(p.get("rho_u", 0.5))

    # ------------------------------------------------------------------
    # Natural rate
    # ------------------------------------------------------------------

    def natural_rate(self) -> float:
        """Natural real interest rate r_n = -log(beta) ≈ (1-beta)/beta."""
        return float(-np.log(self.beta))

    # ------------------------------------------------------------------
    # Solution by undetermined coefficients
    # ------------------------------------------------------------------

    def solve_demand_shock(self) -> dict:
        """Solve for impulse coefficients driven by monetary shock v_t.

        Guess: x_t = psi_xv * v_t,  pi_t = psi_piv * v_t.

        From NKPC (u=0):
            psi_piv*(1 - beta*rho_v) = kappa * psi_xv
            → psi_piv = kappa * psi_xv / (1 - beta*rho_v)

        From IS with Taylor rule substituted:
            psi_xv * [(1-rho_v) + (phi_x + kappa*(phi_pi-rho_v)/(1-beta*rho_v))/sigma]
              = -1/sigma
            → psi_xv = -1 / (sigma * Delta_v)

        where Delta_v = (1-rho_v) + [phi_x + kappa*(phi_pi-rho_v)/(1-beta*rho_v)] / sigma.

        Returns:
            dict with psi_xv, psi_piv, Delta_v.
        """
        rho = self.rho_v
        denom_pi = 1.0 - self.beta * rho
        if abs(denom_pi) < 1e-12:
            raise ValueError("1 - beta*rho_v too close to zero.")
        Delta_v = (1.0 - rho) + (
            self.phi_x + self.kappa * (self.phi_pi - rho) / denom_pi
        ) / self.sigma
        if abs(Delta_v) < 1e-12:
            raise ValueError("Delta_v too close to zero; check parameters.")
        psi_xv = -1.0 / (self.sigma * Delta_v)
        psi_piv = self.kappa * psi_xv / denom_pi
        return {
            "psi_xv": float(psi_xv),
            "psi_piv": float(psi_piv),
            "Delta_v": float(Delta_v),
        }

    def solve_supply_shock(self, rho_u: float | None = None) -> dict:
        """Solve for impulse coefficients driven by cost-push shock u_t.

        Guess: x_t = psi_xu * u_t,  pi_t = psi_piu * u_t.

        From NKPC (v=0):
            psi_piu = (1 + kappa * psi_xu) / (1 - beta*rho_u)

        From IS with Taylor rule substituted:
            psi_xu * D_u = -(phi_pi - rho_u) / (sigma*(1-beta*rho_u))

        where D_u = (1-rho_u) + [phi_x + kappa*(phi_pi-rho_u)/(1-beta*rho_u)] / sigma.

        Args:
            rho_u: Persistence of cost-push shock (defaults to self.rho_u).

        Returns:
            dict with psi_xu, psi_piu, D_u.
        """
        rho = self.rho_u if rho_u is None else float(rho_u)
        denom_pi = 1.0 - self.beta * rho
        if abs(denom_pi) < 1e-12:
            raise ValueError("1 - beta*rho_u too close to zero.")
        D_u = (1.0 - rho) + (
            self.phi_x + self.kappa * (self.phi_pi - rho) / denom_pi
        ) / self.sigma
        if abs(D_u) < 1e-12:
            raise ValueError("D_u too close to zero; check parameters.")
        psi_xu = -(self.phi_pi - rho) / (self.sigma * denom_pi * D_u)
        psi_piu = (1.0 + self.kappa * psi_xu) / denom_pi
        return {
            "psi_xu": float(psi_xu),
            "psi_piu": float(psi_piu),
            "D_u": float(D_u),
        }

    # ------------------------------------------------------------------
    # Impulse response functions
    # ------------------------------------------------------------------

    def irf(
        self,
        shock_type: str = "demand",
        shock_size: float = 0.01,
        T: int = 40,
    ) -> dict:
        """Impulse response to a one-time shock at t=0.

        Args:
            shock_type: 'demand' (monetary policy shock v_t) or
                        'supply' (cost-push shock u_t).
            shock_size: Initial shock magnitude.
            T: Number of periods.

        Returns:
            Dict with arrays time, x, pi, i of length T+1, plus coefficients.
        """
        r_n = self.natural_rate()
        time = np.arange(T + 1)

        if shock_type == "demand":
            sol = self.solve_demand_shock()
            psi_x, psi_pi = sol["psi_xv"], sol["psi_piv"]
            rho = self.rho_v
            shock_path = shock_size * rho ** time
            x_hat = psi_x * shock_path
            pi_hat = psi_pi * shock_path
            # Nominal rate: i_t = r_n + phi_pi*pi + phi_x*x + v_t
            i_hat = self.phi_pi * pi_hat + self.phi_x * x_hat + shock_path
        elif shock_type == "supply":
            sol = self.solve_supply_shock()
            psi_x, psi_pi = sol["psi_xu"], sol["psi_piu"]
            rho = self.rho_u
            shock_path = shock_size * rho ** time
            x_hat = psi_x * shock_path
            pi_hat = psi_pi * shock_path
            # Monetary rule responds to x and pi only (v=0)
            i_hat = self.phi_pi * pi_hat + self.phi_x * x_hat
        else:
            raise ValueError(f"Unknown shock_type '{shock_type}'. Use 'demand' or 'supply'.")

        return {
            "time": time,
            "x": x_hat,
            "pi": pi_hat,
            "i": i_hat,
            "shock": shock_path,
            "shock_type": shock_type,
            "sol": sol,
        }

    # ------------------------------------------------------------------
    # Blanchard-Kahn determinacy
    # ------------------------------------------------------------------

    def system_matrix(self) -> np.ndarray:
        """Reduced-form expectational system matrix M.

        The homogeneous system (no shocks) is:
            E_t[x_{t+1}]  = M[0,0]*x_t + M[0,1]*pi_t
            E_t[pi_{t+1}] = M[1,0]*x_t + M[1,1]*pi_t

        Derived by substituting the Taylor rule into the IS equation and
        using the NKPC to eliminate E_t[pi_{t+1}]:

            M = [[1 + phi_x/sigma + kappa/(beta*sigma),  phi_pi/sigma - 1/(beta*sigma)],
                 [-kappa/beta,                            1/beta                       ]]
        """
        s = self.sigma
        b = self.beta
        k = self.kappa
        fp = self.phi_pi
        fx = self.phi_x
        return np.array([
            [1.0 + fx / s + k / (b * s),  fp / s - 1.0 / (b * s)],
            [-k / b,                        1.0 / b               ],
        ])

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the reduced-form system matrix."""
        return np.linalg.eigvals(self.system_matrix())

    def is_determinate(self) -> bool:
        """True if the Blanchard-Kahn condition holds (both |λ| > 1).

        Both x_t and pi_t are jump variables, so we need both eigenvalues
        outside the unit circle for a unique bounded REE.

        Analytical approximation (Taylor principle):
            kappa*(phi_pi - 1) + (1-beta)*phi_x > 0
        """
        evs = np.abs(self.eigenvalues())
        return bool(np.all(evs > 1.0))

    def blanchard_kahn(
        self,
        phi_pi_grid: np.ndarray,
        phi_x_grid: np.ndarray,
    ) -> np.ndarray:
        """2D determinacy map over (phi_pi, phi_x) grid.

        Args:
            phi_pi_grid: 1-D array of phi_pi values.
            phi_x_grid:  1-D array of phi_x values.

        Returns:
            Boolean array of shape (len(phi_pi_grid), len(phi_x_grid)).
            True → determinate (unique stable REE).
        """
        from copy import deepcopy
        phi_pi_arr = np.asarray(phi_pi_grid, dtype=float)
        phi_x_arr = np.asarray(phi_x_grid, dtype=float)
        result = np.zeros((len(phi_pi_arr), len(phi_x_arr)), dtype=bool)
        for i, fp in enumerate(phi_pi_arr):
            for j, fx in enumerate(phi_x_arr):
                m = NKModel({
                    "beta": self.beta,
                    "sigma": self.sigma,
                    "kappa": self.kappa,
                    "phi_pi": fp,
                    "phi_x": fx,
                    "rho_v": self.rho_v,
                    "rho_u": self.rho_u,
                })
                result[i, j] = m.is_determinate()
        return result

    # ------------------------------------------------------------------
    # Stochastic simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        T: int = 200,
        seed: int | None = None,
        sigma_v: float = 0.01,
        sigma_u: float = 0.005,
    ) -> dict:
        """Stochastic simulation with both demand and supply shocks.

        Args:
            T: Number of periods.
            seed: RNG seed.
            sigma_v: Std dev of monetary shock innovations.
            sigma_u: Std dev of cost-push shock innovations.

        Returns:
            Dict with arrays time, x, pi, i, v, u of length T+1.
        """
        rng = np.random.default_rng(seed)
        sol_v = self.solve_demand_shock()
        sol_u = self.solve_supply_shock()

        eps_v = rng.normal(0.0, sigma_v, size=T)
        eps_u = rng.normal(0.0, sigma_u, size=T)

        v = np.zeros(T + 1)
        u = np.zeros(T + 1)
        for t in range(T):
            v[t + 1] = self.rho_v * v[t] + eps_v[t]
            u[t + 1] = self.rho_u * u[t] + eps_u[t]

        x = sol_v["psi_xv"] * v + sol_u["psi_xu"] * u
        pi = sol_v["psi_piv"] * v + sol_u["psi_piu"] * u
        i = self.natural_rate() + self.phi_pi * pi + self.phi_x * x + v

        return {
            "time": np.arange(T + 1),
            "x": x,
            "pi": pi,
            "i": i,
            "v": v,
            "u": u,
        }

    # ------------------------------------------------------------------
    # Policy frontier (variance trade-off)
    # ------------------------------------------------------------------

    def policy_frontier(
        self,
        phi_pi_range: np.ndarray | None = None,
        sigma_v: float = 0.01,
        sigma_u: float = 0.005,
        T_sim: int = 5000,
        n_draws: int = 30,
        seed: int = 0,
    ) -> dict:
        """Variance frontier: var(pi) vs var(x) as phi_pi varies.

        Traces the efficiency frontier by varying phi_pi while keeping
        phi_x fixed. For each phi_pi, computes simulated variances via
        Monte Carlo.

        Args:
            phi_pi_range: Grid of phi_pi values (default 1.01 to 3.0).
            sigma_v: Std dev of monetary shock.
            sigma_u: Std dev of cost-push shock.
            T_sim: Simulation length per draw.
            n_draws: Number of Monte Carlo draws per point.
            seed: RNG seed.

        Returns:
            dict with phi_pi_range, var_pi, var_x.
        """
        if phi_pi_range is None:
            phi_pi_range = np.linspace(1.05, 4.0, 30)
        phi_pi_arr = np.asarray(phi_pi_range, dtype=float)
        rng = np.random.default_rng(seed)

        var_pi_list = []
        var_x_list = []

        for fp in phi_pi_arr:
            m = NKModel({
                "beta": self.beta,
                "sigma": self.sigma,
                "kappa": self.kappa,
                "phi_pi": float(fp),
                "phi_x": self.phi_x,
                "rho_v": self.rho_v,
                "rho_u": self.rho_u,
            })
            if not m.is_determinate():
                var_pi_list.append(np.nan)
                var_x_list.append(np.nan)
                continue
            pis, xs = [], []
            for _ in range(n_draws):
                s = m.simulate(
                    T=T_sim, seed=int(rng.integers(int(1e9))),
                    sigma_v=sigma_v, sigma_u=sigma_u,
                )
                pis.append(float(np.var(s["pi"][1:], ddof=1)))
                xs.append(float(np.var(s["x"][1:], ddof=1)))
            var_pi_list.append(float(np.mean(pis)))
            var_x_list.append(float(np.mean(xs)))

        return {
            "phi_pi": phi_pi_arr,
            "var_pi": np.array(var_pi_list),
            "var_x": np.array(var_x_list),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test():
    model = NKModel()

    r_n = model.natural_rate()
    assert r_n > 0, "Natural rate must be positive."
    print(f"Natural rate r_n = {r_n:.5f}")

    sol_v = model.solve_demand_shock()
    assert sol_v["psi_xv"] < 0, "Demand shock: psi_xv must be negative."
    assert sol_v["psi_piv"] < 0, "Demand shock: psi_piv must be negative."
    print(f"Demand shock: psi_xv={sol_v['psi_xv']:.5f}, psi_piv={sol_v['psi_piv']:.5f}")

    sol_u = model.solve_supply_shock()
    assert sol_u["psi_piu"] > 0, "Supply shock: psi_piu must be positive."
    assert sol_u["psi_xu"] < 0, "Supply shock: psi_xu must be negative."
    print(f"Supply shock:  psi_xu={sol_u['psi_xu']:.5f}, psi_piu={sol_u['psi_piu']:.5f}")

    irf_d = model.irf("demand", shock_size=0.01, T=30)
    assert irf_d["x"][0] < 0, "Monetary tightening contracts output."
    assert irf_d["i"][0] > 0, "Nominal rate rises on impact."

    irf_s = model.irf("supply", shock_size=0.01, T=30)
    assert irf_s["pi"][0] > 0, "Supply shock raises inflation."
    assert irf_s["x"][0] < 0, "Supply shock contracts output."

    assert model.is_determinate(), "Default params must yield determinacy."
    m_ind = NKModel({"beta": 0.99, "sigma": 1.0, "kappa": 0.1,
                     "phi_pi": 0.5, "phi_x": 0.0, "rho_v": 0.5, "rho_u": 0.5})
    assert not m_ind.is_determinate(), "phi_pi=0.5 phi_x=0 must be indeterminate."

    sim = model.simulate(T=200, seed=0)
    assert len(sim["x"]) == 201
    assert np.all(np.isfinite(sim["pi"]))

    print("All ch07 smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
