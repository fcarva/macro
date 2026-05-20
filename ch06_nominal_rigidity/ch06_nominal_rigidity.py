"""Core Nominal Rigidity models for Chapter 6.

Implements three complementary frameworks following Romer Chapter 6
(pp. 238–305):

1. MenuCostModel — Mankiw (1985) monopolist facing a fixed menu cost z.
   Firms compare the profit gain from adjusting prices to the menu cost.
   Aggregate externalities arise because each firm ignores the demand
   effects of other firms' pricing decisions (Ball–Romer coordination
   failure).

2. CalvoModel — Time-dependent pricing à la Calvo (1983). Each period a
   firm adjusts its price with probability (1-alpha) independently of past
   history. Aggregation yields the New Keynesian Phillips Curve (NKPC).

3. AggregateSupplyDemand — Stripped-down AD-AS model that embeds nominal
   rigidity via the NKPC slope kappa. Used to illustrate how demand and
   supply shocks propagate when prices are sticky.

Reference: Romer, D. (2019). Advanced Macroeconomics, 5th ed. Ch. 6.
"""

from __future__ import annotations

import numpy as np

from params import NR, NK


# ---------------------------------------------------------------------------
# 1. Menu-Cost Model (Mankiw 1985)
# ---------------------------------------------------------------------------


class MenuCostModel:
    """Monopolist facing a fixed menu cost z (Mankiw 1985 / Romer §6.2).

    The firm sets price p to maximise profit given constant-elasticity demand.
    Under flexible prices the optimal log-price tracks the aggregate price
    level plus a markup.  When aggregate demand changes, a firm must choose
    whether to pay cost z to re-optimise or to leave its price unchanged.

    The profit function is approximately quadratic in the deviation from the
    optimal price (second-order Taylor expansion), so the welfare gain from
    adjusting is:

        G(p_dev) = (eta - 1) / 2 * p_dev^2

    where p_dev = log p - log p* is the log-deviation from the optimal price.
    A firm adjusts iff G(p_dev) >= z, i.e. |p_dev| >= threshold(z).

    Parameters
    ----------
    eta : float
        Absolute value of demand elasticity (default 4.0 → markup ≈ 1/3).
    """

    def __init__(self, params: dict | None = None):
        p = dict(NR if params is None else params)
        self.eta = float(p.get("eta", 4.0))
        if self.eta <= 1.0:
            raise ValueError("Demand elasticity eta must be > 1 for markup to be well-defined.")
        self.markup = self.eta / (self.eta - 1.0)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def optimal_price_change(self, aggregate_demand_change: float) -> float:
        """Log-optimal price change in response to an aggregate demand shock.

        Under monopolistic competition the optimal log-price is:
            p* = log(markup) + log(W) + ... ~ log(markup) + demand_shock
        so the optimal price change equals the demand change one-for-one.

        Args:
            aggregate_demand_change: Log-change in nominal aggregate demand.

        Returns:
            Optimal log price change (equals aggregate_demand_change).
        """
        return float(aggregate_demand_change)

    def profit_gain(self, price_deviation: float | np.ndarray) -> float | np.ndarray:
        """Second-order welfare (profit) gain from adjusting price.

        Quadratic approximation around the optimal price p*:
            G(p_dev) = (eta - 1) / 2 * p_dev^2

        The gain is zero at p_dev=0 and grows with the square of the
        deviation from optimal.

        Args:
            price_deviation: Log-deviation from optimal price p*.

        Returns:
            Profit gain G >= 0.
        """
        p = np.asarray(price_deviation, dtype=float)
        return (self.eta - 1.0) / 2.0 * p ** 2

    def adjustment_threshold(self, menu_cost: float) -> float:
        """Minimum |p_dev| at which a firm is willing to pay the menu cost.

        From G(p_dev) = z:  p_dev^2 = 2*z/(eta-1)
        Hence threshold = sqrt(2*z/(eta-1)).

        Args:
            menu_cost: Fixed menu cost z >= 0.

        Returns:
            Threshold for price adjustment (>= 0).
        """
        if menu_cost < 0:
            raise ValueError("Menu cost z must be >= 0.")
        return float(np.sqrt(2.0 * menu_cost / (self.eta - 1.0)))

    def aggregate_demand_externality(
        self,
        n_firms: int,
        menu_cost: float,
        demand_shock: float,
    ) -> dict:
        """Compute private vs. social gain from adjustment (Ball-Romer).

        With n_firms symmetric firms, each firm's private gain from
        adjusting is G(p_dev) = (eta-1)/2 * demand_shock^2.

        The social gain (which internalises the aggregate demand
        externality) is larger: each adjusting firm raises aggregate
        demand for others, generating additional welfare.  Here we
        approximate the social gain as n_firms times the private gain
        (coordination failure upper bound).

        A firm does NOT adjust (coordination failure) when:
            private_gain < menu_cost <= social_gain

        Args:
            n_firms: Number of symmetric firms.
            menu_cost: Fixed menu cost z.
            demand_shock: Log-change in aggregate nominal demand.

        Returns:
            Dict with private_gain, social_gain, threshold, adjusts, externality.
        """
        private_gain = float(self.profit_gain(demand_shock))
        # Social gain: full coordination raises each firm's gain proportionally
        social_gain = float(n_firms) * private_gain
        threshold = self.adjustment_threshold(menu_cost)
        adjusts = abs(demand_shock) >= threshold
        externality = social_gain - private_gain
        return {
            "private_gain": private_gain,
            "social_gain": social_gain,
            "threshold": threshold,
            "adjusts": bool(adjusts),
            "externality": externality,
            "coordination_failure": (not adjusts) and (social_gain >= menu_cost),
        }

    def price_rigidity_region(
        self,
        menu_cost_grid: np.ndarray,
        demand_shock_grid: np.ndarray,
    ) -> np.ndarray:
        """2D array indicating whether a firm adjusts.

        Each cell [i, j] is True if |demand_shock_grid[j]| >=
        adjustment_threshold(menu_cost_grid[i]).

        Args:
            menu_cost_grid: 1-D array of menu cost values z.
            demand_shock_grid: 1-D array of demand shock magnitudes.

        Returns:
            Boolean array of shape (len(menu_cost_grid), len(demand_shock_grid)).
        """
        z = np.asarray(menu_cost_grid, dtype=float)
        d = np.asarray(demand_shock_grid, dtype=float)
        # threshold[i] = sqrt(2*z[i]/(eta-1))
        thresholds = np.sqrt(2.0 * np.maximum(z, 0.0) / (self.eta - 1.0))  # shape (M,)
        # adjusts[i, j] = |d[j]| >= thresholds[i]
        adjusts = np.abs(d[np.newaxis, :]) >= thresholds[:, np.newaxis]
        return adjusts


# ---------------------------------------------------------------------------
# 2. Calvo Pricing Model → NKPC
# ---------------------------------------------------------------------------


class CalvoModel:
    """Time-dependent Calvo (1983) pricing and NKPC derivation.

    Each period a firm keeps its price unchanged with probability alpha
    (independently of history) and re-optimises with probability 1-alpha.
    Aggregating optimal reset prices yields the New Keynesian Phillips Curve:

        pi_t = beta * E[pi_{t+1}] + kappa * x_t

    where x_t is the output gap and kappa = (1-alpha)*(1-alpha*beta)/alpha
    * (omega + 1/sigma) is the slope of the NKPC.

    Parameters
    ----------
    alpha : float
        Fraction of firms NOT adjusting each period (default 0.75).
    beta : float
        Household discount factor (default 0.99).
    sigma : float
        Inverse of intertemporal elasticity of substitution (default 1.0).
    omega : float
        Inverse Frisch elasticity of labour supply (default 1.0).
    """

    def __init__(self, params: dict | None = None):
        p = dict(NR if params is None else params)
        nk = NK
        self.alpha = float(p.get("alpha_calvo", 0.75))
        self.beta = float(p.get("beta", nk.get("beta", 0.99)))
        self.sigma = float(p.get("sigma", nk.get("sigma", 1.0)))
        self.omega = float(p.get("omega", 1.0))
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

    # ------------------------------------------------------------------
    # NKPC slope
    # ------------------------------------------------------------------

    def nkpc_slope(self) -> float:
        """NKPC slope kappa.

        Derived from aggregating optimal reset prices under Calvo timing:

            kappa = (1-alpha)*(1-alpha*beta)/alpha * (omega + 1/sigma)

        Returns:
            kappa > 0.
        """
        return (
            (1.0 - self.alpha)
            * (1.0 - self.alpha * self.beta)
            / self.alpha
            * (self.omega + 1.0 / self.sigma)
        )

    # ------------------------------------------------------------------
    # Inflation dynamics
    # ------------------------------------------------------------------

    def pi_dynamics(
        self,
        x_path: np.ndarray,
        pi0: float = 0.0,
        T: int | None = None,
    ) -> np.ndarray:
        """Simulate inflation from NKPC given path of output gap.

        Uses backward induction (terminal condition pi_{T+1}=0):
            pi_t = beta * pi_{t+1} + kappa * x_t

        Args:
            x_path: Array of output gaps x_0, ..., x_{T-1} (length T).
            pi0: Initial inflation (for reference; not used in backward solve).
            T: Optional override for length; defaults to len(x_path).

        Returns:
            Inflation array pi_0, ..., pi_{T-1} of length T.
        """
        x = np.asarray(x_path, dtype=float)
        n = len(x) if T is None else int(T)
        kappa = self.nkpc_slope()
        pi = np.zeros(n)
        pi_next = 0.0  # terminal condition
        for t in range(n - 1, -1, -1):
            pi[t] = self.beta * pi_next + kappa * x[t]
            pi_next = pi[t]
        return pi

    def price_level_path(self, pi_path: np.ndarray) -> np.ndarray:
        """Cumulative log price level from a series of inflation rates.

        p_t = p_{t-1} + pi_t,  p_{-1} = 0.

        Args:
            pi_path: Array of period inflation rates.

        Returns:
            Cumulative price level (log) array of same length.
        """
        return np.cumsum(np.asarray(pi_path, dtype=float))

    def calvo_weight(self, s: int | np.ndarray) -> float | np.ndarray:
        """Probability that a price set s periods ago is still in effect.

        Under Calvo timing: P(unchanged for s periods) = alpha^s.

        Args:
            s: Number of periods since last price change.

        Returns:
            Weight alpha^s in [0, 1].
        """
        s_arr = np.asarray(s, dtype=float)
        return self.alpha ** s_arr


# ---------------------------------------------------------------------------
# 3. Aggregate Supply–Demand (AD-AS with nominal rigidity)
# ---------------------------------------------------------------------------


class AggregateSupplyDemand:
    """Simple AD-AS model incorporating the NKPC as the aggregate supply curve.

    Equations:
        AS (NKPC):  pi = beta*pi_expect + kappa*x  (supply)
        AD (IS):    x = demand - sigma*pi           (demand, simplified)

    Equilibrium is the intersection of AS and AD for given expectations
    pi_expect and exogenous shifters.

    Parameters
    ----------
    sigma : float
        IS slope (inverse interest-rate sensitivity of output gap).
    kappa : float
        NKPC slope (from CalvoModel or directly calibrated).
    beta : float
        Discount factor used in AS.
    """

    def __init__(self, params: dict | None = None):
        p = dict(NK if params is None else params)
        self.sigma = float(p.get("sigma", 1.0))
        self.kappa = float(p.get("kappa", 0.1))
        self.beta = float(p.get("beta", 0.99))

    def equilibrium(
        self,
        demand_shock: float = 0.0,
        supply_shock: float = 0.0,
        pi_expect: float = 0.0,
    ) -> tuple[float, float]:
        """Solve for equilibrium output gap x and inflation pi.

        System:
            pi = beta*pi_expect + kappa*x + supply_shock   (AS)
            x  = demand_shock - sigma*pi                   (AD)

        Substituting AD into AS:
            pi = beta*pi_expect + kappa*(demand_shock - sigma*pi) + supply_shock
            pi*(1 + kappa*sigma) = beta*pi_expect + kappa*demand_shock + supply_shock
            pi = (beta*pi_expect + kappa*demand_shock + supply_shock) / (1 + kappa*sigma)

        Args:
            demand_shock: Rightward shift in AD curve.
            supply_shock: Upward shift in AS curve (cost push).
            pi_expect: Expected inflation entering AS.

        Returns:
            (x_eq, pi_eq) equilibrium output gap and inflation.
        """
        pi_eq = (
            self.beta * pi_expect + self.kappa * demand_shock + supply_shock
        ) / (1.0 + self.kappa * self.sigma)
        x_eq = demand_shock - self.sigma * pi_eq
        return float(x_eq), float(pi_eq)

    def as_curve(
        self,
        x_range: np.ndarray,
        pi_expect: float = 0.0,
        supply_shock: float = 0.0,
    ) -> np.ndarray:
        """AS (NKPC) schedule: pi = beta*pi_expect + kappa*x + supply_shock.

        Args:
            x_range: Array of output gap values.
            pi_expect: Expected inflation (shifts intercept).
            supply_shock: Cost-push shock (shifts intercept).

        Returns:
            Inflation values along AS curve.
        """
        x = np.asarray(x_range, dtype=float)
        return self.beta * pi_expect + self.kappa * x + supply_shock

    def ad_curve(
        self,
        pi_range: np.ndarray,
        demand: float = 0.0,
    ) -> np.ndarray:
        """AD schedule (inverted IS): x = demand - sigma*pi.

        Args:
            pi_range: Array of inflation values.
            demand: Demand shifter (rightward shift).

        Returns:
            Output gap values along AD curve.
        """
        pi = np.asarray(pi_range, dtype=float)
        return demand - self.sigma * pi


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test():
    """Quick sanity checks for all three models."""
    # MenuCostModel
    mc = MenuCostModel()
    assert mc.markup > 1.0
    g = mc.profit_gain(0.1)
    assert g > 0.0
    thr = mc.adjustment_threshold(0.01)
    assert thr > 0.0
    adj = mc.price_rigidity_region(
        np.linspace(0.001, 0.05, 20),
        np.linspace(-0.1, 0.1, 30),
    )
    assert adj.shape == (20, 30)
    print(f"MenuCostModel OK  (markup={mc.markup:.4f}, G(0.1)={g:.5f}, thr={thr:.4f})")

    # CalvoModel
    calvo = CalvoModel()
    kappa = calvo.nkpc_slope()
    assert kappa > 0.0
    x_path = 0.01 * np.ones(20)
    pi = calvo.pi_dynamics(x_path)
    assert len(pi) == 20
    w = calvo.calvo_weight(np.arange(10))
    assert float(w[0]) == 1.0
    print(f"CalvoModel OK  (kappa={kappa:.5f})")

    # AggregateSupplyDemand
    ads = AggregateSupplyDemand()
    x_eq, pi_eq = ads.equilibrium(demand_shock=0.02)
    assert abs(x_eq) < 0.02 + 1e-9
    print(f"AggregateSupplyDemand OK  (x={x_eq:.5f}, pi={pi_eq:.5f})")

    print("All ch06 smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
