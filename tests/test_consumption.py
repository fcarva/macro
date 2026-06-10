"""Unit tests for Consumption (Ch. 8)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch08_consumption.ch08_consumption import (
    BufferStockModel,
    CampbellMankiw,
    HallRandomWalk,
    PermanentIncomeModel,
)
from params import CONSUMPTION


class TestPermanentIncomeModel(unittest.TestCase):
    def setUp(self):
        self.pih = PermanentIncomeModel(CONSUMPTION)

    def test_consumption_equals_ybar_at_deterministic_ss(self):
        c, _ = self.pih.permanent_income(a=0.0, y=self.pih.y_bar)
        self.assertAlmostEqual(c, self.pih.y_bar, places=8)

    def test_constant_income_implies_constant_consumption(self):
        sim = self.pih.simulate(T=50, a0=0.0, seed=None)
        # Force constant income by overriding the simulated path manually.
        y_const = np.full(51, self.pih.y_bar)
        c_path = []
        a = 0.0
        for t in range(50):
            c, _ = self.pih.permanent_income(a=a, y=y_const[t])
            c_path.append(c)
            a = (1.0 + self.pih.r) * (a + y_const[t] - c)
        self.assertTrue(np.allclose(c_path, self.pih.y_bar, atol=1e-6))

    def test_simulate_returns_finite_paths(self):
        sim = self.pih.simulate(T=200, seed=1)
        self.assertTrue(np.all(np.isfinite(sim["c"])))
        self.assertTrue(np.all(np.isfinite(sim["a"])))

    def test_random_walk_autocorrelation_near_zero(self):
        rwt = self.pih.random_walk_test(T=5000, seed=0)
        self.assertLess(abs(rwt["autocorr_slope"]), 0.1)


class TestHallRandomWalk(unittest.TestCase):
    def test_euler_holds_when_beta_equals_inverse_gross_rate(self):
        hall = HallRandomWalk({"beta": 1.0 / 1.03, "r": 0.03})
        self.assertTrue(hall.euler_holds())

    def test_euler_fails_for_arbitrary_beta(self):
        hall = HallRandomWalk({"beta": 0.96, "r": 0.03})
        self.assertFalse(hall.euler_holds())

    def test_martingale_mean_increment_near_zero(self):
        hall = HallRandomWalk({"beta": 1.0 / 1.03, "r": 0.03})
        result = hall.test_martingale_property(n_sims=500, T=50, seed=0)
        self.assertLess(abs(result["mean_increment"]), 0.01)


class TestBufferStockModel(unittest.TestCase):
    def setUp(self):
        self.model = BufferStockModel({**CONSUMPTION, "n_grid": 25})
        self.sol = self.model.solve(max_iter=500)

    def test_policy_nonnegative(self):
        self.assertTrue(np.all(self.sol["policy_c"] >= 0.0))

    def test_policy_below_resources(self):
        m = (1.0 + self.model.r) * self.model.a_grid[:, None] + self.model.y_states[None, :]
        self.assertTrue(np.all(self.sol["policy_c"] <= m + 1e-8))

    def test_policy_monotonic_in_assets(self):
        for iy in range(2):
            diffs = np.diff(self.sol["policy_c"][:, iy])
            self.assertTrue(np.all(diffs >= -1e-8))

    def test_panel_simulation_finite(self):
        panel = self.model.simulate_panel(N=20, T=50, seed=0, burn_in=10)
        self.assertTrue(np.all(np.isfinite(panel["a"])))
        self.assertTrue(np.all(panel["a"] >= -1e-8))


class TestCampbellMankiw(unittest.TestCase):
    def test_lambda_recovered_in_large_sample(self):
        cm = CampbellMankiw({"lambda_rt": 0.4, "sigma_y": 0.10})
        sim = cm.simulate(T=20000, seed=0)
        est = cm.estimate_lambda(sim["dc"], sim["dy"])
        self.assertAlmostEqual(est["lambda_hat"], cm.lam, delta=0.05)

    def test_lambda_zero_means_no_excess_sensitivity(self):
        cm = CampbellMankiw({"lambda_rt": 0.0, "sigma_y": 0.10})
        sim = cm.simulate(T=20000, seed=1)
        est = cm.estimate_lambda(sim["dc"], sim["dy"])
        self.assertAlmostEqual(est["lambda_hat"], 0.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
