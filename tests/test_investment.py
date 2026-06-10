"""Unit tests for Investment / Tobin's q (Ch. 9)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch09_investment.ch09_investment import AdjustmentCostFirm, TobinQModel
from params import INVESTMENT


class TestTobinQModel(unittest.TestCase):
    def setUp(self):
        self.model = TobinQModel(INVESTMENT)
        self.ss = self.model.steady_state()

    def test_steady_state_q_formula(self):
        expected_q = 1.0 + self.model.a * self.model.delta
        self.assertAlmostEqual(self.ss["q_star"], expected_q, places=10)

    def test_steady_state_satisfies_isoclines(self):
        self.assertAlmostEqual(self.model.k_dot(self.ss["K_star"], self.ss["q_star"]), 0.0, places=8)
        self.assertAlmostEqual(self.model.q_dot(self.ss["K_star"], self.ss["q_star"]), 0.0, places=8)

    def test_steady_state_positive(self):
        self.assertGreater(self.ss["K_star"], 0.0)
        self.assertGreater(self.ss["q_star"], 1.0)

    def test_saddle_point_eigenvalues(self):
        eigvals, _ = self.model.eigen()
        n_negative = int(np.sum(eigvals.real < 0))
        n_positive = int(np.sum(eigvals.real > 0))
        self.assertEqual(n_negative, 1)
        self.assertEqual(n_positive, 1)

    def test_saddle_path_converges(self):
        saddle = self.model.find_saddle_path(self.ss["K_star"] * 0.8)
        sim = saddle["simulation"]
        self.assertAlmostEqual(sim["K"][-1], self.ss["K_star"], delta=1e-2)
        self.assertAlmostEqual(sim["q"][-1], self.ss["q_star"], delta=1e-2)

    def test_productivity_shock_raises_steady_state_capital(self):
        irf = self.model.irf("productivity", shock_size=0.05)
        self.assertGreater(irf["new_steady_state"]["K_star"], irf["old_steady_state"]["K_star"])
        # q* is unchanged by a productivity shock.
        self.assertAlmostEqual(irf["new_steady_state"]["q_star"], irf["old_steady_state"]["q_star"], places=10)

    def test_interest_rate_shock_lowers_steady_state_capital(self):
        irf = self.model.irf("interest", shock_size=0.25)
        self.assertLess(irf["new_steady_state"]["K_star"], irf["old_steady_state"]["K_star"])

    def test_k_locus_is_constant(self):
        self.assertAlmostEqual(self.model.k_locus(), self.ss["q_star"], places=10)

    def test_q_locus_decreasing_in_capital(self):
        k_grid = np.linspace(self.ss["K_star"] * 0.5, self.ss["K_star"] * 1.5, 20)
        q_low, _ = self.model.q_locus(k_grid)
        self.assertTrue(np.all(np.diff(q_low) <= 0.0))


class TestAdjustmentCostFirm(unittest.TestCase):
    def setUp(self):
        self.firm = AdjustmentCostFirm(INVESTMENT)

    def test_investment_rate_formula(self):
        q = 1.2
        self.assertAlmostEqual(self.firm.investment_rate(q), (q - 1.0) / self.firm.a, places=10)

    def test_zero_net_investment_at_q_equal_one(self):
        self.assertAlmostEqual(self.firm.investment_rate(1.0), 0.0, places=10)

    def test_hayashi_average_equals_marginal_q(self):
        self.assertTrue(self.firm.hayashi_holds(K=5.0, marginal_q=1.2))

    def test_i_k_schedule_is_linear(self):
        q_grid = np.linspace(0.8, 1.6, 10)
        _, i_k = self.firm.i_k_schedule(q_grid)
        slope = np.diff(i_k) / np.diff(q_grid)
        self.assertTrue(np.allclose(slope, 1.0 / self.firm.a))


if __name__ == "__main__":
    unittest.main()
