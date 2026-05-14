"""Unit tests for the RBC model (Chapter 5)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch05_rbc.ch05_rbc import LaborLeisureConditions, RBCModel
from params import RBC, clone_params


class RBCModelTests(unittest.TestCase):
    def setUp(self):
        self.model = RBCModel()

    # ------------------------------------------------------------------
    # Steady state
    # ------------------------------------------------------------------

    def test_steady_state_values_positive(self):
        ss = self.model.steady_state()
        for key in ("k_star", "y_star", "c_star", "i_star", "r_star", "w_star"):
            self.assertGreater(ss[key], 0.0, msg=f"{key} must be positive")

    def test_steady_state_shares_sum_to_one(self):
        ss = self.model.steady_state()
        self.assertAlmostEqual(ss["cy"] + ss["iy"], 1.0, places=10)

    def test_beta_euler_condition(self):
        ss = self.model.steady_state()
        # Euler: beta*(r*+1-delta) = 1
        lhs = self.model.beta * (ss["r_star"] + 1.0 - self.model.delta)
        self.assertAlmostEqual(lhs, 1.0, places=10)

    def test_production_consistency(self):
        ss = self.model.steady_state()
        y_check = ss["k_star"] ** self.model.alpha
        self.assertAlmostEqual(ss["y_star"], y_check, places=10)

    # ------------------------------------------------------------------
    # Log-linearisation
    # ------------------------------------------------------------------

    def test_log_linear_stable(self):
        ll = self.model.log_linearize()
        self.assertLess(abs(ll["A"]), 1.0, msg="|A| must be < 1 for stability")

    def test_log_linear_a_k_positive(self):
        ll = self.model.log_linearize()
        self.assertGreater(ll["a_k"], 0.0,
                           msg="Higher capital → higher consumption (a_k > 0)")

    def test_log_linear_a_z_positive(self):
        ll = self.model.log_linearize()
        self.assertGreater(ll["a_z"], 0.0,
                           msg="Positive TFP → higher consumption (a_z > 0)")

    def test_lower_beta_lowers_capital(self):
        baseline = self.model.steady_state()
        impatient = RBCModel(clone_params(RBC, {"beta": 0.95}))
        lowered = impatient.steady_state()
        self.assertLess(lowered["k_star"], baseline["k_star"])

    # ------------------------------------------------------------------
    # IRF
    # ------------------------------------------------------------------

    def test_irf_starts_at_zero_for_k(self):
        irf = self.model.irf(shock_size=0.01, T=30)
        self.assertAlmostEqual(float(irf["k"][0]), 0.0, places=12)

    def test_irf_y_jumps_on_impact(self):
        irf = self.model.irf(shock_size=0.01, T=30)
        # On impact: y_hat = z_hat (since k_hat=0, alpha*0 + 0.01 = 0.01)
        self.assertAlmostEqual(float(irf["y"][0]), 0.01, places=10)

    def test_irf_investment_more_volatile_than_output(self):
        irf = self.model.irf(shock_size=0.01, T=30)
        self.assertGreater(float(irf["i"][0]), float(irf["y"][0]),
                           msg="Investment amplification: I jumps more than Y on impact")

    def test_irf_consumption_less_volatile_than_output(self):
        irf = self.model.irf(shock_size=0.01, T=30)
        self.assertLess(float(irf["c"][0]), float(irf["y"][0]),
                        msg="Consumption smoothing: C jumps less than Y on impact")

    def test_irf_decays_to_zero(self):
        irf = self.model.irf(shock_size=0.01, T=200)
        for key in ("k", "y", "c", "i"):
            self.assertLess(abs(float(irf[key][-1])), 1e-4,
                            msg=f"{key} must decay to zero (stable model)")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def test_simulate_correct_length(self):
        sim = self.model.simulate(T=100, seed=0)
        for key in ("k", "z", "y", "c", "i"):
            self.assertEqual(len(sim[key]), 101)

    def test_simulate_starts_at_zero(self):
        sim = self.model.simulate(T=100, seed=0)
        for key in ("k", "z"):
            self.assertAlmostEqual(float(sim[key][0]), 0.0, places=12)

    def test_simulate_finite(self):
        sim = self.model.simulate(T=200, seed=42)
        for key in ("y", "c", "i"):
            self.assertTrue(np.all(np.isfinite(sim[key])))

    # ------------------------------------------------------------------
    # Capital stability (Lista I Q4)
    # ------------------------------------------------------------------

    def test_capital_stability_indicator(self):
        stab = self.model.capital_stability()
        self.assertTrue(stab["is_stable"])
        self.assertAlmostEqual(
            stab["equilibrium_investment"],
            self.model.delta * self.model.steady_state()["k_star"],
            places=10,
        )

    # ------------------------------------------------------------------
    # Labour-leisure conditions (Lista I Q5–7)
    # ------------------------------------------------------------------

    def test_optimal_leisure_formula(self):
        llc = LaborLeisureConditions(b=2.0)
        c, w = 1.0, 2.0
        l_opt = llc.optimal_leisure(c, w)
        self.assertAlmostEqual(float(l_opt), 2.0 * 1.0 / 2.0, places=10)

    def test_marginal_disutility_formula(self):
        llc = LaborLeisureConditions(b=2.0)
        l = 0.5
        mud = llc.marginal_disutility_labor(l)
        self.assertAlmostEqual(float(mud), 2.0 / 0.5, places=10)

    def test_leisure_increases_with_b(self):
        c, w = 1.0, 1.5
        l_low = LaborLeisureConditions(b=1.0).optimal_leisure(c, w)
        l_high = LaborLeisureConditions(b=3.0).optimal_leisure(c, w)
        self.assertGreater(float(l_high), float(l_low))

    def test_leisure_direction_to_interest_rate(self):
        llc = LaborLeisureConditions(b=2.0)
        self.assertEqual(llc.leisure_response_to_interest_rate(), "decrease")

    def test_leisure_direction_to_future_wages(self):
        llc = LaborLeisureConditions(b=2.0)
        self.assertEqual(llc.leisure_response_to_future_wages(), "increase")

    def test_calibrate_b_positive(self):
        llc = LaborLeisureConditions(b=2.0)
        b_cal = llc.calibrate_b(self.model, n_star=0.33)
        self.assertGreater(b_cal, 0.0)


if __name__ == "__main__":
    unittest.main()
