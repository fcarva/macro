"""Unit tests for Nominal Rigidity (Ch. 6) and New Keynesian DSGE (Ch. 7)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch06_nominal_rigidity.ch06_nominal_rigidity import (
    AggregateSupplyDemand,
    CalvoModel,
    MenuCostModel,
)
from ch07_dsge_nk.ch07_nk import NKModel
from params import NK, NR, clone_params


# ===========================================================================
# Chapter 6 — Nominal Rigidity
# ===========================================================================


class TestMenuCostModel(unittest.TestCase):
    def setUp(self):
        self.mc = MenuCostModel(NR)

    def test_markup_above_one(self):
        self.assertGreater(self.mc.markup, 1.0)

    def test_profit_gain_is_quadratic(self):
        g1 = self.mc.profit_gain(0.1)
        g2 = self.mc.profit_gain(0.2)
        self.assertAlmostEqual(g2, 4.0 * g1, places=10,
                               msg="G(2x) = 4*G(x) for a quadratic.")

    def test_profit_gain_nonnegative(self):
        for dev in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            self.assertGreaterEqual(float(self.mc.profit_gain(dev)), 0.0)

    def test_adjustment_threshold_positive(self):
        thr = self.mc.adjustment_threshold(0.01)
        self.assertGreater(thr, 0.0)

    def test_adjustment_threshold_increases_with_cost(self):
        thr_low = self.mc.adjustment_threshold(0.005)
        thr_high = self.mc.adjustment_threshold(0.020)
        self.assertLess(thr_low, thr_high)

    def test_no_adjustment_inside_threshold(self):
        z = 0.01
        thr = self.mc.adjustment_threshold(z)
        # Deviation inside threshold → does NOT adjust
        inside_dev = thr * 0.9
        adj_arr = self.mc.price_rigidity_region(
            np.array([z]), np.array([inside_dev])
        )
        self.assertFalse(bool(adj_arr[0, 0]))

    def test_adjustment_outside_threshold(self):
        z = 0.01
        thr = self.mc.adjustment_threshold(z)
        outside_dev = thr * 1.1
        adj_arr = self.mc.price_rigidity_region(
            np.array([z]), np.array([outside_dev])
        )
        self.assertTrue(bool(adj_arr[0, 0]))

    def test_rigidity_region_shape(self):
        z_grid = np.linspace(0.001, 0.05, 15)
        d_grid = np.linspace(-0.1, 0.1, 20)
        arr = self.mc.price_rigidity_region(z_grid, d_grid)
        self.assertEqual(arr.shape, (15, 20))

    def test_aggregate_demand_externality_social_gt_private(self):
        ext = self.mc.aggregate_demand_externality(
            n_firms=100, menu_cost=0.01, demand_shock=0.05
        )
        self.assertGreater(ext["social_gain"], ext["private_gain"])

    def test_optimal_price_change_equals_demand_shock(self):
        for shock in [-0.05, 0.0, 0.03]:
            self.assertAlmostEqual(
                self.mc.optimal_price_change(shock), shock, places=12
            )


class TestCalvoModel(unittest.TestCase):
    def setUp(self):
        self.calvo = CalvoModel(NR)

    def test_nkpc_slope_positive(self):
        kappa = self.calvo.nkpc_slope()
        self.assertGreater(kappa, 0.0)

    def test_nkpc_slope_decreases_with_alpha(self):
        calvo_sticky = CalvoModel({"alpha_calvo": 0.90, "beta": 0.99,
                                   "sigma": 1.0, "omega": 1.0})
        calvo_flex = CalvoModel({"alpha_calvo": 0.50, "beta": 0.99,
                                 "sigma": 1.0, "omega": 1.0})
        self.assertLess(calvo_sticky.nkpc_slope(), calvo_flex.nkpc_slope())

    def test_calvo_weight_at_zero_is_one(self):
        self.assertAlmostEqual(float(self.calvo.calvo_weight(0)), 1.0, places=12)

    def test_calvo_weight_decays(self):
        w0 = float(self.calvo.calvo_weight(0))
        w5 = float(self.calvo.calvo_weight(5))
        self.assertGreater(w0, w5)

    def test_pi_dynamics_length(self):
        x_path = 0.01 * np.ones(24)
        pi = self.calvo.pi_dynamics(x_path)
        self.assertEqual(len(pi), 24)

    def test_pi_dynamics_positive_for_positive_gap(self):
        x_path = 0.02 * np.ones(12)
        pi = self.calvo.pi_dynamics(x_path)
        self.assertTrue(np.all(pi > 0.0))

    def test_price_level_cumulative(self):
        pi = np.array([0.01, 0.02, 0.01])
        p = self.calvo.price_level_path(pi)
        self.assertAlmostEqual(float(p[-1]), 0.04, places=12)


class TestAggregateSupplyDemand(unittest.TestCase):
    def setUp(self):
        self.ads = AggregateSupplyDemand(NK)

    def test_baseline_equilibrium_is_zero(self):
        x, pi = self.ads.equilibrium()
        self.assertAlmostEqual(x, 0.0, places=10)
        self.assertAlmostEqual(pi, 0.0, places=10)

    def test_positive_demand_shock_raises_x_and_pi(self):
        x, pi = self.ads.equilibrium(demand_shock=0.03)
        self.assertGreater(x, 0.0)
        self.assertGreater(pi, 0.0)

    def test_supply_shock_raises_pi_lowers_x(self):
        x, pi = self.ads.equilibrium(supply_shock=0.02)
        self.assertLess(x, 0.0)
        self.assertGreater(pi, 0.0)

    def test_as_curve_slope_kappa(self):
        x_range = np.array([0.0, 0.01, 0.02])
        pi_vals = self.ads.as_curve(x_range)
        slope = (pi_vals[1] - pi_vals[0]) / (x_range[1] - x_range[0])
        self.assertAlmostEqual(slope, self.ads.kappa, places=10)

    def test_ad_curve_slope_minus_sigma(self):
        pi_range = np.array([0.0, 0.01, 0.02])
        x_vals = self.ads.ad_curve(pi_range)
        slope = (x_vals[1] - x_vals[0]) / (pi_range[1] - pi_range[0])
        self.assertAlmostEqual(slope, -self.ads.sigma, places=10)


# ===========================================================================
# Chapter 7 — New Keynesian DSGE
# ===========================================================================


class TestNKModel(unittest.TestCase):
    def setUp(self):
        self.model = NKModel(NK)

    # ------------------------------------------------------------------
    # Natural rate
    # ------------------------------------------------------------------

    def test_natural_rate_positive(self):
        self.assertGreater(self.model.natural_rate(), 0.0)

    def test_natural_rate_formula(self):
        r_n = self.model.natural_rate()
        self.assertAlmostEqual(r_n, -np.log(self.model.beta), places=12)

    # ------------------------------------------------------------------
    # Demand shock solution
    # ------------------------------------------------------------------

    def test_demand_shock_psi_xv_negative(self):
        sol = self.model.solve_demand_shock()
        self.assertLess(sol["psi_xv"], 0.0,
                        msg="Monetary tightening contracts output (ψxv < 0).")

    def test_demand_shock_psi_piv_negative(self):
        sol = self.model.solve_demand_shock()
        self.assertLess(sol["psi_piv"], 0.0,
                        msg="Monetary tightening lowers inflation (ψπv < 0).")

    def test_demand_shock_delta_v_positive(self):
        sol = self.model.solve_demand_shock()
        self.assertGreater(sol["Delta_v"], 0.0)

    def test_demand_shock_nkpc_link(self):
        sol = self.model.solve_demand_shock()
        kappa = self.model.kappa
        beta = self.model.beta
        rho = self.model.rho_v
        expected_ratio = kappa / (1.0 - beta * rho)
        actual_ratio = sol["psi_piv"] / sol["psi_xv"]
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=10)

    # ------------------------------------------------------------------
    # Supply shock solution
    # ------------------------------------------------------------------

    def test_supply_shock_psi_piu_positive(self):
        sol = self.model.solve_supply_shock()
        self.assertGreater(sol["psi_piu"], 0.0,
                           msg="Cost-push shock raises inflation (ψπu > 0).")

    def test_supply_shock_psi_xu_negative(self):
        sol = self.model.solve_supply_shock()
        self.assertLess(sol["psi_xu"], 0.0,
                        msg="Cost-push shock contracts output (ψxu < 0).")

    # ------------------------------------------------------------------
    # IRF — demand shock
    # ------------------------------------------------------------------

    def test_irf_demand_x_negative_on_impact(self):
        irf = self.model.irf("demand", shock_size=0.01, T=30)
        self.assertLess(float(irf["x"][0]), 0.0)

    def test_irf_demand_pi_negative_on_impact(self):
        irf = self.model.irf("demand", shock_size=0.01, T=30)
        self.assertLess(float(irf["pi"][0]), 0.0)

    def test_irf_demand_i_positive_on_impact(self):
        irf = self.model.irf("demand", shock_size=0.01, T=30)
        self.assertGreater(float(irf["i"][0]), 0.0)

    def test_irf_demand_decays(self):
        irf = self.model.irf("demand", shock_size=0.01, T=200)
        for key in ("x", "pi"):
            self.assertLess(abs(float(irf[key][-1])), 1e-4,
                            msg=f"{key} must decay to zero.")

    def test_irf_demand_length(self):
        irf = self.model.irf("demand", T=20)
        self.assertEqual(len(irf["x"]), 21)

    # ------------------------------------------------------------------
    # IRF — supply shock
    # ------------------------------------------------------------------

    def test_irf_supply_pi_positive_on_impact(self):
        irf = self.model.irf("supply", shock_size=0.01, T=30)
        self.assertGreater(float(irf["pi"][0]), 0.0)

    def test_irf_supply_x_negative_on_impact(self):
        irf = self.model.irf("supply", shock_size=0.01, T=30)
        self.assertLess(float(irf["x"][0]), 0.0)

    def test_irf_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            self.model.irf("unknown_shock", T=10)

    # ------------------------------------------------------------------
    # Blanchard-Kahn
    # ------------------------------------------------------------------

    def test_default_params_determinate(self):
        self.assertTrue(self.model.is_determinate())

    def test_low_phi_pi_indeterminate(self):
        m = NKModel(clone_params(NK, {"phi_pi": 0.5, "phi_x": 0.0}))
        self.assertFalse(m.is_determinate())

    def test_bk_grid_shape(self):
        phi_pi_grid = np.linspace(0.0, 3.0, 10)
        phi_x_grid = np.linspace(0.0, 2.0, 8)
        result = self.model.blanchard_kahn(phi_pi_grid, phi_x_grid)
        self.assertEqual(result.shape, (10, 8))

    def test_bk_determinate_at_high_phi_pi(self):
        m = NKModel(clone_params(NK, {"phi_pi": 3.0, "phi_x": 0.5}))
        self.assertTrue(m.is_determinate())

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def test_simulate_length(self):
        sim = self.model.simulate(T=100, seed=0)
        self.assertEqual(len(sim["x"]), 101)

    def test_simulate_starts_at_zero(self):
        sim = self.model.simulate(T=100, seed=0)
        for key in ("v", "u"):
            self.assertAlmostEqual(float(sim[key][0]), 0.0, places=12)

    def test_simulate_finite(self):
        sim = self.model.simulate(T=200, seed=42)
        for key in ("x", "pi", "i"):
            self.assertTrue(np.all(np.isfinite(sim[key])))

    # ------------------------------------------------------------------
    # Policy frontier
    # ------------------------------------------------------------------

    def test_policy_frontier_returns_arrays(self):
        frontier = self.model.policy_frontier(
            phi_pi_range=np.linspace(1.1, 2.5, 5),
            T_sim=500, n_draws=5, seed=0,
        )
        self.assertIn("var_pi", frontier)
        self.assertIn("var_x", frontier)
        self.assertEqual(len(frontier["phi_pi"]), 5)

    def test_policy_frontier_higher_phi_pi_lowers_var_pi(self):
        frontier = self.model.policy_frontier(
            phi_pi_range=np.array([1.2, 2.5, 4.0]),
            T_sim=2000, n_draws=10, seed=0,
        )
        vpi = frontier["var_pi"]
        mask = np.isfinite(vpi)
        if mask.sum() >= 2:
            self.assertGreater(vpi[mask][0], vpi[mask][-1],
                               msg="Higher φπ → lower inflation variance.")


if __name__ == "__main__":
    unittest.main()
