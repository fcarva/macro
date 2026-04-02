import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch02_rck_diamond.ch02_rck import RCKModel
from params import clone_params


class RCKModelTests(unittest.TestCase):
    def setUp(self):
        self.model = RCKModel()

    def test_steady_state_conditions_hold(self):
        steady = self.model.steady_state()
        self.assertAlmostEqual(float(self.model.k_dot(steady["k_star"], steady["c_star"])), 0.0, places=10)
        self.assertAlmostEqual(float(self.model.c_dot(steady["k_star"], steady["c_star"])), 0.0, places=10)

    def test_numeric_and_analytic_steady_states_match(self):
        analytic = self.model.steady_state()
        numeric = self.model.steady_state(numeric=True)
        self.assertAlmostEqual(analytic["k_star"], numeric["k_star"], places=8)
        self.assertAlmostEqual(analytic["c_star"], numeric["c_star"], places=8)

    def test_shooting_finds_stable_path(self):
        steady = self.model.steady_state()
        saddle = self.model.find_saddle_path(0.85 * steady["k_star"], T=120.0)
        simulation = saddle["simulation"]
        self.assertEqual(simulation["terminal_reason"], "completed")
        self.assertLess(abs(simulation["k"][-1] - steady["k_star"]) / steady["k_star"], 0.03)
        self.assertLess(abs(simulation["c"][-1] - steady["c_star"]) / steady["c_star"], 0.05)

    def test_lower_rho_raises_steady_state_capital(self):
        baseline = self.model.steady_state()
        patient = RCKModel(clone_params(self.model.params, {"rho": 0.75 * self.model.rho}))
        lowered = patient.steady_state()
        self.assertGreater(lowered["k_star"], baseline["k_star"])

    def test_higher_government_spending_shifts_k_locus_down(self):
        steady = self.model.steady_state()
        bigger_state = RCKModel(clone_params(self.model.params, {"G": 0.05 * self.model.f(steady["k_star"])}))
        self.assertLess(
            float(bigger_state.k_locus(steady["k_star"])),
            float(self.model.k_locus(steady["k_star"])),
        )


if __name__ == "__main__":
    unittest.main()
