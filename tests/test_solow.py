import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow import SolowModel


class SolowModelTests(unittest.TestCase):
    def setUp(self):
        self.model = SolowModel()

    def test_steady_state_condition_holds(self):
        steady = self.model.steady_state()
        gap = self.model.s * self.model.f(steady["k_star"]) - self.model.dilution * steady["k_star"]
        self.assertAlmostEqual(float(gap), 0.0, places=10)

    def test_golden_rule_condition_holds(self):
        golden = self.model.golden_rule()
        marginal_product_gap = self.model.f_prime(golden["k_gold"]) - self.model.dilution
        self.assertAlmostEqual(float(marginal_product_gap), 0.0, places=10)

    def test_transition_converges_from_below_and_above(self):
        steady = self.model.steady_state()
        below = self.model.transition_path(0.5 * steady["k_star"], T=120.0, dt=0.1)
        above = self.model.transition_path(1.5 * steady["k_star"], T=120.0, dt=0.1)
        self.assertLess(abs(below["k"][-1] - steady["k_star"]) / steady["k_star"], 0.02)
        self.assertLess(abs(above["k"][-1] - steady["k_star"]) / steady["k_star"], 0.02)

    def test_growth_accounting_closes(self):
        data = pd.DataFrame(
            {
                "output": [100.0, 103.5, 107.2, 110.8],
                "capital": [250.0, 258.0, 266.5, 275.0],
                "labor": [50.0, 50.6, 51.2, 51.8],
            },
            index=[2000, 2001, 2002, 2003],
        )
        accounting = self.model.growth_accounting(data)
        max_gap = accounting["residual_gap"].abs().fillna(0.0).max()
        self.assertLess(max_gap, 1e-10)


if __name__ == "__main__":
    unittest.main()
