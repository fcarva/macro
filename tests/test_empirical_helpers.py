import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow_empirics import perpetual_inventory
from ch02_rck_diamond.ch02_rck_empirics import annualize_daily_rate, annualize_monthly_inflation, compute_real_rate


class EmpiricalHelperTests(unittest.TestCase):
    def test_perpetual_inventory_is_positive(self):
        investment = pd.Series([10.0, 11.0, 12.0, 12.5], index=[2000, 2001, 2002, 2003])
        capital = perpetual_inventory(investment, depreciation_rate=0.05)
        self.assertTrue((capital > 0).all())
        self.assertEqual(len(capital), len(investment))

    def test_real_rate_pipeline(self):
        nominal = annualize_daily_rate(pd.Series([0.04, 0.05]))
        inflation = annualize_monthly_inflation(pd.Series([0.3, 0.4]))
        nominal.index = pd.Index(["2000-01-31", "2000-02-29"])
        inflation.index = pd.Index(["2000-01-31", "2000-02-29"])
        real = compute_real_rate(nominal, inflation)
        self.assertIn("real_rate", real.columns)
        self.assertEqual(len(real), 2)


if __name__ == "__main__":
    unittest.main()
