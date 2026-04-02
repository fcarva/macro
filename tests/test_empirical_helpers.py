import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ch01_solow.ch01_solow_empirics import build_brazil_solow_metadata, perpetual_inventory
from ch02_rck_diamond.ch02_rck_empirics import (
    annualize_daily_rate,
    annualize_monthly_inflation,
    build_rck_brazil_metadata,
    compute_real_rate,
)
from data_utils import (
    aggregate_quarterly_to_annual,
    compute_validation_residuals,
    fetch_bcb_sgs_series,
    normalize_sidra_tidy,
)


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

    def test_aggregate_quarterly_to_annual_sum(self):
        quarterly = pd.DataFrame(
            {
                "period": ["200001", "200002", "200003", "200004", "200001", "200002", "200003", "200004"],
                "period_label": ["q1", "q2", "q3", "q4", "q1", "q2", "q3", "q4"],
                "series_id": ["90707", "90707", "90707", "90707", "93406", "93406", "93406", "93406"],
                "series_name": ["GDP", "GDP", "GDP", "GDP", "FBCF", "FBCF", "FBCF", "FBCF"],
                "value": [10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0],
                "unit": ["BRL"] * 8,
                "source": ["IBGE"] * 8,
                "dataset_id": ["CNT"] * 8,
                "frequency": ["quarterly"] * 8,
            }
        )
        annual = aggregate_quarterly_to_annual(quarterly, aggregation="sum")
        annual = annual.set_index("series_id")
        self.assertAlmostEqual(annual.loc["90707", "value"], 46.0)
        self.assertAlmostEqual(annual.loc["93406", "value"], 10.0)
        self.assertEqual(annual.loc["90707", "aggregation"], "quarterly_to_annual_sum")

    def test_normalize_sidra_tidy_columns(self):
        raw = pd.DataFrame(
            {
                "D2C": ["2000", "2001"],
                "D2N": ["2000", "2001"],
                "D3C": ["9808", "93"],
                "D3N": ["PIB - valores correntes", "Populacao residente"],
                "V": ["1000", "150"],
                "MN": ["Milhoes de Reais", "Mil pessoas"],
            }
        )
        tidy = normalize_sidra_tidy(
            raw,
            dataset_id="CNA 6784",
            source="IBGE SIDRA",
            frequency="annual",
            period_code_col="D2C",
            period_label_col="D2N",
            series_code_col="D3C",
            series_name_col="D3N",
        )
        self.assertTrue({"period", "series_id", "series_name", "value", "unit", "source", "dataset_id"}.issubset(tidy.columns))
        self.assertEqual(tidy.loc[0, "series_id"], "9808")
        self.assertEqual(tidy.loc[1, "series_name"], "Populacao residente")
        self.assertAlmostEqual(tidy.loc[0, "value"], 1000.0)

    @patch("data_utils.sgs.get")
    def test_fetch_bcb_sgs_series_returns_date_value(self, mock_get):
        mock_get.return_value = pd.DataFrame(
            {"value": [0.04, 0.05]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        result = fetch_bcb_sgs_series(11, start_date="2024-01-01", end_date="2024-01-10")
        self.assertEqual(list(result.columns), ["date", "value"])
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))

    def test_validation_residuals_have_expected_series(self):
        annualized = pd.DataFrame(
            {
                "period": ["2000", "2000", "2000"],
                "period_label": ["2000", "2000", "2000"],
                "series_id": ["90707", "93404", "93406"],
                "series_name": ["GDP", "Consumption", "FBCF"],
                "value": [100.0, 60.0, 20.0],
                "unit": ["BRL", "BRL", "BRL"],
                "source": ["IBGE", "IBGE", "IBGE"],
                "dataset_id": ["CNT", "CNT", "CNT"],
                "frequency": ["annual", "annual", "annual"],
            }
        )
        annual = pd.DataFrame(
            {
                "period": ["2000", "2000", "2000"],
                "period_label": ["2000", "2000", "2000"],
                "series_id": ["90707", "93404", "93406"],
                "series_name": ["GDP", "Consumption", "FBCF"],
                "value": [99.0, 61.0, 21.0],
                "unit": ["BRL", "BRL", "BRL"],
                "source": ["IBGE", "IBGE", "IBGE"],
                "dataset_id": ["SCN", "SCN", "SCN"],
                "frequency": ["annual", "annual", "annual"],
            }
        )
        residuals = compute_validation_residuals(annualized, annual)
        self.assertEqual(set(residuals["series_id"]), {"90707", "93404", "93406"})
        self.assertIn("residual", residuals.columns)
        self.assertIn("residual_pct_of_annual", residuals.columns)

    def test_brazil_metadata_blocks_do_not_reference_world_bank(self):
        solow_metadata = build_brazil_solow_metadata()
        rck_metadata = build_rck_brazil_metadata()
        solow_brazil_text = json.dumps(solow_metadata["brazil"]["sources"]).lower()
        rck_brazil_text = json.dumps(rck_metadata["brazil"]["sources"]).lower()
        self.assertNotIn("world bank", solow_brazil_text)
        self.assertNotIn("world bank", rck_brazil_text)


if __name__ == "__main__":
    unittest.main()
