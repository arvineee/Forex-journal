"""Integrity checks for the synthetic SMC demo dataset (no database needed)."""
import importlib.util
import json
import os
import unittest
from datetime import datetime

from PIL import Image

from app.ai.chart_intelligence import _normalise_analysis, confluence_score, TIMEFRAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "smc_demo", "data")
MANIFEST = os.path.join(DATA, "scenarios.json")


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_smc_demo", os.path.join(ROOT, "smc_demo", "generate_smc_demo.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.path.exists(MANIFEST), "run smc_demo/generate_smc_demo.py first")
class SMCDemoManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MANIFEST, encoding="utf-8") as handle:
            cls.payload = json.load(handle)
        cls.items = cls.payload["scenarios"]

    def test_manifest_is_marked_synthetic_and_unique(self):
        self.assertTrue(self.payload["synthetic"])
        self.assertEqual(len({s["n"] for s in self.items}), len(self.items))

    def test_each_setup_has_four_timeframe_charts(self):
        for s in self.items:
            self.assertEqual(tuple(s["images"]), TIMEFRAMES)
            for tf in TIMEFRAMES:
                path = os.path.join(DATA, "charts", s["images"][tf])
                with Image.open(path) as image:
                    self.assertGreater(image.width, 800)

    def test_trade_geometry_and_outcome_are_consistent(self):
        for s in self.items:
            if s["direction"] == "BUY":
                self.assertTrue(s["sl"] < s["entry_price"] < s["tp"], s["n"])
            else:
                self.assertTrue(s["tp"] < s["entry_price"] < s["sl"], s["n"])
            self.assertEqual(s["pnl"] > 0, s["outcome"] == "win", s["n"])
            self.assertGreater(datetime.fromisoformat(s["exit_time"]), datetime.fromisoformat(s["entry_time"]))

    def test_analyses_match_the_app_schema_without_clamping(self):
        for s in self.items:
            parsed = _normalise_analysis(s["analysis"])
            for key, value in s["analysis"]["features"].items():
                self.assertAlmostEqual(parsed["features"][key], value, places=6)
            score, band = confluence_score(parsed["features"])
            self.assertTrue(0 <= score <= 1)
            self.assertIn(band, {"low", "medium", "high"})

    def test_result_charts_are_kept_out_of_the_analysis_inputs(self):
        for s in self.items:
            self.assertNotIn("RESULT", s["images"])


class SMCGeneratorTests(unittest.TestCase):
    def test_same_seed_reproduces_the_same_trade(self):
        gen = _load_generator()
        day = datetime(2026, 10, 6)
        a = gen.build_scenario(1, 7, "EURUSD", "BUY", day)
        b = gen.build_scenario(1, 7, "EURUSD", "BUY", day)
        self.assertEqual((a["entry_price"], a["exit_price"], a["outcome"]),
                         (b["entry_price"], b["exit_price"], b["outcome"]))

    def test_chart_inputs_stop_at_entry_candle(self):
        gen = _load_generator()
        sc = gen.build_scenario(1, 11, "GBPUSD", "SELL", datetime(2026, 10, 7))
        self.assertLess(sc["E"], sc["exit_i"])


if __name__ == "__main__":
    unittest.main()
