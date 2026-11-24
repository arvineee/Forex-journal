import unittest

from app.ai.chart_intelligence import _bounded, _normalise_analysis, TIMEFRAMES


class ChartIntelligenceNormalizationTests(unittest.TestCase):
    def test_all_timeframes_are_present_even_when_model_omits_one(self):
        result = _normalise_analysis({"timeframes": {"4H": {"bias": "bullish"}}})
        self.assertEqual(tuple(result["timeframes"].keys()), TIMEFRAMES)
        self.assertEqual(result["timeframes"]["4H"]["bias"], "bullish")
        self.assertEqual(result["timeframes"]["15M"]["bias"], "unclear")

    def test_feature_scores_are_clamped(self):
        result = _normalise_analysis({
            "features": {"trend_alignment": 4, "structure_quality": -2}
        })
        self.assertEqual(result["features"]["trend_alignment"], 1.0)
        self.assertEqual(result["features"]["structure_quality"], 0.0)

    def test_invalid_numbers_use_safe_default(self):
        self.assertEqual(_bounded("not-a-number", 0, 1, 0.5), 0.5)


class AdaptiveSMCPlaybookUnitTests(unittest.TestCase):
    def test_feature_bands_are_conservative(self):
        from app.ai.chart_intelligence import _feature_band
        self.assertEqual(_feature_band(0.39), "weak")
        self.assertEqual(_feature_band(0.40), "mixed")
        self.assertEqual(_feature_band(0.60), "mixed")
        self.assertEqual(_feature_band(0.61), "strong")

    def test_session_features_are_not_used_as_playbook_edges(self):
        from app.ai.chart_intelligence import SMC_PLAYBOOK_FEATURES
        self.assertNotIn("session_quality", SMC_PLAYBOOK_FEATURES)
        self.assertNotIn("london_liquidity_sweep", SMC_PLAYBOOK_FEATURES)
        self.assertNotIn("asian_range_sweep", SMC_PLAYBOOK_FEATURES)


if __name__ == "__main__":
    unittest.main()
