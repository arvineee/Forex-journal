"""Plan-vs-outcome review (loaded by path; no Flask needed)."""
import importlib.util
import os
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "app", "ai", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


tp, pr = _load("trade_plan"), _load("plan_review")
PLAN = tp.normalise_trade_plan({"current_price": 1.0850, "order_type": "market", "entry": 1.0850,
                                "stop_loss": 1.0820, "take_profit": 1.0910}, "BUY")


def trade(direction="BUY", entry=1.0850, exit_=1.0910, status="closed"):
    return types.SimpleNamespace(direction=direction, entry_price=entry, exit_price=exit_, status=status, pnl=1)


class PlanReviewTests(unittest.TestCase):
    def test_followed_and_tp_hit(self):
        r = pr.review_plan(PLAN, trade())
        self.assertEqual((r["outcome"], r["followed"], r["realised_r"]), ("tp_hit", True, 2.0))

    def test_followed_and_stopped_out(self):
        r = pr.review_plan(PLAN, trade(exit_=1.0820))
        self.assertEqual((r["outcome"], r["followed"], r["realised_r"]), ("sl_hit", True, -1.0))

    def test_manual_exit_is_not_a_verdict_on_the_plan(self):
        r = pr.review_plan(PLAN, trade(exit_=1.0870))
        self.assertEqual((r["outcome"], r["followed"]), ("manual_exit", False))

    def test_late_entry_is_a_deviation(self):
        r = pr.review_plan(PLAN, trade(entry=1.0870, exit_=1.0910))
        self.assertFalse(r["followed"])

    def test_wrong_side_and_open_trade(self):
        self.assertFalse(pr.review_plan(PLAN, trade(direction="SELL"))["side_matches"])
        r = pr.review_plan(PLAN, trade(exit_=None, status="open"))
        self.assertEqual(r["status"], "open")

    def test_invalid_or_missing_plans_are_skipped(self):
        self.assertIsNone(pr.review_plan(None, trade()))
        self.assertIsNone(pr.review_plan(tp.empty_plan("WAIT"), trade()))
        self.assertIsNone(pr.review_plan(PLAN, None))

    def test_summary(self):
        reviews = [pr.review_plan(PLAN, t) for t in (trade(), trade(exit_=1.0820), trade(exit_=1.0870), trade(exit_=None, status="open"))]
        s = pr.summarise_reviews(reviews)
        self.assertEqual((s["plans_reviewed"], s["closed"], s["open"], s["tp_hit"], s["sl_hit"], s["manual_exit"]), (4, 3, 1, 1, 1, 1))
        self.assertEqual(s["plan_hit_rate"], 50.0)
        self.assertFalse(s["enough_data"])
        self.assertEqual(pr.summarise_reviews([])["plans_reviewed"], 0)


if __name__ == "__main__":
    unittest.main()
