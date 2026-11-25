"""Trade-plan validation (loaded by path, so no Flask or database is needed)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("trade_plan", os.path.join(ROOT, "app", "ai", "trade_plan.py"))
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


def plan(**kw):
    base = {"current_price": 1.0850, "order_type": "market", "entry": 1.0850, "stop_loss": 1.0820,
            "take_profit": 1.0910, "trigger": "", "expiry": "", "rationale": "test"}
    base.update(kw)
    return base


class TradePlanTests(unittest.TestCase):
    def test_valid_buy_market(self):
        p = tp.normalise_trade_plan(plan(), "BUY")
        self.assertTrue(p["available"] and p["valid"])
        self.assertEqual(p["risk_reward"], 2.0)
        self.assertEqual(p["side"], "BUY")

    def test_valid_sell_market(self):
        p = tp.normalise_trade_plan(plan(stop_loss=1.0880, take_profit=1.0790), "SELL")
        self.assertTrue(p["valid"], p["problems"])

    def test_buy_with_inverted_levels_is_rejected(self):
        p = tp.normalise_trade_plan(plan(stop_loss=1.0900, take_profit=1.0800), "BUY")
        self.assertFalse(p["valid"])
        self.assertTrue(p["problems"])

    def test_wait_requires_a_pending_order(self):
        self.assertFalse(tp.normalise_trade_plan(plan(), "WAIT")["valid"])
        ok = tp.normalise_trade_plan(plan(order_type="buy_limit", entry=1.0830, stop_loss=1.0805,
                                          take_profit=1.0900, trigger="15M closes back above OB"), "WAIT")
        self.assertTrue(ok["valid"], ok["problems"])
        self.assertEqual(ok["order_label"], "Buy limit (pending)")

    def test_pending_order_on_wrong_side_of_price_is_rejected(self):
        bad = tp.normalise_trade_plan(plan(order_type="buy_limit", entry=1.0870, stop_loss=1.0840,
                                           take_profit=1.0950), "WAIT")
        self.assertFalse(bad["valid"])
        stop = tp.normalise_trade_plan(plan(order_type="sell_stop", entry=1.0800, stop_loss=1.0830,
                                            take_profit=1.0720), "WAIT")
        self.assertTrue(stop["valid"], stop["problems"])

    def test_missing_levels_or_unclear_call_gives_no_order(self):
        self.assertFalse(tp.normalise_trade_plan(plan(entry=None), "BUY")["available"])
        self.assertFalse(tp.normalise_trade_plan(plan(order_type="none"), "WAIT")["available"])
        self.assertFalse(tp.normalise_trade_plan(plan(), "unclear")["available"])
        self.assertFalse(tp.normalise_trade_plan(None, "BUY")["available"])

    def test_garbage_numbers_are_dropped_and_low_rr_warns(self):
        self.assertFalse(tp.normalise_trade_plan(plan(stop_loss="abc"), "BUY")["available"])
        p = tp.normalise_trade_plan(plan(take_profit=1.0860), "BUY")
        self.assertTrue(p["valid"] and any("Reward-to-risk" in w for w in p["warnings"]))

    def test_normalising_twice_is_stable(self):
        once = tp.normalise_trade_plan(plan(), "BUY")
        twice = tp.normalise_trade_plan(once, "BUY")
        self.assertEqual((once["entry"], once["stop_loss"], once["take_profit"], once["valid"]),
                         (twice["entry"], twice["stop_loss"], twice["take_profit"], twice["valid"]))

    def test_demo_helper_builds_a_valid_plan(self):
        raw = tp.plan_from_levels("SELL", 2562.28, 2565.28, 2554.35)
        self.assertTrue(tp.normalise_trade_plan(raw, "SELL")["valid"])


if __name__ == "__main__":
    unittest.main()
