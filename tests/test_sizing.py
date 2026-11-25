"""Lot-size suggestion using the journal's own P&L maths (no Flask needed)."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


tp = _load("trade_plan", ("app", "ai", "trade_plan.py"))
sz = _load("sizing", ("app", "ai", "sizing.py"))
helpers = _load("journal_helpers", ("app", "utils", "helpers.py"))
calc = helpers.calculate_pnl


def plan(symbol_prices):
    return tp.normalise_trade_plan(dict(order_type="market", current_price=symbol_prices[0], entry=symbol_prices[0],
                                        stop_loss=symbol_prices[1], take_profit=symbol_prices[2]), "BUY")


class SizingTests(unittest.TestCase):
    def test_one_percent_risk_costs_about_one_percent(self):
        p = plan((1.0850, 1.0820, 1.0910))
        s = sz.suggest_size(p, 10000, 1, calc, "EURUSD")
        self.assertGreater(s["lots"], 0)
        self.assertLessEqual(s["loss_at_sl"], 100.0)
        self.assertGreater(s["loss_at_sl"], 90.0)          # rounding down to 0.01 lot loses < 10%
        self.assertAlmostEqual(s["profit_at_tp"] / s["loss_at_sl"], 2.0, delta=0.05)

    def test_never_rounds_risk_up(self):
        s = sz.suggest_size(plan((1.0850, 1.0820, 1.0910)), 10000, 0.5, calc, "EURUSD")
        self.assertLessEqual(s["loss_at_sl"], 50.0)

    def test_gold_and_jpy_work(self):
        g = tp.normalise_trade_plan(dict(order_type="market", current_price=2650, entry=2650, stop_loss=2640, take_profit=2680), "BUY")
        self.assertGreater(sz.suggest_size(g, 10000, 1, calc, "XAUUSD")["lots"], 0)
        j = tp.normalise_trade_plan(dict(order_type="market", current_price=151.4, entry=151.4, stop_loss=151.0, take_profit=152.4), "BUY")
        self.assertGreater(sz.suggest_size(j, 10000, 1, calc, "USDJPY")["lots"], 0)

    def test_tiny_account_warns_instead_of_overrisking_silently(self):
        s = sz.suggest_size(plan((1.0850, 1.0820, 1.0910)), 100, 1, calc, "EURUSD")
        self.assertEqual(s["lots"], 0.01)
        self.assertTrue(s["warnings"])

    def test_no_sizing_for_bad_inputs(self):
        self.assertIsNone(sz.suggest_size(None, 10000, 1, calc, "EURUSD"))
        self.assertIsNone(sz.suggest_size(plan((1.0850, 1.0820, 1.0910)), 0, 1, calc, "EURUSD"))
        self.assertIsNone(sz.suggest_size(plan((1.0850, 1.0820, 1.0910)), 10000, 50, calc, "EURUSD"))
        self.assertIsNone(sz.suggest_size(tp.empty_plan("WAIT"), 10000, 1, calc, "EURUSD"))


if __name__ == "__main__":
    unittest.main()
