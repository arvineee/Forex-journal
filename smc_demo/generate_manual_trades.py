#!/usr/bin/env python3
"""Build 3 hand-entry test trades (with 4-timeframe charts) for the SMC demo account.

    python smc_demo/generate_manual_trades.py

These are NOT part of scenarios.json, so seed_smc_demo.py never loads them. You enter them yourself
(Journal -> New Trade for the trade, Chart Lab for the 4 screenshots) to watch the AI react to setups
it has never seen. Each trade is picked, deterministically, to represent a different setup quality:

    A  textbook A+ sequence     (expected: high confluence)
    B  weak / forced setup      (expected: low confluence, many conflicts)
    C  mixed, "looks ok" setup  (expected: medium confluence; the CHoCH is wick-only)

Output goes to smc_demo/manual_trades/: one folder per trade (4 entry charts + a result chart you open
only AFTER the AI has answered), ENTRY_SHEET.md (what to type) and ANSWER_KEY.md (what the generator
knows about each chart - read it after, to grade the AI).

ALL DATA IS SYNTHETIC.
"""
import importlib.util
import os
import shutil
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gen", os.path.join(HERE, "generate_smc_demo.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

OUT = os.path.join(HERE, "manual_trades")
FIRST_MANUAL_IDX = 51          # setups 1-50 are the seeded demo trades


def _is_a(sc):
    lm = sc["lm"]
    return (sc["smean"] > 0.78 and lm["swept"] and lm["choch_ok"] and lm["x_sign"] > 0
            and lm["zone"] in ("OB", "FVG") and lm["f"] < 0.5 and sc["outcome"] == "win")


def _is_b(sc):
    lm = sc["lm"]
    return (sc["smean"] < 0.42 and lm["x_sign"] <= 0 and not lm["swept"] and lm["f"] >= 0.5
            and sc["outcome"] == "loss")


def _is_c(sc):
    lm = sc["lm"]
    return (0.50 < sc["smean"] < 0.64 and lm["x_sign"] > 0 and lm["swept"] and not lm["choch_ok"]
            and lm["fvg"] is not None and sc["outcome"] == "win")


PLAN = [  # label, symbol, direction, trading day (Mon-Thu, unused by the seeded trades), test, headline
    ("A", "EURUSD", "BUY", datetime(2026, 11, 11), _is_a, "Textbook A+ sequence"),
    ("B", "XAUUSD", "SELL", datetime(2026, 11, 16), _is_b, "Weak / forced setup"),
    ("C", "GBPJPY", "BUY", datetime(2026, 11, 17), _is_c, "Mixed setup with a wick-only CHoCH"),
]


def find(symbol, direction, day, test):
    for seed in range(9_000_000, 9_000_600):
        sc = gen.build_scenario(0, seed, symbol, direction, day)
        if test(sc):
            return seed, sc
    raise RuntimeError(f"no seed matched for {symbol} {direction}")


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    sheet = ["# Manual entry sheet - 3 test trades", "",
             "Synthetic data. Enter each trade by hand, then upload its 4 charts to **Chart Lab** (4H, 1H, 30M, 15M, "
             "in that order) with the same symbol and direction, and link the analysis to the trade.",
             "Do **not** open the `RESULT` image or `ANSWER_KEY.md` until the AI has answered.", ""]
    key = ["# Answer key - open AFTER the AI has analysed the charts", "",
           "What the generator built into each chart. Compare it with the AI's confluences, conflicts and "
           "feature scores. The AI is *not* expected to match every number - look for the right overall read.", ""]
    for k, (label, symbol, direction, day, test, headline) in enumerate(PLAN):
        idx = FIRST_MANUAL_IDX + k
        seed, sc = find(symbol, direction, day, test)
        sc["idx"] = idx
        folder = os.path.join(OUT, f"trade_{label}_{symbol}_{direction}")
        os.makedirs(folder)
        for tf in gen.TFS:
            gen.draw_chart(sc, tf, os.path.join(folder, f"{symbol}_{direction}_{tf}.png"))
        gen.draw_chart(sc, "15M", os.path.join(folder, f"RESULT_do_not_open_until_AI_answers_{symbol}_15M.png"), result=True)

        lm, dec = sc["lm"], sc["dec"]
        size, pnl, pct = 0.01, None, None
        per_lot = abs(gen.calculate_pnl(direction, sc["entry_price"], sc["sl"], 1.0, symbol)[0]) or 1.0
        size = max(0.01, round(100.0 / per_lot, 2))
        pnl, pct = gen.calculate_pnl(direction, sc["entry_price"], sc["exit_price"], size, symbol)
        fm = lambda v: f"{v:.{dec}f}"
        win = sc["outcome"] == "win"
        sheet += [
            f"## Trade {label} - {symbol} {direction}", "",
            "| Field | Enter this |", "|---|---|",
            f"| Currency Pair | {symbol} |", f"| Direction | {direction} |",
            f"| Entry Price | {fm(sc['entry_price'])} |", f"| Exit Price | {fm(sc['exit_price'])} |",
            f"| Position Size (Lots) | {size} |",
            f"| Entry Time | {sc['entry_time'].strftime('%Y-%m-%d %H:%M')} |",
            f"| Exit Time | {sc['exit_time'].strftime('%Y-%m-%d %H:%M')} |",
            f"| Strategy | {gen.STRATEGY} |", "| Timeframe | 15 Minutes |",
            f"| Chart Lab: Current price | {fm(sc['entry_price'])} |",
            f"| Chart Lab: Planned entry time | {sc['entry_time'].strftime('%Y-%m-%d %H:%M')} (times are UTC) |",
            "| Chart Lab: Risk per trade | 1% |",
            f"| Stop loss / take profit (for your notes) | SL {fm(sc['sl'])} / TP {fm(sc['tp'])} (planned R:R 1:{sc['params']['rr']}) |",
            f"| Trade Notes | [MANUAL TEST] {symbol} {direction}. Planned SL {fm(sc['sl'])}, TP {fm(sc['tp'])}. |",
            f"| Emotional State | {'disciplined' if win else 'impulsive'} |",
            f"| Self Rating | {3 if label == 'C' else 5 if win else 2} |",
            f"| Result you should see after saving | {'WIN' if win else 'LOSS'}, P&L about {pnl:+.2f} ({pct:+.2f}%) |",
            f"| Chart files | `{os.path.basename(folder)}/` -> 4H, 1H, 30M, 15M |", ""]
        yn = lambda b: "yes" if b else "no"
        key += [
            f"## Trade {label} - {symbol} {direction}: {headline}", "",
            f"- Generator confluence score: **{sc['smean']:.2f}** (low < 0.40, high > 0.70)",
            f"- HTF aligned with trade: {yn(lm['x_sign'] > 0)} | liquidity sweep of Asian range: {yn(lm['swept'])} "
            f"| CHoCH confirmed by a body close: {yn(lm['choch_ok'])} | BOS: {yn(lm['bos_ok'])}",
            f"- FVG present: {yn(lm['fvg'] is not None)} | entry zone: {lm['zone']} at {lm['f'] * 100:.0f}% of the S-H leg "
            f"({'discount' if direction == 'BUY' else 'premium'} side: {yn(lm['f'] < 0.5)})",
            f"- Planned R:R: 1:{sc['params']['rr']} | simulated outcome: **{sc['outcome'].upper()}** ({pnl:+.2f})",
            f"- Seed {seed} (reproducible)", ""]
        sc["idx"] = idx
        print(f"{label}: {symbol} {direction} seed={seed} confluence={sc['smean']:.2f} {sc['outcome']} pnl={pnl}")
    with open(os.path.join(OUT, "ENTRY_SHEET.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sheet))
    with open(os.path.join(OUT, "ANSWER_KEY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(key))


if __name__ == "__main__":
    main()
