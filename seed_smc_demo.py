#!/usr/bin/env python3
"""Seed a dummy user with synthetic SMC chart analyses and simulated trades.

    python seed_smc_demo.py                 # create/refresh the demo user + data
    python seed_smc_demo.py --remove        # delete the demo user and its files
    python seed_smc_demo.py --password X    # choose the demo password

Input comes from smc_demo/data (made by smc_demo/generate_smc_demo.py). Only the demo
user (default: smc_demo) is touched; no other user's data is read or changed.
No Gemini key is needed: the saved analyses are inserted directly, in the same JSON
schema Chart Lab stores, and every payload is passed through the app's own normaliser.
"""
import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timedelta

os.environ.setdefault("ENV", "development")   # lets create_app() start without a production SECRET_KEY

from sqlalchemy import inspect  # noqa: E402

from app import create_app, db  # noqa: E402
from app.models import AccountBalance, Trade, TradeImage, TradingStrategy, User  # noqa: E402
from app.ai.models import AIChartAnalysis, AIInsight, AILearningLog, TradePattern  # noqa: E402
from app.ai.chart_intelligence import (  # noqa: E402
    _normalise_analysis, build_smc_playbook, learn_chart_outcomes)
from app.ai.trade_plan import plan_from_levels  # noqa: E402
from app.ai.session_context import classify_session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TFS = ("4H", "1H", "30M", "15M")
START_BALANCE = 10000.0


def purge(user, upload_dir):
    """Remove everything previously seeded for this demo user (and only this user)."""
    for row in AIChartAnalysis.query.filter_by(user_id=user.id).all():
        db.session.delete(row)
    for trade in Trade.query.filter_by(user_id=user.id).all():
        db.session.delete(trade)          # cascades to TradeImage rows
    TradingStrategy.query.filter_by(user_id=user.id).delete()
    AccountBalance.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    for pattern in (f"chart_{user.id}_smcdemo_*", f"result_{user.id}_smcdemo_*"):
        for path in glob.glob(os.path.join(upload_dir, pattern)):
            os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "smc_demo", "data"))
    ap.add_argument("--username", default="smc_demo")
    ap.add_argument("--email", default="smc_demo@example.com")
    ap.add_argument("--password", default="SmcDemo#2026")
    ap.add_argument("--remove", action="store_true", help="delete the demo user, trades, analyses and image files")
    ap.add_argument("--create-tables", action="store_true",
                    help="db.create_all() if tables are missing (prefer: flask db upgrade)")
    ap.add_argument("--run-trade-ai", action="store_true",
                    help="also run the original per-trade ForexAI engine for the AI Insights page")
    a = ap.parse_args()

    app = create_app()
    with app.app_context():
        upload_dir = app.config["UPLOAD_FOLDER"]
        names = set(inspect(db.engine).get_table_names())
        if "ai_chart_analyses" not in names:
            if a.create_tables:
                db.create_all()
            else:
                sys.exit("The ai_chart_analyses table is missing. Run `flask db upgrade` first "
                         "(or re-run with --create-tables on a throwaway database).")

        user = User.query.filter_by(username=a.username).first()
        if a.remove:
            if user:
                purge(user, upload_dir)
                for model in (AIInsight, AILearningLog, TradePattern):   # only exist if --run-trade-ai was used
                    model.query.filter_by(user_id=user.id).delete()
                db.session.delete(user)
                db.session.commit()
                print(f"Removed demo user '{a.username}' and its data.")
            else:
                print("Nothing to remove.")
            return

        manifest_path = os.path.join(a.data, "scenarios.json")
        if not os.path.exists(manifest_path):
            sys.exit(f"{manifest_path} not found. Run: python smc_demo/generate_smc_demo.py")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        scenarios = manifest["scenarios"]
        if any(s.get("pnl") is None for s in scenarios):
            sys.exit("Manifest has no P&L. Regenerate it from the project root so app/utils/helpers.py loads.")

        if not user:
            user = User(username=a.username, email=a.email)
            db.session.add(user)
        user.set_password(a.password)
        db.session.commit()
        purge(user, upload_dir)

        wins = sum(1 for s in scenarios if s["pnl"] > 0)
        net = sum(s["pnl"] for s in scenarios)
        strategy = TradingStrategy(
            name=scenarios[0]["strategy"], user_id=user.id, is_active=True,
            description="Asian-range liquidity sweep, CHoCH/BOS displacement, OB/FVG retracement entry (synthetic demo).",
            rules=json.dumps(["4H bias aligned", "Liquidity sweep of the Asian range", "CHoCH/BOS with displacement",
                              "Entry from order block / FVG in discount (buys) or premium (sells)", "Risk 1% per trade"]),
            success_rate=round(100.0 * wins / len(scenarios), 1), avg_pnl=round(net / len(scenarios), 2))
        db.session.add(strategy)
        db.session.flush()

        first_entry = datetime.fromisoformat(min(s["entry_time"] for s in scenarios))
        last_exit = datetime.fromisoformat(max(s["exit_time"] for s in scenarios))
        db.session.add(AccountBalance(user_id=user.id, balance=START_BALANCE,
                                      date=first_entry - timedelta(days=1), notes="[DEMO] Starting balance"))
        db.session.add(AccountBalance(user_id=user.id, balance=round(START_BALANCE + net, 2),
                                      date=last_exit, notes="[DEMO] Balance after simulated SMC trades"))

        copied = []
        try:
            for s in scenarios:
                n = s["n"]
                entry_time = datetime.fromisoformat(s["entry_time"])
                trade = Trade(
                    symbol=s["symbol"], direction=s["direction"], entry_price=s["entry_price"],
                    exit_price=s["exit_price"], size=s["size"], pnl=s["pnl"], pnl_percent=s["pnl_percent"],
                    entry_time=entry_time, exit_time=datetime.fromisoformat(s["exit_time"]), status="closed",
                    strategy=s["strategy"], timeframe="15m", notes=s["notes"], emotions=s["emotions"],
                    mistakes=s["mistakes"] or None, rating=s["rating"], user_id=user.id, strategy_id=strategy.id)
                db.session.add(trade)
                db.session.flush()

                stored = {}
                for tf in TFS:
                    name = f"chart_{user.id}_smcdemo_{n:02d}_{tf}.png"
                    dst = os.path.join(upload_dir, name)
                    shutil.copyfile(os.path.join(a.data, "charts", s["images"][tf]), dst)
                    copied.append(dst); stored[tf] = name
                db.session.add(TradeImage(filename=stored["15M"], image_type="entry_chart", trade_id=trade.id))
                if s.get("result_image"):
                    rname = f"result_{user.id}_smcdemo_{n:02d}.png"
                    dst = os.path.join(upload_dir, rname)
                    shutil.copyfile(os.path.join(a.data, "results", s["result_image"]), dst)
                    copied.append(dst)
                    db.session.add(TradeImage(filename=rname, image_type="exit_chart", trade_id=trade.id))

                analysis = dict(s["analysis"])
                if "trade_plan" not in analysis:                  # manifests made before the trade-plan feature
                    analysis["trade_plan"] = plan_from_levels(
                        s["direction"], s["entry_price"], s["sl"], s["tp"],
                        "Market entry on the OB/FVG retest; stop beyond the sweep extreme, target at the opposing liquidity.")
                parsed = _normalise_analysis(analysis)   # same schema/clamping as live Gemini output
                parsed["session"] = classify_session(entry_time)   # demo times are UTC; shown only, features untouched
                db.session.add(AIChartAnalysis(
                    user_id=user.id, trade_id=trade.id, symbol=s["symbol"], direction=s["direction"],
                    analysis_json=json.dumps(parsed), image_files_json=json.dumps(stored),
                    outcome_label=1 if s["pnl"] > 0 else 0 if s["pnl"] < 0 else None,
                    created_at=entry_time - timedelta(minutes=5)))   # captured before entry: no hindsight
            db.session.commit()
        except Exception:
            db.session.rollback()
            for path in copied:
                if os.path.exists(path):
                    os.remove(path)
            raise

        print(f"Seeded {len(scenarios)} setups for '{a.username}': {wins} wins / {len(scenarios) - wins} losses, "
              f"net P&L {net:.2f} (simulated).")
        print(f"Login  ->  username: {a.username}   password: {a.password}")
        playbook = build_smc_playbook(user.id)
        learning = learn_chart_outcomes(user.id)
        print(f"SMC playbook: {playbook['sample_size']} trades, {len(playbook['findings'])} findings with >= "
              f"{playbook['minimum_bucket_samples']} trades per group.")
        if learning.get("ready"):
            m = learning["metrics"]
            print(f"Chart model trained: balanced accuracy {m['balanced_accuracy']}, Brier {m['brier_score']} (cross-validated).")
        else:
            print("Chart model not trained yet: " + learning["message"])
        if a.run_trade_ai:
            try:
                from app.ai.learning_engine import ForexAI
                ForexAI(user.id).analyze_and_learn()
                print("Per-trade AI insights generated.")
            except Exception as exc:  # optional extra; never fail the seed over it
                print("Skipped per-trade AI insights:", exc)


if __name__ == "__main__":
    main()
