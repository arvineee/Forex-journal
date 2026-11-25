"""Compare the AI's saved trade plan with what the linked journal trade actually did.

Pure Python (no Flask), so it is unit-tested on its own. A trade is duck-typed: it needs
``direction``, ``entry_price``, ``exit_price``, ``pnl`` and ``status`` attributes.

The journal stores only entry and exit prices, not the price path, so "take profit hit" and
"stop loss hit" are inferred from where the trade was closed (within TOUCH_R of the level).
A trade closed in between is a manual exit and says nothing about whether the plan would have worked.
"""

TOUCH_R = 0.15          # exit within 15% of the planned risk of a level counts as that level being hit
ENTRY_TOLERANCE_R = 0.25  # entry within 25% of the planned risk of the planned entry counts as "at plan"


def review_plan(plan, trade):
    """Return a small dict describing how one trade related to one plan (or None if there is nothing to compare)."""
    if not isinstance(plan, dict) or not plan.get("available") or not plan.get("valid") or trade is None:
        return None
    entry, sl, tp = plan["entry"], plan["stop_loss"], plan["take_profit"]
    risk = abs(entry - sl)
    if not risk:
        return None
    side = plan["side"]
    out = {"status": "open", "side_matches": trade.direction == side, "entry_gap_r": None, "realised_r": None,
           "outcome": None, "followed": None, "label": "Trade still open", "plan_rr": plan.get("risk_reward")}
    if trade.direction != side:
        out.update(status="closed" if trade.exit_price else "open", followed=False,
                   label=f"You traded {trade.direction}; the plan was {side}")
        return out
    gap = (trade.entry_price - entry) / risk
    out["entry_gap_r"] = round(gap, 2)
    if not trade.exit_price or getattr(trade, "status", "") != "closed":
        return out

    move = (trade.exit_price - trade.entry_price) if side == "BUY" else (trade.entry_price - trade.exit_price)
    out["realised_r"] = round(move / risk, 2)
    out["status"] = "closed"
    hit_tp = (trade.exit_price >= tp - TOUCH_R * risk) if side == "BUY" else (trade.exit_price <= tp + TOUCH_R * risk)
    hit_sl = (trade.exit_price <= sl + TOUCH_R * risk) if side == "BUY" else (trade.exit_price >= sl - TOUCH_R * risk)
    out["outcome"] = "tp_hit" if hit_tp else "sl_hit" if hit_sl else "manual_exit"
    at_plan = abs(gap) <= ENTRY_TOLERANCE_R
    if out["outcome"] == "manual_exit":
        out.update(followed=False, label="Closed manually before the plan's stop or target")
    elif not at_plan:
        out.update(followed=False, label=f"Entered {abs(out['entry_gap_r'])}R away from the planned entry")
    else:
        out.update(followed=True, label="Followed the plan: take profit hit" if out["outcome"] == "tp_hit"
                   else "Followed the plan: stopped out")
    return out


def summarise_reviews(reviews):
    """Aggregate review dicts into the numbers shown on the accuracy panel."""
    reviews = [r for r in reviews if r]
    closed = [r for r in reviews if r["status"] == "closed" and r["realised_r"] is not None]
    resolved = [r for r in closed if r["outcome"] in ("tp_hit", "sl_hit")]
    followed = [r for r in closed if r["followed"]]
    deviated = [r for r in closed if not r["followed"]]
    avg = lambda rows: round(sum(r["realised_r"] for r in rows) / len(rows), 2) if rows else None
    return {
        "plans_reviewed": len(reviews), "closed": len(closed), "open": len(reviews) - len(closed),
        "tp_hit": sum(1 for r in closed if r["outcome"] == "tp_hit"),
        "sl_hit": sum(1 for r in closed if r["outcome"] == "sl_hit"),
        "manual_exit": sum(1 for r in closed if r["outcome"] == "manual_exit"),
        "plan_hit_rate": round(100 * sum(1 for r in resolved if r["outcome"] == "tp_hit") / len(resolved), 1) if resolved else None,
        "followed": len(followed), "deviated": len(deviated),
        "avg_r_followed": avg(followed), "avg_r_deviated": avg(deviated), "avg_r_all": avg(closed),
        "enough_data": len(resolved) >= 10,
    }
