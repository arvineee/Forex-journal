"""Suggest a lot size so that a stop-out costs a chosen % of the account.

Pure Python: the journal's own ``calculate_pnl`` is passed in, so lot sizes use exactly the
same pip-value maths as the rest of the app (and tests can pass a stub).
"""
import math

MIN_LOT = 0.01


def suggest_size(plan, balance, risk_pct, calculate_pnl, symbol):
    """Return sizing for a valid, available plan, or None. Lots are rounded DOWN to 0.01."""
    if not isinstance(plan, dict) or not plan.get("available") or not plan.get("valid"):
        return None
    try:
        balance, risk_pct = float(balance), float(risk_pct)
    except (TypeError, ValueError):
        return None
    if balance <= 0 or not 0 < risk_pct <= 10:
        return None
    side, entry, sl, tp = plan["side"], plan["entry"], plan["stop_loss"], plan["take_profit"]
    per_lot_loss = abs(calculate_pnl(side, entry, sl, 1.0, symbol)[0] or 0)
    per_lot_gain = abs(calculate_pnl(side, entry, tp, 1.0, symbol)[0] or 0)
    if not per_lot_loss:
        return None
    risk_amount = balance * risk_pct / 100.0
    lots = math.floor(risk_amount / per_lot_loss * 100 + 1e-9) / 100.0
    warnings = []
    if lots < MIN_LOT:
        lots = MIN_LOT
        warnings.append(f"Even the minimum {MIN_LOT} lot risks more than {risk_pct:g}% of the balance "
                        f"(about {MIN_LOT * per_lot_loss:.2f}). Reduce the risk or skip this setup.")
    return {"lots": round(lots, 2), "risk_pct": risk_pct, "balance": round(balance, 2),
            "risk_amount": round(risk_amount, 2), "loss_at_sl": round(lots * per_lot_loss, 2),
            "profit_at_tp": round(lots * per_lot_gain, 2), "warnings": warnings}
