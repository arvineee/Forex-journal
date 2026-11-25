"""Turn the vision model's proposed trade plan into a validated, display-ready plan.

Pure Python (no Flask imports) so it can be unit-tested on its own.

Rules enforced here, whatever the model returned:
  * BUY / SELL  -> a market order (entry near the last visible price) or a same-side pending order.
  * WAIT        -> a PENDING order (buy/sell limit or stop) is required; a market order is rejected.
  * BUY side: stop_loss < entry < take_profit.  SELL side: take_profit < entry < stop_loss.
  * Pending orders must sit on the correct side of the current price
    (buy limit below, sell limit above, buy stop above, sell stop below).
Plans that break these rules are kept for transparency but flagged ``valid=False`` so the UI can
refuse to present them as usable. Prices come from reading a screenshot, so they are estimates.
"""

ORDER_LABELS = {
    "market": "Market order",
    "buy_limit": "Buy limit (pending)",
    "sell_limit": "Sell limit (pending)",
    "buy_stop": "Buy stop (pending)",
    "sell_stop": "Sell stop (pending)",
    "none": "No order",
}
PENDING = {"buy_limit", "sell_limit", "buy_stop", "sell_stop"}
MIN_SENSIBLE_RR = 1.5


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0 or number in (float("inf"), float("-inf")):
        return None
    return number


def _text(value, limit):
    return str(value).strip()[:limit] if value not in (None, "") else ""


def _decimals(*values):
    """Display precision: enough decimals for the finest price given (max 5)."""
    best = 2
    for v in values:
        if v is None:
            continue
        text = f"{v:.5f}".rstrip("0")
        best = max(best, min(5, len(text.split(".")[1]) if "." in text else 0))
    return best


def empty_plan(action="NONE", reason="No trade plan was returned for this analysis."):
    return {"available": False, "valid": False, "action": action, "order_type": "none",
            "order_label": ORDER_LABELS["none"], "side": None, "current_price": None, "entry": None,
            "stop_loss": None, "take_profit": None, "risk": None, "reward": None, "risk_reward": None,
            "decimals": 2, "trigger": "", "expiry": "", "rationale": reason, "problems": [], "warnings": []}


def normalise_trade_plan(raw, setup_direction):
    action = str(setup_direction or "").strip().upper()
    if action not in ("BUY", "SELL", "WAIT"):
        return empty_plan("NONE", "The analysis did not reach a BUY, SELL or WAIT decision, so no order is proposed.")
    if not isinstance(raw, dict):
        return empty_plan(action)

    order_type = str(raw.get("order_type", "none")).strip().lower().replace(" ", "_")
    if order_type not in ORDER_LABELS:
        order_type = "none"
    plan = empty_plan(action, _text(raw.get("rationale"), 600) or "No rationale returned.")
    plan.update(trigger=_text(raw.get("trigger"), 400), expiry=_text(raw.get("expiry"), 300))
    entry, sl, tp = _num(raw.get("entry")), _num(raw.get("stop_loss")), _num(raw.get("take_profit"))
    current = _num(raw.get("current_price"))
    plan["current_price"] = current

    if order_type == "none" or None in (entry, sl, tp):
        why = ("The model could not read prices reliably enough to propose levels."
               if order_type != "none" else "The model did not propose an order.")
        plan["rationale"] = (plan["rationale"] + " " + why).strip()
        return plan

    side = "BUY" if order_type.startswith("buy") else "SELL" if order_type.startswith("sell") else action
    problems, warnings = [], []
    if action in ("BUY", "SELL") and side != action:
        problems.append(f"The setup is {action} but the proposed order is a {side} order.")
    if action == "WAIT" and order_type not in PENDING:
        problems.append("A WAIT decision must come with a pending order (limit or stop), not a market order.")
    if side == "BUY" and not sl < entry < tp:
        problems.append("For a BUY, stop loss must be below entry and take profit above it.")
    if side == "SELL" and not tp < entry < sl:
        problems.append("For a SELL, take profit must be below entry and stop loss above it.")
    if current is not None and order_type in PENDING:
        wrong = {"buy_limit": entry >= current, "sell_limit": entry <= current,
                 "buy_stop": entry <= current, "sell_stop": entry >= current}[order_type]
        if wrong:
            problems.append(f"A {order_type.replace('_', ' ')} must be placed "
                            f"{'below' if order_type in ('buy_limit', 'sell_stop') else 'above'} the current price.")
    if current is None and order_type in PENDING:
        warnings.append("Current price was not readable, so the pending-order side could not be checked.")

    risk, reward = abs(entry - sl), abs(tp - entry)
    rr = round(reward / risk, 2) if risk else None
    if rr is not None and rr < MIN_SENSIBLE_RR:
        warnings.append(f"Reward-to-risk is only 1:{rr}; below {MIN_SENSIBLE_RR} the setup needs a high hit rate.")
    if order_type == "market" and current is not None and abs(entry - current) > 0.25 * risk:
        warnings.append("Market entry is far from the current price; use a pending order instead.")
    if order_type in PENDING and not plan["trigger"]:
        warnings.append("No activation/cancel condition was given for the pending order.")

    plan.update(available=True, valid=not problems, order_type=order_type, order_label=ORDER_LABELS[order_type],
                side=side, entry=entry, stop_loss=sl, take_profit=tp, risk=round(risk, 6), reward=round(reward, 6),
                risk_reward=rr, decimals=_decimals(entry, sl, tp, current), problems=problems, warnings=warnings)
    return plan


def plan_from_levels(direction, entry, sl, tp, rationale="Levels taken from the simulated setup."):
    """Build a market-order plan from known levels (used to backfill the synthetic demo analyses)."""
    side = str(direction).upper()
    return {"current_price": entry, "order_type": "market", "entry": entry, "stop_loss": sl, "take_profit": tp,
            "rationale": rationale, "trigger": "", "expiry": ""} if side in ("BUY", "SELL") else None
