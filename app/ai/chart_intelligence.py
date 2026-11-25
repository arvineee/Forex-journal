"""Multitimeframe chart analysis and outcome-labelled learning.

Vision output is stored as observations. Predictive probabilities are withheld
until enough user-owned, outcome-labelled chart analyses exist.
"""
import io
import time
import json
import logging
import os
import re
import uuid
from datetime import datetime

from PIL import Image
from flask import current_app
from werkzeug.utils import secure_filename

from app import db
from app.models import Trade
from app.ai.models import AIChartAnalysis
from app.ai.trade_plan import normalise_trade_plan

logger = logging.getLogger(__name__)
TIMEFRAMES = ("4H", "1H", "30M", "15M")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _gemini_model():
    import google.generativeai as genai
    key = current_app.config.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Chart analysis requires GEMINI_API_KEY.")
    genai.configure(api_key=key)
    return genai.GenerativeModel(current_app.config.get("GEMINI_MODEL", "gemini-2.0-flash"))


def _image_from_upload(upload):
    filename = secure_filename(upload.filename or "")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Use PNG, JPG, JPEG or WEBP chart screenshots.")
    raw = upload.read(MAX_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Each chart image must be between 1 byte and 8 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # Smaller images upload/process much faster and avoid Gemini 504 timeouts.
        image.thumbnail((1600, 1600))
    except Exception as exc:
        raise ValueError("One of the uploaded files is not a valid image.") from exc
    return filename, raw, image


def _normalise_analysis(data):
    """Constrain model output to stable, numeric fields used by learning."""
    if not isinstance(data, dict):
        raise ValueError("The vision model returned an invalid analysis.")
    structure = data.get("timeframes", {})
    normalised_tf = {}
    for tf in TIMEFRAMES:
        item = structure.get(tf, {}) if isinstance(structure, dict) else {}
        normalised_tf[tf] = {
            "bias": str(item.get("bias", "unclear"))[:40],
            "structure": str(item.get("structure", "unclear"))[:300],
            "liquidity": str(item.get("liquidity", "not identified"))[:300],
            "key_levels": str(item.get("key_levels", "not identified"))[:300],
            "confidence": _bounded(item.get("confidence"), 0, 1, 0),
        }
    features = data.get("features", {})
    feature_names = (
        "trend_alignment", "structure_quality", "liquidity_sweep",
        "asian_range_sweep", "london_liquidity_sweep",
        "bos_choch_confirmation", "order_block_quality", "fvg_quality",
        "ob_fvg_retracement", "premium_discount_location",
        "entry_location", "risk_reward_quality", "session_quality",
    )
    numeric_features = {name: _bounded(features.get(name), 0, 1, 0.5) for name in feature_names}
    result = {
        "market_bias": str(data.get("market_bias", "unclear"))[:40],
        "setup_direction": str(data.get("setup_direction", "unclear"))[:20],
        "setup_type": str(data.get("setup_type", "not identified"))[:100],
        "timeframes": normalised_tf,
        "features": numeric_features,
        "confluences": _string_list(data.get("confluences")),
        "conflicts": _string_list(data.get("conflicts")),
        "invalidation": str(data.get("invalidation", "Not identified; define before entry."))[:500],
        "summary": str(data.get("summary", "No summary returned."))[:1500],
        "risk_notes": _string_list(data.get("risk_notes")),
        "chart_readability": str(data.get("chart_readability", "unknown"))[:100],
    }
    # Entry / stop / take-profit (or a pending order when the call is WAIT), checked for sane geometry.
    result["trade_plan"] = normalise_trade_plan(data.get("trade_plan"), result["setup_direction"])
    return result


def _bounded(value, low, high, default):
    try:
        number = float(value)
        if number != number:
            return default
        return max(low, min(high, number))
    except (TypeError, ValueError):
        return default


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(x)[:250] for x in value[:12]]


def analyse_four_timeframes(user_id, symbol, uploads, direction=None, trade_id=None):
    """Analyse and persist one set of 4 timeframe screenshots."""
    symbol = re.sub(r"[^A-Za-z0-9._/-]", "", (symbol or "").upper())[:20]
    if not symbol:
        raise ValueError("Enter a valid trading symbol.")
    if not isinstance(uploads, dict) or any(not uploads.get(tf) or not uploads[tf].filename for tf in TIMEFRAMES):
        raise ValueError("Upload all four charts: 4H, 1H, 30M and 15M.")

    trade = None
    if trade_id:
        try:
            trade = Trade.query.filter_by(id=int(trade_id), user_id=user_id).first()
        except (TypeError, ValueError):
            trade = None
        if not trade:
            raise ValueError("The selected trade was not found in your journal.")
        if trade.symbol.upper() != symbol:
            raise ValueError("The chart symbol must match the linked journal trade.")

    image_payloads, image_rows = [], {}
    for tf in TIMEFRAMES:
        filename, raw, image = _image_from_upload(uploads[tf])
        image_payloads.append((tf, image))
        stored_name = f"chart_{user_id}_{uuid.uuid4().hex}.{filename.rsplit('.', 1)[-1].lower()}"
        target = os.path.join(current_app.config["UPLOAD_FOLDER"], stored_name)
        with open(target, "wb") as out:
            out.write(raw)
        image_rows[tf] = stored_name

    prompt = """
You are a cautious, evidence-led forex market-structure analyst. You will receive
FOUR screenshots for the SAME instrument, in this exact order: 4H, 1H, 30M, 15M.
Use the 4H for broad directional context, 1H for structure and dealing range,
30M for setup development, and 15M for execution/confirmation. Do not invent
candles, prices, levels, volume, indicators or chart details that are not visible.
If an item is unreadable, say unclear. Distinguish observed facts from inference.
Assess market structure, BOS/CHoCH, liquidity pools and sweeps, order blocks,
fair value gaps, premium/discount, likely invalidation and timeframe alignment.
Specifically inspect the Asian session high/low range, whether London swept either
side, whether a clear BOS followed the sweep, and whether price retraced into a
valid order block or FVG. Treat this as a possible SMC sequence, not an assumption:
only mark a sweep or session relationship when it is visible or the chart labels
make it verifiable. If session boundaries/times are not visible, score session
features as 0.5 (unknown), not as confirmed. Do not force a trade. Explicitly
identify conflicts and missing evidence.
Return ONLY valid JSON with this schema:
{
 "market_bias":"bullish|bearish|range|unclear",
 "setup_direction":"BUY|SELL|WAIT",
 "setup_type":"short label",
 "timeframes":{
  "4H":{"bias":"...","structure":"...","liquidity":"...","key_levels":"...","confidence":0.0},
  "1H":{"bias":"...","structure":"...","liquidity":"...","key_levels":"...","confidence":0.0},
  "30M":{"bias":"...","structure":"...","liquidity":"...","key_levels":"...","confidence":0.0},
  "15M":{"bias":"...","structure":"...","liquidity":"...","key_levels":"...","confidence":0.0}
 },
 "features":{
  "trend_alignment":0.0,"structure_quality":0.0,"liquidity_sweep":0.0,
  "asian_range_sweep":0.0,"london_liquidity_sweep":0.0,
  "bos_choch_confirmation":0.0,"order_block_quality":0.0,"fvg_quality":0.0,
  "ob_fvg_retracement":0.0,"premium_discount_location":0.0,
  "entry_location":0.0,"risk_reward_quality":0.0,"session_quality":0.5
 },
 "confluences":["..."],"conflicts":["..."],
 "invalidation":"...", "summary":"...", "risk_notes":["..."],
 "chart_readability":"clear|mixed|poor",
 "trade_plan":{
  "current_price":0.0,
  "order_type":"market|buy_limit|sell_limit|buy_stop|sell_stop|none",
  "entry":0.0,"stop_loss":0.0,"take_profit":0.0,
  "trigger":"what must happen before this order is valid / when to place it",
  "expiry":"when to cancel the order if it has not filled",
  "rationale":"why these three levels"
 }
}
TRADE PLAN RULES (follow exactly):
- If setup_direction is BUY or SELL: order_type is "market" (entry = the last visible
  15M close / current price) unless the best entry is clearly a retest of an order block or
  FVG, in which case use the same-side limit order ("buy_limit" for BUY, "sell_limit" for SELL).
- If setup_direction is WAIT: you MUST propose the most probable PENDING order for when the
  setup completes (buy_limit/sell_limit at an order block or FVG, or buy_stop/sell_stop at a
  structure break), with entry, stop_loss and take_profit, plus the trigger and expiry.
  Never use "market" with WAIT.
- Read every price from the price axis or the labels printed on the charts. Interpolate
  between axis ticks if needed. If prices are not readable, set order_type to "none" and
  leave the numbers null; never guess or invent levels.
- stop_loss goes just beyond the invalidation level (sweep extreme or the order block /
  structure that, if broken, kills the idea). take_profit goes at the next opposing
  liquidity pool or swing. A BUY needs stop_loss < entry < take_profit; a SELL needs
  take_profit < entry < stop_loss. A buy_limit is below the current price, a sell_limit above,
  a buy_stop above, a sell_stop below.
- Aim for at least 1:1.5 reward-to-risk; if the structure cannot offer it, say so in
  rationale and in risk_notes instead of stretching the take profit.
- current_price must be the last visible close on the 15M chart.
For each numeric feature use a score from 0 (very poor/absent) to 1 (strong).
Do not interpret these feature scores as a win probability. session_quality must
be 0.5 unless the trading session/time is actually provided.
"""
    model = _gemini_model()
    content = [prompt, f"Instrument: {symbol}. User's intended direction: {direction or 'not specified'}."]
    for tf, image in image_payloads:
        content.extend([f"{tf} timeframe chart:", image])
    response = None
    last_exc = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                content,
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
                request_options={"timeout": 150},
            )
            break
        except Exception as exc:  # 504 DeadlineExceeded / 503 ServiceUnavailable are retryable
            last_exc = exc
            name = type(exc).__name__
            if name not in ("DeadlineExceeded", "ServiceUnavailable", "InternalServerError", "ResourceExhausted"):
                raise
            logger.warning("Gemini attempt %s failed (%s); retrying", attempt + 1, name)
            time.sleep(2 * (attempt + 1))
    if response is None:
        raise RuntimeError("The AI service timed out. Please try again in a minute.") from last_exc
    text = (getattr(response, "text", "") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    parsed = _normalise_analysis(json.loads(text))

    record = AIChartAnalysis(
        user_id=user_id,
        trade_id=trade.id if trade else None,
        symbol=symbol,
        direction=(direction or parsed["setup_direction"] or "").upper()[:8],
        analysis_json=json.dumps(parsed),
        image_files_json=json.dumps(image_rows),
        outcome_label=(1 if trade.pnl > 0 else 0 if trade.pnl < 0 else None) if trade and trade.pnl is not None else None,
    )
    db.session.add(record)
    db.session.commit()
    return record, parsed


def learn_chart_outcomes(user_id):
    """Train/cache a calibrated model from unique, outcome-labelled chart trades."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss
    import numpy as np
    import hashlib
    import joblib

    raw_rows = (AIChartAnalysis.query.filter_by(user_id=user_id).filter(
        AIChartAnalysis.outcome_label.isnot(None), AIChartAnalysis.trade_id.isnot(None)
    ).order_by(AIChartAnalysis.created_at.asc()).all())
    by_trade = {}
    for row in raw_rows:
        by_trade[row.trade_id] = row
    rows = list(by_trade.values())
    if len(rows) < 40:
        return {"ready": False, "labelled_samples": len(rows), "minimum_samples": 40,
                "message": "Keep linking chart analyses to completed trades. Probability estimates unlock after 40 labelled chart examples."}

    X, y = [], []
    keys = ("trend_alignment", "structure_quality", "liquidity_sweep",
            "asian_range_sweep", "london_liquidity_sweep",
            "bos_choch_confirmation", "order_block_quality", "fvg_quality",
            "ob_fvg_retracement", "premium_discount_location",
            "entry_location", "risk_reward_quality", "session_quality")
    for row in rows:
        try:
            features = json.loads(row.analysis_json).get("features", {})
            X.append([_bounded(features.get(k), 0, 1, 0.5) for k in keys])
            y.append(int(row.outcome_label))
        except Exception:
            continue
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if len(y) < 40 or min(counts) < 10:
        return {"ready": False, "labelled_samples": len(y), "minimum_per_class": 10,
                "wins": int(counts[1]), "losses": int(counts[0]),
                "message": "The chart dataset needs at least 10 winning and 10 losing examples, with 40 labelled examples overall."}

    fingerprint_source = "|".join(
        f"{row.trade_id}:{row.outcome_label}:{row.id}" for row in rows
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    model_dir = current_app.config.get("AI_MODEL_DIR", "/tmp/forex_ai_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"user_{user_id}_chart_model.pkl")
    meta_path = os.path.join(model_dir, f"user_{user_id}_chart_meta.json")
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("fingerprint") == fingerprint:
                model = joblib.load(model_path)
                return {"ready": True, "model": model, "metrics": meta["metrics"], "features": list(keys)}
        except Exception:
            logger.warning("Could not load cached chart model for user %s; retraining.", user_id)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=19)
    model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=250, max_depth=5,
            min_samples_leaf=3, class_weight="balanced", random_state=19),
        method="sigmoid", cv=cv
    )
    # Out-of-fold metrics are more informative than training-set accuracy.
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "ready": True, "labelled_samples": len(y), "wins": int(counts[1]),
        "losses": int(counts[0]),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 3),
        "brier_score": round(float(brier_score_loss(y, probs)), 3),
        "features": list(keys),
        "message": "Model trained on your labelled chart history. Metrics are cross-validated and may change as new trades are added."
    }
    model.fit(X, y)
    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"fingerprint": fingerprint, "metrics": metrics}, handle)
    return {"ready": True, "model": model, "metrics": metrics, "features": list(keys)}


def predict_from_chart(user_id, parsed):
    """Return no numeric probability until the personal chart model is ready."""
    learned = learn_chart_outcomes(user_id)
    if not learned.get("ready"):
        return {"probability_available": False, "learning_status": learned}
    import numpy as np
    vector = [[_bounded(parsed["features"].get(k), 0, 1, 0.5) for k in learned["features"]]]
    probability = float(learned["model"].predict_proba(np.asarray(vector))[0][1])
    return {
        "probability_available": True,
        "win_probability": round(probability * 100, 1),
        "model_metrics": learned["metrics"],
        "disclaimer": "This is a historical, calibrated estimate for patterns similar to your labelled chart setups, not a guarantee or a standalone reason to trade."
    }


def compare_chart_outcomes(user_id):
    """Compare the feature profile of this user's winning and losing chart setups."""
    raw_rows = (AIChartAnalysis.query.filter_by(user_id=user_id).filter(
        AIChartAnalysis.outcome_label.isnot(None), AIChartAnalysis.trade_id.isnot(None)
    ).order_by(AIChartAnalysis.created_at.asc()).all())
    by_trade = {}
    for row in raw_rows:
        by_trade[row.trade_id] = row
    rows = list(by_trade.values())
    keys = ("trend_alignment", "structure_quality", "liquidity_sweep",
            "asian_range_sweep", "london_liquidity_sweep",
            "bos_choch_confirmation", "order_block_quality", "fvg_quality",
            "ob_fvg_retracement", "premium_discount_location",
            "entry_location", "risk_reward_quality", "session_quality")
    groups = {0: [], 1: []}
    for row in rows:
        try:
            features = json.loads(row.analysis_json).get("features", {})
            groups[int(row.outcome_label)].append(
                [_bounded(features.get(k), 0, 1, 0.5) for k in keys]
            )
        except Exception:
            continue
    if not groups[0] or not groups[1]:
        return {"wins": len(groups[1]), "losses": len(groups[0]), "features": []}
    import numpy as np
    win_avg = np.mean(groups[1], axis=0)
    loss_avg = np.mean(groups[0], axis=0)
    return {
        "wins": len(groups[1]), "losses": len(groups[0]),
        "features": [
            {"name": key.replace("_", " ").title(),
             "win_average": round(float(win_avg[i]), 2),
             "loss_average": round(float(loss_avg[i]), 2),
             "difference": round(float(win_avg[i] - loss_avg[i]), 2)}
            for i, key in enumerate(keys)
        ]
    }


# ---------------------------------------------------------------------------
# Adaptive SMC playbook: outcome-conditioned evidence, not generic rules.
# ---------------------------------------------------------------------------

SMC_PLAYBOOK_FEATURES = {
    "trend_alignment": "Higher-timeframe alignment",
    "structure_quality": "Market structure clarity",
    "liquidity_sweep": "Liquidity sweep quality",
    "bos_choch_confirmation": "BOS / CHoCH confirmation",
    "order_block_quality": "Order block quality",
    "fvg_quality": "Fair value gap quality",
    "ob_fvg_retracement": "OB / FVG retracement",
    "premium_discount_location": "Premium / discount location",
    "entry_location": "Entry location",
    "risk_reward_quality": "Risk-to-reward quality",
}


def _feature_band(value):
    if value < 0.4:
        return "weak"
    if value > 0.6:
        return "strong"
    return "mixed"


def build_smc_playbook(user_id, minimum_bucket_samples=8):
    """Learn which SMC conditions correlate with this user's realised outcomes.

    One latest chart analysis is used per trade. Session-specific features are
    intentionally omitted because screenshots often do not prove session time.
    """
    raw_rows = (AIChartAnalysis.query.filter_by(user_id=user_id)
                .filter(AIChartAnalysis.trade_id.isnot(None))
                .order_by(AIChartAnalysis.created_at.asc()).all())
    if not raw_rows:
        return {"sample_size": 0, "minimum_bucket_samples": minimum_bucket_samples,
                "baseline": None, "findings": [], "smc_score": None,
                "message": "Link chart analyses to completed trades to start building your personal SMC playbook."}

    trades = Trade.query.filter(Trade.user_id == user_id,
                                Trade.id.in_([row.trade_id for row in raw_rows])).all()
    all_trade_by_id = {trade.id: trade for trade in trades}
    latest_by_trade = {}
    for row in raw_rows:
        linked = all_trade_by_id.get(row.trade_id)
        if linked and linked.status == "closed" and linked.pnl is not None:
            row.outcome_label = 1 if linked.pnl > 0 else 0 if linked.pnl < 0 else None
            latest_by_trade[row.trade_id] = row
    db.session.commit()
    rows = list(latest_by_trade.values())
    trade_by_id = {trade.id: trade for trade in trades
                   if trade.status == "closed" and trade.pnl is not None and trade.pnl != 0}
    observations = []
    for row in rows:
        trade = trade_by_id.get(row.trade_id)
        if not trade:
            continue
        try:
            features = json.loads(row.analysis_json).get("features", {})
            clean = {key: _bounded(features.get(key), 0, 1, 0.5)
                     for key in SMC_PLAYBOOK_FEATURES}
            observations.append({"trade_id": trade.id, "pnl": float(trade.pnl),
                                 "win": int(trade.pnl > 0), "features": clean,
                                 "symbol": trade.symbol, "direction": trade.direction})
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    if not observations:
        return {"sample_size": 0, "minimum_bucket_samples": minimum_bucket_samples,
                "baseline": None, "findings": [], "smc_score": None,
                "message": "No linked, closed trades with non-zero recorded P&L are available yet."}

    sample_size = len(observations)
    baseline = {
        "trades": sample_size,
        "win_rate": round(100 * sum(item["win"] for item in observations) / sample_size, 1),
        "expectancy": round(sum(item["pnl"] for item in observations) / sample_size, 2),
    }

    buckets = {}
    for item in observations:
        for key, label in SMC_PLAYBOOK_FEATURES.items():
            band = _feature_band(item["features"][key])
            buckets.setdefault((key, band), []).append(item)

    findings = []
    for (key, band), members in buckets.items():
        if len(members) < minimum_bucket_samples:
            continue
        win_rate = 100 * sum(item["win"] for item in members) / len(members)
        expectancy = sum(item["pnl"] for item in members) / len(members)
        delta = win_rate - baseline["win_rate"]
        if delta >= 10 and expectancy > baseline["expectancy"]:
            signal = "favourable association"
        elif delta <= -10 and expectancy < baseline["expectancy"]:
            signal = "review / potential weakness"
        else:
            signal = "no clear edge yet"
        findings.append({
            "feature": label, "feature_key": key, "band": band,
            "trades": len(members), "win_rate": round(win_rate, 1),
            "expectancy": round(expectancy, 2), "win_rate_delta": round(delta, 1),
            "signal": signal,
        })
    findings.sort(key=lambda item: (item["win_rate_delta"], item["trades"]), reverse=True)

    # A simple transparent confluence score; it is not a prediction.
    core_keys = ("trend_alignment", "structure_quality", "liquidity_sweep",
                 "bos_choch_confirmation", "order_block_quality", "fvg_quality",
                 "ob_fvg_retracement", "premium_discount_location", "entry_location")
    score_groups = {"low": [], "medium": [], "high": []}
    for item in observations:
        score = sum(item["features"][key] for key in core_keys) / len(core_keys)
        band = "low" if score < 0.4 else "high" if score > 0.7 else "medium"
        score_groups[band].append(item)
    score_summary = []
    for band in ("low", "medium", "high"):
        members = score_groups[band]
        if members:
            score_summary.append({
                "band": band, "trades": len(members),
                "win_rate": round(100 * sum(item["win"] for item in members) / len(members), 1),
                "expectancy": round(sum(item["pnl"] for item in members) / len(members), 2),
            })

    return {
        "sample_size": sample_size, "minimum_bucket_samples": minimum_bucket_samples,
        "baseline": baseline, "findings": findings[:12],
        "smc_score": {"description": "Average of nine visible SMC confluence scores; not a win probability.",
                       "groups": score_summary},
        "message": ("Associations are calculated from your linked trade history. They are descriptive, "
                    "not proof that a condition causes a win. Small groups are hidden until they meet "
                    f"the minimum of {minimum_bucket_samples} trades."),
    }


# ---------------------------------------------------------------------------
# Single-setup SMC confluence view (used by the chart analysis detail page)
# ---------------------------------------------------------------------------

SMC_CORE_KEYS = (
    "trend_alignment", "structure_quality", "liquidity_sweep",
    "bos_choch_confirmation", "order_block_quality", "fvg_quality",
    "ob_fvg_retracement", "premium_discount_location", "entry_location",
)  # same nine factors build_smc_playbook() averages into its confluence score


def confluence_score(features):
    """Mean of the nine core SMC factor scores (0-1) and its low/medium/high band."""
    features = features if isinstance(features, dict) else {}
    values = [_bounded(features.get(key), 0, 1, 0.5) for key in SMC_CORE_KEYS]
    score = sum(values) / len(values)
    band = "low" if score < 0.4 else "high" if score > 0.7 else "medium"
    return round(score, 3), band


def score_smc_setup(user_id, parsed):
    """Explain one analysed setup against the user's own SMC history.

    Returns the factor-by-factor bands, the confluence score and, when the user has
    enough linked closed trades, how setups in the same confluence band performed.
    This is descriptive context, not a win probability.
    """
    features = (parsed or {}).get("features", {})
    score, band = confluence_score(features)
    playbook = build_smc_playbook(user_id)
    groups = ((playbook.get("smc_score") or {}).get("groups")) or []
    history = next((g for g in groups if g["band"] == band), None)
    factors = [
        {"name": label, "key": key, "score": round(_bounded(features.get(key), 0, 1, 0.5), 2),
         "band": _feature_band(_bounded(features.get(key), 0, 1, 0.5))}
        for key, label in SMC_PLAYBOOK_FEATURES.items()
    ]
    return {"score": score, "band": band, "history": history,
            "baseline": playbook.get("baseline"), "factors": factors,
            "sample_size": playbook.get("sample_size", 0)}
