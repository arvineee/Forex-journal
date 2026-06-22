"""
AI Chatbot for Forex Trading Journal
Powered by Google Gemini — real LLM responses grounded in the user's own trade data.
"""
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from flask import current_app
from app import db
from app.models import Trade, TradingStrategy
from app.ai.models import AIInsight, AILearningLog
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client (lazy-loaded so the import does not blow up if the SDK is
# absent; the route will return a graceful error instead)
# ---------------------------------------------------------------------------
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY") or current_app.config.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel("gemini-3.1-flash-lite")
        return _gemini_client
    except Exception as exc:
        logger.error("Failed to initialise Gemini client: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Conversation persistence helpers
# ---------------------------------------------------------------------------

def _load_history_from_db(user_id: int, limit: int = 20) -> List[Dict]:
    """Load the most recent chat turns from the learning log table."""
    row = (
        AILearningLog.query
        .filter_by(user_id=user_id, event_type="chatbot_conversation")
        .order_by(AILearningLog.created_at.desc())
        .first()
    )
    if not row:
        return []
    try:
        data = json.loads(row.event_data)
        return data.get("messages", [])[-limit:]
    except Exception as exc:
        logger.warning("Could not restore chat history for user %s: %s", user_id, exc)
        return []


def _save_history_to_db(user_id: int, messages: List[Dict]):
    """Persist the latest conversation snapshot."""
    try:
        log = AILearningLog(
            user_id=user_id,
            event_type="chatbot_conversation",
            event_data=json.dumps({"messages": messages[-40:]}),  # keep last 40 turns
            learned_insights=json.dumps([]),
        )
        db.session.add(log)
        db.session.commit()
    except Exception as exc:
        logger.error("Failed to save chat history for user %s: %s", user_id, exc)
        db.session.rollback()


# ---------------------------------------------------------------------------
# Trading context builder
# ---------------------------------------------------------------------------

def _build_trading_context(user_id: int) -> str:
    """
    Compile a concise, structured snapshot of the user's trading performance
    to inject into the Gemini system prompt.
    """
    trades = Trade.query.filter_by(user_id=user_id).order_by(Trade.entry_time.desc()).all()

    if not trades:
        return "The user has no trades recorded yet."

    closed = [t for t in trades if t.pnl is not None]
    total = len(trades)
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl < 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    total_pnl = sum(t.pnl for t in closed)
    avg_pnl = total_pnl / len(closed) if closed else 0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Symbol breakdown
    sym_map: Dict[str, dict] = {}
    for t in closed:
        s = sym_map.setdefault(t.symbol, {"count": 0, "pnl": 0.0, "wins": 0})
        s["count"] += 1; s["pnl"] += t.pnl; s["wins"] += int(t.pnl > 0)
    sym_lines = []
    for sym, s in sorted(sym_map.items(), key=lambda x: -x[1]["pnl"])[:5]:
        wr = s["wins"] / s["count"] * 100
        sym_lines.append(f"  {sym}: {s['count']} trades, {wr:.1f}% WR, total P&L ${s['pnl']:.2f}")

    # Emotion breakdown
    emo_map: Dict[str, dict] = {}
    for t in closed:
        if not t.emotions:
            continue
        e = emo_map.setdefault(t.emotions.lower(), {"count": 0, "pnl": 0.0, "wins": 0})
        e["count"] += 1; e["pnl"] += t.pnl; e["wins"] += int(t.pnl > 0)
    emo_lines = []
    for emo, e in sorted(emo_map.items(), key=lambda x: -x[1]["count"])[:5]:
        wr = e["wins"] / e["count"] * 100
        emo_lines.append(f"  {emo}: {e['count']} trades, {wr:.1f}% WR")

    # Strategy breakdown
    strat_map: Dict[str, dict] = {}
    for t in closed:
        if not t.strategy:
            continue
        s = strat_map.setdefault(t.strategy, {"count": 0, "pnl": 0.0, "wins": 0})
        s["count"] += 1; s["pnl"] += t.pnl; s["wins"] += int(t.pnl > 0)
    strat_lines = []
    for strat, s in sorted(strat_map.items(), key=lambda x: -x[1]["pnl"])[:5]:
        wr = s["wins"] / s["count"] * 100
        strat_lines.append(f"  {strat}: {s['count']} trades, {wr:.1f}% WR, total ${s['pnl']:.2f}")

    # Recent 5 trades
    recent_lines = []
    for t in trades[:5]:
        status = f"P&L ${t.pnl:.2f}" if t.pnl is not None else "open"
        recent_lines.append(
            f"  #{t.id} {t.symbol} {t.direction} @ {t.entry_price} → {status}"
            + (f" | emotion: {t.emotions}" if t.emotions else "")
        )

    # AI insights already computed
    insights = (
        AIInsight.query
        .filter_by(user_id=user_id, is_active=True)
        .order_by(AIInsight.confidence_score.desc())
        .limit(8)
        .all()
    )
    insight_lines = []
    for ins in insights:
        try:
            data = json.loads(ins.insight_value)
            insight_lines.append(f"  [{data.get('type', '?')}] {data.get('value', '')}")
        except Exception:
            pass

    context = f"""
=== TRADER PROFILE ===
Total trades: {total}
Closed trades: {len(closed)}
Win rate: {win_rate:.1f}%
Net P&L: ${total_pnl:.2f}
Avg P&L per trade: ${avg_pnl:.2f}
Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f}
Risk:Reward ratio: 1:{rr:.2f}
Profit factor: {profit_factor:.2f}

=== SYMBOL PERFORMANCE (top 5) ===
{chr(10).join(sym_lines) or '  No data'}

=== EMOTION IMPACT (top 5) ===
{chr(10).join(emo_lines) or '  Emotions not recorded'}

=== STRATEGY PERFORMANCE (top 5) ===
{chr(10).join(strat_lines) or '  Strategies not recorded'}

=== RECENT TRADES (last 5) ===
{chr(10).join(recent_lines) or '  No recent trades'}

=== AI INSIGHTS ALREADY DETECTED ===
{chr(10).join(insight_lines) or '  Run full analysis to generate insights'}
""".strip()

    return context


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are ForexMind AI — a professional trading coach and data analyst embedded inside a Forex trading journal app.

Your job:
- Analyse the trader's real data (provided below) and give specific, data-driven answers.
- Be concise, warm, and direct. Avoid generic financial disclaimers unless specifically relevant.
- Use markdown formatting (bold, bullet points) to structure longer answers.
- If the trader asks something you cannot answer from their data, say so clearly and offer what you CAN tell them.
- Never fabricate statistics. If you don't have enough data, say so.
- Keep replies under 300 words unless the user asks for a detailed report.
- Suggest one follow-up question at the end of each response when it would be helpful.

You have access to the following live snapshot of this trader's journal:

{trading_context}
""".strip()


# ---------------------------------------------------------------------------
# Main chatbot class
# ---------------------------------------------------------------------------

class ForexChatbot:
    """
    Gemini-powered trading assistant.

    Conversation history is persisted to the database so context survives
    across Flask requests (which create a new instance each time).
    """

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._history: Optional[List[Dict]] = None  # lazy-loaded

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Send the user's message to Gemini with full trading context and history.
        Returns a dict with 'text' and optional 'suggestions'.
        """
        history = self._get_history()

        # Append the new user turn
        history.append({"role": "user", "content": message, "timestamp": datetime.utcnow().isoformat()})

        try:
            trading_context = _build_trading_context(self.user_id)
            system = SYSTEM_PROMPT.format(trading_context=trading_context)
            reply_text = self._call_gemini(system, history)
        except Exception as exc:
            logger.error("Gemini call failed for user %s: %s", self.user_id, exc)
            reply_text = (
                "I'm having trouble connecting to the AI service right now. "
                "Please try again in a moment."
            )

        history.append({
            "role": "assistant",
            "content": reply_text,
            "timestamp": datetime.utcnow().isoformat(),
        })

        self._history = history

        # Persist every 5 assistant turns to avoid hammering the DB
        assistant_turns = sum(1 for m in history if m["role"] == "assistant")
        if assistant_turns % 5 == 0:
            _save_history_to_db(self.user_id, history)

        return {
            "text": reply_text,
            "suggestions": self._get_contextual_suggestions(message, reply_text),
        }

    def get_conversation_history(self, limit: int = 30) -> List[Dict]:
        history = self._get_history()
        return [
            {"role": m["role"], "content": m["content"], "timestamp": m.get("timestamp", "")}
            for m in history[-limit:]
        ]

    def clear_history(self):
        self._history = []
        _save_history_to_db(self.user_id, [])

    # ------------------------------------------------------------------ #
    #  Gemini call                                                         #
    # ------------------------------------------------------------------ #

    def _call_gemini(self, system_prompt: str, history: List[Dict]) -> str:
        """
        Build the Gemini request from the system prompt + conversation history.
        We prepend the system prompt as the first user turn (the Gemini
        Python SDK uses `system_instruction` for true system prompts — we
        support both approaches for compatibility with older SDK versions).
        """
        client = _get_gemini_client()

        # Build the message list Gemini expects
        # System prompt goes in as a model-acknowledged instruction
        gemini_history = []
        for m in history[:-1]:  # all but the last (current) user message
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        # Start a chat session with history
        chat = client.start_chat(history=gemini_history)

        # The current user message, prepended with the system context on the
        # first turn so Gemini always has the trading data in scope.
        current_msg = history[-1]["content"]
        if len(gemini_history) == 0:
            current_msg = f"{system_prompt}\n\n---\n\nUser: {current_msg}"
        else:
            # Re-inject context summary every 10 turns to avoid it drifting out of context
            if len(gemini_history) % 10 == 0:
                current_msg = (
                    f"[Context refresh — trader data as of now]\n"
                    f"{system_prompt}\n\n---\n\n{current_msg}"
                )

        response = chat.send_message(current_msg)
        return response.text.strip()

    # ------------------------------------------------------------------ #
    #  Context-aware follow-up suggestions                                #
    # ------------------------------------------------------------------ #

    def _get_contextual_suggestions(self, user_msg: str, reply: str) -> List[str]:
        """
        Generate relevant follow-up button suggestions based on what was just discussed.
        """
        msg_lower = user_msg.lower()
        reply_lower = reply.lower()

        if any(w in msg_lower for w in ("emotion", "feeling", "mood", "anxious", "fear")):
            return [
                "Which emotion makes me trade best?",
                "How do I avoid trading emotionally?",
                "Show my discipline score",
            ]
        if any(w in msg_lower for w in ("symbol", "pair", "eurusd", "gbpusd", "gold")):
            return [
                "What's my worst performing pair?",
                "Should I stop trading any pairs?",
                "Compare BUY vs SELL on my best pair",
            ]
        if any(w in msg_lower for w in ("risk", "drawdown", "stop loss", "position size")):
            return [
                "What is my average risk per trade?",
                "How can I reduce drawdown?",
                "Am I over-leveraged?",
            ]
        if any(w in msg_lower for w in ("performance", "overall", "how am i", "win rate")):
            return [
                "Break down performance by month",
                "What's dragging my win rate down?",
                "Give me my top 3 improvements",
            ]
        if any(w in msg_lower for w in ("strategy", "setup", "approach")):
            return [
                "Which strategy has the highest profit factor?",
                "Should I drop any strategies?",
                "How consistent is my best strategy?",
            ]
        if "revenge" in reply_lower or "streak" in reply_lower:
            return [
                "How many revenge trades have I made?",
                "What triggers my losing streaks?",
                "How do I break a losing streak?",
            ]

        # Default
        return [
            "What should I improve first?",
            "Analyse my recent trades",
            "Show my key statistics",
        ]

    # ------------------------------------------------------------------ #
    #  History management                                                  #
    # ------------------------------------------------------------------ #

    def _get_history(self) -> List[Dict]:
        if self._history is None:
            self._history = _load_history_from_db(self.user_id)
        return self._history

