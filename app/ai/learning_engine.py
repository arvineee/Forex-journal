"""
AI Learning Engine for Forex Trading Journal
Advanced pattern detection, predictive analytics, and personalized insights
"""
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from app import db
from app.models import Trade, AccountBalance
from app.ai.models import AIInsight, TradePattern, AILearningLog

logger = logging.getLogger(__name__)


class ForexAI:
    """
    AI Learning Engine — analyses user trades and surfaces actionable insights.

    Design principles
    -----------------
    * analyze_and_learn() has two modes: lightweight (on every save) and full
      (scheduled every 6 hours).  Pass full=True only from the scheduler.
    * The RandomForest model is cached to disk per user and only retrained when
      the trade count has grown by ≥ 10 since last training.
    * Every except block logs the error so failures are visible in the Flask log.
    """

    MODEL_DIR = "/tmp/forex_ai_models"

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.trades: List[Trade] = []
        self.account_balances: List[AccountBalance] = []
        self.insights: Dict[str, AIInsight] = {}
        self.patterns: Dict[str, TradePattern] = {}
        self.prediction_model: Optional[RandomForestClassifier] = None
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    def load_data(self):
        """Load user trading data and existing AI state from the database."""
        self.trades = (
            Trade.query
            .filter_by(user_id=self.user_id)
            .order_by(Trade.entry_time.asc())
            .all()
        )
        self.account_balances = (
            AccountBalance.query
            .filter_by(user_id=self.user_id)
            .order_by(AccountBalance.date.asc())
            .all()
        )
        self.insights = {
            i.insight_key: i
            for i in AIInsight.query.filter_by(user_id=self.user_id, is_active=True).all()
        }
        self.patterns = {
            p.pattern_type: p
            for p in TradePattern.query.filter_by(user_id=self.user_id).all()
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_and_learn(self, new_trade: Optional[Trade] = None, full: bool = False) -> dict:
        """
        Run AI analysis.

        Parameters
        ----------
        new_trade : Trade, optional
            If provided, run targeted analysis for the new / edited trade.
        full : bool
            False (default) → lightweight pass suitable for on-save triggers.
            True            → complete analysis, used by the scheduler.
        """
        self.load_data()

        if not self.trades:
            return {}

        learned = {}

        if not full and new_trade:
            # Lightweight: only the analyses that are cheap and immediately useful
            learned.update(self._analyze_new_trade(new_trade))
            learned.update(self._detect_revenge_trading())
            learned.update(self._analyze_streaks_and_sequences())
            self._save_insights(learned)
            self._log_learning_session(learned, mode="lightweight")
            return learned

        # Full analysis (scheduler or manual trigger)
        learned.update(self._analyze_emotional_impact())
        learned.update(self._analyze_time_patterns())
        learned.update(self._analyze_symbol_performance())
        learned.update(self._analyze_strategy_performance())
        learned.update(self._analyze_risk_management())
        learned.update(self._analyze_streaks_and_sequences())
        learned.update(self._analyze_market_conditions())
        learned.update(self._analyze_trade_duration())
        learned.update(self._analyze_entry_exit_quality())
        learned.update(self._detect_revenge_trading())
        learned.update(self._analyze_psychological_biases())
        learned.update(self._calculate_advanced_metrics())
        learned.update(self._detect_behavioral_patterns())
        learned.update(self._predict_trade_success())
        learned.update(self._generate_recommendations(learned))

        if new_trade:
            learned.update(self._analyze_new_trade(new_trade))

        self._save_insights(learned)
        self._log_learning_session(learned, mode="full")
        return learned

    def get_personalized_insights(self, limit: int = 20) -> List[dict]:
        """Return the top insights for this user, ordered by confidence."""
        self.load_data()
        rows = (
            AIInsight.query
            .filter_by(user_id=self.user_id, is_active=True)
            .order_by(
                AIInsight.confidence_score.desc(),
                AIInsight.last_updated.desc()
            )
            .limit(limit)
            .all()
        )
        result = []
        for row in rows:
            try:
                data = json.loads(row.insight_value)
                data["id"] = row.id
                data["confidence"] = row.confidence_score
                data["last_updated"] = row.last_updated.isoformat()
                result.append(data)
            except Exception as exc:
                logger.warning("Failed to parse insight %s: %s", row.id, exc)
        return result

    # ------------------------------------------------------------------ #
    #  Emotional analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_emotional_impact(self) -> dict:
        insights = {}
        emotion_stats: Dict[str, dict] = defaultdict(
            lambda: {"count": 0, "pnl": 0.0, "wins": 0, "total_duration": 0.0}
        )

        for trade in self.trades:
            if not (trade.emotions and trade.pnl is not None):
                continue
            s = emotion_stats[trade.emotions.lower()]
            s["count"] += 1
            s["pnl"] += trade.pnl
            s["wins"] += int(trade.pnl > 0)
            if trade.exit_time and trade.entry_time:
                s["total_duration"] += (
                    (trade.exit_time - trade.entry_time).total_seconds() / 3600
                )

        for emotion, s in emotion_stats.items():
            if s["count"] < 3:
                continue
            win_rate = s["wins"] / s["count"] * 100
            avg_pnl = s["pnl"] / s["count"]
            avg_dur = s["total_duration"] / s["count"]
            confidence = min(s["count"] / 10, 1.0)

            insights[f"emotion_{emotion}_analysis"] = {
                "type": "emotion_analysis",
                "key": f"emotion_{emotion}_analysis",
                "value": f"{emotion.title()}: {win_rate:.1f}% win rate, avg P&L ${avg_pnl:.2f}",
                "confidence": confidence,
                "metadata": {
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "trade_count": s["count"],
                    "avg_duration_hours": avg_dur,
                },
                "recommendation": self._get_emotion_recommendation(emotion, win_rate, avg_pnl),
            }

            if win_rate < 30 and s["count"] >= 5:
                insights[f"emotion_{emotion}_warning"] = {
                    "type": "emotion_analysis",
                    "key": f"emotion_{emotion}_warning",
                    "value": f"⚠️ Trading while {emotion} has only a {win_rate:.1f}% success rate",
                    "confidence": 0.9,
                    "priority": "high",
                    "recommendation": (
                        f"Avoid trading when feeling {emotion}. Step away and journal first."
                    ),
                }

        if len(emotion_stats) >= 3:
            stability = self._analyze_emotional_stability(emotion_stats)
            if stability:
                insights["emotional_stability"] = stability

        return insights

    def _analyze_emotional_stability(self, emotion_stats: dict) -> dict:
        scores = [
            (s["wins"] / s["count"] * 100)
            for s in emotion_stats.values()
            if s["count"] >= 3
        ]
        if len(scores) < 3:
            return {}
        stability = max(0.0, 1 - np.std(scores) / 100)
        return {
            "type": "emotion_analysis",
            "key": "emotional_stability",
            "value": f"Emotional Stability Score: {stability:.2f} / 1.00",
            "confidence": 0.8,
            "metadata": {"stability_score": stability, "emotion_variance": np.std(scores)},
            "interpretation": "High" if stability > 0.7 else "Moderate" if stability > 0.5 else "Low",
        }

    def _get_emotion_recommendation(self, emotion: str, win_rate: float, avg_pnl: float) -> Optional[str]:
        low_map = {
            "anxious": "Practice a pre-trade breathing routine. Consider paper trading when anxious.",
            "fearful": "Build confidence with reduced position sizes. Focus on process, not P&L.",
            "impulsive": "Implement a mandatory 5-minute pause before entering any trade.",
            "greedy": "Set profit targets before opening the trade and use limit orders.",
            "frustrated": "Take a mandatory 24-hour break when frustrated.",
            "angry": "Do not trade when angry. Journal your feelings first.",
        }
        if win_rate < 30:
            return low_map.get(emotion, "Avoid trading in this emotional state.")
        if win_rate > 70 and emotion in ("confident", "disciplined", "focused"):
            return f"Excellent — trading while {emotion} is your strength. Replicate these conditions."
        return None

    # ------------------------------------------------------------------ #
    #  Streaks & sequences                                                 #
    # ------------------------------------------------------------------ #

    def _analyze_streaks_and_sequences(self) -> dict:
        insights = {}
        if len(self.trades) < 5:
            return insights

        sorted_trades = sorted(
            [t for t in self.trades if t.pnl is not None],
            key=lambda t: t.entry_time,
        )

        current_streak = 0
        current_type = None
        max_win_streak = max_loss_streak = 0
        post_win = []
        post_loss = []

        for idx, trade in enumerate(sorted_trades):
            is_win = trade.pnl > 0
            if current_type == is_win:
                current_streak += 1
            else:
                current_streak = 1
                current_type = is_win
            max_win_streak = max(max_win_streak, current_streak) if is_win else max_win_streak
            max_loss_streak = max(max_loss_streak, current_streak) if not is_win else max_loss_streak

            if idx >= 3:
                prev = sorted_trades[idx - 3: idx]
                if all(t.pnl > 0 for t in prev):
                    post_win.append(trade.pnl)
                elif all(t.pnl < 0 for t in prev):
                    post_loss.append(trade.pnl)

        insights["streak_analysis"] = {
            "type": "behavioral_pattern",
            "key": "streak_analysis",
            "value": f"Longest win streak: {max_win_streak} | Longest loss streak: {max_loss_streak}",
            "confidence": 0.8,
            "metadata": {"max_win_streak": max_win_streak, "max_loss_streak": max_loss_streak},
        }

        if post_win:
            avg = np.mean(post_win)
            insights["post_win_streak_behavior"] = {
                "type": "behavioral_pattern",
                "key": "post_win_streak_behavior",
                "value": f"After 3+ consecutive wins: avg P&L ${avg:.2f}",
                "confidence": min(len(post_win) / 10, 0.9),
                "metadata": {"avg_pnl": avg, "sample_size": len(post_win)},
                "recommendation": "Guard against overconfidence after winning streaks." if avg < 0 else None,
            }

        if post_loss:
            avg = np.mean(post_loss)
            insights["post_loss_streak_behavior"] = {
                "type": "behavioral_pattern",
                "key": "post_loss_streak_behavior",
                "value": f"After 3+ consecutive losses: avg P&L ${avg:.2f}",
                "confidence": min(len(post_loss) / 10, 0.9),
                "metadata": {"avg_pnl": avg, "sample_size": len(post_loss)},
                "recommendation": "Take a break after losing streaks to reset mentally." if avg < 0 else None,
            }

        return insights

    # ------------------------------------------------------------------ #
    #  Revenge trading detection                                           #
    # ------------------------------------------------------------------ #

    def _detect_revenge_trading(self) -> dict:
        insights = {}
        if len(self.trades) < 10:
            return insights

        sorted_trades = sorted(self.trades, key=lambda t: t.entry_time)
        revenge_candidates = []

        for i in range(1, len(sorted_trades)):
            prev = sorted_trades[i - 1]
            curr = sorted_trades[i]
            if prev.pnl is None or prev.pnl >= 0 or curr.pnl is None:
                continue

            score = 0
            time_gap = None
            if prev.exit_time:
                time_gap = (curr.entry_time - prev.exit_time).total_seconds() / 60
                if time_gap < 30:
                    score += 3

            if prev.size and curr.size and curr.size > prev.size * 1.5:
                score += 2

            if curr.symbol == prev.symbol:
                score += 2

            if curr.emotions and any(
                e in curr.emotions.lower()
                for e in ("frustrated", "angry", "impulsive", "anxious")
            ):
                score += 3

            if score >= 5:
                revenge_candidates.append(
                    {"trade_id": curr.id, "score": score, "pnl": curr.pnl, "time_gap_minutes": time_gap}
                )

        if revenge_candidates:
            pnls = [r["pnl"] for r in revenge_candidates]
            insights["revenge_trading_detected"] = {
                "type": "behavioral_pattern",
                "key": "revenge_trading_detected",
                "value": f"⚠️ {len(revenge_candidates)} potential revenge trade(s) detected",
                "confidence": 0.85,
                "priority": "high",
                "metadata": {
                    "count": len(revenge_candidates),
                    "avg_pnl": float(np.mean(pnls)),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls) * 100,
                    "trades": revenge_candidates,
                },
                "recommendation": (
                    "Enforce a 1-hour cooldown after any losing trade. "
                    "Write in your journal before re-entering the market."
                ),
            }

        return insights

    # ------------------------------------------------------------------ #
    #  Market conditions                                                   #
    # ------------------------------------------------------------------ #

    def _analyze_market_conditions(self) -> dict:
        insights = {}
        if len(self.trades) < 15:
            return insights

        rows = [
            {"date": t.entry_time.date(), "pnl": t.pnl or 0.0, "direction": 1 if t.direction == "BUY" else -1}
            for t in self.trades if t.pnl is not None
        ]
        if len(rows) < 15:
            return insights

        df = pd.DataFrame(rows)
        volatility = df.groupby("date")["pnl"].sum().std()
        direction_consistency = df.groupby("direction")["pnl"].mean().abs().mean()

        insights["market_adaptation"] = {
            "type": "market_analysis",
            "key": "market_adaptation",
            "value": f"Daily P&L volatility: ${volatility:.2f}",
            "confidence": 0.7,
            "metadata": {
                "daily_volatility": float(volatility),
                "directional_consistency": float(direction_consistency),
            },
            "interpretation": "High" if volatility > 50 else "Moderate" if volatility > 20 else "Low",
        }
        return insights

    # ------------------------------------------------------------------ #
    #  Trade duration                                                      #
    # ------------------------------------------------------------------ #

    def _analyze_trade_duration(self) -> dict:
        insights = {}
        rows = [
            {
                "duration": (t.exit_time - t.entry_time).total_seconds() / 3600,
                "pnl": t.pnl,
                "is_win": t.pnl > 0,
            }
            for t in self.trades
            if t.exit_time and t.entry_time and t.pnl is not None
        ]
        if len(rows) < 10:
            return insights

        df = pd.DataFrame(rows)
        df["bucket"] = pd.cut(
            df["duration"],
            bins=[0, 1, 4, 24, 168, float("inf")],
            labels=["<1h", "1-4h", "4-24h", "1-7d", ">7d"],
        )
        stats = df.groupby("bucket", observed=True).agg(count=("pnl", "count"), avg_pnl=("pnl", "mean"), win_rate=("is_win", "mean"))

        best = stats[stats["count"] >= 3]["avg_pnl"].idxmax() if (stats["count"] >= 3).any() else None
        if best is not None:
            row = stats.loc[best]
            insights["optimal_trade_duration"] = {
                "type": "time_pattern",
                "key": "optimal_trade_duration",
                "value": f"Best holding period: {best} (avg P&L ${row['avg_pnl']:.2f}, {row['win_rate']*100:.1f}% WR)",
                "confidence": 0.75,
                "metadata": {
                    "best_duration": str(best),
                    "avg_pnl": float(row["avg_pnl"]),
                    "win_rate": float(row["win_rate"]) * 100,
                    "sample_size": int(row["count"]),
                },
                "recommendation": f"Focus on trades in the {best} timeframe.",
            }
        return insights

    # ------------------------------------------------------------------ #
    #  Entry / exit quality                                                #
    # ------------------------------------------------------------------ #

    def _analyze_entry_exit_quality(self) -> dict:
        insights = {}
        closed = [t for t in self.trades if t.pnl is not None and t.pnl_percent is not None]
        if len(closed) < 10:
            return insights

        # Use the user's own median as the benchmark — adapts to their style
        pct_values = [t.pnl_percent for t in closed]
        user_median = np.median(pct_values)
        threshold_early = user_median * 0.3
        threshold_optimal = user_median

        counts = {"early": 0, "optimal": 0, "stop_out": 0}
        for t in closed:
            p = t.pnl_percent
            if p < -abs(user_median) * 0.5:
                counts["stop_out"] += 1
            elif 0 < p < threshold_early:
                counts["early"] += 1
            else:
                counts["optimal"] += 1

        total = sum(counts.values())
        if total < 10:
            return insights

        optimal_rate = counts["optimal"] / total * 100
        early_rate = counts["early"] / total * 100

        insights["exit_quality_analysis"] = {
            "type": "behavioral_pattern",
            "key": "exit_quality_analysis",
            "value": f"Optimal exits: {optimal_rate:.1f}% | Early exits: {early_rate:.1f}%",
            "confidence": 0.7,
            "metadata": counts,
            "recommendation": "Practice letting winners run to your full target." if early_rate > 30 else None,
        }
        return insights

    # ------------------------------------------------------------------ #
    #  Psychological biases                                                #
    # ------------------------------------------------------------------ #

    def _analyze_psychological_biases(self) -> dict:
        insights = {}
        if len(self.trades) < 20:
            return insights

        # Loss aversion: holding losers longer than winners
        win_durations, loss_durations = [], []
        for t in self.trades:
            if t.exit_time and t.entry_time and t.pnl is not None:
                dur = (t.exit_time - t.entry_time).total_seconds() / 3600
                (win_durations if t.pnl > 0 else loss_durations).append(dur)

        if win_durations and loss_durations:
            avg_win_dur = np.mean(win_durations)
            avg_loss_dur = np.mean(loss_durations)
            if avg_loss_dur > avg_win_dur * 1.5:
                ratio = avg_loss_dur / avg_win_dur
                insights["loss_aversion_detected"] = {
                    "type": "psychological_bias",
                    "key": "loss_aversion_detected",
                    "value": f"⚠️ You hold losing trades {ratio:.1f}× longer than winning trades",
                    "confidence": 0.8,
                    "priority": "high",
                    "metadata": {"avg_win_duration_h": avg_win_dur, "avg_loss_duration_h": avg_loss_dur, "ratio": ratio},
                    "recommendation": "Set hard stop-losses before entering every trade and honour them unconditionally.",
                }

        # Recency bias
        recent = sorted(self.trades, key=lambda t: t.entry_time, reverse=True)[:10]
        recent_pnl = sum(t.pnl for t in recent if t.pnl is not None)
        overall_pnl = sum(t.pnl for t in self.trades if t.pnl is not None)
        if recent_pnl > 0 and overall_pnl < 0:
            insights["recency_bias_warning"] = {
                "type": "psychological_bias",
                "key": "recency_bias_warning",
                "value": "Recent wins may be masking a negative overall P&L",
                "confidence": 0.7,
                "metadata": {"recent_pnl": recent_pnl, "overall_pnl": overall_pnl},
                "recommendation": "Review your full trade history, not just the most recent trades.",
            }

        # Confirmation bias: repeatedly trading the same losing setup
        setup_stats: Dict[str, dict] = defaultdict(lambda: {"count": 0, "losses": 0})
        for t in self.trades:
            if t.strategy and t.pnl is not None:
                key = f"{t.symbol}_{t.direction}_{t.strategy}"
                setup_stats[key]["count"] += 1
                if t.pnl < 0:
                    setup_stats[key]["losses"] += 1

        bad = [
            (k, s["losses"] / s["count"], s["count"])
            for k, s in setup_stats.items()
            if s["count"] >= 5 and s["losses"] / s["count"] > 0.7
        ]
        if bad:
            worst, loss_rate, count = max(bad, key=lambda x: x[2])
            parts = worst.split("_")
            readable = parts[2] if len(parts) >= 3 else worst
            insights["confirmation_bias_detected"] = {
                "type": "psychological_bias",
                "key": "confirmation_bias_detected",
                "value": f"⚠️ Repeating losing setup '{readable}' ({loss_rate*100:.0f}% loss rate over {count} trades)",
                "confidence": 0.85,
                "priority": "high",
                "metadata": {"setup": worst, "loss_rate": loss_rate, "trade_count": count},
                "recommendation": "Stop trading this setup. Backtest and rethink the entry criteria.",
            }

        return insights

    # ------------------------------------------------------------------ #
    #  Advanced metrics                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_advanced_metrics(self) -> dict:
        insights = {}
        if len(self.trades) < 20:
            return insights

        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        if not pnls:
            return insights

        avg = np.mean(pnls)
        std = np.std(pnls)
        sharpe = avg / std if std > 0 else 0.0

        cum = np.cumsum(pnls)
        max_drawdown = float(np.max(np.maximum.accumulate(cum) - cum))

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        insights["advanced_metrics"] = {
            "type": "performance_metrics",
            "key": "advanced_metrics",
            "value": (
                f"Sharpe: {sharpe:.2f} | Profit Factor: {profit_factor:.2f} | "
                f"Expectancy: ${expectancy:.2f}/trade | Max Drawdown: ${max_drawdown:.2f}"
            ),
            "confidence": 0.9,
            "metadata": {
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "win_rate_pct": win_rate * 100,
            },
            "interpretation": self._interpret_advanced_metrics(sharpe, profit_factor, expectancy),
        }
        return insights

    def _interpret_advanced_metrics(self, sharpe: float, profit_factor: float, expectancy: float) -> str:
        parts = []
        parts.append(
            "Excellent risk-adjusted returns" if sharpe > 1.5
            else "Good risk-adjusted returns" if sharpe > 0.5
            else "Poor risk-adjusted returns"
        )
        parts.append(
            "Strong profit factor" if profit_factor > 2
            else "Adequate profit factor" if profit_factor > 1.5
            else "Weak profit factor — review losing trades"
        )
        parts.append(
            f"Positive expectancy (${expectancy:.2f} per trade)" if expectancy > 0
            else f"Negative expectancy (${expectancy:.2f} per trade) — system is unprofitable"
        )
        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    #  Behavioral pattern clustering (DBSCAN)                             #
    # ------------------------------------------------------------------ #

    def _detect_behavioral_patterns(self) -> dict:
        """
        Cluster trades by (hour, size, direction) to find repeating setups.
        Replaces the previous empty stub.
        """
        insights = {}
        if len(self.trades) < 15:
            return insights

        try:
            from sklearn.cluster import DBSCAN

            rows = []
            trade_refs = []
            for t in self.trades:
                if t.pnl is None or t.entry_time is None:
                    continue
                rows.append([
                    t.entry_time.hour,
                    t.entry_time.weekday(),
                    1 if t.direction == "BUY" else 0,
                    float(t.size or 0),
                ])
                trade_refs.append(t)

            if len(rows) < 15:
                return insights

            X = StandardScaler().fit_transform(np.array(rows))
            labels = DBSCAN(eps=0.6, min_samples=4).fit_predict(X)

            cluster_stats: Dict[int, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
            for label, trade in zip(labels, trade_refs):
                if label == -1:
                    continue
                s = cluster_stats[label]
                s["count"] += 1
                s["pnl"] += trade.pnl
                s["wins"] += int(trade.pnl > 0)

            for label, s in cluster_stats.items():
                if s["count"] < 4:
                    continue
                win_rate = s["wins"] / s["count"] * 100
                avg_pnl = s["pnl"] / s["count"]
                insights[f"behavioral_cluster_{label}"] = {
                    "type": "behavioral_pattern",
                    "key": f"behavioral_cluster_{label}",
                    "value": (
                        f"Trade cluster #{label}: {s['count']} trades, "
                        f"{win_rate:.1f}% WR, avg P&L ${avg_pnl:.2f}"
                    ),
                    "confidence": min(s["count"] / 20, 0.85),
                    "metadata": {"cluster_id": int(label), "count": int(s["count"]), "win_rate": float(win_rate), "avg_pnl": float(avg_pnl)},
                    "recommendation": (
                        f"This recurring setup has {win_rate:.0f}% win rate — "
                        + ("lean into it." if win_rate > 60 else "review or avoid it.")
                    ),
                }
        except Exception as exc:
            logger.warning("Behavioral clustering failed: %s", exc)

        return insights

    # ------------------------------------------------------------------ #
    #  Time patterns                                                       #
    # ------------------------------------------------------------------ #

    def _analyze_time_patterns(self) -> dict:
        insights = {}
        sessions = {
            "asian_session": (0, 8),
            "london_session": (8, 16),
            "ny_session": (13, 21),
            "london_ny_overlap": (13, 16),
        }
        stats: Dict[str, dict] = {k: {"count": 0, "pnl": 0.0, "wins": 0} for k in sessions}

        for t in self.trades:
            if t.entry_time is None or t.pnl is None:
                continue
            h = t.entry_time.hour
            for slot, (start, end) in sessions.items():
                if start <= h < end:
                    stats[slot]["count"] += 1
                    stats[slot]["pnl"] += t.pnl
                    stats[slot]["wins"] += int(t.pnl > 0)

        best_slot, best_wr = None, 0.0
        for slot, s in stats.items():
            if s["count"] < 3:
                continue
            wr = s["wins"] / s["count"] * 100
            avg = s["pnl"] / s["count"]
            if wr > best_wr:
                best_wr, best_slot = wr, slot
            insights[f"session_{slot}"] = {
                "type": "time_pattern",
                "key": f"session_{slot}",
                "value": f"{slot.replace('_', ' ').title()}: {wr:.1f}% WR, avg ${avg:.2f}",
                "confidence": min(s["count"] / 10, 0.9),
                "metadata": {"win_rate": wr, "avg_pnl": avg, "trade_count": s["count"]},
            }

        if best_slot:
            insights["optimal_trading_session"] = {
                "type": "time_pattern",
                "key": "optimal_trading_session",
                "value": f"Best session: {best_slot.replace('_', ' ').title()} ({best_wr:.1f}% WR)",
                "confidence": 0.85,
                "recommendation": f"Prioritise trades during the {best_slot.replace('_', ' ')}.",
            }
        return insights

    # ------------------------------------------------------------------ #
    #  Symbol performance                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_symbol_performance(self) -> dict:
        insights = {}
        sym_stats: Dict[str, dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0, "pnl_list": []})

        for t in self.trades:
            if t.pnl is None:
                continue
            s = sym_stats[t.symbol]
            s["count"] += 1
            s["pnl"] += t.pnl
            s["wins"] += int(t.pnl > 0)
            s["pnl_list"].append(t.pnl)

        for symbol, s in sym_stats.items():
            if s["count"] < 3:
                continue
            wr = s["wins"] / s["count"] * 100
            avg = s["pnl"] / s["count"]
            consistency = max(0.0, 1 - np.std(s["pnl_list"]) / (abs(avg) + 1))
            insights[f"symbol_{symbol}_performance"] = {
                "type": "symbol_analysis",
                "key": f"symbol_{symbol}_performance",
                "value": f"{symbol}: {wr:.1f}% WR, avg ${avg:.2f}, consistency {consistency:.2f}",
                "confidence": min(s["count"] / 15, 0.95),
                "metadata": {"win_rate": wr, "avg_pnl": avg, "consistency": consistency, "trade_count": s["count"]},
            }
        return insights

    # ------------------------------------------------------------------ #
    #  Strategy performance                                                #
    # ------------------------------------------------------------------ #

    def _analyze_strategy_performance(self) -> dict:
        insights = {}
        strat_stats: Dict[str, dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})

        for t in self.trades:
            if not (t.strategy and t.pnl is not None):
                continue
            s = strat_stats[t.strategy]
            s["count"] += 1
            s["pnl"] += t.pnl
            s["wins"] += int(t.pnl > 0)

        for strat, s in strat_stats.items():
            if s["count"] < 3:
                continue
            wr = s["wins"] / s["count"] * 100
            avg = s["pnl"] / s["count"]
            insights[f"strategy_{strat}_performance"] = {
                "type": "strategy_analysis",
                "key": f"strategy_{strat}_performance",
                "value": f"{strat}: {wr:.1f}% WR, avg ${avg:.2f} ({s['count']} trades)",
                "confidence": min(s["count"] / 10, 0.9),
                "metadata": {"win_rate": wr, "avg_pnl": avg, "trade_count": s["count"]},
            }
        return insights

    # ------------------------------------------------------------------ #
    #  Risk management                                                     #
    # ------------------------------------------------------------------ #

    def _analyze_risk_management(self) -> dict:
        insights = {}
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        if len(pnls) < 5:
            return insights

        avg_win = float(np.mean([p for p in pnls if p > 0])) if any(p > 0 for p in pnls) else 0.0
        avg_loss = float(abs(np.mean([p for p in pnls if p < 0]))) if any(p < 0 for p in pnls) else 1.0
        rr = avg_win / avg_loss if avg_loss > 0 else 0.0

        insights["risk_reward_analysis"] = {
            "type": "risk_analysis",
            "key": "risk_reward_analysis",
            "value": f"Risk:Reward Ratio 1:{rr:.2f} | Max win ${max(pnls):.2f} | Max loss ${min(pnls):.2f}",
            "confidence": 0.85,
            "metadata": {"ratio": rr, "avg_win": avg_win, "avg_loss": avg_loss, "max_loss": min(pnls), "max_win": max(pnls)},
            "recommendation": "Aim for at least a 1:2 risk-reward ratio." if rr < 2 else None,
        }
        return insights

    # ------------------------------------------------------------------ #
    #  Predictive ML model (cached)                                       #
    # ------------------------------------------------------------------ #

    def _predict_trade_success(self) -> dict:
        insights = {}
        if len(self.trades) < 30:
            return insights

        model_path = os.path.join(self.MODEL_DIR, f"user_{self.user_id}_rf.pkl")
        meta_path = os.path.join(self.MODEL_DIR, f"user_{self.user_id}_meta.json")

        # Check if retraining is needed
        current_count = len(self.trades)
        retrain = True
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if current_count - meta.get("trained_on", 0) < 10:
                    retrain = False
            except Exception as exc:
                logger.warning("Could not read model meta: %s", exc)

        try:
            from sklearn.model_selection import train_test_split

            if retrain:
                features, labels = [], []
                for i, t in enumerate(self.trades):
                    if t.pnl is None:
                        continue
                    fv = self._extract_trade_features(t, i)
                    if fv:
                        features.append(fv)
                        labels.append(1 if t.pnl > 0 else 0)

                if len(features) < 30:
                    return insights

                X, y = np.array(features), np.array(labels)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)

                joblib.dump(model, model_path)
                with open(meta_path, "w") as f:
                    json.dump({"trained_on": current_count, "accuracy": accuracy}, f)

                self.prediction_model = model
            else:
                if os.path.exists(model_path):
                    self.prediction_model = joblib.load(model_path)
                with open(meta_path) as f:
                    meta = json.load(f)
                accuracy = meta.get("accuracy", 0)

            if self.prediction_model:
                feature_names = self._get_feature_names()
                importances = self.prediction_model.feature_importances_
                top = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
                insights["predictive_model"] = {
                    "type": "prediction",
                    "key": "predictive_model",
                    "value": f"ML prediction model: {accuracy*100:.1f}% accuracy on held-out trades",
                    "confidence": min(accuracy, 0.9),
                    "metadata": {
                        "accuracy": accuracy,
                        "sample_size": current_count,
                        "top_features": [(n, float(v)) for n, v in top],
                        "retrained": retrain,
                    },
                    "interpretation": f"Top success factors: {', '.join(n for n, _ in top)}",
                }
        except Exception as exc:
            logger.warning("ML model training failed: %s", exc)

        return insights

    def _extract_trade_features(self, trade: Trade, index: int) -> Optional[List[float]]:
        try:
            emotion_map = {
                "confident": 1.0, "disciplined": 0.85, "focused": 0.8,
                "neutral": 0.5, "anxious": 0.3, "greedy": 0.2, "fearful": 0.1,
            }
            recent = [t for t in self.trades[:index] if t.pnl is not None][-5:]
            recent_wr = sum(1 for t in recent if t.pnl > 0) / len(recent) if recent else 0.5

            sym_hist = [t for t in self.trades[:index] if t.symbol == trade.symbol and t.pnl is not None]
            sym_wr = sum(1 for t in sym_hist if t.pnl > 0) / len(sym_hist) if sym_hist else 0.5

            return [
                float(trade.entry_time.hour),
                float(trade.entry_time.weekday()),
                1.0 if trade.direction == "BUY" else 0.0,
                float(trade.size or 0),
                recent_wr,
                emotion_map.get(trade.emotions or "", 0.5),
                sym_wr,
            ]
        except Exception as exc:
            logger.debug("Feature extraction failed for trade %s: %s", trade.id, exc)
            return None

    def _get_feature_names(self) -> List[str]:
        return ["Hour of Day", "Day of Week", "Direction", "Position Size",
                "Recent Win Rate", "Emotional State", "Symbol Win Rate"]

    # ------------------------------------------------------------------ #
    #  New trade analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_new_trade(self, new_trade: Trade) -> dict:
        insights = {}
        similar = [
            t for t in self.trades
            if t.symbol == new_trade.symbol
            and t.direction == new_trade.direction
            and t.strategy == new_trade.strategy
            and t.pnl is not None
        ]
        if similar:
            wins = sum(1 for t in similar if t.pnl > 0)
            wr = wins / len(similar) * 100
            insights["trade_probability"] = {
                "type": "trade_analysis",
                "key": "trade_probability",
                "value": f"Historical success rate for this exact setup: {wr:.1f}%",
                "confidence": min(len(similar) / 10, 0.9),
                "metadata": {"win_rate": wr, "sample_size": len(similar)},
                "recommendation": (
                    "Strong setup — proceed with normal sizing." if wr > 65
                    else "Weak historical performance — reduce size or skip." if wr < 35
                    else "Mixed history — trade with caution."
                ),
            }

        # Predict with ML model if available
        if self.prediction_model:
            try:
                fv = self._extract_trade_features(new_trade, len(self.trades))
                if fv:
                    prob = self.prediction_model.predict_proba([fv])[0][1]
                    insights["ml_trade_prediction"] = {
                        "type": "prediction",
                        "key": "ml_trade_prediction",
                        "value": f"ML win probability for this trade: {prob*100:.1f}%",
                        "confidence": 0.75,
                        "metadata": {"win_probability": prob},
                        "recommendation": (
                            "ML model is bullish on this setup." if prob > 0.65
                            else "ML model flags this as a higher-risk trade." if prob < 0.4
                            else "ML model is neutral on this setup."
                        ),
                    }
            except Exception as exc:
                logger.warning("ML prediction for new trade failed: %s", exc)

        return insights

    # ------------------------------------------------------------------ #
    #  Recommendations (synthesises all other analyses)                   #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(self, learned: dict) -> dict:
        """
        Build ranked, actionable recommendations from the completed analyses.
        This replaces the previous empty stub.
        """
        recommendations = {}
        if len(self.trades) < 10:
            return {
                "more_data_needed": {
                    "type": "recommendation",
                    "key": "more_data_needed",
                    "value": "Keep journaling — personalised recommendations unlock after 10 trades.",
                    "confidence": 0.5,
                }
            }

        rank = 0

        def add(key, value, rec, confidence, meta=None):
            nonlocal rank
            rank += 1
            recommendations[key] = {
                "type": "recommendation",
                "key": key,
                "value": value,
                "recommendation": rec,
                "confidence": confidence,
                "rank": rank,
                "metadata": meta or {},
            }

        # Pull findings from already-computed analyses
        if "revenge_trading_detected" in learned:
            m = learned["revenge_trading_detected"]["metadata"]
            add("rec_revenge_trading",
                f"⚠️ Revenge trading detected ({m['count']} instance(s))",
                "Enforce a mandatory 1-hour cooldown after any losing trade.",
                0.9, m)

        if "loss_aversion_detected" in learned:
            add("rec_loss_aversion",
                "⚠️ You hold losers significantly longer than winners",
                "Pre-set your stop-loss before entering every trade and never widen it.",
                0.85)

        if "confirmation_bias_detected" in learned:
            m = learned["confirmation_bias_detected"]["metadata"]
            add("rec_confirmation_bias",
                f"⚠️ Repeating a losing setup ({m['loss_rate']*100:.0f}% loss rate)",
                "Stop trading this setup. Backtest alternatives before returning to it.",
                0.85, m)

        if "optimal_trading_session" in learned:
            m = learned.get("optimal_trading_session", {})
            add("rec_session",
                f"You perform best in: {m.get('value', 'a specific session')}",
                "Restrict active trading to your best-performing session.",
                0.8, m)

        if "optimal_trade_duration" in learned:
            m = learned["optimal_trade_duration"].get("metadata", {})
            add("rec_duration",
                f"Your optimal holding period is {m.get('best_duration', 'unknown')}",
                f"Aim to close trades within the {m.get('best_duration', 'optimal')} window.",
                0.75, m)

        if "risk_reward_analysis" in learned:
            meta = learned["risk_reward_analysis"]["metadata"]
            if meta.get("ratio", 0) < 2:
                add("rec_rr_ratio",
                    f"Risk:Reward ratio is 1:{meta['ratio']:.2f} (below target)",
                    "Review your take-profit targets. Aim for a minimum 1:2 ratio.",
                    0.8, meta)

        if "exit_quality_analysis" in learned:
            meta = learned["exit_quality_analysis"]["metadata"]
            if meta.get("early", 0) / max(sum(meta.values()), 1) > 0.3:
                add("rec_early_exits",
                    "You frequently exit trades before reaching full potential",
                    "Use trailing stops instead of manual exits to let winners run.",
                    0.75, meta)

        # General baseline if nothing else fired
        if not recommendations:
            add("rec_baseline_journaling",
                "Consistent journaling is your most important tool",
                "Record emotions, strategy, and notes on every trade for deeper AI insights.",
                0.6)

        return recommendations

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save_insights(self, insights: dict):
        for key, data in insights.items():
            try:
                existing = self.insights.get(key)
                if existing:
                    existing.insight_value = json.dumps(data)
                    existing.confidence_score = data.get("confidence", 0.5)
                    existing.data_points += 1
                    existing.last_updated = datetime.utcnow()
                else:
                    new_row = AIInsight(
                        user_id=self.user_id,
                        insight_type=data.get("type", "general"),
                        insight_key=key,
                        insight_value=json.dumps(data),
                        confidence_score=data.get("confidence", 0.5),
                        data_points=1,
                    )
                    db.session.add(new_row)
            except Exception as exc:
                logger.error("Failed to save insight '%s': %s", key, exc)
        try:
            db.session.commit()
        except Exception as exc:
            logger.error("DB commit failed in _save_insights: %s", exc)
            db.session.rollback()

    def _log_learning_session(self, insights: dict, mode: str = "full"):
        try:
            log = AILearningLog(
                user_id=self.user_id,
                event_type=f"ai_analysis_{mode}",
                event_data=json.dumps({
                    "trade_count": len(self.trades),
                    "timestamp": datetime.utcnow().isoformat(),
                    "mode": mode,
                    "insight_categories": list({i.get("type") for i in insights.values()}),
                }),
                learned_insights=json.dumps(list(insights.keys())),
            )
            db.session.add(log)
            db.session.commit()
        except Exception as exc:
            logger.error("Failed to log learning session: %s", exc)
            db.session.rollback()


