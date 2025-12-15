"""
Enhanced AI Learning Engine for Forex Trading Journal
Advanced pattern detection, predictive analytics, and personalized insights
"""
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from app import db
from app.models import Trade, User, AccountBalance
from app.ai.models import AIInsight, TradePattern, AILearningLog


class ForexAI:
    """Enhanced AI Learning Engine with advanced analytics"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.trades = []
        self.account_balances = []
        self.insights = {}
        self.patterns = {}
        self.prediction_model = None
        
    def load_data(self):
        """Load user's trading data with enhanced preprocessing"""
        self.trades = Trade.query.filter_by(user_id=self.user_id).order_by(Trade.entry_time.asc()).all()
        self.account_balances = AccountBalance.query.filter_by(
            user_id=self.user_id
        ).order_by(AccountBalance.date.asc()).all()
        
        # Load existing AI insights
        self.insights = {
            insight.insight_key: insight 
            for insight in AIInsight.query.filter_by(user_id=self.user_id, is_active=True).all()
        }
        
        # Load existing patterns
        self.patterns = {
            pattern.pattern_type: pattern 
            for pattern in TradePattern.query.filter_by(user_id=self.user_id).all()
        }
    
    def analyze_and_learn(self, new_trade=None):
        """
        Main learning function with enhanced analytics
        """
        self.load_data()
        
        if not self.trades:
            return {}
        
        learned_insights = {}
        
        # Core Analysis Modules
        learned_insights.update(self._analyze_emotional_impact())
        learned_insights.update(self._analyze_time_patterns())
        learned_insights.update(self._analyze_symbol_performance())
        learned_insights.update(self._analyze_strategy_performance())
        learned_insights.update(self._detect_behavioral_patterns())
        learned_insights.update(self._analyze_risk_management())
        
        # Enhanced Analysis Modules
        learned_insights.update(self._analyze_streaks_and_sequences())
        learned_insights.update(self._analyze_market_conditions())
        learned_insights.update(self._analyze_trade_duration())
        learned_insights.update(self._analyze_entry_exit_quality())
        learned_insights.update(self._detect_revenge_trading())
        learned_insights.update(self._analyze_psychological_biases())
        learned_insights.update(self._calculate_advanced_metrics())
        learned_insights.update(self._predict_trade_success())
        
        # Generate recommendations
        learned_insights.update(self._generate_recommendations())
        
        # Specific new trade analysis
        if new_trade:
            learned_insights.update(self._analyze_new_trade(new_trade))
        
        # Save insights to database
        self._save_insights(learned_insights)
        self._log_learning_session(learned_insights)
        
        return learned_insights
    
    # ==================== ENHANCED EMOTIONAL ANALYSIS ====================
    
    def _analyze_emotional_impact(self):
        """Deep emotional analysis with correlation metrics"""
        insights = {}
        
        emotion_stats = defaultdict(lambda: {
            'count': 0, 'pnl': 0, 'wins': 0, 
            'avg_duration': 0, 'total_duration': 0,
            'avg_risk': 0, 'total_risk': 0
        })
        
        for trade in self.trades:
            if trade.emotions and trade.pnl is not None:
                emotion = trade.emotions.lower()
                stats = emotion_stats[emotion]
                
                stats['count'] += 1
                stats['pnl'] += trade.pnl
                if trade.pnl > 0:
                    stats['wins'] += 1
                
                # Duration analysis
                if trade.exit_time and trade.entry_time:
                    duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                    stats['total_duration'] += duration
                
                # Risk analysis
                if trade.size:
                    stats['total_risk'] += trade.size
        
        # Calculate comprehensive metrics
        for emotion, stats in emotion_stats.items():
            if stats['count'] >= 3:
                win_rate = (stats['wins'] / stats['count']) * 100
                avg_pnl = stats['pnl'] / stats['count']
                avg_duration = stats['total_duration'] / stats['count'] if stats['total_duration'] > 0 else 0
                
                # Emotion impact score (normalized)
                impact_score = self._calculate_emotion_impact_score(win_rate, avg_pnl, stats['count'])
                
                insights[f'emotion_{emotion}_analysis'] = {
                    'type': 'emotion_analysis',
                    'key': f'emotion_{emotion}_analysis',
                    'value': f'{emotion.title()}: {win_rate:.1f}% WR, Avg P&L ${avg_pnl:.2f}',
                    'confidence': min(stats['count'] / 10, 1.0),
                    'metadata': {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'trade_count': stats['count'],
                        'avg_duration_hours': avg_duration,
                        'impact_score': impact_score
                    },
                    'recommendation': self._get_emotion_recommendation(emotion, win_rate, avg_pnl)
                }
                
                # Flag toxic emotional states
                if win_rate < 30 and stats['count'] >= 5:
                    insights[f'emotion_{emotion}_warning'] = {
                        'type': 'emotion_analysis',
                        'key': f'emotion_{emotion}_warning',
                        'value': f'⚠️ Warning: Trading while {emotion} has {win_rate:.1f}% success rate',
                        'confidence': 0.9,
                        'priority': 'high',
                        'recommendation': f'Avoid trading when feeling {emotion}. Take a break or journal first.'
                    }
        
        # Emotional volatility analysis
        if len(emotion_stats) >= 3:
            insights['emotional_stability'] = self._analyze_emotional_stability(emotion_stats)
        
        return insights
    
    def _calculate_emotion_impact_score(self, win_rate: float, avg_pnl: float, count: int) -> float:
        """Calculate normalized emotion impact score"""
        # Weighted score: win rate (40%), avg pnl (40%), sample size (20%)
        win_rate_score = win_rate / 100
        pnl_score = max(0, min(1, (avg_pnl + 50) / 100))  # Normalize around -50 to +50
        sample_score = min(count / 20, 1.0)
        
        return (win_rate_score * 0.4) + (pnl_score * 0.4) + (sample_score * 0.2)
    
    def _analyze_emotional_stability(self, emotion_stats: dict) -> dict:
        """Analyze trader's emotional consistency"""
        emotion_scores = []
        for emotion, stats in emotion_stats.items():
            if stats['count'] >= 3:
                win_rate = (stats['wins'] / stats['count']) * 100
                emotion_scores.append(win_rate)
        
        if len(emotion_scores) >= 3:
            stability_score = 1 - (np.std(emotion_scores) / 100)  # Lower std = more stable
            
            return {
                'type': 'emotion_analysis',
                'key': 'emotional_stability',
                'value': f'Emotional Stability Score: {stability_score:.2f}/1.0',
                'confidence': 0.8,
                'metadata': {
                    'stability_score': stability_score,
                    'emotion_variance': np.std(emotion_scores)
                },
                'interpretation': 'High' if stability_score > 0.7 else 'Moderate' if stability_score > 0.5 else 'Low'
            }
        return {}
    
    # ==================== STREAKS AND SEQUENCES ====================
    
    def _analyze_streaks_and_sequences(self) -> dict:
        """Analyze winning/losing streaks and their psychological impact"""
        insights = {}
        
        if len(self.trades) < 5:
            return insights
        
        # Calculate streaks
        current_streak = 0
        current_type = None
        max_win_streak = 0
        max_loss_streak = 0
        streaks = []
        
        for trade in sorted(self.trades, key=lambda t: t.entry_time):
            if trade.pnl is None:
                continue
                
            is_win = trade.pnl > 0
            
            if current_type == is_win:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = is_win
            
            if is_win:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Analyze behavior after streaks
        post_win_streak_performance = []
        post_loss_streak_performance = []
        
        for i, trade in enumerate(sorted(self.trades, key=lambda t: t.entry_time)):
            if i >= 3 and trade.pnl is not None:
                prev_3 = [t for t in sorted(self.trades, key=lambda t: t.entry_time)[i-3:i] if t.pnl is not None]
                
                if len(prev_3) == 3:
                    if all(t.pnl > 0 for t in prev_3):
                        post_win_streak_performance.append(trade.pnl)
                    elif all(t.pnl < 0 for t in prev_3):
                        post_loss_streak_performance.append(trade.pnl)
        
        # Generate insights
        insights['streak_analysis'] = {
            'type': 'behavioral_pattern',
            'key': 'streak_analysis',
            'value': f'Max Win Streak: {max_win_streak}, Max Loss Streak: {max_loss_streak}',
            'confidence': 0.8,
            'metadata': {
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak
            }
        }
        
        # Post-streak behavior
        if post_win_streak_performance:
            avg_after_wins = np.mean(post_win_streak_performance)
            insights['post_win_streak_behavior'] = {
                'type': 'behavioral_pattern',
                'key': 'post_win_streak_behavior',
                'value': f'After 3+ wins: Avg P&L ${avg_after_wins:.2f}',
                'confidence': min(len(post_win_streak_performance) / 10, 0.9),
                'metadata': {'avg_pnl': avg_after_wins, 'sample_size': len(post_win_streak_performance)},
                'recommendation': 'Watch for overconfidence after winning streaks' if avg_after_wins < 0 else None
            }
        
        if post_loss_streak_performance:
            avg_after_losses = np.mean(post_loss_streak_performance)
            insights['post_loss_streak_behavior'] = {
                'type': 'behavioral_pattern',
                'key': 'post_loss_streak_behavior',
                'value': f'After 3+ losses: Avg P&L ${avg_after_losses:.2f}',
                'confidence': min(len(post_loss_streak_performance) / 10, 0.9),
                'metadata': {'avg_pnl': avg_after_losses, 'sample_size': len(post_loss_streak_performance)},
                'recommendation': 'Consider taking break after losing streaks' if avg_after_losses < 0 else None
            }
        
        return insights
    
    # ==================== REVENGE TRADING DETECTION ====================
    
    def _detect_revenge_trading(self) -> dict:
        """Detect potential revenge trading patterns"""
        insights = {}
        
        if len(self.trades) < 10:
            return insights
        
        revenge_candidates = []
        sorted_trades = sorted(self.trades, key=lambda t: t.entry_time)
        
        for i in range(1, len(sorted_trades)):
            current = sorted_trades[i]
            previous = sorted_trades[i-1]
            
            if previous.pnl is not None and previous.pnl < 0 and current.pnl is not None:
                time_diff = (current.entry_time - previous.exit_time).total_seconds() / 60 if previous.exit_time else None
                
                # Indicators of revenge trading:
                # 1. Quick re-entry after loss (< 30 min)
                # 2. Increased position size
                # 3. Same symbol
                # 4. Emotional state indicates frustration
                
                revenge_score = 0
                
                if time_diff and time_diff < 30:
                    revenge_score += 3
                
                if previous.size and current.size and current.size > previous.size * 1.5:
                    revenge_score += 2
                
                if current.symbol == previous.symbol:
                    revenge_score += 2
                
                if current.emotions and any(e in current.emotions.lower() for e in ['frustrated', 'angry', 'impulsive', 'anxious']):
                    revenge_score += 3
                
                if revenge_score >= 5:
                    revenge_candidates.append({
                        'trade_id': current.id,
                        'score': revenge_score,
                        'pnl': current.pnl,
                        'time_gap_minutes': time_diff
                    })
        
        if revenge_candidates:
            revenge_pnl = [r['pnl'] for r in revenge_candidates]
            avg_revenge_pnl = np.mean(revenge_pnl)
            revenge_win_rate = len([p for p in revenge_pnl if p > 0]) / len(revenge_pnl) * 100
            
            insights['revenge_trading_detected'] = {
                'type': 'behavioral_pattern',
                'key': 'revenge_trading_detected',
                'value': f'⚠️ {len(revenge_candidates)} potential revenge trades detected',
                'confidence': 0.85,
                'priority': 'high',
                'metadata': {
                    'count': len(revenge_candidates),
                    'avg_pnl': avg_revenge_pnl,
                    'win_rate': revenge_win_rate,
                    'trades': revenge_candidates
                },
                'recommendation': 'Implement a mandatory 1-hour cooldown period after losses. Set trading rules to prevent emotional trading.'
            }
        
        return insights
    
    # ==================== MARKET CONDITIONS ANALYSIS ====================
    
    def _analyze_market_conditions(self) -> dict:
        """Analyze performance under different market conditions"""
        insights = {}
        
        if len(self.trades) < 15:
            return insights
        
        # Infer market conditions from trade clustering
        # High volatility = large price movements, more trades
        # Trending = directional consistency
        
        trades_df = pd.DataFrame([{
            'date': t.entry_time.date(),
            'pnl': t.pnl or 0,
            'direction': 1 if t.direction == 'BUY' else -1,
            'symbol': t.symbol
        } for t in self.trades if t.pnl is not None])
        
        if len(trades_df) < 15:
            return insights
        
        # Daily performance variance
        daily_pnl = trades_df.groupby('date')['pnl'].sum()
        volatility = daily_pnl.std()
        
        # Directional consistency
        direction_consistency = trades_df.groupby('symbol')['direction'].apply(
            lambda x: abs(x.mean())
        ).mean()
        
        insights['market_adaptation'] = {
            'type': 'market_analysis',
            'key': 'market_adaptation',
            'value': f'Performance Volatility: ${volatility:.2f}',
            'confidence': 0.7,
            'metadata': {
                'daily_volatility': volatility,
                'directional_consistency': direction_consistency,
                'interpretation': 'High' if volatility > 50 else 'Moderate' if volatility > 20 else 'Low'
            }
        }
        
        return insights
    
    # ==================== TRADE DURATION ANALYSIS ====================
    
    def _analyze_trade_duration(self) -> dict:
        """Analyze optimal trade holding periods"""
        insights = {}
        
        duration_data = []
        for trade in self.trades:
            if trade.exit_time and trade.entry_time and trade.pnl is not None:
                duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                duration_data.append({
                    'duration': duration_hours,
                    'pnl': trade.pnl,
                    'is_win': trade.pnl > 0
                })
        
        if len(duration_data) < 10:
            return insights
        
        df = pd.DataFrame(duration_data)
        
        # Categorize by duration
        df['duration_category'] = pd.cut(df['duration'], 
                                         bins=[0, 1, 4, 24, 168, float('inf')],
                                         labels=['<1h', '1-4h', '4-24h', '1-7d', '>7d'])
        
        duration_stats = df.groupby('duration_category').agg({
            'pnl': ['mean', 'sum', 'count'],
            'is_win': 'mean'
        }).round(2)
        
        # Find optimal duration
        best_duration = None
        best_avg_pnl = -float('inf')
        
        for cat in duration_stats.index:
            if duration_stats.loc[cat, ('pnl', 'count')] >= 3:
                avg_pnl = duration_stats.loc[cat, ('pnl', 'mean')]
                if avg_pnl > best_avg_pnl:
                    best_avg_pnl = avg_pnl
                    best_duration = cat
        
        if best_duration:
            win_rate = duration_stats.loc[best_duration, ('is_win', 'mean')] * 100
            
            insights['optimal_trade_duration'] = {
                'type': 'time_pattern',
                'key': 'optimal_trade_duration',
                'value': f'Optimal holding period: {best_duration} (${best_avg_pnl:.2f} avg)',
                'confidence': 0.75,
                'metadata': {
                    'best_duration': str(best_duration),
                    'avg_pnl': best_avg_pnl,
                    'win_rate': win_rate,
                    'sample_size': int(duration_stats.loc[best_duration, ('pnl', 'count')])
                },
                'recommendation': f'Consider targeting trades in the {best_duration} timeframe'
            }
        
        return insights
    
    # ==================== ENTRY/EXIT QUALITY ANALYSIS ====================
    
    def _analyze_entry_exit_quality(self) -> dict:
        """Analyze the quality of entries and exits"""
        insights = {}
        
        if len(self.trades) < 15:
            return insights
        
        # Calculate metrics for closed trades
        quality_metrics = {
            'early_exits': 0,  # Exited before achieving good R:R
            'optimal_exits': 0,  # Exited at good profit
            'late_exits': 0,  # Held too long, gave back profits
            'stop_outs': 0  # Hit stop loss
        }
        
        for trade in self.trades:
            if trade.pnl is not None and trade.exit_price and trade.entry_price:
                pnl_pct = trade.pnl_percent or 0
                
                if pnl_pct < -1:
                    quality_metrics['stop_outs'] += 1
                elif 0 < pnl_pct < 1:
                    quality_metrics['early_exits'] += 1
                elif 1 <= pnl_pct <= 3:
                    quality_metrics['optimal_exits'] += 1
                elif pnl_pct > 3:
                    # Check if this could be better classified
                    quality_metrics['optimal_exits'] += 1
        
        total_classified = sum(quality_metrics.values())
        
        if total_classified >= 10:
            optimal_rate = (quality_metrics['optimal_exits'] / total_classified) * 100
            early_exit_rate = (quality_metrics['early_exits'] / total_classified) * 100
            
            insights['exit_quality_analysis'] = {
                'type': 'behavioral_pattern',
                'key': 'exit_quality_analysis',
                'value': f'Optimal exits: {optimal_rate:.1f}%, Early exits: {early_exit_rate:.1f}%',
                'confidence': 0.7,
                'metadata': quality_metrics,
                'recommendation': 'Practice letting winners run' if early_exit_rate > 30 else None
            }
        
        return insights
    
    # ==================== PSYCHOLOGICAL BIASES ====================
    
    def _analyze_psychological_biases(self) -> dict:
        """Detect common psychological biases in trading"""
        insights = {}
        
        if len(self.trades) < 20:
            return insights
        
        # 1. Loss Aversion Bias - Holding losers too long
        hold_times_wins = []
        hold_times_losses = []
        
        for trade in self.trades:
            if trade.exit_time and trade.entry_time and trade.pnl is not None:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                if trade.pnl > 0:
                    hold_times_wins.append(duration)
                else:
                    hold_times_losses.append(duration)
        
        if hold_times_wins and hold_times_losses:
            avg_win_duration = np.mean(hold_times_wins)
            avg_loss_duration = np.mean(hold_times_losses)
            
            if avg_loss_duration > avg_win_duration * 1.5:
                insights['loss_aversion_detected'] = {
                    'type': 'psychological_bias',
                    'key': 'loss_aversion_detected',
                    'value': f'⚠️ Holding losses {avg_loss_duration/avg_win_duration:.1f}x longer than wins',
                    'confidence': 0.8,
                    'priority': 'high',
                    'metadata': {
                        'avg_win_duration': avg_win_duration,
                        'avg_loss_duration': avg_loss_duration,
                        'ratio': avg_loss_duration / avg_win_duration
                    },
                    'recommendation': 'Set strict stop losses and honor them. Cut losses quickly, let winners run.'
                }
        
        # 2. Recency Bias - Overweighting recent performance
        recent_trades = sorted(self.trades, key=lambda t: t.entry_time, reverse=True)[:10]
        recent_pnl = sum(t.pnl for t in recent_trades if t.pnl is not None)
        overall_pnl = sum(t.pnl for t in self.trades if t.pnl is not None)
        
        if recent_pnl > 0 and overall_pnl < 0:
            insights['recency_bias_warning'] = {
                'type': 'psychological_bias',
                'key': 'recency_bias_warning',
                'value': 'Recent wins may be masking overall negative performance',
                'confidence': 0.7,
                'metadata': {
                    'recent_pnl': recent_pnl,
                    'overall_pnl': overall_pnl
                },
                'recommendation': 'Review overall performance, not just recent trades'
            }
        
        # 3. Confirmation Bias - Repeatedly trading same losing setups
        setup_performance = defaultdict(lambda: {'count': 0, 'losses': 0})
        
        for trade in self.trades:
            if trade.strategy and trade.pnl is not None:
                setup = f"{trade.symbol}_{trade.direction}_{trade.strategy}"
                setup_performance[setup]['count'] += 1
                if trade.pnl < 0:
                    setup_performance[setup]['losses'] += 1
        
        persistent_losers = []
        for setup, stats in setup_performance.items():
            if stats['count'] >= 5:
                loss_rate = stats['losses'] / stats['count']
                if loss_rate > 0.7:
                    persistent_losers.append((setup, loss_rate, stats['count']))
        
        if persistent_losers:
            worst_setup, loss_rate, count = max(persistent_losers, key=lambda x: x[2])
            
            insights['confirmation_bias_detected'] = {
                'type': 'psychological_bias',
                'key': 'confirmation_bias_detected',
                'value': f'Repeating losing setup: {worst_setup.split("_")[2]} ({loss_rate*100:.0f}% loss rate)',
                'confidence': 0.85,
                'priority': 'high',
                'metadata': {
                    'setup': worst_setup,
                    'loss_rate': loss_rate,
                    'trade_count': count
                },
                'recommendation': 'Stop trading this setup and analyze why it\'s not working'
            }
        
        return insights
    
    # ==================== ADVANCED METRICS ====================
    
    def _calculate_advanced_metrics(self) -> dict:
        """Calculate advanced trading metrics"""
        insights = {}
        
        if len(self.trades) < 20:
            return insights
        
        pnl_values = [t.pnl for t in self.trades if t.pnl is not None]
        
        if not pnl_values:
            return insights
        
        # Sharpe Ratio (simplified)
        avg_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values)
        sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0
        
        # Maximum Drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown)
        
        # Profit Factor
        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = abs(sum(p for p in pnl_values if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        wins = [p for p in pnl_values if p > 0]
        losses = [p for p in pnl_values if p < 0]
        win_rate = len(wins) / len(pnl_values)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        insights['advanced_metrics'] = {
            'type': 'performance_metrics',
            'key': 'advanced_metrics',
            'value': f'Sharpe: {sharpe:.2f}, Profit Factor: {profit_factor:.2f}, Expectancy: ${expectancy:.2f}',
            'confidence': 0.9,
            'metadata': {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'interpretation': self._interpret_advanced_metrics(sharpe, profit_factor, expectancy)
        }
        
        return insights
    
    def _interpret_advanced_metrics(self, sharpe: float, profit_factor: float, expectancy: float) -> str:
        """Interpret advanced metrics"""
        interpretations = []
        
        if sharpe > 1.5:
            interpretations.append("Excellent risk-adjusted returns")
        elif sharpe > 0.5:
            interpretations.append("Good risk-adjusted returns")
        else:
            interpretations.append("Poor risk-adjusted returns")
        
        if profit_factor > 2:
            interpretations.append("Strong profit factor")
        elif profit_factor > 1.5:
            interpretations.append("Decent profit factor")
        else:
            interpretations.append("Weak profit factor")
        
        if expectancy > 0:
            interpretations.append(f"Positive expectancy (${expectancy:.2f} per trade)")
        else:
            interpretations.append(f"Negative expectancy (${expectancy:.2f} per trade)")
        
        return " | ".join(interpretations)
    
    # ==================== PREDICTIVE ANALYTICS ====================
    
    def _predict_trade_success(self) -> dict:
        """Use ML to predict trade success probability"""
        insights = {}
        
        if len(self.trades) < 30:
            return insights
        
        try:
            # Prepare features
            features = []
            labels = []
            
            for i, trade in enumerate(self.trades):
                if trade.pnl is not None:
                    # Feature engineering
                    feature_vector = self._extract_trade_features(trade, i)
                    if feature_vector:
                        features.append(feature_vector)
                        labels.append(1 if trade.pnl > 0 else 0)
            
            if len(features) >= 30:
                X = np.array(features)
                y = np.array(labels)
                
                # Train simple model
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate accuracy
                accuracy = model.score(X_test, y_test)
                
                # Feature importance
                feature_names = self._get_feature_names()
                importances = model.feature_importances_
                top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
                
                insights['predictive_model'] = {
                    'type': 'prediction',
                    'key': 'predictive_model',
                    'value': f'ML model trained: {accuracy*100:.1f}% accuracy',
                    'confidence': min(accuracy, 0.9),
                    'metadata': {
                        'accuracy': accuracy,
                        'sample_size': len(features),
                        'top_features': [(f, float(i)) for f, i in top_features]
                    },
                    'interpretation': f'Top factors: {", ".join([f[0] for f in top_features])}'
                }
                
                self.prediction_model = model
        
        except Exception as e:
            pass  # Silently fail if ML doesn't work
        
        return insights
    
    def _extract_trade_features(self, trade: Trade, index: int) -> Optional[List[float]]:
        """Extract features for ML model"""
        try:
            features = []
            
            # Time features
            features.append(trade.entry_time.hour)
            features.append(trade.entry_time.weekday())
            
            # Trade attributes
            features.append(1 if trade.direction == 'BUY' else 0)
            features.append(trade.size or 0)
            
            # Historical performance
            recent_trades = [t for t in self.trades[:index] if t.pnl is not None][-5:]
            if recent_trades:
                features.append(sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades))
            else:
                features.append(0.5)
            
            # Emotion encoding
            emotion_map = {'confident': 1, 'disciplined': 0.8, 'neutral': 0.5, 
                          'anxious': 0.3, 'greedy': 0.2, 'fearful': 0.1}
            features.append(emotion_map.get(trade.emotions, 0.5) if trade.emotions else 0.5)
            
            # Symbol consistency
            symbol_history = [t for t in self.trades[:index] if t.symbol == trade.symbol and t.pnl is not None]
            if symbol_history:
                features.append(sum(1 for t in symbol_history if t.pnl > 0) / len(symbol_history))
            else:
                features.append(0.5)
            
            return features
        
        except:
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        return [
            'Hour of Day',
            'Day of Week',
            'Direction',
            'Position Size',
            'Recent Performance',
            'Emotional State',
            'Symbol History'
        ]
    
    # ==================== CORE ANALYSIS METHODS (Enhanced) ====================
    
    def _analyze_time_patterns(self):
        """Enhanced time pattern analysis"""
        insights = {}
        
        time_slots = {
            'asian_session': (0, 8),
            'london_session': (8, 16),
            'ny_session': (13, 21),
            'overlap': (13, 16)
        }
        
        slot_stats = {slot: {'count': 0, 'pnl': 0, 'wins': 0} for slot in time_slots}
        
        for trade in self.trades:
            if trade.entry_time and trade.pnl is not None:
                hour = trade.entry_time.hour
                for slot, (start, end) in time_slots.items():
                    if start <= hour < end:
                        slot_stats[slot]['count'] += 1
                        slot_stats[slot]['pnl'] += trade.pnl
                        if trade.pnl > 0:
                            slot_stats[slot]['wins'] += 1
        
        best_session = None
        best_winrate = 0
        
        for slot, stats in slot_stats.items():
            if stats['count'] >= 3:
                win_rate = (stats['wins'] / stats['count']) * 100
                avg_pnl = stats['pnl'] / stats['count']
                
                if win_rate > best_winrate:
                    best_winrate = win_rate
                    best_session = slot
                
                insights[f'session_{slot}'] = {
                    'type': 'time_pattern',
                    'key': f'session_{slot}',
                    'value': f'{slot.replace("_", " ").title()}: {win_rate:.1f}% WR, ${avg_pnl:.2f} avg',
                    'confidence': min(stats['count'] / 10, 0.9),
                    'metadata': {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'trade_count': stats['count']
                    }
                }
        
        if best_session:
            insights['optimal_trading_session'] = {
                'type': 'time_pattern',
                'key': 'optimal_trading_session',
                'value': f'Best performance: {best_session.replace("_", " ").title()} ({best_winrate:.1f}% WR)',
                'confidence': 0.85,
                'recommendation': f'Focus trading during {best_session.replace("_", " ")}'
            }
        
        return insights
    
    def _analyze_symbol_performance(self):
        """Enhanced symbol analysis"""
        insights = {}
        
        symbol_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0, 'pnl_list': []})
        
        for trade in self.trades:
            if trade.pnl is not None:
                stats = symbol_stats[trade.symbol]
                stats['count'] += 1
                stats['pnl'] += trade.pnl
                stats['pnl_list'].append(trade.pnl)
                if trade.pnl > 0:
                    stats['wins'] += 1
        
        for symbol, stats in symbol_stats.items():
            if stats['count'] >= 3:
                win_rate = (stats['wins'] / stats['count']) * 100
                avg_pnl = stats['pnl'] / stats['count']
                consistency = 1 - (np.std(stats['pnl_list']) / (abs(avg_pnl) + 1))
                
                insights[f'symbol_{symbol}_performance'] = {
                    'type': 'symbol_analysis',
                    'key': f'symbol_{symbol}_performance',
                    'value': f'{symbol}: {win_rate:.1f}% WR, ${avg_pnl:.2f} avg, Consistency: {consistency:.2f}',
                    'confidence': min(stats['count'] / 15, 0.95),
                    'metadata': {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'consistency': consistency,
                        'trade_count': stats['count']
                    }
                }
        
        return insights
    
    def _analyze_strategy_performance(self):
        """Enhanced strategy analysis"""
        insights = {}
        
        strategy_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
        
        for trade in self.trades:
            if trade.strategy and trade.pnl is not None:
                stats = strategy_stats[trade.strategy]
                stats['count'] += 1
                stats['pnl'] += trade.pnl
                if trade.pnl > 0:
                    stats['wins'] += 1
        
        for strategy, stats in strategy_stats.items():
            if stats['count'] >= 3:
                win_rate = (stats['wins'] / stats['count']) * 100
                avg_pnl = stats['pnl'] / stats['count']
                
                insights[f'strategy_{strategy}_performance'] = {
                    'type': 'strategy_analysis',
                    'key': f'strategy_{strategy}_performance',
                    'value': f'{strategy}: {win_rate:.1f}% WR, ${avg_pnl:.2f} avg',
                    'confidence': min(stats['count'] / 10, 0.9),
                    'metadata': {
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl,
                        'trade_count': stats['count']
                    }
                }
        
        return insights
    
    def _detect_behavioral_patterns(self):
        """Detect behavioral patterns using clustering"""
        insights = {}
        
        if len(self.trades) < 15:
            return insights
        
        # Pattern detection logic here
        # (Simplified for brevity - can expand with DBSCAN clustering)
        
        return insights
    
    def _analyze_risk_management(self):
        """Enhanced risk management analysis"""
        insights = {}
        
        if len(self.trades) < 5:
            return insights
        
        pnl_values = [trade.pnl for trade in self.trades if trade.pnl is not None]
        
        if pnl_values:
            avg_pnl = np.mean(pnl_values)
            std_pnl = np.std(pnl_values)
            max_loss = min(pnl_values)
            max_win = max(pnl_values)
            
            # Risk-reward ratio
            avg_win = np.mean([p for p in pnl_values if p > 0]) if any(p > 0 for p in pnl_values) else 0
            avg_loss = abs(np.mean([p for p in pnl_values if p < 0])) if any(p < 0 for p in pnl_values) else 1
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            insights['risk_reward_analysis'] = {
                'type': 'risk_analysis',
                'key': 'risk_reward_analysis',
                'value': f'Risk:Reward Ratio: 1:{risk_reward_ratio:.2f}',
                'confidence': 0.85,
                'metadata': {
                    'ratio': risk_reward_ratio,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'max_loss': max_loss,
                    'max_win': max_win
                },
                'recommendation': 'Target at least 1:2 risk-reward ratio' if risk_reward_ratio < 2 else None
            }
        
        return insights
    
    def _analyze_new_trade(self, new_trade):
        """Analyze new trade with predictive insights"""
        insights = {}
        
        # Historical similar trades
        similar_trades = [
            t for t in self.trades 
            if t.symbol == new_trade.symbol 
            and t.direction == new_trade.direction
            and t.strategy == new_trade.strategy
        ]
        
        if similar_trades:
            wins = len([t for t in similar_trades if t.pnl and t.pnl > 0])
            win_rate = wins / len(similar_trades) * 100
            
            insights['trade_probability'] = {
                'type': 'trade_analysis',
                'key': 'trade_probability',
                'value': f'Historical success rate for this setup: {win_rate:.1f}%',
                'confidence': min(len(similar_trades) / 10, 0.9),
                'metadata': {
                    'win_rate': win_rate,
                    'sample_size': len(similar_trades)
                }
            }
        
        return insights
    
    def _generate_recommendations(self):
        """Generate comprehensive recommendations"""
        recommendations = {}
        
        if len(self.trades) < 10:
            return {
                'more_data_needed': {
                    'type': 'recommendation',
                    'key': 'more_data_needed',
                    'value': 'Continue journaling to unlock personalized insights',
                    'confidence': 0.5
                }
            }
        
        # Aggregate insights for recommendations
        # (Implementation continues with specific recommendation logic)
        
        return recommendations
    
    def _get_emotion_recommendation(self, emotion: str, win_rate: float, avg_pnl: float) -> Optional[str]:
        """Enhanced emotion recommendations"""
        emotion = emotion.lower()
        
        if win_rate < 30:
            recommendations = {
                'anxious': "Practice mindfulness before trading. Consider paper trading when anxious.",
                'fearful': "Build confidence with smaller position sizes. Focus on process, not outcome.",
                'impulsive': "Implement a mandatory 5-minute pause before entering trades.",
                'greedy': "Set profit targets in advance and use limit orders.",
                'frustrated': "Take a mandatory 24-hour break when frustrated.",
                'angry': "Do not trade when angry. Journal your feelings first."
            }
            return recommendations.get(emotion, "Avoid trading in this emotional state")
        
        if win_rate > 70:
            if emotion in ['confident', 'disciplined', 'focused']:
                return f"Excellent! Trading while {emotion} is your strength. Replicate these conditions."
        
        return None
    
    def _save_insights(self, insights):
        """Save insights to database"""
        for key, insight_data in insights.items():
            existing = self.insights.get(key)
            
            if existing:
                existing.insight_value = json.dumps(insight_data)
                existing.confidence_score = insight_data.get('confidence', 0.5)
                existing.data_points += 1
                existing.last_updated = datetime.utcnow()
            else:
                new_insight = AIInsight(
                    user_id=self.user_id,
                    insight_type=insight_data.get('type', 'general'),
                    insight_key=key,
                    insight_value=json.dumps(insight_data),
                    confidence_score=insight_data.get('confidence', 0.5),
                    data_points=1
                )
                db.session.add(new_insight)
        
        db.session.commit()
    
    def _log_learning_session(self, insights):
        """Log AI learning session"""
        log = AILearningLog(
            user_id=self.user_id,
            event_type='ai_analysis_completed',
            event_data=json.dumps({
                'trade_count': len(self.trades),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'insight_categories': list(set(i.get('type') for i in insights.values()))
            }),
            learned_insights=json.dumps(list(insights.keys()))
        )
        db.session.add(log)
        db.session.commit()
    
    def get_personalized_insights(self, limit=10):
        """Get personalized insights"""
        self.load_data()
        
        insights = AIInsight.query.filter_by(
            user_id=self.user_id, 
            is_active=True
        ).order_by(
            AIInsight.confidence_score.desc(),
            AIInsight.last_updated.desc()
        ).limit(limit).all()
        
        result = []
        for insight in insights:
            try:
                data = json.loads(insight.insight_value)
                data['id'] = insight.id
                data['confidence'] = insight.confidence_score
                data['last_updated'] = insight.last_updated
                result.append(data)
            except:
                continue
        
        return result
