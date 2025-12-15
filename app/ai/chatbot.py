"""
Enhanced AI Chatbot for Forex Trading Assistant
Advanced NLP, context awareness, and conversational intelligence
"""
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from flask import session
from app import db
from app.models import Trade, TradingStrategy
from app.ai.models import AIInsight, AILearningLog
import numpy as np


class ConversationContext:
    """Manages conversation context and memory"""
    
    def __init__(self):
        self.entities = {}  # Extracted entities (symbols, dates, strategies)
        self.last_intent = None
        self.last_topic = None
        self.mentioned_trades = []
        self.focus_symbol = None
        self.focus_strategy = None
        self.focus_timeframe = None
        
    def update(self, intent: str, entities: dict):
        """Update context with new information"""
        self.last_intent = intent
        self.entities.update(entities)
        
        if 'symbol' in entities:
            self.focus_symbol = entities['symbol']
        if 'strategy' in entities:
            self.focus_strategy = entities['strategy']
        if 'timeframe' in entities:
            self.focus_timeframe = entities['timeframe']


class ForexChatbot:
    """Enhanced conversational AI for trading analysis"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.context = ConversationContext()
        self.conversation_history = deque(maxlen=50)  # Last 50 messages
        self.user_preferences = {}
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process user message with advanced NLP"""
        self._load_context()
        
        # Store user message
        user_msg = {
            'role': 'user',
            'content': message,
            'timestamp': datetime.utcnow()
        }
        self.conversation_history.append(user_msg)
        
        # Extract entities and intent
        entities = self._extract_entities(message)
        intent = self._analyze_intent(message, entities)
        
        # Update conversation context
        self.context.update(intent, entities)
        
        # Generate contextual response
        response = self._generate_response(message, intent, entities)
        
        # Store bot response
        bot_msg = {
            'role': 'assistant',
            'content': response['text'],
            'intent': intent,
            'timestamp': datetime.utcnow()
        }
        self.conversation_history.append(bot_msg)
        
        # Periodic conversation backup
        if len(self.conversation_history) % 10 == 0:
            self._save_conversation_snapshot()
        
        return response
    
    # ==================== ENTITY EXTRACTION ====================
    
    def _extract_entities(self, message: str) -> dict:
        """Extract entities from message using pattern matching"""
        entities = {}
        message_lower = message.lower()
        
        # Currency symbols
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 
                  'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'XAU/USD', 'GOLD']
        for symbol in symbols:
            if symbol.lower() in message_lower or symbol in message:
                entities['symbol'] = symbol
                break
        
        # Time references
        time_patterns = {
            r'\btoday\b': 'today',
            r'\byesterday\b': 'yesterday',
            r'\bthis week\b': 'this_week',
            r'\blast week\b': 'last_week',
            r'\bthis month\b': 'this_month',
            r'\blast month\b': 'last_month',
            r'\brecent\b': 'recent'
        }
        
        for pattern, timeframe in time_patterns.items():
            if re.search(pattern, message_lower):
                entities['timeframe'] = timeframe
                break
        
        # Trade ID extraction
        trade_id_match = re.search(r'trade\s*#?(\d+)', message_lower)
        if trade_id_match:
            entities['trade_id'] = int(trade_id_match.group(1))
        
        # Numbers (for analysis)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', message)
        if numbers:
            entities['numbers'] = [float(n) for n in numbers]
        
        # Strategy names (check against user's strategies)
        strategies = self._get_user_strategies()
        for strategy in strategies:
            if strategy.lower() in message_lower:
                entities['strategy'] = strategy
                break
        
        # Emotions
        emotions = ['confident', 'anxious', 'fearful', 'greedy', 'disciplined', 
                   'impulsive', 'frustrated', 'excited', 'neutral', 'calm']
        for emotion in emotions:
            if emotion in message_lower:
                entities['emotion'] = emotion
                break
        
        return entities
    
    # ==================== INTENT CLASSIFICATION ====================
    
    def _analyze_intent(self, message: str, entities: dict) -> str:
        """Advanced intent classification with context awareness"""
        message_lower = message.lower()
        
        # Context-aware greeting
        if self._is_greeting(message_lower):
            return 'greeting'
        
        # Performance queries
        if self._is_performance_query(message_lower):
            return 'performance_summary'
        
        # Comparative analysis
        if self._is_comparison_query(message_lower):
            return 'comparison_analysis'
        
        # Specific trade analysis
        if 'trade_id' in entities or self._mentions_specific_trade(message_lower):
            return 'trade_analysis'
        
        # Emotional analysis
        if self._is_emotion_query(message_lower):
            return 'emotion_analysis'
        
        # Pattern detection
        if self._is_pattern_query(message_lower):
            return 'pattern_analysis'
        
        # Recommendations
        if self._is_recommendation_query(message_lower):
            return 'recommendations'
        
        # Symbol-specific analysis
        if 'symbol' in entities or self._mentions_symbol(message_lower):
            return 'symbol_analysis'
        
        # Time-based analysis
        if self._is_time_query(message_lower):
            return 'time_analysis'
        
        # Risk management
        if self._is_risk_query(message_lower):
            return 'risk_analysis'
        
        # Strategy analysis
        if 'strategy' in entities or self._is_strategy_query(message_lower):
            return 'strategy_analysis'
        
        # Statistical queries
        if self._is_stats_query(message_lower):
            return 'statistics'
        
        # What-if scenarios
        if self._is_whatif_query(message_lower):
            return 'scenario_analysis'
        
        # Goal tracking
        if self._is_goal_query(message_lower):
            return 'goal_tracking'
        
        # Learning/educational
        if self._is_learning_query(message_lower):
            return 'education'
        
        # Farewell
        if self._is_farewell(message_lower):
            return 'farewell'
        
        # Follow-up question
        if self._is_followup(message_lower):
            return self._get_followup_intent()
        
        # Default
        return 'general_chat'
    
    # Intent helper methods
    def _is_greeting(self, msg: str) -> bool:
        return bool(re.search(r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b', msg))
    
    def _is_performance_query(self, msg: str) -> bool:
        patterns = [
            r'\bhow am i doing\b',
            r'\bmy performance\b',
            r'\bmy results\b',
            r'\bshow.*performance\b',
            r'\boverall.*stats\b',
            r'\bwin rate\b',
            r'\bp&l\b|\bpnl\b',
            r'\btotal.*profit\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_comparison_query(self, msg: str) -> bool:
        patterns = [
            r'\bcompare\b',
            r'\bvs\b|\bversus\b',
            r'\bbetter.*or\b',
            r'\bwhich.*best\b',
            r'\bdifference.*between\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _mentions_specific_trade(self, msg: str) -> bool:
        return bool(re.search(r'\btrade\s*#?\d+\b|\blast trade\b|\blatest trade\b|\brecent trade\b', msg))
    
    def _is_emotion_query(self, msg: str) -> bool:
        patterns = [
            r'\bemotion\b',
            r'\bfeeling\b',
            r'\bmood\b',
            r'\bmental\b',
            r'\bpsycholog\b',
            r'\banxious\b',
            r'\bgreedy\b',
            r'\bfearful\b',
            r'\bconfident\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_pattern_query(self, msg: str) -> bool:
        patterns = [
            r'\bpattern\b',
            r'\btrend\b',
            r'\bhabit\b',
            r'\bbehavior\b',
            r'\broutine\b',
            r'\bnotice\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_recommendation_query(self, msg: str) -> bool:
        patterns = [
            r'\bwhat should i do\b',
            r'\brecommendation\b',
            r'\badvice\b',
            r'\bsuggestion\b',
            r'\bimprove\b',
            r'\bget better\b',
            r'\bhelp me\b',
            r'\btips?\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _mentions_symbol(self, msg: str) -> bool:
        symbols = ['eur', 'gbp', 'usd', 'jpy', 'aud', 'cad', 'nzd', 'gold', 'xau']
        return any(s in msg for s in symbols)
    
    def _is_time_query(self, msg: str) -> bool:
        patterns = [
            r'\bwhen\b',
            r'\btime\b',
            r'\bday\b',
            r'\bhour\b',
            r'\bmorning\b',
            r'\bafternoon\b',
            r'\bevening\b',
            r'\bnight\b',
            r'\bsession\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_risk_query(self, msg: str) -> bool:
        patterns = [
            r'\brisk\b',
            r'\bstop loss\b',
            r'\btake profit\b',
            r'\bposition size\b',
            r'\blot size\b',
            r'\brisk management\b',
            r'\bdrawdown\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_strategy_query(self, msg: str) -> bool:
        patterns = [
            r'\bstrategy\b',
            r'\bsetup\b',
            r'\bapproach\b',
            r'\bmethod\b',
            r'\btechnique\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_stats_query(self, msg: str) -> bool:
        patterns = [
            r'\bstatistics\b',
            r'\baverage\b',
            r'\bmean\b',
            r'\bmedian\b',
            r'\bcount\b',
            r'\bhow many\b',
            r'\btotal\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_whatif_query(self, msg: str) -> bool:
        patterns = [
            r'\bwhat if\b',
            r'\bscenario\b',
            r'\bif i\b',
            r'\bwould.*happen\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_goal_query(self, msg: str) -> bool:
        patterns = [
            r'\bgoal\b',
            r'\btarget\b',
            r'\bobjective\b',
            r'\bprogress\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_learning_query(self, msg: str) -> bool:
        patterns = [
            r'\bhow do i\b',
            r'\bwhat is\b',
            r'\bexplain\b',
            r'\bteach me\b',
            r'\blearn\b'
        ]
        return any(re.search(p, msg) for p in patterns)
    
    def _is_farewell(self, msg: str) -> bool:
        return bool(re.search(r'\b(bye|goodbye|see you|farewell|thanks|thank you)\b', msg))
    
    def _is_followup(self, msg: str) -> bool:
        """Detect if this is a follow-up question"""
        followup_patterns = [
            r'\bwhat about\b',
            r'\bhow about\b',
            r'\band\b.*\?',
            r'\balso\b',
            r'\bmore\b',
            r'\bother\b',
            r'\belse\b'
        ]
        return any(re.search(p, msg) for p in followup_patterns)
    
    def _get_followup_intent(self) -> str:
        """Return intent based on conversation context"""
        return self.context.last_intent or 'general_chat'
    
    # ==================== RESPONSE GENERATION ====================
    
    def _generate_response(self, message: str, intent: str, entities: dict) -> Dict[str, Any]:
        """Generate contextual response"""
        
        # Intent routing
        response_methods = {
            'greeting': self._respond_greeting,
            'performance_summary': self._respond_performance,
            'comparison_analysis': self._respond_comparison,
            'trade_analysis': self._respond_trade_analysis,
            'emotion_analysis': self._respond_emotion_analysis,
            'pattern_analysis': self._respond_pattern_analysis,
            'recommendations': self._respond_recommendations,
            'symbol_analysis': self._respond_symbol_analysis,
            'time_analysis': self._respond_time_analysis,
            'risk_analysis': self._respond_risk_analysis,
            'strategy_analysis': self._respond_strategy_analysis,
            'statistics': self._respond_statistics,
            'scenario_analysis': self._respond_scenario,
            'goal_tracking': self._respond_goals,
            'education': self._respond_education,
            'farewell': self._respond_farewell,
            'general_chat': self._respond_general
        }
        
        handler = response_methods.get(intent, self._respond_general)
        return handler(message, entities)
    
    def _respond_greeting(self, message: str, entities: dict) -> dict:
        """Personalized greeting"""
        trades = Trade.query.filter_by(user_id=self.user_id).all()
        
        greetings = [
            f"Hi! I've been analyzing your {len(trades)} trades. What would you like to know?",
            f"Hello! Ready to dive into your trading data. You have {len(trades)} trades on record.",
            f"Hey there! I'm here to help you improve your trading. What's on your mind?"
        ]
        
        import random
        greeting = random.choice(greetings)
        
        return {
            'text': greeting,
            'suggestions': [
                "How's my performance?",
                "What patterns did you find?",
                "Give me recommendations",
                "Analyze my emotions"
            ]
        }
    
    def _respond_performance(self, message: str, entities: dict) -> dict:
        """Enhanced performance summary"""
        timeframe = entities.get('timeframe', 'all')
        trades = self._get_trades_by_timeframe(timeframe)
        
        if not trades:
            return {
                'text': "No trades found for this period. Start journaling to unlock insights!",
                'suggestions': ["How do I track trades?", "What should I record?"]
            }
        
        # Calculate metrics
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        total_pnl = sum(t.pnl or 0 for t in trades)
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        # Advanced metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        profit_factor = (sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in losing_trades))) if losing_trades else float('inf')
        
        # Best and worst
        best_trade = max(trades, key=lambda x: x.pnl or 0) if trades else None
        worst_trade = min(trades, key=lambda x: x.pnl or 0) if trades else None
        
        text = f"""📊 **Performance Summary** ({timeframe.replace('_', ' ').title()})

**Overview:**
• Total Trades: {len(trades)}
• Win Rate: {win_rate:.1f}%
• Net P&L: ${total_pnl:.2f}
• Profit Factor: {profit_factor:.2f}

**Win/Loss Profile:**
• Winners: {len(winning_trades)} (Avg: ${avg_win:.2f})
• Losers: {len(losing_trades)} (Avg: ${avg_loss:.2f})
• Risk:Reward: 1:{(avg_win/avg_loss if avg_loss > 0 else 0):.2f}

**Best & Worst:**
• Best: +${best_trade.pnl:.2f} ({best_trade.symbol})
• Worst: ${worst_trade.pnl:.2f} ({worst_trade.symbol})
"""
        
        # Add insight
        if win_rate > 50 and profit_factor > 1.5:
            text += "\n✅ You're on the right track! Focus on consistency."
        elif win_rate < 40:
            text += "\n⚠️ Win rate needs improvement. Let's analyze what's not working."
        
        return {
            'text': text,
            'suggestions': [
                "What's my best symbol?",
                "Analyze my emotions",
                "What patterns do you see?",
                "Give me recommendations"
            ]
        }
    
    def _respond_comparison(self, message: str, entities: dict) -> dict:
        """Compare different aspects"""
        # Implementation for comparison queries
        return {
            'text': "Comparison analysis coming soon! What specifically would you like to compare?",
            'suggestions': [
                "Compare EUR/USD vs GBP/USD",
                "Compare my strategies",
                "Compare morning vs evening trades"
            ]
        }
    
    def _respond_trade_analysis(self, message: str, entities: dict) -> dict:
        """Analyze specific trade"""
        trade_id = entities.get('trade_id')
        
        if not trade_id:
            # Get latest trade
            trade = Trade.query.filter_by(user_id=self.user_id).order_by(Trade.entry_time.desc()).first()
        else:
            trade = Trade.query.filter_by(user_id=self.user_id, id=trade_id).first()
        
        if not trade:
            return {
                'text': "Trade not found. Try asking about your latest trade or specify a trade ID.",
                'suggestions': ["Analyze my latest trade", "Show my recent trades"]
            }
        
        # Comprehensive trade analysis
        text = f"""🔍 **Trade #{trade.id} Analysis**

**Details:**
• Symbol: {trade.symbol}
• Direction: {trade.direction}
• Entry: {trade.entry_price}
• Exit: {trade.exit_price or 'Open'}
• P&L: ${trade.pnl:.2f if trade.pnl else 'Pending'}
• Size: {trade.size} lots

**Context:**
"""
        
        if trade.emotions:
            text += f"• Emotion: {trade.emotions}\n"
        if trade.strategy:
            text += f"• Strategy: {trade.strategy}\n"
        if trade.notes:
            text += f"• Notes: {trade.notes[:100]}...\n"
        
        # Similar trades analysis
        similar_trades = Trade.query.filter_by(
            user_id=self.user_id,
            symbol=trade.symbol,
            direction=trade.direction
        ).filter(Trade.id != trade.id, Trade.pnl.isnot(None)).all()
        
        if similar_trades:
            wins = len([t for t in similar_trades if t.pnl > 0])
            win_rate = (wins / len(similar_trades)) * 100
            text += f"\n**Historical Context:**\n"
            text += f"• Similar trades: {len(similar_trades)} ({win_rate:.1f}% WR)\n"
            
            if win_rate > 60:
                text += "✅ This setup has historically performed well!"
            elif win_rate < 40:
                text += "⚠️ This setup has struggled historically. Review carefully."
        
        return {
            'text': text,
            'suggestions': [
                "What's my best trade?",
                "Analyze my worst trade",
                "Show similar trades"
            ]
        }
    
    def _respond_emotion_analysis(self, message: str, entities: dict) -> dict:
        """Deep emotional analysis"""
        emotion_filter = entities.get('emotion')
        
        insights = AIInsight.query.filter_by(
            user_id=self.user_id,
            insight_type='emotion_analysis',
            is_active=True
        ).order_by(AIInsight.confidence_score.desc()).limit(10).all()
        
        text = "🧠 **Emotional Intelligence Analysis**\n\n"
        
        if insights:
            for insight in insights:
                try:
                    data = json.loads(insight.insight_value)
                    text += f"• {data.get('value', '')}\n"
                    
                    if 'recommendation' in data and data['recommendation']:
                        text += f"  💡 {data['recommendation']}\n"
                    
                    if 'priority' in data and data['priority'] == 'high':
                        text += "  🚨 High Priority\n"
                    
                    text += "\n"
                except:
                    continue
        else:
            text += "Not enough emotional data yet. Remember to log your emotions with each trade!\n\n"
            text += "**Why emotions matter:**\n"
            text += "• They drive decision-making\n"
            text += "• Pattern recognition reveals triggers\n"
            text += "• Self-awareness improves discipline\n"
        
        return {
            'text': text,
            'suggestions': [
                "How does anxiety affect me?",
                "When am I most disciplined?",
                "Show emotional stability score"
            ]
        }
    
    def _respond_pattern_analysis(self, message: str, entities: dict) -> dict:
        """Pattern detection and analysis"""
        insights = AIInsight.query.filter(
            AIInsight.user_id == self.user_id,
            AIInsight.insight_type.in_(['behavioral_pattern', 'time_pattern']),
            AIInsight.is_active == True
        ).order_by(AIInsight.confidence_score.desc()).limit(8).all()
        
        text = "🔎 **Pattern Analysis**\n\n"
        
        if insights:
            text += "I've detected these patterns in your trading:\n\n"
            
            for insight in insights:
                try:
                    data = json.loads(insight.insight_value)
                    confidence = data.get('confidence', 0)
                    
                    text += f"{'🟢' if confidence > 0.7 else '🟡'} {data.get('value', '')}\n"
                    
                    if 'recommendation' in data and data['recommendation']:
                        text += f"   → {data['recommendation']}\n"
                    
                    text += "\n"
                except:
                    continue
        else:
            text += "Need more data to detect patterns. Keep journaling!\n\n"
            text += "**Patterns I look for:**\n"
            text += "• Time-based success rates\n"
            text += "• Winning/losing streaks\n"
            text += "• Revenge trading tendencies\n"
            text += "• Symbol preferences\n"
        
        return {
            'text': text,
            'suggestions': [
                "Do I revenge trade?",
                "What's my best time to trade?",
                "Analyze my streaks"
            ]
        }
    
    def _respond_recommendations(self, message: str, entities: dict) -> dict:
        """Personalized recommendations"""
        insights = AIInsight.query.filter_by(
            user_id=self.user_id,
            insight_type='recommendation',
            is_active=True
        ).order_by(AIInsight.confidence_score.desc()).limit(5).all()
        
        text = "💡 **Personalized Recommendations**\n\n"
        
        if insights:
            for i, insight in enumerate(insights, 1):
                try:
                    data = json.loads(insight.insight_value)
                    text += f"**{i}. {data.get('value', '')}**\n"
                    
                    if 'recommendation' in data:
                        text += f"   {data['recommendation']}\n"
                    
                    text += "\n"
                except:
                    continue
        else:
            # Generic recommendations
            trades = Trade.query.filter_by(user_id=self.user_id).all()
            
            text += "Based on best practices:\n\n"
            text += "1. **Journal Consistently**: Record emotions and notes for every trade\n"
            text += "2. **Risk Management**: Risk only 1-2% per trade\n"
            text += "3. **Review Weekly**: Analyze patterns every week\n"
            text += "4. **Stay Disciplined**: Follow your trading plan\n"
            text += "5. **Continuous Learning**: Study both wins and losses\n"
            
            if len(trades) < 20:
                text += "\n📊 Take 20+ trades for personalized recommendations"
        
        return {
            'text': text,
            'suggestions': [
                "What should I improve first?",
                "How can I be more consistent?",
                "Risk management tips"
            ]
        }
    
    def _respond_symbol_analysis(self, message: str, entities: dict) -> dict:
        """Symbol-specific performance"""
        symbol = entities.get('symbol') or self.context.focus_symbol
        
        if not symbol:
            # Show all symbols
            trades = Trade.query.filter_by(user_id=self.user_id).all()
            symbols = {}
            
            for trade in trades:
                if trade.pnl is not None:
                    if trade.symbol not in symbols:
                        symbols[trade.symbol] = {'count': 0, 'pnl': 0, 'wins': 0}
                    symbols[trade.symbol]['count'] += 1
                    symbols[trade.symbol]['pnl'] += trade.pnl
                    if trade.pnl > 0:
                        symbols[trade.symbol]['wins'] += 1
            
            text = "📈 **Symbol Performance**\n\n"
            
            for sym, stats in sorted(symbols.items(), key=lambda x: x[1]['pnl'], reverse=True):
                win_rate = (stats['wins'] / stats['count']) * 100
                text += f"• {sym}: {win_rate:.1f}% WR, ${stats['pnl']:.2f} total ({stats['count']} trades)\n"
            
            return {
                'text': text,
                'suggestions': [
                    "Analyze EUR/USD",
                    "Which symbol is best for me?",
                    "Should I avoid any symbols?"
                ]
            }
        
        # Specific symbol analysis
        trades = Trade.query.filter_by(user_id=self.user_id, symbol=symbol).all()
        
        if not trades:
            return {
                'text': f"No trades found for {symbol}. Try another symbol?",
                'suggestions': ["Show all symbols", "EUR/USD analysis"]
            }
        
        wins = len([t for t in trades if t.pnl and t.pnl > 0])
        win_rate = (wins / len(trades)) * 100 if trades else 0
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        
        text = f"""📊 **{symbol} Analysis**

**Performance:**
• Total Trades: {len(trades)}
• Win Rate: {win_rate:.1f}%
• Total P&L: ${total_pnl:.2f}
• Avg P&L: ${total_pnl/len(trades):.2f}

**Direction Breakdown:**
"""
        
        buy_trades = [t for t in trades if t.direction == 'BUY']
        sell_trades = [t for t in trades if t.direction == 'SELL']
        
        if buy_trades:
            buy_wins = len([t for t in buy_trades if t.pnl and t.pnl > 0])
            text += f"• BUY: {(buy_wins/len(buy_trades)*100):.1f}% WR ({len(buy_trades)} trades)\n"
        
        if sell_trades:
            sell_wins = len([t for t in sell_trades if t.pnl and t.pnl > 0])
            text += f"• SELL: {(sell_wins/len(sell_trades)*100):.1f}% WR ({len(sell_trades)} trades)\n"
        
        return {
            'text': text,
            'suggestions': [
                f"What's my best time for {symbol}?",
                "Compare to other symbols",
                "Show losing trades"
            ]
        }
    
    def _respond_time_analysis(self, message: str, entities: dict) -> dict:
        """Time-based analysis"""
        insights = AIInsight.query.filter_by(
            user_id=self.user_id,
            insight_type='time_pattern',
            is_active=True
        ).order_by(AIInsight.confidence_score.desc()).limit(5).all()
        
        text = "⏰ **Time-Based Performance**\n\n"
        
        if insights:
            for insight in insights:
                try:
                    data = json.loads(insight.insight_value)
                    text += f"• {data.get('value', '')}\n"
                except:
                    continue
        else:
            text += "Need more trades to identify optimal trading times.\n\n"
            text += "**Key Times to Track:**\n"
            text += "• Asian Session (00:00-08:00 UTC)\n"
            text += "• London Session (08:00-16:00 UTC)\n"
            text += "• New York Session (13:00-21:00 UTC)\n"
            text += "• Overlap Period (13:00-16:00 UTC)\n"
        
        return {
            'text': text,
            'suggestions': [
                "When do I trade best?",
                "Morning vs evening?",
                "Analyze by day of week"
            ]
        }
    
    def _respond_risk_analysis(self, message: str, entities: dict) -> dict:
        """Risk management analysis"""
        insights = AIInsight.query.filter_by(
            user_id=self.user_id,
            insight_type='risk_analysis',
            is_active=True
        ).order_by(AIInsight.confidence_score.desc()).limit(5).all()
        
        text = "🛡️ **Risk Management Analysis**\n\n"
        
        if insights:
            for insight in insights:
                try:
                    data = json.loads(insight.insight_value)
                    text += f"• {data.get('value', '')}\n"
                    if 'recommendation' in data:
                        text += f"  💡 {data['recommendation']}\n"
                    text += "\n"
                except:
                    continue
        else:
            text += "**Risk Management Fundamentals:**\n\n"
        
        text += "**Best Practices:**\n"
        text += "1. Risk 1-2% per trade maximum\n"
        text += "2. Use stop losses on every trade\n"
        text += "3. Aim for min 1:2 risk:reward ratio\n"
        text += "4. Avoid revenge trading after losses\n"
        text += "5. Review risk weekly\n"
        
        return {
            'text': text,
            'suggestions': [
                "Calculate position size",
                "What's my risk:reward ratio?",
                "Am I risking too much?"
            ]
        }
    
    def _respond_strategy_analysis(self, message: str, entities: dict) -> dict:
        """Strategy performance analysis"""
        strategy = entities.get('strategy') or self.context.focus_strategy
        
        insights = AIInsight.query.filter_by(
            user_id=self.user_id,
            insight_type='strategy_analysis',
            is_active=True
        ).order_by(AIInsight.confidence_score.desc()).limit(5).all()
        
        text = "🎯 **Strategy Analysis**\n\n"
        
        if insights:
            for insight in insights:
                try:
                    data = json.loads(insight.insight_value)
                    text += f"• {data.get('value', '')}\n"
                except:
                    continue
        else:
            text += "Track your strategy with each trade to see what works best!\n"
        
        return {
            'text': text,
            'suggestions': [
                "What's my best strategy?",
                "Compare strategies",
                "Should I change my approach?"
            ]
        }
    
    def _respond_statistics(self, message: str, entities: dict) -> dict:
        """Statistical analysis"""
        trades = Trade.query.filter_by(user_id=self.user_id).all()
        
        if not trades:
            return {
                'text': "No trades to analyze yet!",
                'suggestions': ["How do I start?"]
            }
        
        pnl_values = [t.pnl for t in trades if t.pnl is not None]
        
        text = "📊 **Trading Statistics**\n\n"
        
        if pnl_values:
            text += f"**Descriptive Stats:**\n"
            text += f"• Mean P&L: ${np.mean(pnl_values):.2f}\n"
            text += f"• Median P&L: ${np.median(pnl_values):.2f}\n"
            text += f"• Std Deviation: ${np.std(pnl_values):.2f}\n"
            text += f"• Max Win: ${max(pnl_values):.2f}\n"
            text += f"• Max Loss: ${min(pnl_values):.2f}\n"
        
        return {
            'text': text,
            'suggestions': [
                "Calculate profit factor",
                "What's my Sharpe ratio?",
                "Show expectancy"
            ]
        }
    
    def _respond_scenario(self, message: str, entities: dict) -> dict:
        """What-if scenario analysis"""
        return {
            'text': "Scenario analysis helps you plan ahead. What scenario would you like to explore?",
            'suggestions': [
                "What if I only traded mornings?",
                "What if I doubled position size?",
                "What if I avoided anxious trades?"
            ]
        }
    
    def _respond_goals(self, message: str, entities: dict) -> dict:
        """Goal tracking"""
        return {
            'text': "Goal tracking helps you stay focused. Set goals in your dashboard to track progress!",
            'suggestions': [
                "How do I set a goal?",
                "What are good trading goals?",
                "Show my progress"
            ]
        }
    
    def _respond_education(self, message: str, entities: dict) -> dict:
        """Educational content"""
        return {
            'text': "I can help explain trading concepts. What would you like to learn about?",
            'suggestions': [
                "What's a good win rate?",
                "How to calculate risk:reward?",
                "What's revenge trading?"
            ]
        }
    
    def _respond_farewell(self, message: str, entities: dict) -> dict:
        """Goodbye message"""
        farewells = [
            "Happy trading! Remember to journal every trade. 📝",
            "Good luck with your trades! Stay disciplined. 💪",
            "See you next time! Keep learning from every trade. 📈"
        ]
        
        import random
        return {
            'text': random.choice(farewells),
            'suggestions': []
        }
    
    def _respond_general(self, message: str, entities: dict) -> dict:
        """General conversational response"""
        return {
            'text': "I'm here to help analyze your trading! I can provide insights on:\n\n"
                   "• 📊 Performance & statistics\n"
                   "• 🧠 Emotional patterns\n"
                   "• 🔍 Behavioral analysis\n"
                   "• 💡 Personalized recommendations\n"
                   "• ⏰ Time-based patterns\n"
                   "• 🎯 Strategy effectiveness\n\n"
                   "What would you like to explore?",
            'suggestions': [
                "How's my performance?",
                "Analyze my emotions",
                "What patterns do you see?",
                "Give me recommendations"
            ]
        }
    
    # ==================== HELPER METHODS ====================
    
    def _get_trades_by_timeframe(self, timeframe: str) -> List[Trade]:
        """Get trades filtered by timeframe"""
        query = Trade.query.filter_by(user_id=self.user_id)
        
        if timeframe == 'today':
            start = datetime.utcnow().replace(hour=0, minute=0, second=0)
            query = query.filter(Trade.entry_time >= start)
        elif timeframe == 'yesterday':
            start = datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=1)
            end = datetime.utcnow().replace(hour=0, minute=0, second=0)
            query = query.filter(Trade.entry_time >= start, Trade.entry_time < end)
        elif timeframe == 'this_week':
            start = datetime.utcnow() - timedelta(days=7)
            query = query.filter(Trade.entry_time >= start)
        elif timeframe == 'last_week':
            start = datetime.utcnow() - timedelta(days=14)
            end = datetime.utcnow() - timedelta(days=7)
            query = query.filter(Trade.entry_time >= start, Trade.entry_time < end)
        elif timeframe == 'this_month':
            start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
            query = query.filter(Trade.entry_time >= start)
        elif timeframe == 'last_month':
            from dateutil.relativedelta import relativedelta
            start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0) - relativedelta(months=1)
            end = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
            query = query.filter(Trade.entry_time >= start, Trade.entry_time < end)
        elif timeframe == 'recent':
            query = query.order_by(Trade.entry_time.desc()).limit(10)
            return query.all()
        
        return query.all()
    
    def _get_user_strategies(self) -> List[str]:
        """Get user's defined strategies"""
        strategies = TradingStrategy.query.filter_by(user_id=self.user_id).all()
        return [s.name for s in strategies]
    
    def _load_context(self):
        """Load user context and preferences"""
        trades = Trade.query.filter_by(user_id=self.user_id).count()
        
        # Could expand this to load user preferences
        self.user_preferences = {
            'trade_count': trades,
            'last_analysis': datetime.utcnow()
        }
    
    def _save_conversation_snapshot(self):
        """Save conversation snapshot to logs"""
        try:
            last_intents = [msg.get('intent') for msg in list(self.conversation_history)[-5:] if msg['role'] == 'assistant']
            
            log = AILearningLog(
                user_id=self.user_id,
                event_type='chatbot_conversation',
                event_data=json.dumps({
                    'messages': len(self.conversation_history),
                    'recent_intents': last_intents
                }),
                learned_insights=json.dumps([])
            )
            db.session.add(log)
            db.session.commit()
        except:
            pass
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict]:
        """Get recent conversation history"""
        history = list(self.conversation_history)[-limit:]
        
        # Format for frontend
        formatted = []
        for msg in history:
            formatted.append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']
            })
        
        return formatted
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.context = ConversationContext()
