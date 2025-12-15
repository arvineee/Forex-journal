"""
AI Learning Models for Forex Journal
"""
from datetime import datetime
from app import db
import json

class AIInsight(db.Model):
    __tablename__ = 'ai_insights'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    insight_type = db.Column(db.String(50), nullable=False)  # pattern, emotion, strategy, risk, improvement
    insight_key = db.Column(db.String(100), nullable=False)   # e.g., 'anxiety_impact', 'best_timeframe'
    insight_value = db.Column(db.Text)  # JSON or string value
    confidence_score = db.Column(db.Float, default=0.0)
    data_points = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<AIInsight {self.insight_type}:{self.insight_key}>'

class TradePattern(db.Model):
    __tablename__ = 'trade_patterns'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pattern_type = db.Column(db.String(50), nullable=False)  # win_pattern, loss_pattern, emotion_pattern
    pattern_data = db.Column(db.Text)  # JSON pattern data
    occurrence_count = db.Column(db.Integer, default=1)
    first_detected = db.Column(db.DateTime, default=datetime.utcnow)
    last_detected = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<TradePattern {self.pattern_type} x{self.occurrence_count}>'

class AILearningLog(db.Model):
    __tablename__ = 'ai_learning_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)  # trade_added, trade_updated, pattern_detected
    event_data = db.Column(db.Text)  # JSON event data
    learned_insights = db.Column(db.Text)  # JSON insights learned
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<AILearningLog {self.event_type} at {self.created_at}>'
