from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)  # Increased from 128 to 256
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trades = db.relationship('Trade', backref='author', lazy='dynamic', cascade='all, delete-orphan')
    account_balances = db.relationship('AccountBalance', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    trading_strategies = db.relationship('TradingStrategy', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    trading_goals = db.relationship('TradingGoal', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    trade_setups = db.relationship('TradeSetup', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Trade(db.Model):
    __tablename__ = 'trades'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False, index=True)
    direction = db.Column(db.String(4), nullable=False)  # BUY/SELL
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    size = db.Column(db.Float, nullable=False)
    pnl = db.Column(db.Float)
    pnl_percent = db.Column(db.Float)
    entry_time = db.Column(db.DateTime, nullable=False, index=True)
    exit_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='open', index=True)  # open/closed
    strategy = db.Column(db.String(100))  # Strategy name (string)
    timeframe = db.Column(db.String(10))
    notes = db.Column(db.Text)
    emotions = db.Column(db.String(100))  # Emotional state during trade
    mistakes = db.Column(db.Text)  # Trade mistakes analysis
    rating = db.Column(db.Integer)  # Self-rating 1-5
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('trading_strategies.id'), nullable=True)  # Foreign key to TradingStrategy
    images = db.relationship('TradeImage', backref='trade', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Trade {self.symbol} {self.direction} {self.entry_price}>'

class TradeImage(db.Model):
    __tablename__ = 'trade_images'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    annotation_data = db.Column(db.Text)  # JSON for drawing annotations
    image_type = db.Column(db.String(50))  # entry_chart, exit_chart, setup, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trade_id = db.Column(db.Integer, db.ForeignKey('trades.id'), nullable=False, index=True)
    
    def __repr__(self):
        return f'<TradeImage {self.filename}>'

class AccountBalance(db.Model):
    __tablename__ = 'account_balances'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    balance = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    notes = db.Column(db.Text)  # Optional notes about deposit/withdrawal
    
    def __repr__(self):
        return f'<AccountBalance ${self.balance} on {self.date}>'

class TradingStrategy(db.Model):
    __tablename__ = 'trading_strategies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    rules = db.Column(db.Text)  # JSON string of strategy rules
    success_rate = db.Column(db.Float)
    avg_pnl = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship to Trade model with cascade
    trades = db.relationship('Trade', backref='linked_strategy', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<TradingStrategy {self.name}>'

class TradeSetup(db.Model):
    __tablename__ = 'trade_setups'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    conditions = db.Column(db.Text)  # Setup conditions in JSON
    image_pattern = db.Column(db.String(255))  # Reference image for pattern
    success_rate = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    def __repr__(self):
        return f'<TradeSetup {self.name}>'

class TradingGoal(db.Model):
    __tablename__ = 'trading_goals'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    target_type = db.Column(db.String(50), nullable=False)  # pnl, win_rate, consistency
    target_value = db.Column(db.Float, nullable=False)
    current_value = db.Column(db.Float, default=0)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    is_achieved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

