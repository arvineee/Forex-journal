from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trades = db.relationship('Trade', backref='author', lazy='dynamic', cascade='all, delete-orphan')
    
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
    strategy = db.Column(db.String(100))
    timeframe = db.Column(db.String(10))
    notes = db.Column(db.Text)
    emotions = db.Column(db.String(100))  # Emotional state during trade
    mistakes = db.Column(db.Text)  # Trade mistakes analysis
    rating = db.Column(db.Integer)  # Self-rating 1-5
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
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

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))
