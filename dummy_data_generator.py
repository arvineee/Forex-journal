#!/usr/bin/env python3
"""
Dummy Data Generator for Forex Journal
Creates realistic trading data to test the AI system
"""
import random
import numpy as np
from datetime import datetime, timedelta, UTC
from app import create_app, db
from app.models import User, Trade, AccountBalance
from app.utils.helpers import calculate_pnl

class DummyDataGenerator:
    def __init__(self, user_id):
        self.user_id = user_id
        self.app = create_app()
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'XAUUSD', 'XAGUSD', 'EURJPY', 'GBPJPY', 'BTCUSD'
        ]
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.strategies = [
            'Breakout', 'Scalping', 'Swing Trading', 'Trend Following',
            'Counter-trend', 'News Trading', 'Mean Reversion'
        ]
        self.emotions = [
            'confident', 'neutral', 'anxious', 'greedy', 'fearful',
            'disciplined', 'impulsive'
        ]
    
    def generate_trades(self, num_trades=3000):
        """Generate realistic dummy trades"""
        with self.app.app_context():
            # Create patterns for realistic data
            patterns = {
                'winning_patterns': [
                    {'symbol': 'EURUSD', 'strategy': 'Breakout', 'win_rate': 0.75},
                    {'symbol': 'XAUUSD', 'strategy': 'Trend Following', 'win_rate': 0.70},
                    {'symbol': 'GBPUSD', 'strategy': 'Swing Trading', 'win_rate': 0.65},
                ],
                'losing_patterns': [
                    {'symbol': 'USDJPY', 'strategy': 'Counter-trend', 'win_rate': 0.35},
                    {'symbol': 'BTCUSD', 'strategy': 'Impulsive', 'win_rate': 0.25},
                ]
            }
            
            trades = []
            # Updated: Start 2 years ago (730 days) to accommodate ~3000 trades
            start_date = datetime.now(UTC) - timedelta(days=730)
            
            # Calculate time step: 730 days * 24 hours = 17520 hours
            # 17520 hours / 3000 trades approx 5.8 hours per trade
            # Use 6 hours increment on average
            avg_hours_per_trade = (730 * 24) / num_trades
            
            for i in range(num_trades):
                # Determine if this trade follows a pattern
                use_pattern = random.random() < 0.6  # 60% follow patterns
                
                if use_pattern and i > 10:
                    # Use a defined pattern
                    if random.random() < 0.7:  # 70% winning patterns
                        pattern = random.choice(patterns['winning_patterns'])
                    else:
                        pattern = random.choice(patterns['losing_patterns'])
                    
                    symbol = pattern['symbol']
                    strategy = pattern['strategy']
                    target_win_rate = pattern['win_rate']
                else:
                    # Random selection
                    symbol = random.choice(self.symbols)
                    strategy = random.choice(self.strategies)
                    target_win_rate = 0.5
                
                # Create trade with realistic data
                direction = random.choice(['BUY', 'SELL'])
                size = round(random.uniform(0.01, 1.0), 2)  # 0.01 to 1.0 lots
                
                # Generate realistic prices based on symbol
                base_price = self._get_base_price(symbol)
                entry_price = base_price * random.uniform(0.999, 1.001)
                entry_price = round(entry_price, 5 if 'JPY' in symbol else 6)
                
                # Determine if trade wins based on pattern
                will_win = random.random() < target_win_rate
                
                # Calculate exit price (winning or losing)
                if will_win:
                    # Winning trade - price moves in favorable direction
                    if direction == 'BUY':
                        exit_price = entry_price * random.uniform(1.0005, 1.005)
                    else:
                        exit_price = entry_price * random.uniform(0.9995, 0.995)
                else:
                    # Losing trade - price moves against
                    if direction == 'BUY':
                        exit_price = entry_price * random.uniform(0.9995, 0.995)
                    else:
                        exit_price = entry_price * random.uniform(1.0005, 1.005)
                
                exit_price = round(exit_price, 5 if 'JPY' in symbol else 6)
                
                # Generate timestamps - Spread trades out evenly over the 730 days
                entry_time = start_date + timedelta(hours=i*avg_hours_per_trade, minutes=random.randint(0, 59))
                exit_time = entry_time + timedelta(hours=random.randint(1, 48))
                
                # Calculate P&L
                pnl, pnl_percent = calculate_pnl(direction, entry_price, exit_price, size, symbol)
                
                # Select emotion based on outcome
                if will_win:
                    emotion = random.choice(['confident', 'disciplined', 'neutral'])
                    rating = random.randint(3, 5)
                    mistakes = random.choice([
                        "Could have held longer for more profit",
                        "Entry was slightly late",
                        None
                    ])
                else:
                    emotion = random.choice(['anxious', 'fearful', 'impulsive', 'greedy'])
                    rating = random.randint(1, 3)
                    mistakes = random.choice([
                        "Overtraded the setup",
                        "Didn't follow stop loss",
                        "Entered too early",
                        "Let emotions dictate the trade"
                    ])
                
                # Create notes
                notes = self._generate_notes(symbol, direction, strategy, will_win)
                
                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    status='closed',
                    strategy=strategy,
                    timeframe=random.choice(self.timeframes),
                    notes=notes,
                    emotions=emotion,
                    mistakes=mistakes,
                    rating=rating,
                    user_id=self.user_id,
                    created_at=entry_time - timedelta(minutes=5)
                )
                
                trades.append(trade)
                db.session.add(trade)
            
            # Create some open trades (3 is fine)
            for i in range(3):
                symbol = random.choice(self.symbols)
                direction = random.choice(['BUY', 'SELL'])
                base_price = self._get_base_price(symbol)
                entry_price = base_price * random.uniform(0.999, 1.001)
                entry_price = round(entry_price, 5 if 'JPY' in symbol else 6)
                
                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    size=round(random.uniform(0.01, 0.5), 2),
                    entry_time=datetime.now(UTC) - timedelta(hours=random.randint(1, 24)),
                    status='open',
                    strategy=random.choice(self.strategies),
                    timeframe=random.choice(self.timeframes),
                    emotions=random.choice(self.emotions),
                    user_id=self.user_id
                )
                db.session.add(trade)
            
            db.session.commit()
            print(f"✅ Generated {num_trades + 3} dummy trades")
            
            return trades
    
    def generate_account_balances(self, initial_balance=10000):
        """Generate dummy account balance history"""
        with self.app.app_context():
            balances = []
            current_balance = initial_balance
            # Updated: Start 2 years ago to match the trade period
            start_date = datetime.now(UTC) - timedelta(days=730)
            
            # Get all trades sorted by entry time
            trades = Trade.query.filter_by(user_id=self.user_id)\
                .order_by(Trade.entry_time).all()
            
            # Updated: Daily balance for 730 days (2 years)
            for i in range(730): 
                date = start_date + timedelta(days=i)
                
                # Add random deposits/withdrawals occasionally
                if random.random() < 0.1:  # 10% chance
                    change = random.uniform(-500, 1000)
                    current_balance += change
                
                # Add P&L from trades on this day
                day_trades = [t for t in trades if t.exit_time and t.exit_time.date() == date.date()]
                for trade in day_trades:
                    if trade.pnl:
                        current_balance += trade.pnl
                
                balance = AccountBalance(
                    user_id=self.user_id,
                    balance=round(current_balance, 2),
                    date=date,
                    notes=random.choice([
                        "Weekly deposit",
                        "Profit withdrawal",
                        "Monthly savings",
                        None
                    ])
                )
                balances.append(balance)
                db.session.add(balance)
            
            # Add current balance
            current_balance_record = AccountBalance(
                user_id=self.user_id,
                balance=round(current_balance, 2),
                date=datetime.now(UTC),
                notes="Current balance"
            )
            db.session.add(current_balance_record)
            
            db.session.commit()
            print(f"✅ Generated {len(balances) + 1} account balance records")
            
            return balances
    
    def generate_emotional_patterns(self):
        """Generate trades with specific emotional patterns for AI to learn"""
        with self.app.app_context():
            # Pattern 1: Anxious emotions lead to losses
            for i in range(10):
                trade = Trade(
                    symbol='EURUSD',
                    direction=random.choice(['BUY', 'SELL']),
                    entry_price=1.0800 + random.uniform(-0.0020, 0.0020),
                    exit_price=1.0800 + random.uniform(-0.0050, -0.0010),  # Loss
                    size=0.1,
                    pnl=random.uniform(-50, -10),
                    pnl_percent=random.uniform(-2, -0.5),
                    entry_time=datetime.now(UTC) - timedelta(days=20-i),
                    exit_time=datetime.now(UTC) - timedelta(days=20-i) + timedelta(hours=2),
                    status='closed',
                    strategy='Scalping',
                    timeframe='5m',
                    emotions='anxious',
                    mistakes='Exited too early due to fear',
                    rating=2,
                    user_id=self.user_id
                )
                db.session.add(trade)
            
            # Pattern 2: Confident emotions lead to wins
            for i in range(10):
                trade = Trade(
                    symbol='XAUUSD',
                    direction='BUY',
                    entry_price=1950 + random.uniform(-10, 10),
                    exit_price=1950 + random.uniform(15, 40),  # Win
                    size=0.05,
                    pnl=random.uniform(25, 100),
                    pnl_percent=random.uniform(1, 4),
                    entry_time=datetime.now(UTC) - timedelta(days=15-i),
                    exit_time=datetime.now(UTC) - timedelta(days=15-i) + timedelta(hours=4),
                    status='closed',
                    strategy='Trend Following',
                    timeframe='1h',
                    emotions='confident',
                    mistakes=None,
                    rating=4,
                    user_id=self.user_id
                )
                db.session.add(trade)
            
            db.session.commit()
            print("✅ Generated emotional pattern trades (20 trades)")
    
    def generate_time_patterns(self):
        """Generate trades with time-based patterns"""
        with self.app.app_context():
            # Winning trades in London session (6-12 UTC)
            for i in range(8):
                entry_time = datetime.now(UTC).replace(
                    hour=random.randint(6, 11),
                    minute=random.randint(0, 59),
                    second=0
                ) - timedelta(days=random.randint(1, 20))
                
                trade = Trade(
                    symbol='GBPUSD',
                    direction=random.choice(['BUY', 'SELL']),
                    entry_price=1.2600 + random.uniform(-0.0020, 0.0020),
                    exit_price=1.2600 + random.uniform(0.0030, 0.0080),  # Win
                    size=0.1,
                    pnl=random.uniform(30, 80),
                    pnl_percent=random.uniform(1.5, 3.5),
                    entry_time=entry_time,
                    exit_time=entry_time + timedelta(hours=random.randint(1, 3)),
                    status='closed',
                    strategy='Breakout',
                    timeframe='15m',
                    emotions='disciplined',
                    rating=4,
                    user_id=self.user_id
                )
                db.session.add(trade)
            
            # Losing trades in Asian session (0-6 UTC)
            for i in range(8):
                entry_time = datetime.now(UTC).replace(
                    hour=random.randint(0, 5),
                    minute=random.randint(0, 59),
                    second=0
                ) - timedelta(days=random.randint(1, 20))
                
                trade = Trade(
                    symbol='USDJPY',
                    direction=random.choice(['BUY', 'SELL']),
                    entry_price=148.00 + random.uniform(-0.50, 0.50),
                    exit_price=148.00 + random.uniform(-1.50, -0.60),  # Loss
                    size=0.1,
                    pnl=random.uniform(-40, -15),
                    pnl_percent=random.uniform(-2, -0.8),
                    entry_time=entry_time,
                    exit_time=entry_time + timedelta(hours=random.randint(2, 6)),
                    status='closed',
                    strategy='Scalping',
                    timeframe='5m',
                    emotions='impulsive',
                    rating=2,
                    user_id=self.user_id
                )
                db.session.add(trade)
            
            db.session.commit()
            print("✅ Generated time pattern trades (16 trades)")
    
    def _get_base_price(self, symbol):
        """Get realistic base price for symbol"""
        base_prices = {
            'EURUSD': 1.08000,
            'GBPUSD': 1.26000,
            'USDJPY': 148.000,
            'AUDUSD': 0.65000,
            'USDCAD': 1.35000,
            'XAUUSD': 1950.00,
            'XAGUSD': 22.500,
            'EURJPY': 160.000,
            'GBPJPY': 186.000,
            'BTCUSD': 42000.00
        }
        return base_prices.get(symbol, 1.00000)
    
    def _generate_notes(self, symbol, direction, strategy, won):
        """Generate realistic trade notes"""
        if won:
            notes = [
                f"Clean {strategy} setup on {symbol}. Entry triggered at support/resistance.",
                f"{direction} signal confirmed with volume. Exited at target.",
                f"Followed {strategy} rules perfectly. Good risk management.",
                f"Market conditions favorable for {direction} on {symbol}."
            ]
        else:
            notes = [
                f"{strategy} setup failed. Market reversed unexpectedly.",
                f"Entered {direction} on {symbol} against the trend. Poor decision.",
                f"Stop loss hit. News event caused volatility.",
                f"Didn't wait for confirmation on {symbol} {direction}."
            ]
        return random.choice(notes)

def create_test_user():
    """Create a test user if not exists"""
    with create_app().app_context():
        # Check if test user exists
        test_user = User.query.filter_by(username='test_trader').first()
        
        if not test_user:
            test_user = User(
                username='test_trader',
                email='test@trader.com',
            )
            test_user.set_password('test123')
            db.session.add(test_user)
            db.session.commit()
            print("✅ Created test user: test_trader (password: test123)")
        
        return test_user.id

def generate_complete_dummy_data():
    """Generate complete dummy dataset for testing"""
    print("🎲 Generating dummy trading data...")
    
    user_id = create_test_user()
    
    # Clear existing data for test user
    with create_app().app_context():
        Trade.query.filter_by(user_id=user_id).delete()
        AccountBalance.query.filter_by(user_id=user_id).delete()
        db.session.commit()
    
    generator = DummyDataGenerator(user_id)
    
    # Generate various types of data
    print("\n📊 Generating basic trades (approx 2900)...")
    generator.generate_trades(2900) # <- Changed to generate 2900 trades
    
    print("\n😊 Generating emotional pattern trades...")
    generator.generate_emotional_patterns()
    
    print("\n⏰ Generating time pattern trades...")
    generator.generate_time_patterns()
    
    print("\n💰 Generating account balances (over 730 days)...")
    generator.generate_account_balances()
    
    # Calculate final counts
    with create_app().app_context():
        total_trades = Trade.query.filter_by(user_id=user_id).count()
        total_balances = AccountBalance.query.filter_by(user_id=user_id).count()

    print("\n✅ Dummy data generation complete!")
    print("\n📈 Data includes:")
    print(f"   - {total_trades} total trades (approx 3000)")
    print("   - 20 emotional pattern trades")
    print("   - 16 time pattern trades")
    print(f"   - {total_balances} account balance records (730 days + current)")
    print("   - Clear patterns for AI to learn")
    
    return user_id

def run_ai_analysis(user_id):
    """Run AI analysis on the generated data"""
    print("\n🤖 Running AI analysis on dummy data...")
    
    from app.ai.learning_engine import ForexAI
    
    with create_app().app_context():
        ai_engine = ForexAI(user_id)
        insights = ai_engine.analyze_and_learn()
        
        print(f"\n✅ AI analyzed {len(ai_engine.trades)} trades")
        print(f"📝 Generated {len(insights)} insights")
        
        # Show top insights
        print("\n🔍 Top AI Insights:")
        personalized_insights = ai_engine.get_personalized_insights(5)
        
        for i, insight in enumerate(personalized_insights, 1):
            print(f"\n{i}. {insight.get('value', 'No insight')}")
            if insight.get('recommendation'):
                print(f"   💡 Recommendation: {insight['recommendation']}")
            print(f"   🎯 Confidence: {insight.get('confidence', 0) * 100:.1f}%")
        
        return insights

if __name__ == '__main__':
    # Generate dummy data
    user_id = generate_complete_dummy_data()
    
    # Run AI analysis
    run_ai_analysis(user_id)
    
    print("\n🎉 All done! Your AI system is now trained with dummy data.")
    print("🔗 Login with: username='test_trader', password='test123'")
    print("🌐 Visit /ai to see AI insights dashboard")

