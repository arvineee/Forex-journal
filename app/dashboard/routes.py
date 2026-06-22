from flask import render_template, jsonify
from flask_login import login_required, current_user
from app.dashboard import dashboard_bp
from app.models import Trade
from app.utils.analysis import TradeAnalyzer
from app.models import AccountBalance 
from app.ai.learning_engine import ForexAI


@dashboard_bp.route('/metrics')          
@login_required
def metrics():
    try:
        trades = Trade.query.filter_by(user_id=current_user.id).all()
        metrics = TradeAnalyzer.calculate_metrics(trades)   
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dashboard_bp.route('/')
@login_required
def index():
    try:
        # Get all trades for the current user
        trades = Trade.query.filter_by(user_id=current_user.id).all()
        
        # Get account balances
        account_balances = AccountBalance.query.filter_by(user_id=current_user.id)\
            .order_by(AccountBalance.date.asc())\
            .all()
        
        # Calculate metrics
        metrics = TradeAnalyzer.calculate_metrics(trades)
        
        # Generate equity curve data
        equity_data = TradeAnalyzer.generate_equity_data(trades)
        
        # Generate account balance curve data
        account_data = []
        running_balance = 0
        for balance in account_balances:
            account_data.append({
                'date': balance.date.isoformat(),
                'balance': balance.balance
            })
        
        # Get recent trades
        recent_trades = TradeAnalyzer.get_recent_trades(trades, 5)

        # --- Win/Loss Distribution ---
        closed = [t for t in trades if t.pnl is not None]
        wins   = sum(1 for t in closed if t.pnl > 0)
        losses = sum(1 for t in closed if t.pnl < 0)
        breakeven = len(closed) - wins - losses
        win_loss_data = {'wins': wins, 'losses': losses, 'breakeven': breakeven}

        # --- Symbol Performance (top 6 by trade count) ---
        from collections import defaultdict
        sym_map = defaultdict(lambda: {'pnl': 0.0, 'count': 0, 'wins': 0})
        for t in closed:
            sym_map[t.symbol]['pnl']   += t.pnl
            sym_map[t.symbol]['count'] += 1
            sym_map[t.symbol]['wins']  += int(t.pnl > 0)
        top_symbols = sorted(sym_map.items(), key=lambda x: x[1]['count'], reverse=True)[:6]
        symbol_data = [
            {
                'symbol':   sym,
                'pnl':      round(s['pnl'], 2),
                'win_rate': round(s['wins'] / s['count'] * 100, 1) if s['count'] else 0,
                'count':    s['count'],
            }
            for sym, s in top_symbols
        ]

        # --- Trading Hours Performance (0-23) ---
        hour_map = defaultdict(lambda: {'pnl': 0.0, 'count': 0, 'wins': 0})
        for t in closed:
            if t.entry_time:
                h = t.entry_time.hour
                hour_map[h]['pnl']   += t.pnl
                hour_map[h]['count'] += 1
                hour_map[h]['wins']  += int(t.pnl > 0)
        trading_hours_data = [
            {
                'hour':     h,
                'pnl':      round(hour_map[h]['pnl'], 2),
                'count':    hour_map[h]['count'],
                'win_rate': round(hour_map[h]['wins'] / hour_map[h]['count'] * 100, 1) if hour_map[h]['count'] else 0,
            }
            for h in range(24)
        ]
        
        # Get current account balance
        current_balance = account_balances[-1].balance if account_balances else 0
        
        # Calculate risk metrics (ADDED THIS)
        risk_metrics = TradeAnalyzer.calculate_risk_metrics(trades, account_balances)
        
        # Calculate account metrics (ADDED THIS)
        account_metrics = TradeAnalyzer.calculate_account_metrics(trades, account_balances)

         #Get AI insights
        ai_engine = ForexAI(current_user.id)
        ai_insights = ai_engine.get_personalized_insights(5)
        
        return render_template('dashboard/index.html', 
                             metrics=metrics, 
                             equity_data=equity_data,
                             account_data=account_data,
                             current_balance=current_balance,
                             recent_trades=recent_trades,
                             risk_metrics=risk_metrics,  
                             account_metrics=account_metrics,
                             ai_insights=ai_insights,
                             win_loss_data=win_loss_data,
                             symbol_data=symbol_data,
                             trading_hours_data=trading_hours_data,
                            )

    
    except Exception as e:
        # Fallback in case of any error
        print(f"Dashboard error: {e}")
        # Provide default risk_metrics (ADDED THIS)
        default_risk_metrics = {
            'avg_risk_per_trade': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'risk_reward_ratio': 0,
            'expectancy': 0
        }
        default_account_metrics = {
            'current_balance': 0,
            'initial_balance': 0,
            'net_profit': 0,
            'total_return': 0
        }
        
        return render_template('dashboard/index.html', 
                             metrics=TradeAnalyzer.calculate_metrics([]), 
                             equity_data=[],
                             account_data=[],
                             current_balance=0,
                             recent_trades=[],
                             risk_metrics=default_risk_metrics, 
                             account_metrics=default_account_metrics)  

