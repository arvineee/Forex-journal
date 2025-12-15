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
                             ai_insights=ai_insights
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
