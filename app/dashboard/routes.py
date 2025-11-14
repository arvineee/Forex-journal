from flask import render_template, jsonify
from flask_login import login_required, current_user
from app.dashboard import dashboard_bp
from app.models import Trade
from app.utils.analysis import TradeAnalyzer

@dashboard_bp.route('/')
@login_required
def index():
    try:
        # Get all trades for the current user
        trades = Trade.query.filter_by(user_id=current_user.id).all()
        
        # Calculate metrics
        metrics = TradeAnalyzer.calculate_metrics(trades)
        
        # Generate equity curve data
        equity_data = TradeAnalyzer.generate_equity_data(trades)
        
        # Get recent trades
        recent_trades = TradeAnalyzer.get_recent_trades(trades, 5)
        
        return render_template('dashboard/index.html', 
                             metrics=metrics, 
                             equity_data=equity_data,
                             recent_trades=recent_trades)
    
    except Exception as e:
        # Fallback in case of any error
        print(f"Dashboard error: {e}")
        return render_template('dashboard/index.html', 
                             metrics=TradeAnalyzer.calculate_metrics([]), 
                             equity_data=[],
                             recent_trades=[])

@dashboard_bp.route('/metrics')
@login_required
def metrics():
    try:
        trades = Trade.query.filter_by(user_id=current_user.id).all()
        metrics = TradeAnalyzer.calculate_metrics(trades)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
