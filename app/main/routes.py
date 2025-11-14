from flask import render_template, Blueprint
from flask_login import current_user
from app.models import Trade
from app.utils.analysis import TradeAnalyzer

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    # If user is logged in, show their metrics on the homepage
    if current_user.is_authenticated:
        try:
            trades = Trade.query.filter_by(user_id=current_user.id).all()
            metrics = TradeAnalyzer.calculate_metrics(trades)
            return render_template('index.html', metrics=metrics)
        except Exception as e:
            print(f"Error loading metrics for homepage: {e}")
            # Fallback: show empty metrics
            metrics = TradeAnalyzer.calculate_metrics([])
            return render_template('index.html', metrics=metrics)
    else:
        # For non-logged in users, don't show metrics
        return render_template('index.html', metrics=None)
