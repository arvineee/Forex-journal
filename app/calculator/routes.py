from flask import render_template
from flask_login import current_user, login_required
from app.calculator import calculator_bp
from app.models import AccountBalance


@calculator_bp.route('/')
@login_required
def index():
    """Advanced forex calculator: position size, pip value, risk/reward, margin, swap."""
    latest_balance = None
    try:
        latest = (
            AccountBalance.query
            .filter_by(user_id=current_user.id)
            .order_by(AccountBalance.date.desc())
            .first()
        )
        if latest:
            latest_balance = latest.balance
    except Exception as e:
        print(f"Error loading latest balance for calculator: {e}")

    return render_template('calculator/calculator.html', latest_balance=latest_balance)

