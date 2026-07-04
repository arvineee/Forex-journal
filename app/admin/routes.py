from flask import render_template
from flask_login import login_required
from app.admin import admin_bp
from app.admin.decorators import admin_required
from app.models import User, LoginActivity

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    all_users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=all_users)

@admin_bp.route('/users/<int:user_id>')
@login_required
@admin_required
def user_detail(user_id):
    user = User.query.get_or_404(user_id)
    logins = user.login_activities.order_by(LoginActivity.login_at.desc()).limit(50).all()
    trade_count = user.trades.count()
    return render_template('admin/user_detail.html', user=user, logins=logins, trade_count=trade_count)

