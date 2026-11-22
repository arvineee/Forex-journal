from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from app import db
from app.account import account_bp
from app.models import AccountBalance, Trade
from app.utils.timezones import valid_timezone, DEFAULT_TIMEZONE, local_to_utc_naive, to_user_timezone


def auto_update_balance_on_close(trade):
    """
    Called whenever a trade is closed (status set to 'closed' with a pnl).
    Finds the latest account balance and adds the trade PnL to create a new record.
    Only runs if pnl is not None.
    """
    if trade.pnl is None:
        return

    latest = (
        AccountBalance.query
        .filter_by(user_id=current_user.id)
        .order_by(AccountBalance.date.desc())
        .first()
    )

    if not latest:
        return  # No starting balance recorded — skip auto-update

    new_balance_amount = round(latest.balance + trade.pnl, 2)
    direction = "profit" if trade.pnl >= 0 else "loss"

    new_balance = AccountBalance(
        balance=new_balance_amount,
        date=trade.exit_time or datetime.utcnow(),
        notes=(
            f"Auto-updated after closing {trade.symbol} {trade.direction} "
            f"({direction}: ${trade.pnl:+.2f}). Trade #{trade.id}."
        ),
        user_id=current_user.id
    )
    db.session.add(new_balance)
    # Caller must commit after this


@account_bp.route('/balance')
@login_required
def balance_history():
    balances = AccountBalance.query.filter_by(user_id=current_user.id)\
        .order_by(AccountBalance.date.desc())\
        .all()
    return render_template('account/balance_history.html', balances=balances)


@account_bp.route('/balance/add', methods=['GET', 'POST'])
@login_required
def add_balance():
    from app.account.forms import AccountBalanceForm
    form = AccountBalanceForm()
    if form.validate_on_submit():
        balance = AccountBalance(
            balance=form.balance.data,
            date=local_to_utc_naive(form.date.data, current_user.timezone),
            notes=form.notes.data,
            user_id=current_user.id
        )
        db.session.add(balance)
        db.session.commit()
        flash('Account balance recorded successfully!', 'success')
        return redirect(url_for('account.balance_history'))

    if request.method == 'GET':
        form.date.data = to_user_timezone(datetime.utcnow(), current_user.timezone).replace(tzinfo=None)
    return render_template('account/add_balance.html', form=form)


@account_bp.route('/balance/<int:balance_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_balance(balance_id):
    from app.account.forms import AccountBalanceForm
    balance = AccountBalance.query.filter_by(
        id=balance_id, user_id=current_user.id
    ).first_or_404()

    form = AccountBalanceForm()

    if form.validate_on_submit():
        balance.balance = form.balance.data
        balance.date = local_to_utc_naive(form.date.data, current_user.timezone)
        balance.notes = form.notes.data
        db.session.commit()
        flash('Account balance updated successfully!', 'success')
        return redirect(url_for('account.balance_history'))

    elif request.method == 'GET':
        form.balance.data = balance.balance
        form.date.data = to_user_timezone(balance.date, current_user.timezone).replace(tzinfo=None)
        form.notes.data = balance.notes

    return render_template('account/edit_balance.html', form=form, balance=balance)


@account_bp.route('/balance/<int:balance_id>/delete', methods=['POST'])
@login_required
def delete_balance(balance_id):
    balance = AccountBalance.query.filter_by(
        id=balance_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(balance)
    db.session.commit()
    flash('Account balance deleted successfully!', 'success')
    return redirect(url_for('account.balance_history'))


@account_bp.route('/balance/current')
@login_required
def get_current_balance():
    """Get the most recent account balance."""
    latest_balance = AccountBalance.query.filter_by(user_id=current_user.id)\
        .order_by(AccountBalance.date.desc())\
        .first()

    if latest_balance:
        return jsonify({
            'balance': latest_balance.balance,
            'date': latest_balance.date.isoformat(),
            'notes': latest_balance.notes
        })
    else:
        return jsonify({'balance': None, 'message': 'No balance records found'})



@account_bp.route('/timezone', methods=['GET', 'POST'])
@login_required
def timezone_settings():
    from app.utils.timezones import COMMON_TIMEZONES
    if request.method == 'POST':
        selected = (request.form.get('timezone') or '').strip()
        if not valid_timezone(selected):
            flash('Please choose a valid timezone.', 'danger')
        else:
            current_user.timezone = selected
            db.session.commit()
            flash('Your timezone preference has been saved.', 'success')
            return redirect(url_for('account.timezone_settings'))
    return render_template('account/timezone.html', timezones=COMMON_TIMEZONES)


@account_bp.route('/timezone/detect', methods=['POST'])
@login_required
def detect_timezone():
    selected = (request.get_json(silent=True) or {}).get('timezone', '')
    if not valid_timezone(selected):
        return jsonify({'success': False, 'message': 'Invalid timezone.'}), 400
    current_user.timezone = selected
    db.session.commit()
    return jsonify({'success': True, 'timezone': selected})
