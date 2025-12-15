# Create new file: app/account/routes.py
from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime
from app import db
from app.account import account_bp
from app.models import AccountBalance
from app.account.forms import AccountBalanceForm

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
    form = AccountBalanceForm()
    if form.validate_on_submit():
        balance = AccountBalance(
            balance=form.balance.data,
            date=form.date.data,
            notes=form.notes.data,
            user_id=current_user.id
        )
        
        db.session.add(balance)
        db.session.commit()
        
        flash('Account balance recorded successfully!', 'success')
        return redirect(url_for('account.balance_history'))
    
    return render_template('account/add_balance.html', form=form)

@account_bp.route('/balance/<int:balance_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_balance(balance_id):
    balance = AccountBalance.query.filter_by(id=balance_id, user_id=current_user.id).first_or_404()
    form = AccountBalanceForm()
    
    if form.validate_on_submit():
        balance.balance = form.balance.data
        balance.date = form.date.data
        balance.notes = form.notes.data
        
        db.session.commit()
        flash('Account balance updated successfully!', 'success')
        return redirect(url_for('account.balance_history'))
    
    elif request.method == 'GET':
        form.balance.data = balance.balance
        form.date.data = balance.date
        form.notes.data = balance.notes
    
    return render_template('account/edit_balance.html', form=form, balance=balance)

@account_bp.route('/balance/<int:balance_id>/delete', methods=['POST'])
@login_required
def delete_balance(balance_id):
    balance = AccountBalance.query.filter_by(id=balance_id, user_id=current_user.id).first_or_404()
    
    db.session.delete(balance)
    db.session.commit()
    
    flash('Account balance deleted successfully!', 'success')
    return redirect(url_for('account.balance_history'))

@account_bp.route('/balance/current')
@login_required
def get_current_balance():
    """Get the most recent account balance"""
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
