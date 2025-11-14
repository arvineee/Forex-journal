from flask import render_template, redirect, url_for, flash, request, jsonify, send_file
from flask_login import login_required, current_user
import os
import json
from datetime import datetime
from app import db
from app.trades import trades_bp
from app.models import Trade, TradeImage
from app.trades.forms import TradeForm
from app.utils.helpers import save_image, allowed_file, calculate_pnl
import csv
import io

@trades_bp.route('/')
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    trades = Trade.query.filter_by(user_id=current_user.id)\
        .order_by(Trade.entry_time.desc())\
        .paginate(page=page, per_page=10)
    return render_template('trades/index.html', trades=trades)

@trades_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_trade():
    form = TradeForm()
    if form.validate_on_submit():
        trade = Trade(
            symbol=form.symbol.data.upper(),
            direction=form.direction.data,
            entry_price=form.entry_price.data,
            exit_price=form.exit_price.data,
            size=form.size.data,
            entry_time=form.entry_time.data,
            exit_time=form.exit_time.data,
            strategy=form.strategy.data,
            timeframe=form.timeframe.data,
            notes=form.notes.data,
            emotions=form.emotions.data,
            mistakes=form.mistakes.data,
            rating=form.rating.data,
            user_id=current_user.id
        )
        
        # Calculate P&L if exit price exists
        if trade.exit_price:
            trade.status = 'closed'
            pnl, pnl_percent = calculate_pnl(trade.direction, trade.entry_price, trade.exit_price, trade.size, trade.symbol)
            trade.pnl = pnl
            trade.pnl_percent = pnl_percent
        
        db.session.add(trade)
        db.session.commit()
        
        # Handle image uploads
        if form.images.data:
            for image in form.images.data:
                if image and allowed_file(image.filename):
                    filename = save_image(image)
                    trade_image = TradeImage(
                        filename=filename,
                        trade_id=trade.id
                    )
                    db.session.add(trade_image)
            
            db.session.commit()
        
        flash('Trade added successfully!', 'success')
        return redirect(url_for('trades.index'))
    
    return render_template('trades/new_trade.html', form=form)

@trades_bp.route('/trade/<int:trade_id>')
@login_required
def view_trade(trade_id):
    trade = Trade.query.filter_by(id=trade_id, user_id=current_user.id).first_or_404()
    return render_template('trades/view_trade.html', trade=trade)

@trades_bp.route('/trade/<int:trade_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_trade(trade_id):
    trade = Trade.query.filter_by(id=trade_id, user_id=current_user.id).first_or_404()
    form = TradeForm()
    
    if form.validate_on_submit():
        # Update trade with form data
        trade.symbol = form.symbol.data.upper()
        trade.direction = form.direction.data
        trade.entry_price = form.entry_price.data
        trade.exit_price = form.exit_price.data
        trade.size = form.size.data
        trade.entry_time = form.entry_time.data
        trade.exit_time = form.exit_time.data
        trade.strategy = form.strategy.data
        trade.timeframe = form.timeframe.data
        trade.notes = form.notes.data
        trade.emotions = form.emotions.data
        trade.mistakes = form.mistakes.data
        trade.rating = form.rating.data
        
        # Update P&L if exit price exists
        if trade.exit_price:
            trade.status = 'closed'
            pnl, pnl_percent = calculate_pnl(trade.direction, trade.entry_price, trade.exit_price, trade.size, trade.symbol)
            trade.pnl = pnl
            trade.pnl_percent = pnl_percent
        else:
            trade.status = 'open'
            trade.pnl = None
            trade.pnl_percent = None
        
        # Handle new image uploads
        if form.images.data:
            for image in form.images.data:
                if image and allowed_file(image.filename):
                    filename = save_image(image)
                    trade_image = TradeImage(
                        filename=filename,
                        trade_id=trade.id
                    )
                    db.session.add(trade_image)
        
        db.session.commit()
        flash('Trade updated successfully!', 'success')
        return redirect(url_for('trades.view_trade', trade_id=trade.id))
    
    elif request.method == 'GET':
        # Pre-populate form with existing trade data
        form.symbol.data = trade.symbol
        form.direction.data = trade.direction
        form.entry_price.data = trade.entry_price
        form.exit_price.data = trade.exit_price
        form.size.data = trade.size
        form.entry_time.data = trade.entry_time
        form.exit_time.data = trade.exit_time
        form.strategy.data = trade.strategy
        form.timeframe.data = trade.timeframe
        form.notes.data = trade.notes
        form.emotions.data = trade.emotions
        form.mistakes.data = trade.mistakes
        form.rating.data = trade.rating
    
    return render_template('trades/edit_trade.html', form=form, trade=trade)

@trades_bp.route('/trade/<int:trade_id>/delete', methods=['POST'])
@login_required
def delete_trade(trade_id):
    trade = Trade.query.filter_by(id=trade_id, user_id=current_user.id).first_or_404()
    
    # Delete associated images first
    for image in trade.images:
        # Delete image file from filesystem
        try:
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image.filename)
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image file: {e}")
    
    db.session.delete(trade)
    db.session.commit()
    
    flash('Trade deleted successfully!', 'success')
    return redirect(url_for('trades.index'))

@trades_bp.route('/export')
@login_required
def export_trades():
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Symbol', 'Direction', 'Entry Price', 'Exit Price', 'Size', 
                    'P&L', 'P&L %', 'Entry Time', 'Exit Time', 'Status', 
                    'Strategy', 'Timeframe', 'Notes', 'Emotions', 'Mistakes', 'Rating'])
    
    # Write data
    for trade in trades:
        writer.writerow([
            trade.symbol, trade.direction, trade.entry_price, trade.exit_price,
            trade.size, trade.pnl, trade.pnl_percent, trade.entry_time,
            trade.exit_time, trade.status, trade.strategy, trade.timeframe,
            trade.notes, trade.emotions, trade.mistakes, trade.rating
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'forex_trades_{datetime.now().strftime("%Y%m%d")}.csv'
    )
