from flask import render_template, redirect, url_for, flash, request, jsonify, send_file, current_app
from flask_login import login_required, current_user
import os
import json
from datetime import datetime
from app import db
from app.trades import trades_bp
from app.models import Trade, TradeImage
from app.trades.forms import TradeForm
from app.utils.helpers import save_image, allowed_file, calculate_pnl
from app.ai.learning_engine import ForexAI
import csv
import io
import numpy as np
from datetime import datetime, timedelta
import json
import pandas as pd  
from app.trades.forms import TradeForm, ExportForm

@trades_bp.route('/')
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    
    # Get filter parameters
    symbol = request.args.get('symbol', '')
    status = request.args.get('status', '')
    strategy = request.args.get('strategy', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Build query
    query = Trade.query.filter_by(user_id=current_user.id)
    
    # Apply filters
    if symbol:
        query = query.filter(Trade.symbol.ilike(f'%{symbol}%'))
    if status:
        query = query.filter_by(status=status)
    if strategy:
        query = query.filter(Trade.strategy.ilike(f'%{strategy}%'))
    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Trade.entry_time >= date_from_obj)
        except ValueError:
            pass
    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d')
            query = query.filter(Trade.entry_time <= date_to_obj)
        except ValueError:
            pass
    
    # Order and paginate
    trades = query.order_by(Trade.entry_time.desc())\
                 .paginate(page=page, per_page=20, error_out=False)
    
    # Get unique values for filter dropdowns
    symbols = db.session.query(Trade.symbol).filter_by(user_id=current_user.id)\
                   .distinct().order_by(Trade.symbol).all()
    symbols = [s[0] for s in symbols]
    
    strategies = db.session.query(Trade.strategy).filter_by(user_id=current_user.id)\
                    .filter(Trade.strategy.isnot(None)).distinct().all()
    strategies = [s[0] for s in strategies if s[0]]
    
    # Trigger AI learning on page load (if needed)
    try:
        ai_engine = ForexAI(current_user.id)
        ai_engine.load_data()
        if len(ai_engine.trades) > 0 and len(ai_engine.trades) % 10 == 0:
            # Learn every 10 trades
            ai_engine.analyze_and_learn()
    except Exception as e:
        print(f"AI learning error: {e}")
    
    return render_template('trades/index.html', 
                         trades=trades, 
                         symbols=symbols,
                         strategies=strategies,
                         current_filters={
                             'symbol': symbol,
                             'status': status,
                             'strategy': strategy,
                             'date_from': date_from,
                             'date_to': date_to
                         })

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
            pnl, pnl_percent = calculate_pnl(trade.direction, trade.entry_price, 
                                           trade.exit_price, trade.size, trade.symbol)
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
        
        # Trigger AI learning
        try:
            ai_engine = ForexAI(current_user.id)
            learned_insights = ai_engine.analyze_and_learn(new_trade=trade)
            
            # Show AI insight if available
            if learned_insights:
                # Get most relevant insight
                for key, insight in learned_insights.items():
                    if 'trade_analysis' in insight.get('type', ''):
                        flash(f'🤖 AI Insight: {insight.get("value", "")}', 'info')
                        break
        except Exception as e:
            print(f"AI learning error: {e}")
        
        flash('Trade added successfully!', 'success')
        return redirect(url_for('trades.index'))
    
    # Pre-fill form with current time
    form.entry_time.data = datetime.now()
    
    # Get AI prediction for current setup (if available)
    ai_prediction = None
    try:
        if request.method == 'GET' and current_user.is_authenticated:
            # Check if there's a recent similar trade
            recent_trades = Trade.query.filter_by(user_id=current_user.id)\
                .order_by(Trade.entry_time.desc()).limit(5).all()
            
            if recent_trades:
                ai_engine = ForexAI(current_user.id)
                ai_engine.load_data()
                
                # Get potential insights for the current symbol if user frequently trades it
                symbol_stats = {}
                for trade in ai_engine.trades:
                    if trade.symbol not in symbol_stats:
                        symbol_stats[trade.symbol] = {'count': 0, 'wins': 0}
                    symbol_stats[trade.symbol]['count'] += 1
                    if trade.pnl and trade.pnl > 0:
                        symbol_stats[trade.symbol]['wins'] += 1
                
                # Find most traded symbol
                if symbol_stats:
                    most_traded = max(symbol_stats.items(), key=lambda x: x[1]['count'])
                    if most_traded[1]['count'] >= 5:
                        win_rate = (most_traded[1]['wins'] / most_traded[1]['count']) * 100
                        if win_rate > 70:
                            ai_prediction = f"Your best performing pair: {most_traded[0]} ({win_rate:.1f}% win rate)"
                        elif win_rate < 30:
                            ai_prediction = f"Consider reviewing your {most_traded[0]} trades ({win_rate:.1f}% win rate)"
    except Exception as e:
        print(f"AI prediction error: {e}")
    
    return render_template('trades/new_trade.html', form=form, ai_prediction=ai_prediction)

@trades_bp.route('/trade/<int:trade_id>')
@login_required
def view_trade(trade_id):
    trade = Trade.query.filter_by(id=trade_id, user_id=current_user.id).first_or_404()
    
    # Get AI insights for this specific trade
    ai_insights = []
    try:
        ai_engine = ForexAI(current_user.id)
        ai_engine.load_data()
        
        # Find similar historical trades
        similar_trades = [
            t for t in ai_engine.trades 
            if t.id != trade.id 
            and t.symbol == trade.symbol 
            and t.direction == trade.direction
            and t.strategy == trade.strategy
            and t.pnl is not None
        ]
        
        if similar_trades:
            win_count = len([t for t in similar_trades if t.pnl > 0])
            win_rate = (win_count / len(similar_trades)) * 100
            avg_pnl = np.mean([t.pnl for t in similar_trades])
            
            ai_insights.append({
                'type': 'historical_pattern',
                'title': f'Historical Performance ({len(similar_trades)} similar trades)',
                'value': f'{win_rate:.1f}% win rate, Avg P&L: ${avg_pnl:.2f}',
                'recommendation': 'High win rate pattern detected' if win_rate > 70 else 
                                'Review this setup carefully' if win_rate < 30 else 
                                'Neutral historical performance'
            })
        
        # Check for emotional patterns
        if trade.emotions:
            emotion_trades = [t for t in ai_engine.trades 
                            if t.emotions == trade.emotions and t.pnl is not None]
            if len(emotion_trades) >= 3:
                emotion_win_rate = len([t for t in emotion_trades if t.pnl > 0]) / len(emotion_trades) * 100
                ai_insights.append({
                    'type': 'emotion_impact',
                    'title': f'Impact of {trade.emotions} emotion',
                    'value': f'{emotion_win_rate:.1f}% win rate with this emotion',
                    'recommendation': 'This emotional state works well for you!' if emotion_win_rate > 60 else
                                    'Consider mindfulness when feeling this way' if emotion_win_rate < 40 else
                                    'Neutral impact from this emotion'
                })
    except Exception as e:
        print(f"AI insights error: {e}")
    
    return render_template('trades/view_trade.html', trade=trade, ai_insights=ai_insights)

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
            pnl, pnl_percent = calculate_pnl(trade.direction, trade.entry_price, 
                                           trade.exit_price, trade.size, trade.symbol)
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
        
        # Trigger AI learning with updated trade
        try:
            ai_engine = ForexAI(current_user.id)
            ai_engine.analyze_and_learn(new_trade=trade)
        except Exception as e:
            print(f"AI learning error: {e}")
        
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
    
    # Get trade info for logging
    trade_info = f"{trade.symbol} {trade.direction} at {trade.entry_price}"
    
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
    
    # Trigger AI re-analysis (since data changed)
    try:
        ai_engine = ForexAI(current_user.id)
        ai_engine.analyze_and_learn()
    except Exception as e:
        print(f"AI re-analysis error after deletion: {e}")
    
    flash(f'Trade {trade_info} deleted successfully!', 'success')
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


@trades_bp.route('/ai/predict', methods=['POST'])
@login_required
def predict_trade():
    """Predict outcome of a potential trade using AI"""
    data = request.get_json()
    
    try:
        ai_engine = ForexAI(current_user.id)
        
        # Create temporary trade object for analysis
        from app.models import Trade
        temp_trade = Trade(
            symbol=data.get('symbol', '').upper(),
            direction=data.get('direction', ''),
            entry_price=float(data.get('entry_price', 0)),
            size=float(data.get('size', 0.01)),
            strategy=data.get('strategy', ''),
            timeframe=data.get('timeframe', ''),
            emotions=data.get('emotions', 'neutral'),
            user_id=current_user.id
        )
        
        # Analyze this specific trade setup
        ai_engine.load_data()
        insights = ai_engine._analyze_new_trade(temp_trade)
        
        # Get similar historical trades
        similar_trades = [
            t for t in ai_engine.trades 
            if t.symbol == temp_trade.symbol 
            and t.direction == temp_trade.direction
            and t.strategy == temp_trade.strategy
            and t.pnl is not None
        ]
        
        prediction = {
            'similar_trades_count': len(similar_trades),
            'insights': list(insights.values()) if insights else [],
            'recommendation': 'No strong historical pattern detected'
        }
        
        if similar_trades:
            win_count = len([t for t in similar_trades if t.pnl > 0])
            win_rate = (win_count / len(similar_trades)) * 100
            avg_pnl = np.mean([t.pnl for t in similar_trades])
            
            prediction.update({
                'historical_win_rate': win_rate,
                'historical_avg_pnl': avg_pnl,
                'recommendation': _get_trade_recommendation(win_rate, avg_pnl)
            })
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _get_trade_recommendation(win_rate, avg_pnl):
    """Get recommendation based on historical data"""
    if win_rate > 70 and avg_pnl > 10:
        return "Strong historical pattern - Consider taking this trade with normal position size"
    elif win_rate > 60 and avg_pnl > 5:
        return "Good historical performance - Consider taking with normal position size"
    elif win_rate < 40 or avg_pnl < -5:
        return "Weak historical pattern - Consider reducing position size or skipping"
    elif win_rate < 30 or avg_pnl < -10:
        return "Poor historical pattern - Consider avoiding this setup"
    else:
        return "Neutral historical pattern - Trade based on current market conditions"


@trades_bp.route('/advanced_export', methods=['GET', 'POST'])
@login_required
def advanced_export():
    """Advanced export options page"""
    form = ExportForm()
    
    # Set default dates if not already set
    if not form.start_date.data:
        form.start_date.data = datetime.now() - timedelta(days=30)
    if not form.end_date.data:
        form.end_date.data = datetime.now()
    
    if form.validate_on_submit():
        try:
            # Get form data
            export_format = form.format.data
            date_range = form.date_range.data
            include_images = form.include_images.data
            include_annotations = form.include_annotations.data
            include_summary = form.include_summary.data
            start_date = form.start_date.data
            end_date = form.end_date.data
            
            # Filter trades based on date range
            base_query = Trade.query.filter_by(user_id=current_user.id)
            
            # Apply date filter
            if date_range != 'all':
                today = datetime.utcnow()
                if date_range == 'today':
                    start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
                    base_query = base_query.filter(Trade.entry_time >= start_date)
                elif date_range == 'week':
                    start_date = today - timedelta(days=7)
                    base_query = base_query.filter(Trade.entry_time >= start_date)
                elif date_range == 'month':
                    start_date = today - timedelta(days=30)
                    base_query = base_query.filter(Trade.entry_time >= start_date)
                elif date_range == 'quarter':
                    start_date = today - timedelta(days=90)
                    base_query = base_query.filter(Trade.entry_time >= start_date)
                elif date_range == 'year':
                    start_date = today - timedelta(days=365)
                    base_query = base_query.filter(Trade.entry_time >= start_date)
                elif date_range == 'custom':
                    # Use custom dates from form
                    if start_date:
                        start_date_obj = datetime.combine(start_date, datetime.min.time())
                        base_query = base_query.filter(Trade.entry_time >= start_date_obj)
                    if end_date:
                        end_date_obj = datetime.combine(end_date, datetime.max.time())
                        base_query = base_query.filter(Trade.entry_time <= end_date_obj)
            
            # Get filtered trades
            trades = base_query.order_by(Trade.entry_time.desc()).all()
            
            if not trades:
                flash('No trades found for the selected criteria.', 'warning')
                return redirect(url_for('trades.advanced_export'))
            
            # Export based on format
            if export_format == 'csv':
                return export_trades_csv(trades, include_images, include_annotations, include_summary)
            elif export_format == 'excel':
                return export_trades_excel(trades, include_images, include_annotations, include_summary)
            elif export_format == 'json':
                return export_trades_json(trades, include_images, include_annotations, include_summary)
            elif export_format == 'pdf':
                return export_trades_pdf(trades, include_images, include_annotations, include_summary)
            else:
                flash('Unsupported export format.', 'error')
                return redirect(url_for('trades.advanced_export'))
                
        except Exception as e:
            flash(f'Export error: {str(e)}', 'error')
            current_app.logger.error(f"Export error: {e}")
            return redirect(url_for('trades.advanced_export'))
    
    return render_template('trades/advanced_export.html', form=form)

def export_trades_csv(trades, include_images=False, include_annotations=False, include_summary=False):
    """Export trades to CSV format"""

    output = io.StringIO()

    # Write header
    headers = ['ID', 'Symbol', 'Direction', 'Entry Price', 'Exit Price', 'Size',
               'P&L', 'P&L %', 'Entry Time', 'Exit Time', 'Status', 'Strategy',
               'Timeframe', 'Notes', 'Emotions', 'Mistakes', 'Rating', 'Created At']

    if include_images:
        headers.extend(['Image Count'])

    if include_annotations:
        headers.extend(['Annotations'])

    output.write(','.join(headers) + '\n')

    # Write data
    for trade in trades:
        row = [
            str(trade.id),
            trade.symbol,
            trade.direction,
            str(trade.entry_price),
            str(trade.exit_price) if trade.exit_price else '',
            str(trade.size),
            str(trade.pnl) if trade.pnl else '',
            str(trade.pnl_percent) if trade.pnl_percent else '',
            trade.entry_time.isoformat() if trade.entry_time else '',
            trade.exit_time.isoformat() if trade.exit_time else '',
            trade.status,
            trade.strategy or '',
            trade.timeframe or '',
            f'"{trade.notes}"' if trade.notes else '',  # Quote notes to handle commas
            trade.emotions or '',
            f'"{trade.mistakes}"' if trade.mistakes else '',  # Quote mistakes
            str(trade.rating) if trade.rating else '',
            trade.created_at.isoformat() if trade.created_at else ''
        ]

        if include_images:
            image_count = trade.images.count()
            row.append(str(image_count))

        if include_annotations:
            annotations = []
            for image in trade.images:
                if image.annotation_data:
                    annotations.append(image.annotation_data)
            row.append(f'"{", ".join(annotations)}"' if annotations else '')

        output.write(','.join(row) + '\n')

    output.seek(0)

    # Create response
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'forex_trades_advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

def export_trades_excel(trades, include_images=False, include_annotations=False, include_summary=False):
    """Export trades to Excel format"""
    try:
        # Try to import pandas for Excel export
        try:
            import pandas as pd
        except ImportError:
            flash('Excel export requires pandas. Please install: pip install pandas', 'error')
            return redirect(url_for('trades.advanced_export'))

        # Prepare data
        data = []
        for trade in trades:
            trade_data = {
                'ID': trade.id,
                'Symbol': trade.symbol,
                'Direction': trade.direction,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price if trade.exit_price else None,
                'Size': trade.size,
                'P&L': trade.pnl,
                'P&L %': trade.pnl_percent,
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'Status': trade.status,
                'Strategy': trade.strategy or '',
                'Timeframe': trade.timeframe or '',
                'Notes': trade.notes or '',
                'Emotions': trade.emotions or '',
                'Mistakes': trade.mistakes or '',
                'Rating': trade.rating,
                'Created At': trade.created_at
            }

            if include_images:
                trade_data['Image Count'] = trade.images.count()

            if include_annotations:
                annotations = []
                for image in trade.images:
                    if image.annotation_data:
                        annotations.append(image.annotation_data)
                trade_data['Annotations'] = ', '.join(annotations) if annotations else ''

            data.append(trade_data)

        df = pd.DataFrame(data)

        # Create Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Trades', index=False)

            if include_summary and len(trades) > 0:
                # Create summary sheet
                summary_data = calculate_summary_statistics(trades)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

        output.seek(0)

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'forex_trades_advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

    except Exception as e:
        flash(f'Excel export error: {str(e)}', 'error')
        return redirect(url_for('trades.advanced_export'))

def export_trades_json(trades, include_images=False, include_annotations=False, include_summary=False):
    """Export trades to JSON format"""

    data = {
        'export_date': datetime.now().isoformat(),
        'trade_count': len(trades),
        'trades': []
    }

    for trade in trades:
        trade_data = {
            'id': trade.id,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_price': float(trade.entry_price) if trade.entry_price else None,
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'size': float(trade.size) if trade.size else None,
            'pnl': float(trade.pnl) if trade.pnl else None,
            'pnl_percent': float(trade.pnl_percent) if trade.pnl_percent else None,
            'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'status': trade.status,
            'strategy': trade.strategy,
            'timeframe': trade.timeframe,
            'notes': trade.notes,
            'emotions': trade.emotions,
            'mistakes': trade.mistakes,
            'rating': trade.rating,
            'created_at': trade.created_at.isoformat() if trade.created_at else None
        }

        if include_images:
            trade_data['images'] = [
                {
                    'filename': image.filename,
                    'image_type': image.image_type,
                    'created_at': image.created_at.isoformat() if image.created_at else None
                }
                for image in trade.images
            ]

        if include_annotations:
            trade_data['annotations'] = [
                {
                    'filename': image.filename,
                    'annotation_data': image.annotation_data
                }
                for image in trade.images if image.annotation_data
            ]

        data['trades'].append(trade_data)

    if include_summary:
        data['summary'] = calculate_summary_statistics(trades)

    # Return as downloadable JSON file
    json_str = json.dumps(data, indent=2, default=str)

    return send_file(
        io.BytesIO(json_str.encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name=f'forex_trades_advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

def export_trades_pdf(trades, include_images=False, include_annotations=False, include_summary=False):
    """Export trades to PDF format (simplified - returns CSV if PDF not available)"""
    flash('PDF export is not yet implemented. Downloading as CSV instead.', 'info')
    return export_trades_csv(trades, include_images, include_annotations, include_summary)

def calculate_summary_statistics(trades):
    """Calculate summary statistics for export"""
    if not trades:
        return {}

    closed_trades = [t for t in trades if t.status == 'closed' and t.pnl is not None]

    summary = {
        'total_trades': len(trades),
        'closed_trades': len(closed_trades),
        'open_trades': len([t for t in trades if t.status == 'open']),
        'winning_trades': len([t for t in closed_trades if t.pnl > 0]),
        'losing_trades': len([t for t in closed_trades if t.pnl < 0]),
        'breakeven_trades': len([t for t in closed_trades if t.pnl == 0]),
    }

    if closed_trades:
        summary['total_pnl'] = sum(t.pnl for t in closed_trades)
        summary['average_pnl'] = sum(t.pnl for t in closed_trades) / len(closed_trades)
        summary['win_rate'] = (summary['winning_trades'] / len(closed_trades)) * 100

        winning_pnl = [t.pnl for t in closed_trades if t.pnl > 0]
        losing_pnl = [t.pnl for t in closed_trades if t.pnl < 0]

        if winning_pnl:
            summary['average_win'] = sum(winning_pnl) / len(winning_pnl)
            summary['largest_win'] = max(winning_pnl)

        if losing_pnl:
            summary['average_loss'] = sum(losing_pnl) / len(losing_pnl)
            summary['largest_loss'] = min(losing_pnl)

        if losing_pnl and winning_pnl:
            if sum(losing_pnl) != 0:
                summary['profit_factor'] = abs(sum(winning_pnl) / sum(losing_pnl))
            else:
                summary['profit_factor'] = float('inf')

    return summary
