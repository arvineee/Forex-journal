"""
AI Routes for Forex Journal
"""
import logging
from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.models import Trade
from . import ai_bp
from .learning_engine import ForexAI
from .chatbot import ForexChatbot
from .models import AIInsight, AILearningLog, AIChartAnalysis
from .chart_intelligence import analyse_four_timeframes, predict_from_chart, learn_chart_outcomes, compare_chart_outcomes, build_smc_playbook

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Dashboard & insights                                               #
# ------------------------------------------------------------------ #

@ai_bp.route('/dashboard')
@login_required
def dashboard():
    """AI Dashboard — show grouped personalized insights."""
    ai_engine = ForexAI(current_user.id)
    insights_data = ai_engine.get_personalized_insights(limit=100)

    grouped = {
        'emotion': [],
        'patterns': [],
        'recommendations': [],
        'risk': [],
        'prediction': [],
    }

    type_map = {
        'emotion_analysis': 'emotion',
        'time_pattern': 'patterns',
        'symbol_analysis': 'patterns',
        'strategy_analysis': 'patterns',
        'behavioral_pattern': 'patterns',
        'market_analysis': 'patterns',
        'recommendation': 'recommendations',
        'risk_analysis': 'risk',
        'risk_management': 'risk',
        'prediction': 'prediction',
        'performance_metrics': 'risk',
    }

    for insight in insights_data:
        bucket = type_map.get(insight.get('type', ''), 'patterns')
        grouped[bucket].append(insight)

    recent_logs = (
        AILearningLog.query
        .filter_by(user_id=current_user.id)
        .order_by(AILearningLog.created_at.desc())
        .limit(10)
        .all()
    )

    return render_template('ai/dashboard.html', insights=grouped, recent_logs=recent_logs)


@ai_bp.route('/analyze', methods=['POST', 'GET'])
@login_required
def analyze():
    """Trigger a full AI analysis of the user's trades."""
    try:
        ai_engine = ForexAI(current_user.id)
        insights = ai_engine.analyze_and_learn(full=True)

        log = AILearningLog(
            user_id=current_user.id,
            event_type='manual_analysis_triggered',
            event_data='{}',
            learned_insights=str(len(insights)),
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Analysis complete — {len(insights)} insights generated.',
            'insight_count': len(insights),
        })
    except Exception as exc:
        logger.error("AI analysis error for user %s: %s", current_user.id, exc)
        return jsonify({'success': False, 'message': 'Analysis failed. Please try again later.'}), 500


@ai_bp.route('/insight/<int:insight_id>', methods=['PUT'])
@login_required
def update_insight(insight_id):
    """Accept/reject insight feedback to adjust confidence scores."""
    data = request.get_json(silent=True) or {}
    feedback = data.get('confidence_feedback')

    if feedback not in ('agree', 'disagree'):
        return jsonify({'success': False, 'message': 'Invalid feedback value'}), 400

    insight = AIInsight.query.get_or_404(insight_id)
    if insight.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorised'}), 403

    if feedback == 'agree':
        insight.confidence_score = min(1.0, (insight.confidence_score or 0.0) + 0.1)
    else:
        insight.confidence_score = max(0.0, (insight.confidence_score or 0.0) - 0.2)
        insight.data_points = max(0, (insight.data_points or 0) - 1)

    db.session.commit()
    return jsonify({'success': True, 'new_confidence': insight.confidence_score})


@ai_bp.route('/insights')
@login_required
def get_insights():
    ai_engine = ForexAI(current_user.id)
    insights = ai_engine.get_personalized_insights(limit=100)
    return jsonify({'success': True, 'insights': insights, 'count': len(insights)})


@ai_bp.route('/patterns')
@login_required
def get_patterns():
    ai_engine = ForexAI(current_user.id)
    ai_engine.load_data()
    return jsonify({
        'success': True,
        'patterns': [
            {
                'id': p.id,
                'type': p.pattern_type,
                'data': p.pattern_data,
                'occurrences': p.occurrence_count,
                'first_detected': p.first_detected.isoformat() if p.first_detected else None,
                'last_detected': p.last_detected.isoformat() if p.last_detected else None,
            }
            for p in ai_engine.patterns.values()
        ],
    })




# ------------------------------------------------------------------ #
# Multi-timeframe chart intelligence                                   #
# ------------------------------------------------------------------ #

@ai_bp.route('/chart-lab', methods=['GET', 'POST'])
@login_required
def chart_lab():
    """Upload 4H/1H/30M/15M charts for a single top-down analysis."""
    result = None
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            record, result = analyse_four_timeframes(
                user_id=current_user.id,
                symbol=request.form.get('symbol', ''),
                direction=request.form.get('direction', ''),
                trade_id=request.form.get('trade_id') or None,
                uploads={tf: request.files.get(tf) for tf in ('4H', '1H', '30M', '15M')},
            )
            prediction = predict_from_chart(current_user.id, result)
            result['analysis_id'] = record.id
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            db.session.rollback()
            logger.exception("Chart analysis failed for user %s", current_user.id)
            error = "Chart analysis failed. Check your Gemini API configuration and try again."

    # Refresh labels when a previously linked journal trade has since closed.
    for chart_row in AIChartAnalysis.query.filter_by(user_id=current_user.id).all():
        if chart_row.trade_id:
            linked_trade = Trade.query.filter_by(id=chart_row.trade_id, user_id=current_user.id).first()
            if linked_trade and linked_trade.pnl is not None:
                chart_row.outcome_label = (1 if linked_trade.pnl > 0 else
                                           0 if linked_trade.pnl < 0 else None)
    db.session.commit()

    recent = (AIChartAnalysis.query.filter_by(user_id=current_user.id)
              .order_by(AIChartAnalysis.created_at.desc()).limit(12).all())
    closed_trades = (Trade.query.filter_by(user_id=current_user.id, status='closed')
                     .order_by(Trade.entry_time.desc()).limit(100).all())
    learning = learn_chart_outcomes(current_user.id)
    if learning.get('ready'):
        learning = learning['metrics']
    outcome_comparison = compare_chart_outcomes(current_user.id)
    return render_template('ai/chart_lab.html', result=result, prediction=prediction,
                           error=error, recent=recent, closed_trades=closed_trades,
                           learning=learning, outcome_comparison=outcome_comparison)


@ai_bp.route('/chart-analysis/<int:analysis_id>/link', methods=['POST'])
@login_required
def link_chart_analysis(analysis_id):
    """Link a chart snapshot to its actual journal trade to create a labelled example."""
    record = AIChartAnalysis.query.filter_by(
        id=analysis_id, user_id=current_user.id
    ).first_or_404()
    trade_id = request.form.get('trade_id', type=int)
    trade = Trade.query.filter_by(id=trade_id, user_id=current_user.id).first()
    if not trade:
        return jsonify({'success': False, 'error': 'Trade not found.'}), 404
    if trade.symbol.upper() != record.symbol.upper():
        return jsonify({'success': False, 'error': 'Trade symbol must match the chart symbol.'}), 400
    record.trade_id = trade.id
    record.outcome_label = (1 if trade.pnl > 0 else 0 if trade.pnl < 0 else None) if trade.pnl is not None else None
    db.session.commit()
    return jsonify({
        'success': True,
        'labelled': record.outcome_label is not None,
        'message': 'Chart linked to the closed trade and outcome recorded.' if record.outcome_label is not None
                   else 'Chart linked. Its outcome will be learned once the trade is closed.'
    })


@ai_bp.route('/chart-learning', methods=['GET'])
@login_required
def chart_learning_status():
    """Refresh outcome labels from linked closed trades and report model readiness."""
    rows = AIChartAnalysis.query.filter_by(user_id=current_user.id).all()
    for row in rows:
        if row.trade_id:
            trade = Trade.query.filter_by(id=row.trade_id, user_id=current_user.id).first()
            if trade and trade.pnl is not None:
                row.outcome_label = 1 if trade.pnl > 0 else 0 if trade.pnl < 0 else None
    db.session.commit()
    status = learn_chart_outcomes(current_user.id)
    if status.get('ready'):
        status = status['metrics']
    return jsonify({'success': True, 'learning': status})


# ------------------------------------------------------------------ #
# Trading Performance Intelligence Centre                             #
# ------------------------------------------------------------------ #

@ai_bp.route('/smc-playbook', methods=['GET'])
@login_required
def smc_playbook_data():
    """Return evidence-based, user-specific SMC condition summaries."""
    return jsonify({'success': True, 'playbook': build_smc_playbook(current_user.id)})


@ai_bp.route('/performance')
@login_required
def performance_centre():
    """Show outcome-based trading statistics and chart-learning diagnostics."""
    from collections import defaultdict
    from statistics import mean

    trades = (Trade.query.filter_by(user_id=current_user.id, status='closed')
              .order_by(Trade.entry_time.asc()).all())
    wins = [t for t in trades if t.pnl is not None and t.pnl > 0]
    losses = [t for t in trades if t.pnl is not None and t.pnl < 0]
    breakeven = [t for t in trades if t.pnl is not None and t.pnl == 0]
    pnl_values = [float(t.pnl) for t in trades if t.pnl is not None]
    gross_profit = sum(float(t.pnl) for t in wins)
    gross_loss = abs(sum(float(t.pnl) for t in losses))

    def rate(part, total):
        return round(100 * len(part) / total, 1) if total else 0.0

    def grouped_stats(key_fn):
        buckets = defaultdict(list)
        for trade in trades:
            if trade.pnl is None:
                continue
            label = key_fn(trade)
            if label:
                buckets[str(label)].append(float(trade.pnl))
        result = []
        for label, values in buckets.items():
            positive = sum(v > 0 for v in values)
            negative = sum(v < 0 for v in values)
            result.append({
                'label': label, 'count': len(values), 'net_pnl': round(sum(values), 2),
                'win_rate': round(100 * positive / len(values), 1),
                'average_pnl': round(mean(values), 2),
                'wins': positive, 'losses': negative,
            })
        return sorted(result, key=lambda x: (x['net_pnl'], x['count']), reverse=True)

    by_symbol = grouped_stats(lambda t: (t.symbol or '').upper())
    by_strategy = grouped_stats(lambda t: (t.strategy or '').strip() or 'Unspecified')
    by_direction = grouped_stats(lambda t: (t.direction or '').upper())
    # Keep this explicitly as recorded entry-hour; do not assume server timezone == London/NY.
    by_entry_hour = grouped_stats(lambda t: f"{t.entry_time.hour:02d}:00–{(t.entry_time.hour + 1) % 24:02d}:00" if t.entry_time else None)

    chart_rows = AIChartAnalysis.query.filter_by(user_id=current_user.id).all()
    labelled_by_trade = {}
    for chart in chart_rows:
        if chart.trade_id:
            linked = Trade.query.filter_by(id=chart.trade_id, user_id=current_user.id, status='closed').first()
            if linked and linked.pnl is not None:
                chart.outcome_label = 1 if linked.pnl > 0 else 0 if linked.pnl < 0 else None
                if chart.outcome_label is not None:
                    labelled_by_trade[linked.id] = chart
    db.session.commit()

    learning = learn_chart_outcomes(current_user.id)
    model_metrics = learning.get('metrics') if learning.get('ready') else None
    learning_status = learning if not learning.get('ready') else None
    comparison = compare_chart_outcomes(current_user.id)
    smc_playbook = build_smc_playbook(current_user.id)

    recent = trades[-10:][::-1]
    stats = {
        'closed': len(trades), 'wins': len(wins), 'losses': len(losses),
        'breakeven': len(breakeven), 'win_rate': rate(wins, len(wins) + len(losses)),
        'net_pnl': round(sum(pnl_values), 2), 'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss else None,
        'average_win': round(mean([float(t.pnl) for t in wins]), 2) if wins else None,
        'average_loss': round(mean([float(t.pnl) for t in losses]), 2) if losses else None,
        'expectancy': round(mean(pnl_values), 2) if pnl_values else 0,
        'labelled_chart_trades': len(labelled_by_trade),
        'chart_analyses': len(chart_rows),
    }
    return render_template('ai/performance.html', stats=stats, by_symbol=by_symbol,
                           by_strategy=by_strategy, by_direction=by_direction,
                           by_entry_hour=by_entry_hour, comparison=comparison, smc_playbook=smc_playbook,
                           model_metrics=model_metrics, learning_status=learning_status,
                           recent=recent)


# ------------------------------------------------------------------ #
#  Chatbot                                                            #
# ------------------------------------------------------------------ #

@ai_bp.route('/chat', methods=['GET'])
@login_required
def chat_interface():
    return render_template('ai/chat.html')


@ai_bp.route('/chat/message', methods=['POST'])
@login_required
def chat_message():
    """Handle a single chatbot message and return the AI reply."""
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()

    if not message:
        return jsonify({'success': False, 'error': 'Empty message'}), 400

    if len(message) > 2000:
        return jsonify({'success': False, 'error': 'Message too long (max 2000 characters)'}), 400

    try:
        bot = ForexChatbot(current_user.id)
        response = bot.process_message(message)
        history = bot.get_conversation_history()

        return jsonify({'success': True, 'response': response, 'history': history})

    except Exception as exc:
        logger.error("Chatbot error for user %s: %s", current_user.id, exc)
        return jsonify({
            'success': False,
            'error': 'AI service temporarily unavailable.',
            'response': {
                'text': (
                    "I'm having trouble connecting to the AI service right now. "
                    "Please try again in a moment."
                ),
                'suggestions': ['How am I doing?', 'Analyse my emotions'],
            },
        })


@ai_bp.route('/chat/history', methods=['GET'])
@login_required
def chat_history():
    try:
        bot = ForexChatbot(current_user.id)
        return jsonify({'success': True, 'history': bot.get_conversation_history()})
    except Exception as exc:
        logger.error("Chat history error for user %s: %s", current_user.id, exc)
        return jsonify({'success': False, 'error': 'Unable to load chat history right now.'}), 500


@ai_bp.route('/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    try:
        bot = ForexChatbot(current_user.id)
        bot.clear_history()
        return jsonify({'success': True, 'message': 'Chat history cleared.'})
    except Exception as exc:
        logger.error("Clear chat error for user %s: %s", current_user.id, exc)
        return jsonify({'success': False, 'error': 'Unable to clear chat history right now.'}), 500


@ai_bp.route('/chat/suggestions', methods=['GET'])
@login_required
def get_chat_suggestions():
    return jsonify({
        'success': True,
        'suggestions': [
            "How am I performing overall?",
            "What's my win rate and profit factor?",
            "Which currency pair should I focus on?",
            "How do my emotions affect my trading?",
            "Am I revenge trading?",
            "What's my best strategy?",
            "When do I trade best during the day?",
            "Analyse my risk management",
            "What psychological biases do I have?",
            "Give me my top 3 improvements",
            "What's my average risk:reward ratio?",
            "Show me my worst performing pair",
            "How has my performance changed recently?",
            "Analyse my latest trade",
            "What patterns have you found in my trading?",
        ],
    })

