"""
AI Routes for Forex Journal
"""
import logging
from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from . import ai_bp
from .learning_engine import ForexAI
from .chatbot import ForexChatbot
from .models import AIInsight, AILearningLog

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
        return jsonify({'success': False, 'message': f'Analysis failed: {exc}'}), 500


@ai_bp.route('/insight/<int:insight_id>', methods=['PUT'])
@login_required
def update_insight(insight_id):
    """Accept/reject insight feedback to adjust confidence scores."""
    data = request.get_json()
    feedback = data.get('confidence_feedback')

    if feedback not in ('agree', 'disagree'):
        return jsonify({'success': False, 'message': 'Invalid feedback value'}), 400

    insight = AIInsight.query.get_or_404(insight_id)
    if insight.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorised'}), 403

    if feedback == 'agree':
        insight.confidence_score = min(1.0, insight.confidence_score + 0.1)
    else:
        insight.confidence_score = max(0.0, insight.confidence_score - 0.2)
        insight.data_points = max(0, insight.data_points - 1)

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
                'first_detected': p.first_detected.isoformat(),
                'last_detected': p.last_detected.isoformat(),
            }
            for p in ai_engine.patterns.values()
        ],
    })


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
    data = request.get_json()
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
        return jsonify({'success': False, 'error': str(exc)}), 500


@ai_bp.route('/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    try:
        bot = ForexChatbot(current_user.id)
        bot.clear_history()
        return jsonify({'success': True, 'message': 'Chat history cleared.'})
    except Exception as exc:
        logger.error("Clear chat error for user %s: %s", current_user.id, exc)
        return jsonify({'success': False, 'error': str(exc)}), 500


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

