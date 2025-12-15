"""
AI Routes for Forex Journal
"""
from flask import render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from . import ai_bp
from .learning_engine import ForexAI
from .chatbot import ForexChatbot  # Add this import
from .models import AIInsight, AILearningLog

@ai_bp.route('/dashboard')
@login_required
def dashboard():
    """AI Dashboard - Show personalized insights"""
    # Get user's AI insights
    ai_engine = ForexAI(current_user.id)
    # FIX: Increase the limit to ensure all generated insights, including emotions/recommendations, are visible
    insights_data = ai_engine.get_personalized_insights(limit=100)
    
    # Group insights by type for the template
    grouped_insights = {
        'emotion': [],
        'patterns': [],
        'recommendations': [],
        'risk': []
    }
    
    for insight in insights_data:
        insight_type = insight.get('type', 'general')
        if insight_type in ['emotion', 'emotion_analysis']:
            grouped_insights['emotion'].append(insight)
        elif insight_type in ['time_pattern', 'symbol_analysis', 'strategy_analysis', 'behavioral_pattern']:
            grouped_insights['patterns'].append(insight)
        elif insight_type == 'recommendation':
            grouped_insights['recommendations'].append(insight)
        elif insight_type in ['risk_analysis', 'risk_management']:
            grouped_insights['risk'].append(insight)
        else:
            # Default to patterns for unknown types
            grouped_insights['patterns'].append(insight)
    
    # Get recent learning logs
    recent_logs = AILearningLog.query.filter_by(
        user_id=current_user.id
    ).order_by(
        AILearningLog.created_at.desc()
    ).limit(10).all()
    
    return render_template('ai/dashboard.html', 
                         insights=grouped_insights,
                         recent_logs=recent_logs)

@ai_bp.route('/analyze', methods=['POST','GET'])
@login_required
def analyze():
    """Trigger manual AI analysis of user's trades"""
    try:
        ai_engine = ForexAI(current_user.id)
        insights = ai_engine.analyze_and_learn()
        
        # Log this analysis
        log = AILearningLog(
            user_id=current_user.id,
            event_type='manual_analysis_triggered',
            event_data='{}',
            learned_insights=str(len(insights))
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Analysis completed! Generated {len(insights)} insights.',
            'insight_count': len(insights)
        })
    except Exception as e:
        current_app.logger.error(f"AI analysis error: {e}")
        return jsonify({
            'success': False,
            'message': f'Analysis failed: {str(e)}'
        }), 500

@ai_bp.route('/insight/<int:insight_id>', methods=['PUT'])
@login_required
def update_insight(insight_id):
    """Update insight based on user feedback"""
    data = request.get_json()
    feedback = data.get('confidence_feedback')
    
    if feedback not in ['agree', 'disagree']:
        return jsonify({'success': False, 'message': 'Invalid feedback'}), 400
    
    insight = AIInsight.query.get_or_404(insight_id)
    
    # Verify ownership
    if insight.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    # Adjust confidence based on feedback
    if feedback == 'agree':
        insight.confidence_score = min(1.0, insight.confidence_score + 0.1)
    else:  # disagree
        insight.confidence_score = max(0.0, insight.confidence_score - 0.2)
        insight.data_points -= 1  # Reduce data points for disagreement
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Feedback recorded',
        'new_confidence': insight.confidence_score
    })

@ai_bp.route('/insights')
@login_required
def get_insights():
    """Get all insights for the user"""
    ai_engine = ForexAI(current_user.id)
    # Increase limit for the API route as well to ensure more data is returned
    insights = ai_engine.get_personalized_insights(limit=100) 
    
    return jsonify({
        'success': True,
        'insights': insights,
        'count': len(insights)
    })

@ai_bp.route('/patterns')
@login_required
def get_patterns():
    """Get detected trading patterns"""
    ai_engine = ForexAI(current_user.id)
    ai_engine.load_data()
    
    # Get all patterns
    patterns = ai_engine.patterns.values()
    
    return jsonify({
        'success': True,
        'patterns': [{
            'id': p.id,
            'type': p.pattern_type,
            'data': p.pattern_data,
            'occurrences': p.occurrence_count,
            'first_detected': p.first_detected.isoformat(),
            'last_detected': p.last_detected.isoformat()
        } for p in patterns]
    })

# ===== CHATBOT ROUTES =====

@ai_bp.route('/chat', methods=['GET'])
@login_required
def chat_interface():
    """Render chatbot interface"""
    return render_template('ai/chat.html')

@ai_bp.route('/chat/message', methods=['POST'])
@login_required
def chat_message():
    """Handle chatbot messages"""
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'success': False, 'error': 'Empty message'})
    
    try:
        # Initialize or get chatbot from session
        chatbot = ForexChatbot(current_user.id)
        
        # Process message
        response = chatbot.process_message(message)
        
        # Get conversation history
        history = chatbot.get_conversation_history()
        
        return jsonify({
            'success': True,
            'response': response,
            'history': history
        })
        
    except Exception as e:
        current_app.logger.error(f"Chatbot error: {e}")
        return jsonify({
            'success': False,
            'error': 'Chatbot service unavailable',
            'response': {
                'text': "I'm having trouble accessing your trading data. Please try again later.",
                'suggestions': []
            }
        })

@ai_bp.route('/chat/history', methods=['GET'])
@login_required
def chat_history():
    """Get chat history"""
    try:
        chatbot = ForexChatbot(current_user.id)
        history = chatbot.get_conversation_history()
        
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@ai_bp.route('/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    """Clear chat history"""
    try:
        chatbot = ForexChatbot(current_user.id)
        chatbot.clear_history()
        
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@ai_bp.route('/chat/suggestions', methods=['GET'])
@login_required
def get_chat_suggestions():
    """Get suggested questions for the chatbot"""
    suggestions = [
        "How am I doing overall?",
        "What's my win rate?",
        "Analyze my emotions impact",
        "What patterns have you found?",
        "Give me trading recommendations",
        "When is my best trading time?",
        "What's my best currency pair?",
        "Analyze my risk management",
        "Show me my biggest loss",
        "What's my average profit per trade?",
        "How many trades have I taken?",
        "What's my worst performing pair?",
        "Should I change my strategy?",
        "How can I improve my trading?",
        "Analyze my latest trade"
    ]
    
    return jsonify({
        'success': True,
        'suggestions': suggestions
    })
