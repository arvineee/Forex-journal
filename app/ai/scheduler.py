"""
Scheduled AI learning tasks
"""
from apscheduler.schedulers.background import BackgroundScheduler
from app import create_app
from app.ai.learning_engine import ForexAI
from app.models import User
from datetime import datetime
import atexit

def learn_from_all_users():
    """Periodically learn from all users' trades"""
    app = create_app()
    
    with app.app_context():
        users = User.query.all()
        
        for user in users:
            try:
                ai_engine = ForexAI(user.id)
                ai_engine.analyze_and_learn()
                print(f"[{datetime.now()}] Learned from user {user.id} ({user.username})")
            except Exception as e:
                print(f"[{datetime.now()}] Error learning from user {user.id}: {e}")

def setup_scheduler():
    """Set up scheduled tasks"""
    scheduler = BackgroundScheduler()
    
    # Run AI learning every 6 hours
    scheduler.add_job(
        func=learn_from_all_users,
        trigger="interval",
        hours=6,
        id="ai_learning_job",
        name="AI Learning from all users",
        replace_existing=True
    )
    
    scheduler.start()
    
    # Shut down scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
