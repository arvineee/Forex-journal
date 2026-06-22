"""
Scheduled AI learning tasks — runs full analysis for all users every 6 hours.
Lightweight analysis is triggered on every trade save from the trades routes.
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from app import create_app
from app.ai.learning_engine import ForexAI
from app.models import User
from datetime import datetime
import atexit

logger = logging.getLogger(__name__)


def learn_from_all_users():
    """Full AI analysis pass — runs on a schedule, not on every trade save."""
    app = create_app()
    with app.app_context():
        users = User.query.all()
        logger.info("[AI Scheduler] Starting full analysis for %d users at %s", len(users), datetime.now())
        for user in users:
            try:
                ai = ForexAI(user.id)
                ai.analyze_and_learn(full=True)
                logger.info("[AI Scheduler] Completed analysis for user %d (%s)", user.id, user.username)
            except Exception as exc:
                logger.error("[AI Scheduler] Error for user %d: %s", user.id, exc)
        logger.info("[AI Scheduler] Full analysis pass complete at %s", datetime.now())


def setup_scheduler():
    """Initialise and start the APScheduler background job."""
    scheduler = BackgroundScheduler(daemon=True)

    scheduler.add_job(
        func=learn_from_all_users,
        trigger="interval",
        hours=6,
        id="ai_learning_job",
        name="Full AI learning pass (all users)",
        replace_existing=True,
        misfire_grace_time=300,  # Allow 5-minute late start before skipping
    )

    scheduler.start()
    logger.info("[AI Scheduler] Scheduler started — full analysis every 6 hours.")
    atexit.register(lambda: scheduler.shutdown(wait=False))
    return scheduler

