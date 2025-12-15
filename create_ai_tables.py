#!/usr/bin/env python3
"""
Create AI tables migration
"""
from app import create_app, db
from flask_migrate import Migrate

app = create_app()
migrate = Migrate(app, db)

with app.app_context():
    # Import AI models so they're registered with SQLAlchemy
    from app.ai.models import AIInsight, TradePattern, AILearningLog
    
    # Create all tables
    db.create_all()
    
    print("✅ AI tables created successfully!")
    print("   - ai_insights")
    print("   - trade_patterns")
    print("   - ai_learning_logs")
