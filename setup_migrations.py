#!/usr/bin/env python3
"""
Manual migration setup for Forex Journal
"""
import os
import sys
from app import create_app, db
from flask_migrate import Migrate

def setup_migrations():
    """Set up database migrations"""
    try:
        app = create_app()
        migrate = Migrate(app, db)
        
        with app.app_context():
            # Initialize migrations if not already done
            from flask_migrate import init, migrate, upgrade
            try:
                init()
                print("✅ Migration repository initialized")
            except:
                print("ℹ️  Migration repository already exists")
            
            # Create initial migration
            migrate(message="Initial migration")
            print("✅ Initial migration created")
            
            # Apply migration
            upgrade()
            print("✅ Database upgraded to latest migration")
            
    except Exception as e:
        print(f"❌ Error during migration setup: {e}")
        print("📋 You can manually run migrations later with:")
        print("   export FLASK_APP=run.py")
        print("   flask db init")
        print("   flask db migrate -m 'Initial migration'")
        print("   flask db upgrade")

if __name__ == '__main__':
    setup_migrations()
