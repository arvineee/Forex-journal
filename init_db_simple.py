#!/usr/bin/env python3
"""
Simplified database initialization script for Forex Journal
"""
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def init_db():
    """Initialize the database with required tables"""
    try:
        from app import create_app, db
        
        app = create_app()
        
        with app.app_context():
            # Create all tables
            db.create_all()
            print("✅ Database tables created successfully!")
            
            # Create upload directory if it doesn't exist
            upload_dir = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                print(f"✅ Created upload directory: {upload_dir}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during database initialization: {e}")

if __name__ == '__main__':
    init_db()
