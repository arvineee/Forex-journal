#!/usr/bin/env python3
"""
Forex Journal - Main Application Entry Point
"""
import os
from app import create_app
from app.ai.scheduler import setup_scheduler
# Create application instance
app = create_app()

@app.context_processor
def utility_processor():
    """Make utility functions available in templates"""
    return dict(len=len, str=str, float=float, int=int)

if __name__ == '__main__':
    # Ensure upload directory exists
    with app.app_context():
        upload_dir = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            print(f"✅ Created upload directory: {upload_dir}")
    
    print("🚀 Starting Forex Journal...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Mobile-friendly interface ready")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)

    if os.environ.get("ENABLE_AI_SCHEDULER", "false").lower() == "true":
        setup_scheduler()
        print("AI learning scheduler started in this process")

    # Run the application
    app.run(
        debug=app.config.get('DEBUG', False),
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', 5000)),
        use_reloader=app.config.get('DEBUG', False)
    )
