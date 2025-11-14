from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure upload directory exists
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    
    # Register blueprints with error handling
    try:
        from app.auth.routes import auth_bp
        from app.trades.routes import trades_bp
        from app.dashboard.routes import dashboard_bp
        from app.main.routes import bp as main_bp
        
        app.register_blueprint(auth_bp, url_prefix='/auth')
        app.register_blueprint(trades_bp, url_prefix='/trades')
        app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
        app.register_blueprint(main_bp)
        
    except Exception as e:
        print(f"Warning: Some blueprints couldn't be registered: {e}")
        # Register at least the main blueprint
        try:
            from app.main.routes import bp as main_bp
            app.register_blueprint(main_bp)
        except:
            pass
    
    return app
