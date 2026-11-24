from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config
from datetime import datetime
import os
from flask_mail import Mail
from flask_wtf.csrf import generate_csrf
from advanced_protection import advanced_protection
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'
migrate = Migrate()
mail = Mail() 

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Respect Config/environment database URI; never hard-code a machine-specific path.
    if not app.config.get('SECRET_KEY'):
        raise RuntimeError('Set a strong SECRET_KEY in the environment before starting the app')
    if not app.config.get('SQLALCHEMY_DATABASE_URI'):
        raise RuntimeError('SQLALCHEMY_DATABASE_URI must be configured')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    mail.init_app(app) 
    advanced_protection.init_app(app)  # this also registers its own blueprint

    @app.context_processor
    def inject_csrf_token():
        return {'csrf_token': generate_csrf}

    @app.template_filter('datetimeformat')
    def datetimeformat(value, format='%Y-%m-%d %H:%M'):
        if value is None:
            return ''
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except:
                return value
        return value.strftime(format)

    
    # Register all application modules explicitly. Fail startup if a module is broken;
    # silently running with missing routes is much harder to diagnose.
    from app.auth.routes import auth_bp
    from app.trades.routes import trades_bp
    from app.dashboard.routes import dashboard_bp
    from app.main.routes import bp as main_bp
    from app.account.routes import account_bp
    from app.ai import ai_bp
    from app.calculator import calculator_bp
    from app.admin import admin_bp

    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(ai_bp, url_prefix='/ai')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(trades_bp, url_prefix='/trades')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(account_bp, url_prefix='/account')
    app.register_blueprint(calculator_bp, url_prefix='/calculator')
    app.register_blueprint(main_bp)

    return app


