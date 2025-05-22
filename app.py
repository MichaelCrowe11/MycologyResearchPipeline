import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flasgger import Swagger
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from config import active_config


# Configure logging
logging.basicConfig(
    level=getattr(logging, active_config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


def create_app(config_object=active_config):
    """Application factory pattern."""
    app = Flask(__name__)
    app.config.from_object(config_object)
    
    # Set secret key
    app.secret_key = os.environ.get("SESSION_SECRET", config_object.SECRET_KEY)
    
    # Setup middleware for running behind proxies
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    # Initialize extensions
    db.init_app(app)
    
    # Setup Swagger API documentation
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs",
    }
    
    swagger = Swagger(app, config=swagger_config)
    
    # Setup Flask-Login
    from flask_login import LoginManager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from models import User
        return User.query.get(int(user_id))
    
    # Register blueprints
    from api_routes import api_bp
    from web_routes import web_bp
    from auth_routes import auth_bp
    from ai_routes import ai_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(web_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(ai_bp, url_prefix='/ai')
    
    # Setup database tables
    with app.app_context():
        import models  # Import models to register them with SQLAlchemy
        db.create_all()
    
    # Add Prometheus metrics endpoint if enabled
    if config_object.ENABLE_METRICS:
        from monitoring import setup_metrics
        setup_metrics(app)
        app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
            '/metrics': make_wsgi_app()
        })
    
    return app
