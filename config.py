import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Base configuration."""
    
    # Flask
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SESSION_SECRET', 'dev-key-for-mycology-research')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///mycology_research.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    
    # API
    API_PORT = int(os.environ.get('PORT', 8000))
    API_HOST = os.environ.get('HOST', '0.0.0.0')
    
    # Web Server
    WEB_PORT = int(os.environ.get('WEB_PORT', 5000))
    WEB_HOST = os.environ.get('WEB_HOST', '0.0.0.0')
    
    # Monitoring
    ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'True').lower() == 'true'
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # File paths
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')
    
    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # Model settings
    MODEL_VERSION = os.environ.get('MODEL_VERSION', 'v1.0')
    

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Set active configuration
active_config = config[os.environ.get('FLASK_ENV', 'default')]
