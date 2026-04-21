import os

class Config:
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'saved_model')
    
    # Server Configuration
    SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT = int(os.getenv('SERVER_PORT', 5000))
    
    # Monitoring Configuration
    METRICS_ENABLED = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', 30))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Security Configuration
    API_KEY = os.getenv('API_KEY', '')  # Optional API key for authentication
    
    # Performance Configuration
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 32))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

# Environment-specific configurations
class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment variable"""
    env = os.getenv('FLASK_ENV', 'default')
    return config.get(env, config['default'])