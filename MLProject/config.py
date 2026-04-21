import os
from typing import Dict, Any

class Config:
    """Production configuration"""
    
    # Server configuration
    SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT = int(os.getenv('SERVER_PORT', 5000))
    WORKERS = int(os.getenv('WORKERS', 4))
    TIMEOUT = int(os.getenv('TIMEOUT', 30))
    GRACEFUL_TIMEOUT = int(os.getenv('GRACEFUL_TIMEOUT', 30))
    MAX_REQUESTS = int(os.getenv('MAX_REQUESTS', 1000))
    MAX_REQUESTS_JITTER = int(os.getenv('MAX_REQUESTS_JITTER', 100))
    
    # Model configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/credit_risk_model')
    MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')
    
    # Metrics configuration
    METRICS_ENABLED = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    
    # Features
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    EXIT_ON_MODEL_FAILURE = os.getenv('EXIT_ON_MODEL_FAILURE', 'false').lower() == 'true'
    ALLOW_MODEL_RELOAD = os.getenv('ALLOW_MODEL_RELOAD', 'false').lower() == 'true'
    ENABLE_DOCS = os.getenv('ENABLE_DOCS', 'false').lower() == 'true'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'info')
    
    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
    VERSION = os.getenv('VERSION', '1.0.0')

def get_config() -> Dict[str, Any]:
    """Get configuration as dictionary"""
    return {k: v for k, v in Config.__dict__.items() 
            if not k.startswith('_') and not callable(v)}