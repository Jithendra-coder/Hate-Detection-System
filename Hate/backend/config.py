"""
Configuration management for Hate Speech Detection API
"""
import os
from typing import Dict, Any


class Config:
    """Base configuration class"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False

    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'lstm_hate_model.h5')
    TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', 'tokenizer.pkl')
    LABELS_PATH = os.environ.get('LABELS_PATH', 'labels.pkl')
    MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH', '100'))
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.5'))

    # API rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    RATELIMIT_DEFAULT = "1000 per hour"

    # Text processing limits
    MAX_TEXT_LENGTH = int(os.environ.get('MAX_TEXT_LENGTH', '1000'))
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '10'))

    # CORS settings
    CORS_ORIGINS = [
        "chrome-extension://*",
        "moz-extension://*",
        "http://localhost:*",
        "https://localhost:*"
    ]

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'hate_speech_api.log')

    # Database (if using one)
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///feedback.db')

    @staticmethod
    def init_app(app):
        """Initialize app with configuration"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    RATELIMIT_ENABLED = False
    MODEL_PATH = 'test_model.h5'  # Use smaller test model


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Log to file in production
        import logging
        from logging.handlers import RotatingFileHandler

        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler(
            'logs/hate_speech_api.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Hate Speech API startup')


class DockerConfig(ProductionConfig):
    """Docker container configuration"""

    @classmethod
    def init_app(cls, app):
        ProductionConfig.init_app(app)

        # Log to stdout in containers
        import logging
        import sys

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)