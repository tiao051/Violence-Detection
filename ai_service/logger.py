"""Logging configuration for AI Service."""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Configure logging for the AI service.
    
    Args:
        log_level: Override log level. If None, uses VERBOSE_LOGGING env var.
    
    Returns:
        Configured logger instance.
    """
    # Determine log level from environment or parameter
    if log_level is None:
        verbose = os.getenv('VERBOSE_LOGGING', 'true').lower() == 'true'
        log_level = logging.INFO if verbose else logging.WARNING
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler - minimal format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - capture INFO and above for investigation, but avoid debug spam
    try:
        # Ensure logs directory exists (relative to project root /app)
        log_dir = "ai_service/logs"
        if not os.path.exists(log_dir):
            # Fallback if running locally inside ai_service dir
            if os.path.exists("logs"):
                log_dir = "logs"
            else:
                os.makedirs(log_dir, exist_ok=True)
                
        file_handler = RotatingFileHandler(
            f"{log_dir}/inference.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging: {e}")
        pass  # Silently fail if can't create log file
    
    # Silence noisy third-party libraries
    logging.getLogger("aiokafka").setLevel(logging.ERROR)
    logging.getLogger("kafka").setLevel(logging.ERROR)
    
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
