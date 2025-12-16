"""Logging configuration."""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

from src.core.config import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Root logger - INFO by default to capture startup events
    # Use DEBUG=true in settings to enable verbose logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    # Format - minimal for production
    # Use a custom formatter to ensure timezone is respected if needed, 
    # but setting TZ env var in Docker is usually sufficient.
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - always enabled for debugging and monitoring
    # Separate from console to capture persistent log history
    try:
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/app.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        root_logger.warning(f"Failed to setup file logging: {e}")
    
    # Silence third-party libraries
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("redis").setLevel(logging.ERROR)
    logging.getLogger("aiokafka").setLevel(logging.ERROR)
    
    # Control verbose libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Reduce access logs
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
