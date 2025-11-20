"""Logging configuration."""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

from src.core.config import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional) — enable when explicitly requested or in production
    if settings.log_to_file or settings.app_env == "production":
        try:
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)

            file_handler = RotatingFileHandler(
                "logs/app.log",
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # If file handler can't be created, fall back to console only
            root_logger.warning(f"Failed to create file log handler: {e}")
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
