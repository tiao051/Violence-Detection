"""Logging configuration."""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

from src.core.config import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Root logger - WARNING by default, only errors and warnings shown
    # Use DEBUG=true in settings to enable verbose logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.WARNING)
    
    # Format - minimal for production
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.WARNING)
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
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            pass  # Silently fail for file handler
    
    # Silence third-party libraries completely
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("redis").setLevel(logging.ERROR)
    logging.getLogger("aiokafka").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
