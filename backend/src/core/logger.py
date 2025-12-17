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
    
    # Avoid duplicate handlers (from Uvicorn reloader)
    if root_logger.handlers:
        print(f"[LOGGER] Clearing {len(root_logger.handlers)} existing handlers")
        root_logger.handlers.clear()
    
    root_logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    # Format - minimal for production
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    print("[LOGGER] Console handler added")
    
    # File handler - always enabled for debugging and monitoring
    try:
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        print(f"[LOGGER] Logs directory ready: {os.path.abspath(logs_dir)}")
        
        log_file = os.path.join(logs_dir, "app.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"[LOGGER] File handler added: {os.path.abspath(log_file)}")
    except Exception as e:
        print(f"[LOGGER] ERROR: Failed to setup file logging: {e}")
        import traceback
        traceback.print_exc()
    
    # Verify handlers
    print(f"[LOGGER] Root logger now has {len(root_logger.handlers)} handlers")
    
    # Silence third-party libraries
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("redis").setLevel(logging.ERROR)
    logging.getLogger("aiokafka").setLevel(logging.ERROR)
    
    # Control verbose libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
