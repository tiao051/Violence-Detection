"""
Standalone Inference Consumer Service

This service runs INDEPENDENTLY from the backend FastAPI server.
It listens to Kafka messages, runs the AI model, and publishes detections to Redis.

IMPORTANT: This should run in a SEPARATE PROCESS/CONTAINER to avoid GIL blocking.

Architecture:
- Entry point for InferenceConsumer (separate from FastAPI)
- Consumes frames from Kafka topic
- Runs ViolenceDetectionModel on GPU/CPU
- Publishes results to Redis
- No interference with API/WebSocket serving

Run this as:
    cd ai_service && python -m inference.inference_consumer_service

Or in Docker:
    docker-compose up inference

"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Add parent directory to path so we can import remonet
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference_consumer import InferenceConsumer
from inference.inference_model import ViolenceDetectionModel, InferenceConfig

# Setup logging - minimal output (only startup + errors)
def setup_logging():
    """Configure logging - WARNING level by default (only errors shown)."""
    os.makedirs("logs", exist_ok=True)
    
    # Check if verbose logging is enabled
    verbose = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'
    log_level = logging.INFO if verbose else logging.WARNING
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler - minimal format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler - only errors
    try:
        file_handler = RotatingFileHandler(
            "logs/inference.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Silently fail
    
    # Silence noisy libraries
    logging.getLogger("aiokafka").setLevel(logging.ERROR)
    logging.getLogger("kafka").setLevel(logging.ERROR)
    
    return logging.getLogger(__name__)

logger = setup_logging()


async def main():
    """Start the inference consumer service."""
    start_time = time.time()
    
    try:
        # Get inference parameters from environment
        model_path = os.getenv('MODEL_PATH')
        inference_device = os.getenv('INFERENCE_DEVICE', 'cpu')
        confidence_threshold = float(os.getenv('VIOLENCE_CONFIDENCE_THRESHOLD', '0.5'))
        
        if not model_path:
            raise ValueError("MODEL_PATH environment variable is required")
        
        # Create config
        config = InferenceConfig(
            model_path=model_path,
            device=inference_device,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize model
        model = ViolenceDetectionModel(config=config)
        
        # Create and start inference consumer
        consumer = InferenceConsumer(model=model)
        await consumer.start()
        
        # Print startup complete (always shown)
        elapsed = time.time() - start_time
        print(f"\n✅ AI Service started in {elapsed:.1f}s | Device: {inference_device} | Threshold: {confidence_threshold}\n")
        
        # Keep running indefinitely
        # Will be stopped by signal (SIGTERM/SIGINT from Docker or Ctrl+C)
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 AI Service stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """
    Entry point for standalone inference service.
    
    This runs in a separate process from backend/main.py to avoid:
    - GIL blocking API requests
    - Frame drops from RTSP pulling
    - WebSocket lag during model inference
    
    Each service has its own Python interpreter with full CPU access.
    """
    asyncio.run(main())
