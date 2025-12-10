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
from .inference_consumer import InferenceConsumer
from .inference_model import ViolenceDetectionModel
from ..remonet.config.model_config import ModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Start the inference consumer service."""
    logger.info("=" * 80)
    logger.info("Starting Inference Consumer Service (Standalone)")
    logger.info("=" * 80)
    
    try:
        # Load model config
        logger.info("Loading violence detection model...")
        model_config = ModelConfig()
        
        # Get inference parameters
        model_path = os.getenv('MODEL_PATH')
        inference_device = os.getenv('INFERENCE_DEVICE', 'cuda')
        confidence_threshold = float(os.getenv('VIOLENCE_CONFIDENCE_THRESHOLD', '0.5'))
        
        # Initialize model
        model = ViolenceDetectionModel(
            config=model_config,
            model_path=model_path,
            inference_device=inference_device,
            confidence_threshold=confidence_threshold
        )
        logger.info(f"Model loaded successfully (device: {inference_device})")
        
        # Create and start inference consumer
        consumer = InferenceConsumer(model=model)
        await consumer.start()
        logger.info("Inference Consumer started")
        
        # Keep running indefinitely
        # Will be stopped by signal (SIGTERM/SIGINT from Docker or Ctrl+C)
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
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
