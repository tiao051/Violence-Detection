"""
Entry point for running the inference consumer service.

This is a standalone service that:
1. Consumes frames from Kafka
2. Performs batch inference
3. Publishes alerts to Redis

Usage:
    python -m backend.src.infrastructure.consumers.inference_consumer_service
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ai_service.inference.inference_consumer import InferenceConsumer
from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig
from backend.src.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for inference consumer service."""
    try:
        logger.info("Initializing Inference Consumer Service...")
        
        # Initialize violence detection model
        model_config = InferenceConfig(
            model_path=os.getenv(
                "MODEL_PATH",
                "../ai_service/training/two-stage/checkpoints/last_model.pt"
            ),
            backbone=os.getenv("MODEL_BACKBONE", "mobilenet_v3_small"),
            device=os.getenv("INFERENCE_DEVICE", "cuda"),
            confidence_threshold=float(os.getenv("VIOLENCE_CONFIDENCE_THRESHOLD", "0.5")),
            num_frames=int(os.getenv("INFERENCE_BUFFER_SIZE", "30")),
        )
        
        logger.info(f"Loading model from {model_config.model_path}")
        model = ViolenceDetectionModel(model_config)
        
        # Initialize inference consumer
        consumer = InferenceConsumer(
            model=model,
            kafka_bootstrap_servers=settings.kafka_bootstrap_servers,
            kafka_topic=settings.kafka_frame_topic,
            kafka_group_id=settings.kafka_consumer_group,
            redis_url=settings.redis_url,
            batch_size=settings.inference_batch_size,
            batch_timeout_ms=settings.inference_batch_timeout_ms,
            alert_cooldown_seconds=settings.alert_cooldown_seconds,
        )
        
        # Start consumer
        await consumer.start()
        
        # Run until interrupted
        logger.info("Inference consumer service started. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(10)
                stats = consumer.get_stats()
                logger.info(
                    f"Stats: frames={stats['frames_consumed']}, "
                    f"processed={stats['frames_processed']}, "
                    f"detections={stats['detections_made']}, "
                    f"fps={stats['fps']:.1f}"
                )
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await consumer.stop()
            logger.info("Inference consumer service stopped.")
    
    except Exception as e:
        logger.error(f"Failed to start inference consumer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
