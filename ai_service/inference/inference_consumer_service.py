"""
Standalone Inference Consumer Service (Spark-based).

Consumes frames from Kafka and performs distributed inference via Spark,
publishing detection results to HDFS (via Kafka) and alerts to Redis.

Run: python -m ai_service.inference.inference_consumer_service
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference_consumer import InferenceConsumer
from inference.inference_model import ViolenceDetectionModel, InferenceConfig
from logger import setup_logging, get_logger

logger = get_logger(__name__)
setup_logging()


async def main():
    """Start the inference consumer with Spark backend."""
    start_time = time.time()
    
    try:
        model_path = os.getenv('MODEL_PATH')
        if not model_path:
            raise ValueError("MODEL_PATH environment variable is required")
        
        logger.info("=" * 80)
        logger.info("VIOLENCE DETECTION - INFERENCE SERVICE (SPARK)")
        logger.info("=" * 80)
        
        config = InferenceConfig(
            model_path=model_path,
            device=os.getenv('INFERENCE_DEVICE', 'cpu'),
            confidence_threshold=float(os.getenv('VIOLENCE_CONFIDENCE_THRESHOLD', '0.5'))
        )
        
        model = ViolenceDetectionModel(config=config)
        
        # Check if Spark should be used (default: True)
        use_spark = os.getenv('USE_SPARK', 'true').lower() == 'true'
        logger.info(f"Inference Mode: {'Distributed (Spark)' if use_spark else 'Local (Single Node)'}")
        
        consumer = InferenceConsumer(model=model, use_spark=use_spark)
        await consumer.start()
        
        elapsed = time.time() - start_time
        logger.info(f"Service started in {elapsed:.1f}s")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Spark Workers: {os.getenv('N_SPARK_WORKERS', '4')}\n")
        
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Service stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())


