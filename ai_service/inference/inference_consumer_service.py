"""
Standalone Inference Consumer Service (Spark-based).

This service runs INDEPENDENTLY from the backend FastAPI server.
It listens to Kafka messages, runs the AI model using distributed Spark workers,
and publishes detections to Redis.

Architecture (Spark-based):
- Entry point for SparkInferenceWorker (separate from FastAPI)
- Consumes frames from Kafka topic (partitioned by camera_id)
- Distributes inference across Spark cluster (N workers)
- Each Spark worker loads model once per partition
- Aggregates results and publishes to Redis
- No interference with API/WebSocket serving

Run this as:
    cd ai_service && python -m inference.inference_consumer_service

"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import remonet
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.spark_worker import SparkInferenceWorker
from inference.inference_consumer import InferenceConsumer
from inference.inference_model import ViolenceDetectionModel, InferenceConfig
from logger import setup_logging, get_logger

logger = get_logger(__name__)
setup_logging()


async def main():
    """Start the Spark-based inference consumer service."""
    start_time = time.time()
    
    try:
        # Get inference parameters from environment
        model_path = os.getenv('MODEL_PATH')
        inference_device = os.getenv('INFERENCE_DEVICE', 'cpu')
        confidence_threshold = float(os.getenv('VIOLENCE_CONFIDENCE_THRESHOLD', '0.5'))
        n_spark_workers = int(os.getenv('N_SPARK_WORKERS', '4'))
        
        if not model_path:
            raise ValueError("MODEL_PATH environment variable is required")
        
        logger.info("=" * 80)
        logger.info("VIOLENCE DETECTION - SPARK-BASED INFERENCE SERVICE")
        logger.info("=" * 80)
        
        # Create Spark worker
        logger.info(f"Initializing Spark cluster with {n_spark_workers} workers...")
        spark_worker = SparkInferenceWorker(
            n_workers=n_spark_workers,
            batch_size=32,
            model_path=model_path,
            device=inference_device,
            kafka_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            kafka_frame_topic=os.getenv('KAFKA_FRAME_TOPIC', 'frames'),
            kafka_result_topic=os.getenv('KAFKA_RESULT_TOPIC', 'inference-results'),
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            alert_confidence_threshold=confidence_threshold,
        )
        
        # Start Spark worker
        spark_worker.start()
        
        # Create traditional inference consumer (for legacy compatibility)
        # It can use Spark worker internally if configured
        config = InferenceConfig(
            model_path=model_path,
            device=inference_device,
            confidence_threshold=confidence_threshold
        )
        
        model = ViolenceDetectionModel(config=config)
        consumer = InferenceConsumer(model=model)
        await consumer.start()
        
        # Print startup complete
        elapsed = time.time() - start_time
        print(f"\nAI Service (Spark-based) started in {elapsed:.1f}s")
        print(f"  Device: {inference_device}")
        print(f"  Spark Workers: {n_spark_workers}")
        print(f"  Confidence Threshold: {confidence_threshold}")
        print(f"  Model: {model_path}\n")
        
        # Keep running indefinitely
        # Will be stopped by signal
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\nAI Service stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """
    Entry point for Spark-based inference service.
    Metrics tracked:
    - Frames processed per batch
    - Processing time per frame
    - Worker utilization distribution
    - Alerts generated
    """
    asyncio.run(main())

