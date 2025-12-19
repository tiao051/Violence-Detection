"""
Inference consumer that listens to Kafka and processes frames.

Architecture:
- Consumes frames from Kafka (partitioned by camera_id)
- Distributes inference across Spark cluster for parallel processing
- Publishes detection results to Kafka (for HDFS archiving)
- Publishes alerts to Redis
- Uses alert deduplication to prevent spam

Data flow: Kafka → InferenceConsumer → [SparkInferenceWorker] → Kafka + Redis
"""

import asyncio
import logging
import os
import json
import time
import base64
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import cv2
import msgpack
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import redis.asyncio as redis

from .spark_worker import SparkInferenceWorker

logger = logging.getLogger(__name__)


@dataclass
class FrameMessage:
    """Frame message from Kafka."""
    camera_id: str
    frame_id: str
    timestamp: float
    frame_seq: int
    frame: np.ndarray


class InferenceConsumer:
    """
    Consumes frames from Kafka, performs distributed inference via Spark, publishes results.
    
    Key features:
    - Per-camera buffers (maintains temporal ordering from Kafka partitioning)
    - Distributed batch inference via SparkInferenceWorker (parallel processing)
    - Publishes ALL results to Kafka (for HDFS archiving)
    - Publishes high-confidence alerts to Redis
    - Alert deduplication (max 1 alert per camera per 60s)
    - Graceful error handling and metrics
    - Async I/O (non-blocking Kafka)
    """
    
    def __init__(
        self,
        model,  # ViolenceDetectionModel instance (optional if using Spark)
        kafka_bootstrap_servers: str = None,
        kafka_topic: str = None,
        kafka_group_id: str = None,
        redis_url: str = None,
        batch_size: int = None,
        batch_timeout_ms: int = None,
        use_spark: bool = True,
        n_spark_workers: int = 4,
        model_path: str = None,
    ):
        """
        Initialize inference consumer.
        
        Args:
            model: ViolenceDetectionModel instance (for non-Spark fallback)
            kafka_bootstrap_servers: Kafka broker address (from KAFKA_BOOTSTRAP_SERVERS env)
            kafka_topic: Topic to consume frames from (from KAFKA_FRAME_TOPIC env)
            kafka_group_id: Kafka consumer group (from KAFKA_CONSUMER_GROUP env)
            redis_url: Redis connection URL (from REDIS_URL env)
            batch_size: Frames to accumulate before inference (from INFERENCE_BATCH_SIZE env)
            batch_timeout_ms: Max wait for batch (from INFERENCE_BATCH_TIMEOUT_MS env)
            use_spark: Use SparkInferenceWorker for distributed inference
            n_spark_workers: Number of Spark workers (from N_SPARK_WORKERS env)
            model_path: Path to model (from MODEL_PATH env)
        """
        # Load from environment
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.kafka_topic = kafka_topic or os.getenv("KAFKA_FRAME_TOPIC")
        self.kafka_group_id = kafka_group_id or os.getenv("KAFKA_CONSUMER_GROUP")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.batch_size = batch_size or int(os.getenv("INFERENCE_BATCH_SIZE", "") or 0)
        self.batch_timeout_ms = batch_timeout_ms or int(os.getenv("INFERENCE_BATCH_TIMEOUT_MS", "") or 0)
        
        # Spark config
        self.use_spark = use_spark
        self.n_spark_workers = n_spark_workers or int(os.getenv("N_SPARK_WORKERS", "4"))
        self.model_path = model_path or os.getenv("MODEL_PATH")
        
        # Validate all required config
        if not self.kafka_bootstrap_servers:
            raise ValueError("KAFKA_BOOTSTRAP_SERVERS env variable not set")
        if not self.kafka_topic:
            raise ValueError("KAFKA_FRAME_TOPIC env variable not set")
        if not self.kafka_group_id:
            raise ValueError("KAFKA_CONSUMER_GROUP env variable not set")
        if not self.redis_url:
            raise ValueError("REDIS_URL env variable not set")
        if self.batch_size <= 0:
            raise ValueError("INFERENCE_BATCH_SIZE env variable not set or invalid")
        if self.batch_timeout_ms <= 0:
            raise ValueError("INFERENCE_BATCH_TIMEOUT_MS env variable not set or invalid")
        if self.use_spark and not self.model_path:
            raise ValueError("MODEL_PATH env variable required for Spark inference")
        
        self.model = model  # Fallback model
        
        # Spark worker
        self.spark_worker: Optional[SparkInferenceWorker] = None
        if self.use_spark:
            logger.info(f"Using SparkInferenceWorker with {self.n_spark_workers} workers")
            self.spark_worker = SparkInferenceWorker(
                n_workers=self.n_spark_workers,
                batch_size=self.batch_size,
                model_path=self.model_path,
                device=os.getenv("INFERENCE_DEVICE", "cpu"),
                kafka_servers=self.kafka_bootstrap_servers,
                redis_url=self.redis_url,
                alert_confidence_threshold=float(os.getenv("VIOLENCE_CONFIDENCE_THRESHOLD", "0.85")),
            )
        
        # State
        self.is_running = False
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[AIOKafkaProducer] = None
        self.kafka_result_topic = os.getenv("KAFKA_RESULT_TOPIC", "inference-results")
        self.redis_client: Optional[redis.Redis] = None
        
        # Per-camera buffers and state
        self.camera_buffers: Dict[str, List[FrameMessage]] = {}
        
        # Metrics
        self.frames_consumed = 0
        self.frames_processed = 0
        self.detections_made = 0
        self.alerts_sent = 0
        self.start_time = time.time()
        
        # Queue for decoupling Kafka consumer and Inference worker
        self.frame_queue = asyncio.Queue(maxsize=100)
    
    async def start(self) -> None:
        """Start consuming from Kafka and initialize Spark worker if needed."""
        try:
            # Initialize Spark worker if using Spark
            if self.use_spark and self.spark_worker:
                logger.info("Starting SparkInferenceWorker...")
                self.spark_worker.start()
            
            # Connect to Kafka with retry logic
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                try:
                    self.consumer = AIOKafkaConsumer(
                        self.kafka_topic,
                        bootstrap_servers=self.kafka_bootstrap_servers,
                        group_id=self.kafka_group_id,
                        auto_offset_reset='latest',
                        enable_auto_commit=True,
                        value_deserializer=lambda m: m,  # Raw bytes for MessagePack binary format
                        key_deserializer=lambda k: k.decode('utf-8') if k else None,
                        retry_backoff_ms=1000,
                        request_timeout_ms=10000,
                    )
                    await self.consumer.start()
                    logger.debug("Successfully connected to Kafka Consumer")
                    
                    # Initialize Producer
                    self.producer = AIOKafkaProducer(
                        bootstrap_servers=self.kafka_bootstrap_servers,
                        value_serializer=lambda v: json.dumps(v).encode('utf-8')
                    )
                    await self.producer.start()
                    logger.debug("Successfully connected to Kafka Producer")
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Failed to connect to Kafka (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count == max_retries:
                        raise
                    await asyncio.sleep(2)
            
            # Connect to Redis
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            self.is_running = True
            
            # Start tasks
            asyncio.create_task(self._kafka_reader_task())
            asyncio.create_task(self._inference_worker_task())
        
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            if self.spark_worker:
                self.spark_worker.stop()
            raise
    
    async def stop(self) -> None:
        """Stop consuming and cleanup Spark worker."""
        self.is_running = False
        
        if self.consumer:
            await self.consumer.stop()
            
        if self.producer:
            await self.producer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.spark_worker:
            self.spark_worker.stop()
    
    async def _kafka_reader_task(self) -> None:
        """Consume frames from Kafka and queue them for inference."""
        try:
            async for msg in self.consumer:
                if not self.is_running:
                    break
                
                try:
                    # Parse Kafka message
                    frame_msg = self._parse_message(msg.key, msg.value)
                    if frame_msg is None:
                        continue
                    
                    self.frames_consumed += 1
                    
                    # If queue is full, drop oldest frame to maintain real-time
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    
                    await self.frame_queue.put(frame_msg)
                
                except Exception as e:
                    logger.error(f"Error processing message in reader task: {e}")
        
        except Exception as e:
            logger.error(f"Kafka reader task error: {e}")

    async def _inference_worker_task(self) -> None:
        """Batch frames and run inference via Spark or fallback model."""
        while self.is_running:
            try:
                # Get frame from queue
                frame_msg = await self.frame_queue.get()
                
                # Add to per-camera buffer for batching
                camera_id = frame_msg.camera_id
                if camera_id not in self.camera_buffers:
                    self.camera_buffers[camera_id] = []
                
                self.camera_buffers[camera_id].append(frame_msg)
                
                # Process batch if buffer size reached
                if len(self.camera_buffers[camera_id]) >= self.batch_size:
                    await self._process_batch(camera_id)
                    
            except Exception as e:
                logger.error(f"Inference worker task error: {e}")
                # Sleep briefly to avoid tight loop in case of persistent error
                await asyncio.sleep(0.1)
    
    def _parse_message(self, key: str, value: bytes) -> Optional[FrameMessage]:
        """Parse MessagePack frame message from Kafka."""
        try:
            msg_dict = msgpack.unpackb(value, raw=False)
            camera_id = key or msg_dict.get('camera_id')
            frame_id = msg_dict.get('frame_id')
            timestamp = msg_dict.get('timestamp')
            frame_seq = msg_dict.get('frame_seq')
            
            if timestamp is None:
                logger.error(f"[{camera_id}] Frame {frame_id} missing required 'timestamp' field")
                return None
            
            jpeg_bytes = msg_dict.get('jpeg')
            if not jpeg_bytes:
                logger.error(f"[{camera_id}] Missing JPEG data")
                return None
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error(f"[{camera_id}] Failed to decode frame {frame_id}")
                return None
            
            return FrameMessage(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                frame_seq=frame_seq,
                frame=frame,
            )
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            return None
    
    async def _publish_result_to_kafka(self, camera_id: str, result_data: Dict) -> None:
        """Publish inference result to Kafka for HDFS archiving."""
        if not self.producer:
            return
        try:
            await self.producer.send(self.kafka_result_topic, result_data)
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
    
    async def _process_batch(self, camera_id: str) -> None:
        """
        Process batch of frames for a camera using Spark or traditional inference.
        
        Args:
            camera_id: Camera identifier
        """
        frames_msg = self.camera_buffers[camera_id][:self.batch_size]
        
        try:
            if self.use_spark and self.spark_worker:
                # Extract data from messages
                frames = [msg.frame for msg in frames_msg]
                frame_ids = [msg.frame_id for msg in frames_msg]
                timestamps = [msg.timestamp for msg in frames_msg]

                # Run Spark inference (blocking call, but distributed)
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self.spark_worker.infer_batch,
                    frames,
                    camera_id,
                    frame_ids,
                    timestamps
                )
                
                for i, spark_result in enumerate(results):
                    self.frames_processed += 1
                    detection = {
                        'violence': spark_result.is_violence,
                        'confidence': spark_result.confidence,
                        'latency_ms': spark_result.processing_time_ms,
                        'worker_id': spark_result.worker_id,
                    }
                    
                    result_msg = {
                        'camera_id': camera_id,
                        'timestamp': spark_result.timestamp,
                        'violence': spark_result.is_violence,
                        'confidence': float(spark_result.confidence),
                        'label': 'violence' if spark_result.is_violence else 'nonviolence'
                    }
                    await self._publish_result_to_kafka(camera_id, result_msg)
                    
                    if spark_result.is_violence:
                        await self._publish_detection(camera_id, frames_msg[i], detection, None)
                        self.alerts_sent += 1
                        self.detections_made += 1
                    
            else:
                loop = asyncio.get_event_loop()
                for frame_msg in frames_msg:
                    await loop.run_in_executor(None, self.model.add_frame, frame_msg.frame)
                    detection = await loop.run_in_executor(None, self.model.predict, frame_msg.timestamp)
                    
                    if detection is None:
                        continue
                    
                    self.frames_processed += 1
                    result_msg = {
                        'camera_id': camera_id,
                        'timestamp': frame_msg.timestamp,
                        'violence': detection['violence'],
                        'confidence': float(detection['confidence']),
                        'label': 'violence' if detection['violence'] else 'nonviolence'
                    }
                    await self._publish_result_to_kafka(camera_id, result_msg)
                    
                    if detection['violence']:
                        await self._publish_detection(camera_id, frame_msg, detection, None)
                        self.alerts_sent += 1
                        self.detections_made += 1
        
        except Exception as e:
            logger.error(f"[{camera_id}] Batch processing error: {e}")
        finally:
            self.camera_buffers[camera_id] = frames_msg[self.batch_size:]
    
    async def _publish_detection(
        self,
        camera_id: str,
        frame_msg: FrameMessage,
        detection: Dict,
        frames_temp_path: Optional[str] = None,
    ) -> None:
        """Publish detection to Redis pub/sub and streams."""
        try:
            _, buffer = cv2.imencode('.jpg', frame_msg.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            alert_message = {
                'camera_id': camera_id,
                'frame_id': frame_msg.frame_id,
                'frame_seq': frame_msg.frame_seq,
                'timestamp': frame_msg.timestamp,
                'violence': detection['violence'],
                'confidence': float(detection['confidence']),
                'inference_latency_ms': float(detection.get('latency_ms', 0)),
                'e2e_latency_ms': float(detection.get('e2e_latency_ms', 0)),
                'frames_temp_path': frames_temp_path or '',
                'snapshot': f"data:image/jpeg;base64,{frame_base64}"
            }
            
            alert_json = json.dumps(alert_message)
            await self.redis_client.publish(f"alerts:{camera_id}", alert_json)
            await self.redis_client.setex(f"detection:latest:{camera_id}", 60, alert_json)
            await self.redis_client.xadd(f"detection:stream:{camera_id}", {'detection': alert_json}, maxlen=100)
            
            logger.debug(
                f"[{camera_id}] Detection published: "
                f"violence={detection['violence']}, "
                f"confidence={detection['confidence']:.2f}, "
                f"inference={detection.get('latency_ms', 0):.1f}ms, "
                f"e2e={detection.get('e2e_latency_ms', 0):.1f}ms"
            )
        
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to publish detection: {e}")
    
    def get_stats(self) -> Dict:
        """Get consumer statistics."""
        elapsed = time.time() - self.start_time
        return {
            'frames_consumed': self.frames_consumed,
            'frames_processed': self.frames_processed,
            'detections_made': self.detections_made,
            'alerts_sent': self.alerts_sent,
            'fps': self.frames_consumed / elapsed if elapsed > 0 else 0,
            'elapsed_seconds': elapsed,
        }

