"""
Inference consumer that listens to Kafka and processes frames.

Architecture:
- Consumes frames from Kafka (partitioned by camera_id)
- Maintains per-camera buffers for temporal continuity
- Performs batch inference on GPU
- Publishes detection results to Redis
- Uses alert deduplication to prevent spam

Data flow: Kafka → inference_consumer → [model] → Redis alerts → event_processor
"""

import asyncio
import logging
import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import cv2
import msgpack
from aiokafka import AIOKafkaConsumer
import redis.asyncio as redis

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
    Consumes frames from Kafka, performs inference, publishes results to Redis.
    
    Key features:
    - Per-camera buffers (maintains temporal ordering from Kafka partitioning)
    - Batch processing (default 4 frames per batch for GPU efficiency)
    - Alert deduplication (max 1 alert per camera per 60s)
    - Graceful error handling and metrics
    - Async I/O (non-blocking)
    """
    
    def __init__(
        self,
        model,  # ViolenceDetectionModel instance
        kafka_bootstrap_servers: str = None,
        kafka_topic: str = None,
        kafka_group_id: str = None,
        redis_url: str = None,
        batch_size: int = None,
        batch_timeout_ms: int = None,
        alert_cooldown_seconds: int = None,
    ):
        """
        Initialize inference consumer.
        
        Args:
            model: ViolenceDetectionModel instance
            kafka_bootstrap_servers: Kafka broker address (from KAFKA_BOOTSTRAP_SERVERS env, required)
            kafka_topic: Topic to consume frames from (from KAFKA_FRAME_TOPIC env, required)
            kafka_group_id: Kafka consumer group (from KAFKA_CONSUMER_GROUP env, required)
            redis_url: Redis connection URL (from REDIS_URL env, required)
            batch_size: Frames to accumulate before inference (from INFERENCE_BATCH_SIZE env, required)
            batch_timeout_ms: Max wait for batch (from INFERENCE_BATCH_TIMEOUT_MS env, required)
            alert_cooldown_seconds: Alert deduplication cooldown (from ALERT_COOLDOWN_SECONDS env, required)
        """
        # Load from environment - all required, fail if missing
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.kafka_topic = kafka_topic or os.getenv("KAFKA_FRAME_TOPIC")
        self.kafka_group_id = kafka_group_id or os.getenv("KAFKA_CONSUMER_GROUP")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.batch_size = batch_size or int(os.getenv("INFERENCE_BATCH_SIZE", "") or 0)
        self.batch_timeout_ms = batch_timeout_ms or int(os.getenv("INFERENCE_BATCH_TIMEOUT_MS", "") or 0)
        self.alert_cooldown_seconds = alert_cooldown_seconds or int(os.getenv("ALERT_COOLDOWN_SECONDS", "") or 0)
        
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
        if self.alert_cooldown_seconds <= 0:
            raise ValueError("ALERT_COOLDOWN_SECONDS env variable not set or invalid")
        
        self.model = model
        
        # State
        self.is_running = False
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Per-camera buffers and state
        self.camera_buffers: Dict[str, List[FrameMessage]] = {}
        self.camera_last_alert: Dict[str, float] = {}  # Timestamp of last alert
        
        # Metrics
        self.frames_consumed = 0
        self.frames_processed = 0
        self.detections_made = 0
        self.alerts_sent = 0
        self.alerts_deduped = 0
        self.start_time = time.time()
    
    async def start(self) -> None:
        """Start consuming from Kafka."""
        try:
            logger.info("Starting Inference Consumer...")
            
            # Connect to Kafka
            # Increase timeouts to handle slow CPU inference (12+ seconds per frame)
            self.consumer = AIOKafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id=self.kafka_group_id,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda m: m,  # Raw bytes for MessagePack binary format
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                # Prevent Kafka from kicking consumer during slow inference
                session_timeout_ms=60000,  # 60 seconds (default 10s)
                heartbeat_interval_ms=20000,  # 20 seconds (should be < session_timeout/3)
                max_poll_interval_ms=300000,  # 5 minutes for slow batch processing
            )
            await self.consumer.start()
            logger.info(f"Kafka consumer started: topic={self.kafka_topic}")
            
            # Connect to Redis
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connected for publishing results")
            
            self.is_running = True
            
            # Start main loop
            asyncio.create_task(self._run())
            logger.info("Inference consumer running")
        
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop consuming."""
        logger.info("Stopping Inference Consumer...")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Inference consumer stopped")
    
    async def _run(self) -> None:
        """Main consumer loop."""
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
                    
                    # Add to per-camera buffer
                    camera_id = frame_msg.camera_id
                    if camera_id not in self.camera_buffers:
                        self.camera_buffers[camera_id] = []
                    
                    self.camera_buffers[camera_id].append(frame_msg)
                    
                    # Process batch if buffer size reached
                    if len(self.camera_buffers[camera_id]) >= self.batch_size:
                        await self._process_batch(camera_id)
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except Exception as e:
            logger.error(f"Consumer error: {e}")
    
    def _parse_message(self, key: str, value: bytes) -> Optional[FrameMessage]:
        """
        Parse Kafka message into FrameMessage.
        
        Expected format (MessagePack binary):
        {
            'camera_id': str,
            'frame_id': str,
            'timestamp': float,
            'frame_seq': int,
            'jpeg': bytes (raw JPEG binary, no Base16 encoding),
            'frame_shape': [height, width, channels]
        }
        """
        try:
            # Unpack MessagePack binary format
            msg_dict = msgpack.unpackb(value, raw=False)
            
            camera_id = key or msg_dict.get('camera_id')
            frame_id = msg_dict.get('frame_id')
            timestamp = msg_dict.get('timestamp')
            frame_seq = msg_dict.get('frame_seq')
            
            # Get JPEG bytes directly (no hex decoding needed)
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
    
    async def _process_batch(self, camera_id: str) -> None:
        """
        Process batch of frames for a camera.
        
        Args:
            camera_id: Camera identifier
        """
        frames = self.camera_buffers[camera_id][:self.batch_size]
        
        try:
            for frame_msg in frames:
                # Add frame to model buffer
                self.model.add_frame(frame_msg.frame)
                
                # Try to get inference result (returns None if buffer not full)
                detection = self.model.predict()
                
                if detection is None:
                    # Model buffer not full yet, continue accumulating
                    continue
                
                self.frames_processed += 1
                
                # Check if should send alert (deduplication)
                if not self._should_alert(camera_id, frame_msg.timestamp):
                    self.alerts_deduped += 1
                    continue
                
                # Publish detection result to Redis
                await self._publish_detection(camera_id, frame_msg, detection)
                self.alerts_sent += 1
                self.detections_made += 1
        
        except Exception as e:
            logger.error(f"[{camera_id}] Batch processing error: {e}")
        
        finally:
            # Remove processed frames from buffer (sliding window)
            self.camera_buffers[camera_id] = frames[self.batch_size:]
    
    def _should_alert(self, camera_id: str, timestamp: float) -> bool:
        """
        Check deduplication: should we send alert for this detection?
        
        Logic: Don't spam alerts. Max 1 per camera per cooldown period.
        
        Args:
            camera_id: Camera identifier
            timestamp: Frame timestamp
        
        Returns:
            True if should send alert, False if within cooldown period
        """
        last_alert = self.camera_last_alert.get(camera_id, 0)
        
        # Check if last alert was within cooldown period
        if timestamp - last_alert < self.alert_cooldown_seconds:
            logger.debug(
                f"[{camera_id}] Alert deduped "
                f"(last alert {timestamp - last_alert:.1f}s ago, "
                f"cooldown {self.alert_cooldown_seconds}s)"
            )
            return False
        
        # Update last alert time
        self.camera_last_alert[camera_id] = timestamp
        return True
    
    async def _publish_detection(
        self,
        camera_id: str,
        frame_msg: FrameMessage,
        detection: Dict,
    ) -> None:
        """
        Publish detection result to Redis.
        
        Uses Redis pub/sub for real-time alerts and stores in streams.
        
        Args:
            camera_id: Camera identifier
            frame_msg: Frame message with metadata
            detection: Detection result from model
        """
        try:
            # Prepare alert message
            alert_message = {
                'camera_id': camera_id,
                'frame_id': frame_msg.frame_id,
                'frame_seq': frame_msg.frame_seq,
                'timestamp': frame_msg.timestamp,
                'violence': detection['violence'],
                'confidence': float(detection['confidence']),
                'latency_ms': float(detection.get('latency_ms', 0)),
            }
            
            alert_json = json.dumps(alert_message)
            
            # Publish via pub/sub (real-time, in-memory)
            await self.redis_client.publish(
                f"alerts:{camera_id}",
                alert_json
            )
            
            # Store latest detection (with TTL)
            await self.redis_client.setex(
                f"detection:latest:{camera_id}",
                self.alert_cooldown_seconds,
                alert_json,
            )
            
            # Append to stream (persistent log)
            # Note: xadd requires string values, use alert_json
            await self.redis_client.xadd(
                f"detection:stream:{camera_id}",
                {'detection': alert_json},
                maxlen=100,  # Keep last 100 detections per camera
            )
            
            logger.info(
                f"[{camera_id}] Detection published: "
                f"violence={detection['violence']}, "
                f"confidence={detection['confidence']:.2f}, "
                f"latency={detection.get('latency_ms', 0):.1f}ms"
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
            'alerts_deduped': self.alerts_deduped,
            'fps': self.frames_consumed / elapsed if elapsed > 0 else 0,
            'elapsed_seconds': elapsed,
        }
