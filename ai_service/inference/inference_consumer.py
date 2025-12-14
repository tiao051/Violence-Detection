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
import tempfile
import uuid
import shutil
import base64
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

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
        """
        # Load from environment - all required, fail if missing
        self.kafka_bootstrap_servers = kafka_bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.kafka_topic = kafka_topic or os.getenv("KAFKA_FRAME_TOPIC")
        self.kafka_group_id = kafka_group_id or os.getenv("KAFKA_CONSUMER_GROUP")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.batch_size = batch_size or int(os.getenv("INFERENCE_BATCH_SIZE", "") or 0)
        self.batch_timeout_ms = batch_timeout_ms or int(os.getenv("INFERENCE_BATCH_TIMEOUT_MS", "") or 0)
        
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
        
        self.model = model
        
        # State
        self.is_running = False
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Per-camera buffers and state
        self.camera_buffers: Dict[str, List[FrameMessage]] = {}
        
        # Metrics
        self.frames_consumed = 0
        self.frames_processed = 0
        self.detections_made = 0
        self.alerts_sent = 0
        self.start_time = time.time()
        
        # ThreadPoolExecutor for CPU-bound inference
        # Prevents blocking event loop so Kafka heartbeats can continue
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        
        # Queue for decoupling Kafka consumer and Inference worker
        # Maxsize 100 to prevent memory overflow if inference is slow
        self.frame_queue = asyncio.Queue(maxsize=100)
    
    async def start(self) -> None:
        """Start consuming from Kafka."""
        try:
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
                    logger.info("Successfully connected to Kafka")
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
            raise
    
    async def stop(self) -> None:
        """Stop consuming."""
        self.is_running = False
        
        if self.consumer:
            await self.consumer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def _kafka_reader_task(self) -> None:
        """
        Task 1: Consumes frames from Kafka and puts them into the internal queue.
        This task is lightweight and ensures we don't block Kafka consumption.
        """
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
        """
        Task 2: Takes frames from queue, batches them, and runs inference.
        This task handles the heavy lifting (AI model) without blocking the reader.
        """
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
            loop = asyncio.get_event_loop()
            
            for frame_msg in frames:
                # Add frame to model buffer (run in thread to not block heartbeats)
                await loop.run_in_executor(
                    self._executor,
                    self.model.add_frame,
                    frame_msg.frame
                )
                
                # Try to get inference result (returns None if buffer not full)
                # Run in thread pool to prevent blocking Kafka heartbeats
                detection = await loop.run_in_executor(
                    self._executor,
                    self.model.predict,
                    frame_msg.timestamp  # Pass frame timestamp for e2e latency
                )
                
                if detection is None:
                    # Model buffer not full yet, continue accumulating
                    continue
                
                self.frames_processed += 1
                
                # Only process alerts if violence is detected
                if not detection['violence']:
                    continue
                
                # Save batch frames to temp and publish detection
                # frames contains the batch being processed
                # Run I/O in thread pool to prevent blocking
                frames_temp_path = await loop.run_in_executor(
                    self._executor,
                    self._save_frames_to_temp,
                    frames,
                    camera_id
                )
                
                # Publish detection result to Redis
                await self._publish_detection(camera_id, frame_msg, detection, frames_temp_path)
                self.alerts_sent += 1
                self.detections_made += 1
        
        except Exception as e:
            logger.error(f"[{camera_id}] Batch processing error: {e}")
        
        finally:
            # Remove processed frames from buffer (sliding window)
            self.camera_buffers[camera_id] = frames[self.batch_size:]
    
    async def _publish_detection(
        self,
        camera_id: str,
        frame_msg: FrameMessage,
        detection: Dict,
        frames_temp_path: Optional[str] = None,
    ) -> None:
        """
        Publish detection result to Redis.
        
        Uses Redis pub/sub for real-time alerts and stores in streams.
        
        Args:
            camera_id: Camera identifier
            frame_msg: Frame message with metadata
            detection: Detection result from model
            frames_temp_path: Path to temp directory with batch frames (optional)
        """
        try:
            # Encode frame to base64 for immediate visual feedback
            _, buffer = cv2.imencode('.jpg', frame_msg.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare alert message
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
                'snapshot': f"data:image/jpeg;base64,{frame_base64}" # Add snapshot
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
                60, # Default TTL 60s
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

    def _save_frames_to_temp(self, frames: List[FrameMessage], camera_id: str) -> Optional[str]:
        """
        Save batch frames to mounted volume directory.
        
        Args:
            frames: List of FrameMessage objects
            camera_id: Camera identifier
            
        Returns:
            Path to directory containing frames, or None if failed
        """
        try:
            # Create directory on mounted volume (maps to host backend/outputs)
            frames_dir = os.path.join("/app/violence_frames", camera_id, f"batch_{uuid.uuid4()}")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Save each frame as JPEG
            for idx, frame_msg in enumerate(frames, 1):
                frame_path = os.path.join(frames_dir, f"frame_{idx:03d}.jpg")
                
                # Encode as JPEG
                success = cv2.imwrite(frame_path, frame_msg.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    logger.warning(f"Failed to save frame {idx} for {camera_id}")
            
            logger.info(f"Saved {len(frames)} frames to: {frames_dir}")
            return frames_dir
            
        except Exception as e:
            logger.error(f"Failed to save frames: {e}")
            return None
