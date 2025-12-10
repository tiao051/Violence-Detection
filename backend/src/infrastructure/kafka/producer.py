"""Kafka producer for frame publishing."""

import json
import logging
from typing import Dict, Any, Optional
import numpy as np
import cv2
from aiokafka import AIOKafkaProducer

from ...core.config import settings

logger = logging.getLogger(__name__)


class KafkaFrameProducer:
    """
    Kafka producer for sending sampled frames.
    
    Features:
    - Resizes frames to model input size (224x224)
    - Compresses as JPEG before sending (reduce size ~10x)
    - Partitions by camera_id to maintain temporal ordering
    - Handles connection management
    - Batches messages for throughput
    
    Data flow: camera_worker → [resize + compress] → Kafka
    """
    
    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        compression_type: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        target_width: int = 224,
        target_height: int = 224,
    ):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker address (from config if not provided)
            topic: Topic to publish frames (from config if not provided)
            compression_type: Compression type (from config if not provided)
            jpeg_quality: JPEG quality (from config if not provided)
            target_width: Target frame width for resize
            target_height: Target frame height for resize
        """
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_frame_topic
        self.compression_type = compression_type or settings.kafka_compression_type
        self.jpeg_quality = jpeg_quality or settings.kafka_jpeg_quality
        self.target_width = target_width
        self.target_height = target_height
        
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_connected = False
        
        # Metrics
        self.frames_sent = 0
        self.frames_failed = 0
        self.total_bytes_sent = 0
    
    async def connect(self) -> None:
        """Connect to Kafka broker."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                compression_type=self.compression_type,
                max_batch_size=1048576,  # 1MB per batch
                linger_ms=10,  # Wait 10ms to batch messages
                acks='1',  # Wait for leader ack (faster than 'all')
            )
            await self.producer.start()
            self.is_connected = True
            logger.info(
                f"Kafka producer connected to {self.bootstrap_servers}, "
                f"topic={self.topic}, compression={self.compression_type}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self.producer:
            await self.producer.stop()
            self.is_connected = False
            logger.info("Kafka producer disconnected")
    
    def _resize_and_compress(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Resize frame to target size and compress as JPEG.
        
        This is the single resize point for optimization.
        Only called here to avoid double resizing.
        
        Args:
            frame: Input frame (BGR, uint8, any resolution)
        
        Returns:
            JPEG bytes or None if failed
        """
        try:
            # Check if resize is needed
            current_h, current_w = frame.shape[:2]
            
            # Only resize if dimensions differ from target
            if (current_w, current_h) != (self.target_width, self.target_height):
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # Compress as JPEG
            success, jpeg_data = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            
            if not success:
                return None
            
            return jpeg_data.tobytes()
        
        except Exception as e:
            logger.error(f"Frame compression error: {e}")
            return None
    
    async def send_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_id: str,
        timestamp: float,
        frame_seq: int,
    ) -> Optional[str]:
        """
        Send frame to Kafka.
        
        Args:
            camera_id: Camera identifier (used as Kafka key for partitioning)
            frame: BGR frame (uint8)
            frame_id: Unique frame ID
            timestamp: Frame timestamp
            frame_seq: Frame sequence number
        
        Returns:
            Message ID or None if failed
        """
        if not self.is_connected:
            logger.warning("Kafka producer not connected, frame dropped")
            return None
        
        try:
            # Resize and compress frame
            jpeg_bytes = self._resize_and_compress(frame)
            
            if jpeg_bytes is None:
                logger.error(f"[{camera_id}] Frame encoding failed")
                self.frames_failed += 1
                return None
            
            # Prepare JSON message with base16 encoded JPEG
            message = {
                'camera_id': camera_id,
                'frame_id': frame_id,
                'timestamp': timestamp,
                'frame_seq': frame_seq,
                'jpeg': jpeg_bytes.hex(),  # Base16 encode
                'frame_shape': [self.target_height, self.target_width, 3],
            }
            
            message_json = json.dumps(message).encode('utf-8')
            
            # Send to Kafka with camera_id as key
            # This ensures all frames from same camera go to same partition
            # and are processed in order by same consumer
            await self.producer.send_and_wait(
                self.topic,
                value=message_json,
                key=camera_id.encode('utf-8'),
            )
            
            self.frames_sent += 1
            self.total_bytes_sent += len(message_json)
            
            return frame_id
        
        except Exception as e:
            logger.error(f"[{camera_id}] Kafka send error: {e}")
            self.frames_failed += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            'frames_sent': self.frames_sent,
            'frames_failed': self.frames_failed,
            'total_bytes_sent': self.total_bytes_sent,
            'avg_frame_size_kb': (
                self.total_bytes_sent / (self.frames_sent * 1024)
                if self.frames_sent > 0 else 0
            ),
        }


# Singleton instance
_kafka_producer: Optional[KafkaFrameProducer] = None


def get_kafka_producer(
    bootstrap_servers: Optional[str] = None,
    **kwargs
) -> KafkaFrameProducer:
    """
    Get or create Kafka producer singleton.
    
    Uses settings from config if not provided.
    """
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = KafkaFrameProducer(
            bootstrap_servers=bootstrap_servers or settings.kafka_bootstrap_servers,
            **kwargs
        )
    return _kafka_producer
