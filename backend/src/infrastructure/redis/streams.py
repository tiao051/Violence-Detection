"""Redis Streams producer for frame metadata and threat detection."""

import json
import logging
from typing import Optional, Dict, Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisStreamProducer:
    """
    Redis Streams producer for publishing frame metadata and threat detection results.
    
    Features:
    - Add frame metadata to Redis Streams
    - Store latest threat detection per camera
    - Auto cleanup old entries (TTL)
    - Minimal payload optimization
    """
    
    def __init__(self, redis_client: redis.Redis, ttl_minutes: int = 5):
        """
        Initialize Redis producer.
        
        Args:
            redis_client: Redis async client instance
            ttl_minutes: Time to live for frames (auto-cleanup)
        """
        self.redis_client = redis_client
        self.ttl_minutes = ttl_minutes
        self.ttl_ms = ttl_minutes * 60 * 1000
        self.detection_ttl_seconds = 60  # Detection results expire after 60 seconds
    
    async def add_frame_metadata(
        self,
        camera_id: str,
        frame_id: str,
        timestamp: float,
        frame_seq: int,
        detection: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Add frame metadata to Redis Stream with optional detection result.
        
        The actual frame data is stored in shared RAM buffer. This only stores metadata
        which includes frame reference and threat detection results if available.
        
        Args:
            camera_id: Camera identifier (e.g., "cam1")
            frame_id: Unique frame ID (UUID)
            timestamp: Frame timestamp (unix time seconds)
            frame_seq: Frame sequence number
            detection: Optional detection result dict with keys:
                - violence: bool (True if violence detected)
                - confidence: float (0-1)
                - class_id: int (0=Violence, 1=NonViolence)
                - buffer_size: int (frames in buffer)
        
        Returns:
            Stream message ID or None if failed
        """
        try:
            stream_key = f"frames:{camera_id}"
            
            # Prepare metadata
            metadata = {
                "frame_id": frame_id,
                "cam": camera_id,
                "ts": str(timestamp),
                "seq": str(frame_seq),
                "location": "ram://frame_buffer",
            }
            
            # Add detection results if available
            if detection:
                metadata["detection"] = json.dumps(detection)
            
            # Add to stream
            message_id = await self.redis_client.xadd(stream_key, metadata)
            
            # Store latest threat detection for quick access
            if detection and detection.get('violence'):
                await self._store_threat_detection(camera_id, detection, timestamp)
            
            return message_id
        
        except Exception as e:
            logger.error(f"Failed to add frame metadata for {camera_id}: {e}")
            return None
    
    async def _store_threat_detection(
        self,
        camera_id: str,
        detection: Dict[str, Any],
        timestamp: float
    ) -> None:
        """
        Store latest threat detection in a separate key for quick access.
        
        Args:
            camera_id: Camera identifier
            detection: Detection result dictionary
            timestamp: Detection timestamp
        """
        try:
            threat_key = f"threat:{camera_id}"
            
            threat_data = {
                "camera_id": camera_id,
                "violence": str(detection.get('violence', False)).lower(),
                "confidence": str(detection.get('confidence', 0.0)),
                "timestamp": str(timestamp),
            }
            
            # Store with TTL
            await self.redis_client.hset(threat_key, mapping=threat_data)
            await self.redis_client.expire(threat_key, self.detection_ttl_seconds)
        
        except Exception as e:
            logger.error(f"Failed to store threat detection for {camera_id}: {e}")
    
    async def get_latest_threat(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Get latest threat detection for a camera.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Latest threat detection dict or None if no threat
        """
        try:
            threat_key = f"threat:{camera_id}"
            threat_data = await self.redis_client.hgetall(threat_key)
            
            if not threat_data:
                return None
            
            return {
                "camera_id": threat_data.get(b'camera_id', b'').decode(),
                "violence": threat_data.get(b'violence', b'false').decode().lower() == 'true',
                "confidence": float(threat_data.get(b'confidence', b'0.0').decode()),
                "timestamp": float(threat_data.get(b'timestamp', b'0').decode()),
            }
        
        except Exception as e:
            logger.error(f"Failed to get latest threat for {camera_id}: {e}")
            return None
    
    async def get_all_threats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get latest threat status for all cameras.
        
        Returns:
            Dictionary mapping camera_id to threat status
        """
        try:
            # Scan for all threat: keys
            cursor = 0
            threats = {}
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match="threat:*",
                    count=100
                )
                
                for key in keys:
                    camera_id = key.decode().split(":", 1)[1]
                    threat = await self.get_latest_threat(camera_id)
                    if threat:
                        threats[camera_id] = threat
                
                if cursor == 0:
                    break
            
            return threats
        
        except Exception as e:
            logger.error(f"Failed to get all threats: {e}")
            return {}


# Singleton instance
_redis_stream_producer: Optional[RedisStreamProducer] = None


def get_redis_stream_producer() -> RedisStreamProducer:
    """Get Redis stream producer singleton."""
    global _redis_stream_producer
    if _redis_stream_producer is None:
        raise RuntimeError("Redis stream producer not initialized")
    return _redis_stream_producer


def set_redis_stream_producer(producer: RedisStreamProducer) -> None:
    """Set Redis stream producer singleton."""
    global _redis_stream_producer
    _redis_stream_producer = producer