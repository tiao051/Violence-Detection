"""Redis Streams producer for frame metadata."""

import logging
from typing import Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisStreamProducer:
    """
    Redis Streams producer for publishing frame metadata.
    
    Features:
    - Add frame metadata to Redis Streams (NOT frame data)
    - Auto cleanup old entries (TTL)
    - Minimal payload (~200 bytes)
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
    
    async def add_frame_metadata(
        self,
        camera_id: str,
        frame_id: str,
        timestamp: float,
        frame_seq: int,
    ) -> Optional[str]:
        """
        Add frame metadata to Redis Stream (reference to RAM buffer).
        
        This only stores metadata - the actual frame is in shared RAM.
        Consumers read metadata from Redis, then fetch frame from RAM buffer.
        
        Args:
            camera_id: Camera identifier
            frame_id: Unique frame ID (UUID)
            timestamp: Frame timestamp (unix time)
            frame_seq: Frame sequence number
        
        Returns:
            Stream message ID or None if failed
        """
        try:
            stream_key = f"frames:{camera_id}"
            
            # Add to stream (metadata only, ~150 bytes, no frame data)
            message_id = await self.redis_client.xadd(
                stream_key,
                {
                    "frame_id": frame_id,
                    "cam": camera_id,
                    "ts": str(timestamp),
                    "seq": str(frame_seq),
                    "location": "ram://frame_buffer",  # Reference to RAM
                }
            )
            
            return message_id
        
        except Exception as e:
            logger.error(f"Redis add_frame_metadata error for {camera_id}: {str(e)}")
            return None
    
    async def cleanup_old_frames(self, camera_id: str) -> int:
        """
        Remove old frame metadata from stream (keep only recent ones).
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Number of entries removed
        """
        try:
            stream_key = f"frames:{camera_id}"
            
            # Get current time (redis returns in seconds)
            current_time_ms = await self.redis_client.time()
            current_time_ms = int(current_time_ms[0] * 1000 + current_time_ms[1] / 1000)
            
            # Calculate cutoff time
            cutoff_time = current_time_ms - self.ttl_ms
            
            # Remove entries older than cutoff
            removed = await self.redis_client.xtrim(
                stream_key,
                minid=f"{cutoff_time}-0",
                approximate=False
            )
            
            if removed > 0:
                logger.debug(f"Cleaned up {removed} old frames from {stream_key}")
            
            return removed
        except Exception as e:
            logger.error(f"Redis cleanup_old_frames error for {camera_id}: {str(e)}")
            return 0
    
    async def get_stream_length(self, camera_id: str) -> int:
        """
        Get number of metadata entries in stream.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Number of entries in stream
        """
        try:
            stream_key = f"frames:{camera_id}"
            length = await self.redis_client.xlen(stream_key)
            return length
        except Exception as e:
            logger.error(f"Redis xlen error for {camera_id}: {str(e)}")
            return 0
