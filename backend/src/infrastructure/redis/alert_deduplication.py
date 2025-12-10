"""Alert deduplication service using Redis."""

import logging
from typing import Optional, Dict, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class AlertDeduplication:
    """
    Deduplicates violence alerts to prevent spam.
    
    Logic:
    - When violence detected, check if alert already exists for this camera
    - If yes and recent (within TTL), skip sending alert
    - If no, create alert entry with TTL
    - TTL = alert cooldown period (default 60s)
    
    Data flow: inference_consumer → [check Redis] → event_processor → Firebase
    
    Example:
        Key: "alert:active:camera_01"
        Value: "1" (timestamp is implicit in Redis TTL)
        TTL: 60 seconds
        
        Result: Max 1 alert per camera per 60 seconds
    """
    
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 60):
        """
        Initialize deduplication service.
        
        Args:
            redis_client: Redis async client
            ttl_seconds: Alert cooldown period (seconds)
                        Don't send duplicate alerts within this period
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
    
    def _get_alert_key(self, camera_id: str) -> str:
        """Get Redis key for camera alert state."""
        return f"alert:active:{camera_id}"
    
    async def should_send_alert(self, camera_id: str) -> bool:
        """
        Check if should send alert for this camera.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            True if should send alert (first time or cooldown expired)
            False if already sent within cooldown period
        """
        key = self._get_alert_key(camera_id)
        
        try:
            # Check if alert already exists (not expired)
            exists = await self.redis_client.exists(key)
            
            if exists:
                logger.debug(
                    f"[{camera_id}] Alert already sent within last {self.ttl_seconds}s, "
                    f"skipping to avoid spam"
                )
                return False
            
            # Create alert entry with TTL
            await self.redis_client.setex(
                key,
                self.ttl_seconds,
                "1"  # Simple flag value
            )
            
            logger.info(
                f"[{camera_id}] Alert entry created (cooldown: {self.ttl_seconds}s)"
            )
            return True
        
        except Exception as e:
            logger.error(f"[{camera_id}] Deduplication check failed: {e}")
            # On error, allow sending (fail-open) - better to send extra alert
            # than to miss an actual violence event
            return True
    
    async def clear_alert(self, camera_id: str) -> None:
        """
        Clear alert state (e.g., when violence stops).
        
        Useful to reset cooldown if situation resolved.
        """
        key = self._get_alert_key(camera_id)
        try:
            deleted = await self.redis_client.delete(key)
            if deleted:
                logger.info(f"[{camera_id}] Alert cleared (can now send new alert)")
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to clear alert: {e}")
    
    async def get_ttl(self, camera_id: str) -> Optional[int]:
        """Get remaining TTL for alert cooldown."""
        key = self._get_alert_key(camera_id)
        try:
            ttl = await self.redis_client.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"[{camera_id}] Failed to get TTL: {e}")
            return None


def get_alert_deduplication(
    redis_client: redis.Redis,
    ttl_seconds: int = 60,
) -> AlertDeduplication:
    """Get deduplication service instance."""
    return AlertDeduplication(redis_client, ttl_seconds)
