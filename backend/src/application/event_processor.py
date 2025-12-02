"""Event Processor for handling real-time threat alerts."""

import logging
import json
import asyncio
import time
from typing import Dict, Optional
import redis.asyncio as redis
from src.infrastructure.storage.event_persistence import get_event_persistence_service

logger = logging.getLogger(__name__)


class EventProcessor:
    """
    Background worker that processes threat alerts.
    
    1. Listens to Redis pub/sub 'threat_alerts'.
    2. Debounces alerts (prevents duplicate events for same incident).
    3. Triggers EventPersistenceService to save video and metadata.
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.persistence_service = get_event_persistence_service()
        self.is_running = False
        self.last_event_time: Dict[str, float] = {}
        self.debounce_seconds = 30.0  # Minimum seconds between events for same camera

    async def start(self) -> None:
        """Start the event processor background task."""
        if self.is_running:
            return
            
        self.is_running = True
        asyncio.create_task(self._run())
        logger.info("Event Processor started")

    async def stop(self) -> None:
        """Stop the event processor."""
        self.is_running = False
        logger.info("Event Processor stopping...")

    async def _run(self) -> None:
        """Main loop listening to Redis."""
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("threat_alerts")
        
        logger.info("Event Processor subscribed to 'threat_alerts'")

        try:
            while self.is_running:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    
                    if message and message['type'] == 'message':
                        await self._handle_message(message['data'])
                    
                    # Small sleep to prevent tight loop if get_message returns immediately
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Event Processor loop error: {e}")
                    await asyncio.sleep(1.0)
        finally:
            await pubsub.unsubscribe("threat_alerts")
            await pubsub.close()
            logger.info("Event Processor stopped")

    async def _handle_message(self, data) -> None:
        """Process a single alert message."""
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            alert = json.loads(data)
            camera_id = alert.get('camera_id')
            
            if not camera_id:
                logger.warning("Received alert without camera_id")
                return

            # Debounce check
            now = time.time()
            last_time = self.last_event_time.get(camera_id, 0)
            
            if now - last_time < self.debounce_seconds:
                logger.debug(f"Skipping duplicate event for {camera_id} (debounce active)")
                return
            
            # Valid new event -> Persist it
            logger.info(f"Processing new violence event for {camera_id} (confidence={alert.get('confidence')})")
            
            # Trigger persistence (Video + Firestore)
            event_id = await self.persistence_service.save_event(camera_id, alert)
            
            if event_id:
                self.last_event_time[camera_id] = now
                logger.info(f"Event processed and saved: {event_id}")
            else:
                logger.warning(f"Failed to save event for {camera_id}")
                
        except json.JSONDecodeError:
            logger.error("Failed to decode alert message JSON")
        except Exception as e:
            logger.error(f"Failed to process alert message: {e}")


# Singleton instance
_event_processor: Optional[EventProcessor] = None

def get_event_processor(redis_client: redis.Redis = None) -> Optional[EventProcessor]:
    global _event_processor
    if _event_processor is None and redis_client:
        _event_processor = EventProcessor(redis_client)
    return _event_processor
