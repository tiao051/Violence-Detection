"""Event Processor for handling real-time threat alerts."""

import logging
import json
import asyncio
import time
import os
import shutil
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
        
        # Active recording sessions: {camera_id: {'start_time': float, 'last_seen': float, 'max_confidence': float}}
        self.active_events: Dict[str, Dict] = {}
        
        # Config
        self.event_timeout_seconds = 5.0  # Wait 5s after last alert to close event
        self.min_event_duration = 1.0     # Ignore blips shorter than 1s
        self.pre_event_padding = 2.0      # Seconds to include before event
        self.post_event_padding = 2.0     # Seconds to include after event

    async def start(self) -> None:
        """Start the event processor background task."""
        if self.is_running:
            return
            
        self.is_running = True
        asyncio.create_task(self._run())
        asyncio.create_task(self._event_monitor_loop()) # New task for monitoring timeouts
        logger.info("Event Processor started with Debounce & Extend logic")

    async def stop(self) -> None:
        """Stop the event processor."""
        self.is_running = False
        logger.info("Event Processor stopping...")

    async def _run(self) -> None:
        """Main loop listening to Redis."""
        pubsub = self.redis_client.pubsub()
        await pubsub.psubscribe("alerts:*")
        
        logger.info("Event Processor subscribed to 'alerts:*' pattern")

        try:
            while self.is_running:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    
                    if message and message['type'] == 'pmessage':
                        await self._handle_message(message['data'])
                    
                    # Small sleep to prevent tight loop if get_message returns immediately
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Event Processor loop error: {e}")
                    await asyncio.sleep(1.0)
        finally:
            await pubsub.punsubscribe("alerts:*")
            await pubsub.close()
            logger.info("Event Processor stopped")

    async def _handle_message(self, data) -> None:
        """Process a single alert message."""
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            alert = json.loads(data)
            camera_id = alert.get('camera_id')
            confidence = alert.get('confidence', 0)
            
            if not camera_id:
                return

            now = time.time()
            
            # Logic: Debounce & Extend
            if camera_id not in self.active_events:
                # START NEW EVENT
                logger.info(f"[{camera_id}] New violence event started (conf={confidence:.2f})")
                self.active_events[camera_id] = {
                    'start_time': now,
                    'last_seen': now,
                    'max_confidence': confidence,
                    'first_alert': alert, # Keep first alert for metadata
                    'frames_temp_paths': [] # Collect all temp paths
                }
            else:
                # EXTEND EXISTING EVENT
                logger.debug(f"[{camera_id}] Extending event (conf={confidence:.2f})")
                self.active_events[camera_id]['last_seen'] = now
                self.active_events[camera_id]['max_confidence'] = max(
                    self.active_events[camera_id]['max_confidence'], 
                    confidence
                )
            
            # Collect temp frames path if available
            frames_path = alert.get('frames_temp_path')
            if frames_path:
                self.active_events[camera_id]['frames_temp_paths'].append(frames_path)
                
        except json.JSONDecodeError:
            logger.error("Failed to decode alert message JSON")
        except Exception as e:
            logger.error(f"Failed to process alert message: {e}")

    async def _event_monitor_loop(self) -> None:
        """
        Background task to monitor and close finished events.
        Checks every 1s if any event has timed out.
        """
        while self.is_running:
            try:
                now = time.time()
                # Create list of keys to avoid runtime error during iteration
                active_cameras = list(self.active_events.keys())
                
                for camera_id in active_cameras:
                    event = self.active_events[camera_id]
                    
                    # Check timeout (No new alerts for X seconds)
                    if now - event['last_seen'] > self.event_timeout_seconds:
                        # Event finished! Process it.
                        duration = event['last_seen'] - event['start_time']
                        
                        logger.info(f"[{camera_id}] Event finished. Duration: {duration:.1f}s. Saving...")
                        
                        # Use the first alert as template, but update confidence
                        final_alert = event['first_alert'].copy()
                        final_alert['confidence'] = event['max_confidence']
                        
                        # Calculate time window for video extraction
                        # Start = First Alert - Padding
                        # End = Last Alert + Padding
                        video_start_time = event['start_time'] - self.pre_event_padding
                        video_end_time = event['last_seen'] + self.post_event_padding
                        
                        # Trigger persistence with time window
                        event_id = await self.persistence_service.save_event(
                            camera_id, 
                            final_alert, 
                            frames_temp_paths=event['frames_temp_paths'], # Pass list of paths
                            start_timestamp=video_start_time,
                            end_timestamp=video_end_time
                        )
                        
                        if event_id:
                            logger.info(f"[{camera_id}] Event saved successfully: {event_id}")
                        
                        # Remove from active list
                        del self.active_events[camera_id]
                        
                        # Cleanup temp frames
                        for path in event['frames_temp_paths']:
                            if path and os.path.exists(path):
                                try:
                                    shutil.rmtree(path)
                                except: pass
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Event monitor error: {e}")
                await asyncio.sleep(1.0)


# Singleton instance
_event_processor: Optional[EventProcessor] = None

def get_event_processor(redis_client: redis.Redis = None) -> Optional[EventProcessor]:
    global _event_processor
    if _event_processor is None and redis_client:
        _event_processor = EventProcessor(redis_client)
    return _event_processor
