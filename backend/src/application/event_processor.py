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
from src.application.security_engine import get_security_engine, init_security_engine

logger = logging.getLogger(__name__)


class EventProcessor:
    """
    Firestore-First Event Processor.
    
    1. First alert → CREATE event in Firestore immediately (status: active)
    2. Higher confidence → UPDATE event in Firestore
    3. Timeout (30s) → FINALIZE event (add video, status: completed)
    
    Redis pub/sub notifications:
    - event_started: New event created
    - event_updated: Event updated with better evidence
    - event_completed: Event finalized with video
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.persistence_service = get_event_persistence_service()
        self.is_running = False
        
        # Initialize SecurityEngine at startup (loads rules into RAM)
        self.security_engine = init_security_engine()
        
        # Active recording sessions: {camera_id: {...}}
        self.active_events: Dict[str, Dict] = {}
        
        # Queue for background severity analysis
        self.severity_queue: asyncio.Queue = asyncio.Queue()
        
        # Config
        self.event_timeout_seconds = 5.0   # Wait 5s after last alert to close event (quick finalize)
        self.min_event_duration = 1.0      # Ignore blips shorter than 1s
        self.max_event_duration = 30.0     # Max 30s from first alert - then it's a new event
        self.pre_event_padding = 2.0       # Seconds to include before event
        self.post_event_padding = 2.0      # Seconds to include after event

    async def start(self) -> None:
        """Start the event processor background task."""
        if self.is_running:
            return
            
        self.is_running = True
        asyncio.create_task(self._run())
        asyncio.create_task(self._event_monitor_loop())  # Task for monitoring timeouts
        asyncio.create_task(self._severity_worker_loop())  # NEW: Background severity analysis
        logger.info("Event Processor started with Debounce & Extend logic + SecurityEngine")

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

            # Ignore system messages (like event_saved updates)
            msg_type = alert.get('type')
            if msg_type in ['event_started', 'event_updated', 'event_completed']:
                logger.debug(f"Ignored own message: {msg_type}")
                return
            
            camera_id = alert.get('camera_id')
            confidence = alert.get('confidence', 0)
            
            if not camera_id:
                return

            # Detection MUST have timestamp
            if 'timestamp' not in alert or alert['timestamp'] is None:
                logger.error(f"[{camera_id}] Alert missing required 'timestamp' field")
                return
            
            detection_timestamp = alert['timestamp']
            
            # FIRESTORE-FIRST LOGIC
            if camera_id not in self.active_events:
                # === FIRST ALERT: CREATE EVENT IN FIRESTORE ===
                event_id = await self.persistence_service.create_event(camera_id, alert)
                
                if not event_id:
                    logger.error(f"[{camera_id}] Failed to create event in Firestore")
                    return
                
                logger.info(f"[{camera_id}] New violence event started (conf={confidence:.2f}, event_id={event_id})")
                
                self.active_events[camera_id] = {
                    'event_id': event_id,  # Firestore document ID
                    'start_time': detection_timestamp,
                    'last_seen': detection_timestamp,
                    'last_processed_at': time.time(),
                    'max_confidence': confidence,
                    'best_alert': alert,
                    'frames_temp_paths': []
                }
                
                # Publish event_started to frontend (severity_level: PENDING - yellow)
                await self._publish_event_notification('event_started', camera_id, {
                    'event_id': event_id,
                    'timestamp': detection_timestamp,
                    'confidence': confidence,
                    'snapshot': alert.get('snapshot', ''),
                    'status': 'active',
                    'severity_level': 'PENDING'  # Will be updated by background worker
                })
                
                # Queue for background severity analysis (non-blocking)
                await self.severity_queue.put({
                    'event_id': event_id,
                    'camera_id': camera_id,
                    'confidence': confidence,
                    'timestamp': detection_timestamp
                })
                
            else:
                # === SUBSEQUENT ALERT: MAYBE UPDATE EVENT ===
                event = self.active_events[camera_id]
                video_duration = detection_timestamp - event['start_time']
                
                # Check if max duration exceeded - force finalize and start new event
                if video_duration > self.max_event_duration:
                    logger.info(f"[{camera_id}] Max duration exceeded ({video_duration:.1f}s). Force finalizing current event...")
                    
                    # Force finalize current event
                    video_start_time = event['start_time'] - self.pre_event_padding
                    video_end_time = event['last_seen'] + self.post_event_padding
                    
                    await self.persistence_service.finalize_event(
                        event['event_id'],
                        camera_id,
                        video_start_time,
                        video_end_time
                    )
                    
                    # Remove from active list
                    del self.active_events[camera_id]
                    
                    # Create new event for continuing violence
                    event_id = await self.persistence_service.create_event(camera_id, alert)
                    if event_id:
                        logger.info(f"[{camera_id}] New violence event started after max duration (conf={confidence:.2f}, event_id={event_id})")
                        
                        self.active_events[camera_id] = {
                            'event_id': event_id,
                            'start_time': detection_timestamp,
                            'last_seen': detection_timestamp,
                            'last_processed_at': time.time(),
                            'max_confidence': confidence,
                            'best_alert': alert,
                            'frames_temp_paths': []
                        }
                    return
                
                # Update timing
                event['last_seen'] = detection_timestamp
                event['last_processed_at'] = time.time()
                
                # Check if this alert has higher confidence
                if confidence > event['max_confidence']:
                    event['max_confidence'] = confidence
                    event['best_alert'] = alert
                    
                    # Update Firestore with better evidence
                    await self.persistence_service.update_event(
                        event['event_id'],
                        camera_id,
                        alert
                    )
                    
                    # Publish event_updated to frontend
                    await self._publish_event_notification('event_updated', camera_id, {
                        'event_id': event['event_id'],
                        'timestamp': event['start_time'],  # Keep original timestamp
                        'confidence': confidence,
                        'snapshot': alert.get('snapshot', ''),
                        'status': 'active'
                    })
                    
                    logger.debug(f"[{camera_id}] Event updated with higher conf={confidence:.2f}")
            
            # Collect temp frames path if available
            frames_path = alert.get('frames_temp_path')
            if frames_path and camera_id in self.active_events:
                self.active_events[camera_id]['frames_temp_paths'].append(frames_path)
                
        except json.JSONDecodeError:
            logger.error("Failed to decode alert message JSON")
        except Exception as e:
            logger.error(f"Failed to process alert message: {e}")

    async def _event_monitor_loop(self) -> None:
        """
        Background task to monitor and finalize events.
        Checks every 1s if any event has timed out.
        """
        while self.is_running:
            try:
                active_cameras = list(self.active_events.keys())
                
                for camera_id in active_cameras:
                    event = self.active_events[camera_id]
                    
                    current_system_time = time.time()
                    last_processed_at = event.get('last_processed_at', current_system_time)
                    
                    is_timeout = (current_system_time - last_processed_at > self.event_timeout_seconds)
                    video_duration = event['last_seen'] - event['start_time']
                    is_max_duration = (video_duration > self.max_event_duration)

                    if is_timeout or is_max_duration:
                        # === FINALIZE EVENT ===
                        reason = f"Timeout ({current_system_time - last_processed_at:.1f}s)" if is_timeout else f"Max Duration ({video_duration:.1f}s)"
                        logger.info(f"[{camera_id}] Event finished [{reason}]. Finalizing...")
                        
                        video_start_time = event['start_time'] - self.pre_event_padding
                        video_end_time = event['last_seen'] + self.post_event_padding
                        
                        # Finalize in Firestore (add video, status: completed)
                        result = await self.persistence_service.finalize_event(
                            event['event_id'],
                            camera_id,
                            video_start_time,
                            video_end_time
                        )
                        
                        if result:
                            video_url = result.get('firebase_video_url')
                            logger.info(f"[{camera_id}] Event finalized: {event['event_id']}")
                            
                            # Publish event_completed to frontend
                            await self._publish_event_notification('event_completed', camera_id, {
                                'event_id': event['event_id'],
                                'timestamp': event['start_time'],
                                'confidence': event['max_confidence'],
                                'video_url': video_url,
                                'status': 'completed'
                            })
                            
                            # Store in Redis for quick lookup
                            try:
                                event_data = {
                                    "id": event['event_id'],
                                    "camera_id": camera_id,
                                    "timestamp": event['start_time'],
                                    "video_url": video_url,
                                    "confidence": event['max_confidence']
                                }
                                await self.redis_client.setex(
                                    f"event:{event['event_id']}",
                                    86400,
                                    json.dumps(event_data)
                                )
                            except Exception as e:
                                logger.error(f"Failed to cache event in Redis: {e}")
                        
                        # Remove from active list
                        del self.active_events[camera_id]
                        
                        # Cleanup temp frames
                        for path in event.get('frames_temp_paths', []):
                            if path and os.path.exists(path):
                                try:
                                    shutil.rmtree(path)
                                except:
                                    pass
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Event monitor error: {e}")
                await asyncio.sleep(1.0)

    async def _severity_worker_loop(self) -> None:
        """
        Background worker for severity analysis.
        
        Picks up events from queue → runs SecurityEngine → updates Firestore → notifies frontend.
        All rules are cached in RAM, so analysis is sub-millisecond.
        """
        logger.info("Severity worker started")
        
        while self.is_running:
            try:
                # Wait for event with timeout (allows graceful shutdown)
                try:
                    event_data = await asyncio.wait_for(
                        self.severity_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                event_id = event_data['event_id']
                camera_id = event_data['camera_id']
                confidence = event_data['confidence']
                timestamp = event_data['timestamp']
                
                # Run severity analysis (rules cached in RAM - very fast)
                analysis_result = self.security_engine.analyze_severity(
                    camera_id=camera_id,
                    confidence=confidence,
                    timestamp=timestamp
                )
                
                severity_level = analysis_result['severity_level']
                severity_score = analysis_result['severity_score']
                analysis_time_ms = analysis_result['analysis_time_ms']
                
                logger.info(
                    f"[{camera_id}] Severity analysis: {severity_level} "
                    f"(score={severity_score:.3f}, time={analysis_time_ms:.3f}ms)"
                )
                
                # Update Firestore with severity
                await self._update_event_severity(
                    event_id, 
                    severity_level, 
                    severity_score,
                    analysis_result
                )
                
                # Publish severity_updated to frontend
                await self._publish_event_notification('severity_updated', camera_id, {
                    'event_id': event_id,
                    'severity_level': severity_level,
                    'severity_score': severity_score,
                    'analysis_time_ms': analysis_time_ms,
                    'rule_matched': analysis_result.get('rule_matched'),
                    'risk_profile': analysis_result.get('risk_profile')
                })
                
                self.severity_queue.task_done()
                
            except Exception as e:
                logger.error(f"Severity worker error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Severity worker stopped")

    async def _update_event_severity(
        self, 
        event_id: str, 
        severity_level: str, 
        severity_score: float,
        analysis_result: Dict
    ) -> bool:
        """Update event in Firestore with severity information."""
        try:
            if not self.persistence_service.db:
                return False
            
            event_ref = self.persistence_service.db.collection('events').document(event_id)
            
            from firebase_admin import firestore as fs
            event_ref.update({
                'severityLevel': severity_level,
                'severityScore': severity_score,
                'severityAnalysis': {
                    'ruleMatched': analysis_result.get('rule_matched'),
                    'riskProfile': analysis_result.get('risk_profile'),
                    'features': analysis_result.get('features', {})
                },
                'updatedAt': fs.SERVER_TIMESTAMP
            })
            
            logger.debug(f"Updated event {event_id} with severity: {severity_level}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update event severity: {e}")
            return False

    async def _publish_event_notification(self, event_type: str, camera_id: str, data: Dict) -> None:
        """Publish event notification to Redis for frontend."""
        try:
            message = {
                "type": event_type,
                "camera_id": camera_id,
                **data
            }
            await self.redis_client.publish(
                f"alerts:{camera_id}",
                json.dumps(message)
            )
            logger.debug(f"[{camera_id}] Published {event_type}")
        except Exception as e:
            logger.error(f"Failed to publish {event_type}: {e}")


# Singleton instance
_event_processor: Optional[EventProcessor] = None

def get_event_processor(redis_client: redis.Redis = None) -> Optional[EventProcessor]:
    global _event_processor
    if _event_processor is None and redis_client:
        _event_processor = EventProcessor(redis_client)
    return _event_processor
