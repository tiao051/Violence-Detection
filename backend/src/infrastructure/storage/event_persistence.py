"""Service for persisting detected violence events to Firebase."""

import logging
import os
import uuid
import tempfile
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from firebase_admin import storage, firestore
from src.infrastructure.memory import get_frame_buffer
from src.infrastructure.notifications.notification_service import get_notification_service
from src.infrastructure.storage.token_repository import get_token_repository

logger = logging.getLogger(__name__)


class EventPersistenceService:
    """
    Handles the persistence of violence events.
    
    Firestore-First Design:
    1. On first alert: CREATE event immediately (status: active)
    2. On higher confidence: UPDATE event (best snapshot)
    3. On timeout: FINALIZE event (add video, status: completed)
    """

    def __init__(self):
        self.frame_buffer = get_frame_buffer()
        # Note: Firebase must be initialized before using this service
        try:
            self.db = firestore.client()
            self.bucket = storage.bucket()
        except Exception as e:
            logger.error(f"Failed to get Firebase clients: {e}")
            self.db = None
            self.bucket = None

    async def create_event(
        self,
        camera_id: str,
        detection: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create a new event in Firestore immediately when first alert arrives.
        Returns event_id if successful.
        """
        if not self.db:
            logger.error("Firestore client not initialized")
            return None

        try:
            owner_uid = self._get_camera_owner(camera_id)
            
            if "timestamp" not in detection or detection["timestamp"] is None:
                raise ValueError(f"Detection missing required 'timestamp' field")
            
            timestamp = datetime.fromtimestamp(detection["timestamp"], tz=timezone.utc)
            
            event_data = {
                "userId": owner_uid,
                "cameraId": camera_id,
                "cameraName": self._get_camera_name(camera_id),
                "timestamp": timestamp,
                "videoUrl": "",  # Will be filled on finalize
                "thumbnailUrl": "",
                "confidence": detection.get("confidence", 0),
                "imageBase64": detection.get("snapshot", ""),
                "status": "active",  # NEW: Event is still ongoing
                "createdAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP
            }
            
            update_time, event_ref = self.db.collection('events').add(event_data)
            logger.info(f"[{camera_id}] Created active event: {event_ref.id}")
            
            return event_ref.id
            
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return None

    async def update_event(
        self,
        event_id: str,
        camera_id: str,
        detection: Dict[str, Any]
    ) -> bool:
        """
        Update existing event with better evidence (higher confidence snapshot).
        """
        if not self.db:
            return False

        try:
            event_ref = self.db.collection('events').document(event_id)
            
            update_data = {
                "confidence": detection.get("confidence", 0),
                "imageBase64": detection.get("snapshot", ""),
                "updatedAt": firestore.SERVER_TIMESTAMP
            }
            
            event_ref.update(update_data)
            logger.info(f"[{camera_id}] Updated event {event_id} with higher confidence: {detection.get('confidence'):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update event {event_id}: {e}")
            return False

    async def finalize_event(
        self,
        event_id: str,
        camera_id: str,
        start_timestamp: float,
        end_timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize event: generate video, upload, update Firestore status to completed.
        Returns dict with video_url if successful.
        """
        if not self.db or not self.bucket:
            logger.error("Firebase clients not initialized")
            return None

        try:
            # 1. Get frames from buffer
            all_frames = self.frame_buffer.get_video_frames(
                camera_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
            )
            
            if not all_frames:
                logger.warning(f"No frames found for finalize: {camera_id}")
                # Still mark as completed but without video
                self.db.collection('events').document(event_id).update({
                    "status": "completed",
                    "updatedAt": firestore.SERVER_TIMESTAMP
                })
                return {"id": event_id, "firebase_video_url": None}
            
            logger.info(f"Retrieved {len(all_frames)} frames for video generation")

            # 2. Generate video
            temp_video_path = self._create_video_file(all_frames, camera_id)
            if not temp_video_path:
                return None

            try:
                # 3. Upload to Firebase Storage
                video_url = self._upload_video(temp_video_path, camera_id)
                
                # 4. Update Firestore with video URL and status
                event_ref = self.db.collection('events').document(event_id)
                event_ref.update({
                    "videoUrl": video_url or "",
                    "status": "completed",
                    "updatedAt": firestore.SERVER_TIMESTAMP
                })
                
                logger.info(f"[{camera_id}] Event finalized: {event_id}")
                
                # 5. Send push notification
                doc = event_ref.get()
                if doc.exists:
                    await self._send_push_notification(camera_id, event_id, doc.to_dict())
                
                return {
                    'id': event_id,
                    'firebase_video_url': video_url
                }

            finally:
                # Cleanup temp file
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Failed to finalize event {event_id}: {e}")
            return None

    async def save_event(
        self, 
        camera_id: str, 
        detection: Dict[str, Any], 
        frames_temp_paths: Optional[List[str]] = None,
        frames_temp_path: Optional[str] = None, # Backward compatibility
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Save a violence event.
        
        Args:
            camera_id: ID of the camera
            detection: Detection result dictionary
            frames_temp_paths: List of paths to temp directories with batch frames (from multiple alerts)
            frames_temp_path: Single path (legacy support)
            start_timestamp: Start time for video clip
            end_timestamp: End time for video clip
            
        Returns:
            Dict with event_id and video paths if successful, None otherwise
        """
        if not self.db or not self.bucket:
            logger.error("Firebase clients not initialized")
            return None

        # Handle legacy single path argument
        if frames_temp_path and not frames_temp_paths:
            frames_temp_paths = [frames_temp_path]
        if not frames_temp_paths:
            frames_temp_paths = []

        logger.info(f"Processing event persistence for camera {camera_id}")

        # 1. Get frames from buffer with time window
        all_frames = self.frame_buffer.get_video_frames(
            camera_id, 
            start_timestamp=start_timestamp, 
            end_timestamp=end_timestamp
        )
        
        if not all_frames:
            logger.warning(f"No frames found in buffer for camera {camera_id} (window: {start_timestamp}-{end_timestamp})")
            return None
        
        if not all_frames:
             logger.warning(f"Buffer completely empty for {camera_id}")
             return None
        
        logger.info(f"Retrieved {len(all_frames)} frames for video generation")

        # 2. Generate MP4 video file (temp location)
        # Use all retrieved frames as they are already filtered by time window
        temp_video_path = self._create_video_file(all_frames, camera_id)
        if not temp_video_path:
            return None

        try:
            # 3. Upload to Firebase Storage
            video_url = self._upload_video(temp_video_path, camera_id)
            if not video_url:
                logger.error("Failed to upload to Firebase")
                return None
            
            # 4. Save to Firestore
            event_id = self._save_to_firestore(camera_id, video_url, detection)
            
            logger.info(f"Event saved successfully: {event_id}")
            
            # 5. Send push notification to user
            await self._send_push_notification(camera_id, event_id, detection)
            
            return {
                'id': event_id,
                'firebase_video_url': video_url
            }

        except Exception as e:
            logger.error(f"Error saving event: {e}")
            return None
        finally:
            # Cleanup temp file
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass

    def _create_video_file(self, frames: List[np.ndarray], camera_id: str) -> Optional[str]:
        """Encode frames to MP4 file."""
        try:
            if not frames:
                return None

            # Create temp file in app directory instead of /tmp to avoid permission issues in Docker
            temp_dir = "/app/tmp"  # Use app directory instead of system /tmp
            os.makedirs(temp_dir, exist_ok=True)  # Ensure directory exists
            filename = f"violence_{camera_id}_{uuid.uuid4()}.mp4"
            filepath = os.path.join(temp_dir, filename)

            # Get dimensions from first frame
            height, width, layers = frames[0].shape
            size = (width, height)

            # Initialize VideoWriter (MJPG codec - works without FFmpeg in Docker)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(filepath, fourcc, 6.0, size)

            for frame in frames:
                out.write(frame)

            out.release()
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Generated video file: {filepath} ({file_size} bytes)")
            
            return filepath
        except Exception as e:
            logger.error(f"Failed to create video file: {e}")
            return None

    def _upload_video(self, file_path: str, camera_id: str) -> Optional[str]:
        """Upload video to Firebase Storage and return public URL."""
        try:
            filename = os.path.basename(file_path)
            blob_path = f"events/{camera_id}/{filename}"
            
            blob = self.bucket.blob(blob_path)
            
            # Upload
            blob.upload_from_filename(file_path, content_type='video/mp4')
            
            # Make public to get a viewable URL
            blob.make_public()
            
            logger.info(f"Uploaded video to: {blob.public_url}")
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            return None

    def _save_to_firestore(self, camera_id: str, video_url: Optional[str], detection: Dict[str, Any]) -> str:
        """Save event document to Firestore."""
        
        # MOCK: Get owner ID (In production this should come from a Camera Service)
        owner_uid = self._get_camera_owner(camera_id)
        
        # Detection MUST have timestamp - no fallback to server timestamp
        if "timestamp" not in detection or detection["timestamp"] is None:
            raise ValueError(f"Detection missing required 'timestamp' field. Detection: {detection}")
        
        try:
            # Convert float timestamp (Unix time) to Firestore-compatible datetime
            timestamp = datetime.fromtimestamp(detection["timestamp"], tz=timezone.utc)
            logger.info(f"Using detection timestamp: {timestamp}")
        except Exception as e:
            raise ValueError(f"Failed to convert detection timestamp {detection.get('timestamp')}: {e}")

        event_data = {
            "userId": owner_uid,
            "cameraId": camera_id,
            "cameraName": self._get_camera_name(camera_id),
            "timestamp": timestamp,
            "videoUrl": video_url,  # Firebase URL (may be empty)
            "thumbnailUrl": "", # TODO: Generate thumbnail
            "confidence": detection.get("confidence"),
            # Store the snapshot image for frontend (same as alert history)
            # detection['snapshot'] = "data:image/jpeg;base64,..." from inference_consumer
            "imageBase64": detection.get("snapshot", "")  # Base64 encoded snapshot frame
        }
        
        # Add to 'events' collection
        update_time, event_ref = self.db.collection('events').add(event_data)
        
        return event_ref.id

    def _get_camera_owner(self, camera_id: str) -> str:
        """Mock function to get camera owner."""
        # Mapping based on auth_routes.py mock data
        # cam1, cam2, cam3 -> user_123
        # cam4, cam5 -> user_456
        if camera_id in ['cam1', 'cam2', 'cam3']:
            return "H2399Gybu8TeslP8zyEyP4uhn0l2" # Updated with real user UID for testing
        elif camera_id in ['cam4', 'cam5']:
            return "user_456"
        
        # Fallback to a default user if you are testing with a specific account
        # Return the UID of the user currently logged in the Flutter app for easier testing
        return "H2399Gybu8TeslP8zyEyP4uhn0l2" # From auth_routes.py mock

    def _get_camera_name(self, camera_id: str) -> str:
        names = {
            "cam1": "Ngã tư Lê Trọng Tấn",
            "cam2": "Ngã tư Cộng Hòa",
            "cam3": "Ngã ba Âu Cơ",
            "cam4": "Ngã tư Hòa Bình",
            "cam5": "Ngã tư Tân Sơn Nhì"
        }
        return names.get(camera_id, f"Camera {camera_id}")

    async def _send_push_notification(self, camera_id: str, event_id: str, detection: Dict[str, Any]) -> None:
        """
        Send push notification to user about violence detection.
        
        Args:
            camera_id: ID of the camera that detected violence
            event_id: ID of the saved event (for deep linking)
            detection: Detection data with confidence score
        """
        try:
            # Get camera owner
            owner_uid = self._get_camera_owner(camera_id)
            
            # Get user's FCM tokens
            token_repo = get_token_repository()
            tokens = token_repo.get_tokens(owner_uid)
            
            if not tokens:
                logger.warning(f"No FCM tokens found for user {owner_uid}, skipping notification")
                return
            
            # Get camera name and format timestamp
            camera_name = self._get_camera_name(camera_id)
            timestamp = datetime.now().strftime("%H:%M")
            confidence = detection.get("confidence", 0.0)
            
            # Send notification
            notification_service = get_notification_service()
            success_count = notification_service.send_multicast(
                tokens=tokens,
                title="⚠️ Phát Hiện Bạo Lực",
                body=f"Camera {camera_name} phát hiện hoạt động đáng ngờ lúc {timestamp}. Bấm để xem!",
                data={"event_id": event_id}  # For deep linking
            )
            
            logger.info(f"Sent notification to {success_count}/{len(tokens)} devices for event {event_id}")
            
        except Exception as e:
            # Don't fail event save if notification fails
            logger.error(f"Failed to send push notification: {e}", exc_info=True)


# Singleton instance
_event_persistence_service: Optional[EventPersistenceService] = None

def get_event_persistence_service() -> EventPersistenceService:
    global _event_persistence_service
    if _event_persistence_service is None:
        _event_persistence_service = EventPersistenceService()
    return _event_persistence_service
