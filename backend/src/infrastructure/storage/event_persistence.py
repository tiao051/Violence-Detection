"""Service for persisting detected violence events to Firebase."""

import logging
import os
import uuid
import tempfile
import shutil
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import firebase_admin
from firebase_admin import storage, firestore
from src.infrastructure.memory import get_frame_buffer
from src.infrastructure.notifications.notification_service import get_notification_service
from src.infrastructure.storage.token_repository import get_token_repository

logger = logging.getLogger(__name__)


class EventPersistenceService:
    """
    Handles the persistence of violence events.
    
    1. Generates video clip from FrameBuffer history.
    2. Uploads video to Firebase Storage.
    3. Saves event metadata to Cloud Firestore.
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

    async def save_event(self, camera_id: str, detection: Dict[str, Any], frames_temp_path: Optional[str] = None) -> Optional[str]:
        """
        Save a violence event.
        
        Args:
            camera_id: ID of the camera
            detection: Detection result dictionary
            frames_temp_path: Optional path to temp directory with batch frames
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.db or not self.bucket:
            logger.error("Firebase clients not initialized")
            return None

        logger.info(f"Processing event persistence for camera {camera_id}")

        # 1. Get all frames from buffer
        all_frames = self.frame_buffer.get_video_frames(camera_id)
        if not all_frames:
            logger.warning(f"No frames found in buffer for camera {camera_id}")
            return None
        
        logger.info(f"Retrieved {len(all_frames)} frames from buffer")

        # 2. Filter frames: only keep from inference batch onwards
        # Inference processes in batches of 30, so violence detection uses last 30 frames
        # We keep: last 30 frames (violent) + some padding before
        # Total: ~30 frames padding + 30 frames violent = 60 frames
        # But if buffer has less, use all
        if len(all_frames) >= 60:
            # Keep last 60 frames (buffer is full, we have enough context)
            frames_to_save = all_frames[-60:]
        elif len(all_frames) >= 30:
            # Keep all frames (buffer not full, but we have violence frames)
            frames_to_save = all_frames
        else:
            # Not enough frames, save what we have
            frames_to_save = all_frames
        
        logger.info(f"Using {len(frames_to_save)} frames for video (out of {len(all_frames)} in buffer)")

        # 3. Generate MP4 video file (temp location)
        temp_video_path = self._create_video_file(frames_to_save, camera_id)
        if not temp_video_path:
            return None

        try:
            # 4. Save to local disk
            local_video_path = self._save_video_locally(temp_video_path, camera_id)
            if not local_video_path:
                logger.warning("Failed to save video locally, continuing with Firebase only")
            
            # 5. Save batch frames to local disk (if available from inference)
            frames_folder_path = None
            if frames_temp_path:
                logger.info(f"Checking frames_temp_path: {frames_temp_path}")
                if os.path.exists(frames_temp_path):
                    logger.info(f"Frames temp path exists, saving batch frames...")
                    frames_folder_path = self._save_batch_frames(frames_temp_path, camera_id, local_video_path)
                else:
                    logger.warning(f"Frames temp path does not exist: {frames_temp_path}")
            else:
                logger.info("No frames_temp_path provided")
            
            # 6. Upload to Firebase Storage
            video_url = self._upload_video(temp_video_path, camera_id)
            if not video_url:
                logger.warning("Failed to upload to Firebase, continuing with local copy only")
            
            # 7. Save to Firestore with both paths
            event_id = self._save_to_firestore(camera_id, video_url, local_video_path, detection, frames_folder_path)
            
            logger.info(f"Event saved successfully: {event_id}")
            
            # 8. Send push notification to user
            await self._send_push_notification(camera_id, event_id, detection)
            
            return event_id

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

    def _save_video_locally(self, temp_path: str, camera_id: str) -> Optional[str]:
        """Save video to local disk at backend/outputs/{camera_id}/"""
        try:
            # Create outputs directory
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", camera_id)
            os.makedirs(outputs_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"violence_{camera_id}_{timestamp}.mp4"
            local_path = os.path.join(outputs_dir, filename)
            
            # Copy temp file to local
            shutil.copy2(temp_path, local_path)
            
            file_size = os.path.getsize(local_path)
            logger.info(f"Saved video locally: {local_path} ({file_size} bytes)")
            
            return local_path
        except Exception as e:
            logger.error(f"Failed to save video locally: {e}")
            return None

    def _save_batch_frames(self, frames_temp_path: str, camera_id: str, local_video_path: Optional[str]) -> Optional[str]:
        """Save batch frames from inference to local disk."""
        try:
            if not os.path.exists(frames_temp_path):
                logger.warning(f"Frames temp path does not exist: {frames_temp_path}")
                return None
            
            # Create frames directory next to video
            # E.g., /app/src/outputs/cam1/violence_cam1_20251212_131441/
            if local_video_path:
                video_name = os.path.splitext(os.path.basename(local_video_path))[0]
                frames_dir = os.path.join(os.path.dirname(local_video_path), video_name)
            else:
                # Fallback if no video path
                outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", camera_id)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                frames_dir = os.path.join(outputs_dir, f"violence_{camera_id}_{timestamp}")
            
            os.makedirs(frames_dir, exist_ok=True)
            
            # Copy all frames from temp directory
            frame_count = 0
            for filename in os.listdir(frames_temp_path):
                if filename.endswith('.jpg'):
                    src = os.path.join(frames_temp_path, filename)
                    dst = os.path.join(frames_dir, filename)
                    shutil.copy2(src, dst)
                    frame_count += 1
            
            logger.info(f"Saved {frame_count} frames to: {frames_dir}")
            return frames_dir
            
        except Exception as e:
            logger.error(f"Failed to save batch frames: {e}")
            return None
            return local_path
        except Exception as e:
            logger.error(f"Failed to save video locally: {e}")
            return None

    def _create_video_file(self, frames: List[np.ndarray], camera_id: str) -> Optional[str]:
        """Encode frames to MP4 file."""
        try:
            if not frames:
                return None

            # Create temp file
            temp_dir = tempfile.gettempdir()
            filename = f"violence_{camera_id}_{uuid.uuid4()}.mp4"
            filepath = os.path.join(temp_dir, filename)

            # Get dimensions from first frame
            height, width, layers = frames[0].shape
            size = (width, height)

            # Initialize VideoWriter (mp4v codec)
            # Note: In some docker containers 'avc1' or 'h264' might be needed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, 6.0, size) # 6 FPS match RTSP sample rate

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

    def _save_to_firestore(self, camera_id: str, video_url: Optional[str], local_video_path: Optional[str], detection: Dict[str, Any], frames_folder_path: Optional[str] = None) -> str:
        """Save event document to Firestore."""
        
        # MOCK: Get owner ID (In production this should come from a Camera Service)
        owner_uid = self._get_camera_owner(camera_id)
        
        event_data = {
            "userId": owner_uid,
            "cameraId": camera_id,
            "cameraName": self._get_camera_name(camera_id),
            "type": "violence",
            "status": "new",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "videoUrl": video_url or "",  # Firebase URL (may be empty)
            "localVideoPath": local_video_path or "",  # Local path (may be empty)
            "framesPath": frames_folder_path or "",  # Frames folder path (may be empty)
            "thumbnailUrl": "", # TODO: Generate thumbnail
            "confidence": detection.get("confidence", 0.0),
            "viewed": False
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
            "cam1": "Front Gate",
            "cam2": "Back Yard",
            "cam3": "Front Door",
            "cam4": "Living Room",
            "cam5": "Garage"
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
