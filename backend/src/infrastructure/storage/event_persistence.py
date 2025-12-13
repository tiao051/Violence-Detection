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
            # Fallback: get all frames if window query failed
            all_frames = self.frame_buffer.get_video_frames(camera_id)
        
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
            # 3. Save to local disk
            local_video_path = self._save_video_locally(temp_video_path, camera_id)
            if not local_video_path:
                logger.warning("Failed to save video locally, continuing with Firebase only")
            
            # 4. Save batch frames to local disk (collect from all temp paths)
            frames_folder_path = None
            if frames_temp_paths:
                logger.info(f"Saving frames from {len(frames_temp_paths)} batches...")
                frames_folder_path = self._save_batch_frames_multi(frames_temp_paths, camera_id, local_video_path)
            
            # 5. Upload to Firebase Storage
            video_url = self._upload_video(temp_video_path, camera_id)
            if not video_url:
                logger.warning("Failed to upload to Firebase, continuing with local copy only")
            
            # 6. Save to Firestore with both paths
            event_id = self._save_to_firestore(camera_id, video_url, local_video_path, detection, frames_folder_path)
            
            logger.info(f"Event saved successfully: {event_id}")
            
            # 7. Send push notification to user
            await self._send_push_notification(camera_id, event_id, detection)
            
            return {
                'id': event_id,
                'local_video_path': local_video_path,
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

    def _save_batch_frames_multi(self, frames_temp_paths: List[str], camera_id: str, local_video_path: Optional[str]) -> Optional[str]:
        """Save batch frames from multiple temp directories to local disk."""
        try:
            # Create frames directory next to video
            if local_video_path:
                video_name = os.path.splitext(os.path.basename(local_video_path))[0]
                frames_dir = os.path.join(os.path.dirname(local_video_path), video_name)
            else:
                outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", camera_id)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                frames_dir = os.path.join(outputs_dir, f"violence_{camera_id}_{timestamp}")
            
            os.makedirs(frames_dir, exist_ok=True)
            
            # Copy all frames from all temp directories
            frame_count = 0
            for temp_path in frames_temp_paths:
                if not os.path.exists(temp_path):
                    continue
                    
                for filename in os.listdir(temp_path):
                    if filename.endswith('.jpg'):
                        # Generate unique name to prevent collision between batches
                        unique_name = f"{uuid.uuid4()}_{filename}"
                        src = os.path.join(temp_path, filename)
                        dst = os.path.join(frames_dir, unique_name)
                        shutil.copy2(src, dst)
                        frame_count += 1
            
            logger.info(f"Saved {frame_count} frames to: {frames_dir}")
            return frames_dir
            
        except Exception as e:
            logger.error(f"Failed to save batch frames: {e}")
            return None

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
            
            # Cleanup old files (keep last 20)
            self._cleanup_local_files(camera_id, max_files=20)
            
            return local_path
        except Exception as e:
            logger.error(f"Failed to save video locally: {e}")
            return None

    def _cleanup_local_files(self, camera_id: str, max_files: int = 20) -> None:
        """Keep only the last max_files videos and their corresponding frame folders."""
        try:
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", camera_id)
            if not os.path.exists(outputs_dir):
                return

            # List all mp4 files
            files = [f for f in os.listdir(outputs_dir) if f.endswith('.mp4')]
            if len(files) <= max_files:
                return

            # Sort by modification time (oldest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)))

            # Files to delete
            files_to_delete = files[:-max_files]

            for filename in files_to_delete:
                file_path = os.path.join(outputs_dir, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old local video: {filename}")
                    
                    # Check for corresponding folder (remove extension .mp4)
                    folder_name = os.path.splitext(filename)[0]
                    folder_path = os.path.join(outputs_dir, folder_name)
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted old local frames folder: {folder_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old file {filename}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up local files: {e}")

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
