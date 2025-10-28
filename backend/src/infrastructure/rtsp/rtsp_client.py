"""RTSP client for streaming video capture."""

import asyncio
import cv2
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RTSPClient:
    """
    RTSP client for pulling frames from RTSP streams.
    
    Features:
    - Connect to RTSP URL
    - Read frames as numpy arrays
    - Automatic reconnection with exponential backoff
    - Health monitoring (FPS tracking, error counting)
    """
    
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 2,
        timeout: int = 10,
    ):
        """
        Initialize RTSP client.
        
        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://localhost:8554/cam1)
            camera_id: Camera identifier
            max_reconnect_attempts: Max reconnection attempts before giving up
            reconnect_delay: Initial delay between reconnection attempts (seconds)
            timeout: Frame read timeout (seconds)
        """
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        
        # Metrics
        self.frames_read = 0
        self.errors_count = 0
        self.last_frame_time: Optional[datetime] = None
        self.fps = 0.0
    
    async def connect(self) -> bool:
        """
        Connect to RTSP stream.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            logger.info(f"[{self.camera_id}] Connecting to {self.rtsp_url}")
            
            # Release old connection if exists
            if self.cap is not None:
                self.cap.release()
            
            # Create new VideoCapture
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Set timeout for frame reading
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            
            # Try to read one frame to verify connection
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.error(f"[{self.camera_id}] Failed to read frame during connection test")
                self.cap.release()
                self.cap = None
                return False
            
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info(f"[{self.camera_id}] Successfully connected to {self.rtsp_url}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {str(e)}")
            self.is_connected = False
            return False
    
    async def read_frame(self) -> Tuple[bool, Optional[bytes]]:
        """
        Read next frame from RTSP stream.
        
        Returns:
            Tuple of (success: bool, frame: numpy.ndarray or None)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning(f"[{self.camera_id}] Failed to read frame")
                self.errors_count += 1
                self.is_connected = False
                return False, None
            
            # Update metrics
            self.frames_read += 1
            self.last_frame_time = datetime.now()
            
            return True, frame
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Frame read error: {str(e)}")
            self.errors_count += 1
            self.is_connected = False
            return False, None
    
    async def reconnect(self) -> bool:
        """
        Reconnect with exponential backoff.
        
        Returns:
            True if successfully reconnected, False if max attempts exceeded
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                f"[{self.camera_id}] Max reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            return False
        
        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
        delay = self.reconnect_delay * (2 ** self.reconnect_attempts)
        self.reconnect_attempts += 1
        
        logger.warning(
            f"[{self.camera_id}] Reconnecting in {delay}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
        )
        
        await asyncio.sleep(delay)
        return await self.connect()
    
    def close(self) -> None:
        """Close RTSP connection."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
        logger.info(f"[{self.camera_id}] RTSP connection closed")
    
    def get_stats(self) -> dict:
        """
        Get connection statistics.
        
        Returns:
            Dictionary with stats (frames_read, errors, fps, last_frame_age_seconds)
        """
        last_frame_age = None
        if self.last_frame_time:
            last_frame_age = (datetime.now() - self.last_frame_time).total_seconds()
        
        return {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "frames_read": self.frames_read,
            "errors_count": self.errors_count,
            "fps": self.fps,
            "last_frame_age_seconds": last_frame_age,
        }
