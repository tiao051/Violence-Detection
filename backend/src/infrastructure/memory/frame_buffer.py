"""In-memory frame buffer for zero-copy frame sharing."""

import logging
from typing import Optional, Dict, List, Deque
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    Thread-safe in-memory frame buffer.
    
    Stores a HISTORY of frames per camera (Ring Buffer).
    Designed for zero-copy frame passing and video generation.
    
    Features:
    - Thread-safe (using locks)
    - Ring buffer (keep last N frames)
    - Fast access
    """

    def __init__(self, max_buffer_size: int = 300):
        """
        Initialize frame buffer.
        
        Args:
            max_buffer_size: Max frames to keep per camera (default 300 = 60s @ 5fps)
                             Increased to allow capturing longer violence events.
        """
        self.max_buffer_size = max_buffer_size
        self.buffers: Dict[str, Deque[np.ndarray]] = {}
        self.metadata: Dict[str, Deque[Dict]] = {}  # Store frame metadata history
        self.locks: Dict[str, threading.RLock] = {}

    def register_camera(self, camera_id: str) -> None:
        """
        Register a camera in the buffer.
        
        Args:
            camera_id: Camera identifier
        """
        if camera_id not in self.buffers:
            self.buffers[camera_id] = deque(maxlen=self.max_buffer_size)
            self.metadata[camera_id] = deque(maxlen=self.max_buffer_size)
            self.locks[camera_id] = threading.RLock()
            logger.debug(f"Registered camera {camera_id} in frame buffer (size={self.max_buffer_size})")

    def put(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_id: str,
        timestamp: float,
        frame_seq: int,
    ) -> None:
        """
        Store frame in buffer (append to history).
        
        Thread-safe operation.
        
        Args:
            camera_id: Camera identifier
            frame: Frame as numpy array (BGR or RGB)
            frame_id: Unique frame ID
            timestamp: Frame timestamp
            frame_seq: Frame sequence number
        """
        self.register_camera(camera_id)

        with self.locks[camera_id]:
            # Append to deque (automatically removes oldest if full)
            self.buffers[camera_id].append(frame)

            # Store metadata
            meta = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "frame_seq": frame_seq,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
            }
            self.metadata[camera_id].append(meta)

    def get(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Retrieve LATEST frame for camera.
        
        Thread-safe operation.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Latest frame (numpy array) or None if not available
        """
        self.register_camera(camera_id)

        with self.locks[camera_id]:
            if not self.buffers[camera_id]:
                return None
            return self.buffers[camera_id][-1]

    def get_video_frames(
        self, 
        camera_id: str, 
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Retrieve sequence of frames for video generation.
        
        Args:
            camera_id: Camera identifier
            start_timestamp: Optional start time filter
            end_timestamp: Optional end time filter
        
        Returns:
            List of frames (numpy arrays)
        """
        self.register_camera(camera_id)
        
        with self.locks[camera_id]:
            # If no time filter, return all frames
            if start_timestamp is None and end_timestamp is None:
                return list(self.buffers[camera_id])
            
            # Filter by timestamp
            frames = []
            buffer_list = list(self.buffers[camera_id])
            metadata_list = list(self.metadata[camera_id])
            
            for frame, meta in zip(buffer_list, metadata_list):
                ts = meta['timestamp']
                
                # Check time window
                if start_timestamp and ts < start_timestamp:
                    continue
                if end_timestamp and ts > end_timestamp:
                    continue
                    
                frames.append(frame)
                
            return frames

    def get_with_metadata(self, camera_id: str) -> tuple:
        """
        Retrieve latest frame and its metadata.
        
        Thread-safe operation.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Tuple of (frame, metadata) or (None, {}) if not available
        """
        self.register_camera(camera_id)

        with self.locks[camera_id]:
            if not self.buffers[camera_id]:
                return None, {}
            return self.buffers[camera_id][-1], self.metadata[camera_id][-1]

    def has_frame(self, camera_id: str) -> bool:
        """
        Check if camera has a frame in buffer.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            True if frame available, False otherwise
        """
        self.register_camera(camera_id)
        return len(self.buffers[camera_id]) > 0

    def memory_usage_mb(self) -> float:
        """
        Estimate memory usage of all frames.
        
        Returns:
            Memory usage in MB
        """
        total_bytes = 0
        for buffer in self.buffers.values():
            for frame in buffer:
                total_bytes += frame.nbytes

        return total_bytes / (1024 * 1024)

    def stats(self) -> Dict:
        """
        Get buffer statistics.
        
        Returns:
            Statistics dictionary
        """
        frame_count = sum(len(b) for b in self.buffers.values())

        return {
            "total_cameras": len(self.buffers),
            "total_frames_buffered": frame_count,
            "memory_usage_mb": self.memory_usage_mb(),
        }


# Global frame buffer instance
_frame_buffer: Optional[FrameBuffer] = None


def get_frame_buffer() -> FrameBuffer:
    """
    Get global frame buffer instance (singleton).
    
    Returns:
        FrameBuffer instance
    """
    global _frame_buffer
    if _frame_buffer is None:
        # Default to 100 frames (approx 16s @ 6fps)
        _frame_buffer = FrameBuffer(max_buffer_size=100)
    return _frame_buffer