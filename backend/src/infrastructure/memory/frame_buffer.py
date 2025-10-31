"""In-memory frame buffer for zero-copy frame sharing."""

import logging
from typing import Optional, Dict
import threading

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    Thread-safe in-memory frame buffer.
    
    Stores latest frame per camera in RAM (keep-last-only strategy).
    Designed for zero-copy frame passing between components.
    
    Features:
    - Thread-safe (using locks)
    - Keep last frame only per camera (memory efficient)
    - Fast access (O(1))
    - No serialization/deserialization
    """

    def __init__(self):
        """Initialize frame buffer."""
        self.buffers: Dict[str, Optional[np.ndarray]] = {}
        self.metadata: Dict[str, Dict] = {}  # Store frame metadata
        self.locks: Dict[str, threading.RLock] = {}

    def register_camera(self, camera_id: str) -> None:
        """
        Register a camera in the buffer.
        
        Args:
            camera_id: Camera identifier
        """
        if camera_id not in self.buffers:
            self.buffers[camera_id] = None
            self.metadata[camera_id] = {}
            self.locks[camera_id] = threading.RLock()
            logger.debug(f"Registered camera {camera_id} in frame buffer")

    def put(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_id: str,
        timestamp: float,
        frame_seq: int,
    ) -> None:
        """
        Store frame in buffer (keep-last-only).
        
        Previous frame is automatically discarded.
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
            # Store frame (latest only, previous is discarded)
            self.buffers[camera_id] = frame

            # Store metadata
            self.metadata[camera_id] = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "frame_seq": frame_seq,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
            }

    def get(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Retrieve latest frame for camera.
        
        Thread-safe operation.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Latest frame (numpy array) or None if not available
        """
        self.register_camera(camera_id)

        with self.locks[camera_id]:
            frame = self.buffers[camera_id]

        return frame

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
            frame = self.buffers[camera_id]
            metadata = self.metadata[camera_id].copy()

        return frame, metadata

    def get_metadata(self, camera_id: str) -> Dict:
        """
        Retrieve metadata for latest frame.
        
        Thread-safe operation.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Metadata dictionary
        """
        self.register_camera(camera_id)

        with self.locks[camera_id]:
            metadata = self.metadata[camera_id].copy()

        return metadata

    def get_all_latest(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get all latest frames across cameras.
        
        Returns:
            Dictionary {camera_id: frame}
        """
        result = {}
        for camera_id in self.buffers:
            result[camera_id] = self.get(camera_id)
        return result

    def has_frame(self, camera_id: str) -> bool:
        """
        Check if camera has a frame in buffer.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            True if frame available, False otherwise
        """
        self.register_camera(camera_id)
        return self.buffers[camera_id] is not None

    def memory_usage_mb(self) -> float:
        """
        Estimate memory usage of all frames.
        
        Returns:
            Memory usage in MB
        """
        total_bytes = 0
        for frame in self.buffers.values():
            if frame is not None:
                total_bytes += frame.nbytes

        return total_bytes / (1024 * 1024)

    def stats(self) -> Dict:
        """
        Get buffer statistics.
        
        Returns:
            Statistics dictionary
        """
        frame_count = sum(1 for f in self.buffers.values() if f is not None)

        return {
            "total_cameras": len(self.buffers),
            "cameras_with_frames": frame_count,
            "memory_usage_mb": self.memory_usage_mb(),
            "metadata": {
                camera_id: self.metadata.get(camera_id, {})
                for camera_id in self.buffers
            }
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
        _frame_buffer = FrameBuffer()
    return _frame_buffer