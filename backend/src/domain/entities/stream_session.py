"""StreamSession Entity - Streaming session business logic"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class StreamStatus(str, Enum):
    """Stream session status"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamSession:
    """
    StreamSession Entity - Represents an active streaming session
    
    Business rules for stream lifecycle management.
    """
    id: Optional[str]
    camera_id: str
    status: StreamStatus
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    frames_processed: int = 0
    fps: float = 0.0
    errors_count: int = 0
    
    def start(self) -> None:
        """Start streaming session"""
        self.status = StreamStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.frames_processed = 0
        self.errors_count = 0
    
    def stop(self) -> None:
        """Stop streaming session"""
        self.status = StreamStatus.STOPPED
        self.stopped_at = datetime.utcnow()
    
    def increment_frame_count(self) -> None:
        """Increment processed frames counter"""
        self.frames_processed += 1
    
    def update_fps(self, fps: float) -> None:
        """Update frames per second"""
        self.fps = fps
    
    def increment_error_count(self) -> None:
        """Increment error counter"""
        self.errors_count += 1
        if self.errors_count > 10:  # Business rule: max 10 errors
            self.status = StreamStatus.ERROR
    
    def is_healthy(self) -> bool:
        """Check if stream is healthy"""
        return (
            self.status == StreamStatus.RUNNING and
            self.errors_count < 5 and
            self.fps > 0
        )
    
    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if not self.started_at:
            return None
        
        end_time = self.stopped_at or datetime.utcnow()
        duration = (end_time - self.started_at).total_seconds()
        return duration
