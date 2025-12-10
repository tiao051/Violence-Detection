"""
Violence Event Schema

Defines the data structure for violence detection events.
This schema is used for both mock data generation and real data from Firestore.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class ViolenceEvent:
    """
    Represents a single violence detection event.
    
    Attributes:
        event_id: Unique identifier for the event
        camera_id: ID of the camera that detected the event
        camera_name: Human-readable name of the camera
        timestamp: When the event occurred
        confidence: Model confidence score (0.0 to 1.0)
        user_id: Owner of the camera
        video_url: URL to the recorded video clip
        status: Event status (new, viewed, resolved)
        viewed: Whether the event has been viewed
        
    Derived attributes (computed):
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        is_weekend: Whether event occurred on weekend
        time_period: Morning/Afternoon/Evening/Night
    """
    
    event_id: str
    camera_id: str
    camera_name: str
    timestamp: datetime
    confidence: float
    user_id: str = "user_123"
    video_url: str = ""
    status: str = "new"
    viewed: bool = False
    
    # Derived attributes
    @property
    def hour(self) -> int:
        """Hour of day (0-23)."""
        return self.timestamp.hour
    
    @property
    def day_of_week(self) -> int:
        """Day of week (0=Monday, 6=Sunday)."""
        return self.timestamp.weekday()
    
    @property
    def day_name(self) -> str:
        """Day name (Monday, Tuesday, etc.)."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[self.day_of_week]
    
    @property
    def is_weekend(self) -> bool:
        """Whether event occurred on weekend."""
        return self.day_of_week >= 5
    
    @property
    def time_period(self) -> str:
        """
        Time period of day.
        - Morning: 6-12
        - Afternoon: 12-18
        - Evening: 18-22
        - Night: 22-6
        """
        hour = self.hour
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Night"
    
    @property
    def severity(self) -> str:
        """
        Severity level based on confidence score.
        - Low: < 0.6
        - Medium: 0.6 - 0.8
        - High: >= 0.8
        """
        if self.confidence < 0.6:
            return "Low"
        elif self.confidence < 0.8:
            return "Medium"
        else:
            return "High"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "user_id": self.user_id,
            "video_url": self.video_url,
            "status": self.status,
            "viewed": self.viewed,
            # Derived
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "day_name": self.day_name,
            "is_weekend": self.is_weekend,
            "time_period": self.time_period,
            "severity": self.severity,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ViolenceEvent":
        """Create ViolenceEvent from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            event_id=data.get("event_id", ""),
            camera_id=data.get("camera_id", ""),
            camera_name=data.get("camera_name", ""),
            timestamp=timestamp,
            confidence=data.get("confidence", 0.0),
            user_id=data.get("user_id", "user_123"),
            video_url=data.get("video_url", ""),
            status=data.get("status", "new"),
            viewed=data.get("viewed", False),
        )
    
    def __repr__(self) -> str:
        return f"ViolenceEvent({self.camera_name}, {self.timestamp}, conf={self.confidence:.2f})"
