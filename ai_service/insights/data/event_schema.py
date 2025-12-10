"""
Violence Event Schema

Defines the data structure for violence detection events.
This schema matches the Firestore 'events' collection structure.

Firestore fields:
- userId: Owner of the camera
- cameraId: Camera identifier
- cameraName: Human-readable camera name
- type: Event type (always "violence")
- status: Event status ("new", "viewed", "resolved")
- timestamp: When the event occurred (Firestore Timestamp)
- videoUrl: URL to the recorded video clip
- thumbnailUrl: URL to thumbnail image
- confidence: Model confidence score (0.0 to 1.0)
- viewed: Whether the event has been viewed
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict


@dataclass
class ViolenceEvent:
    """
    Represents a single violence detection event.
    
    Matches Firestore 'events' collection schema for easy integration.
    
    Core attributes (from Firestore):
        event_id: Document ID in Firestore
        camera_id: ID of the camera (cameraId)
        camera_name: Human-readable name (cameraName)
        timestamp: When the event occurred
        confidence: Model confidence score (0.0 to 1.0)
        user_id: Owner of the camera (userId)
        event_type: Always "violence"
        status: Event status (new, viewed, resolved)
        video_url: URL to the recorded video clip (videoUrl)
        thumbnail_url: URL to thumbnail (thumbnailUrl)
        viewed: Whether the event has been viewed
        
    Derived attributes (computed from timestamp/confidence):
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        day_name: Day name (Monday, Tuesday, etc.)
        is_weekend: Whether event occurred on weekend
        time_period: Morning/Afternoon/Evening/Night
        severity: Low/Medium/High based on confidence
    """
    
    # Core fields (match Firestore)
    event_id: str
    camera_id: str  # cameraId in Firestore
    camera_name: str  # cameraName in Firestore
    timestamp: datetime
    confidence: float
    user_id: str = ""  # userId in Firestore
    event_type: str = "violence"  # type in Firestore
    status: str = "new"
    video_url: str = ""  # videoUrl in Firestore
    thumbnail_url: str = ""  # thumbnailUrl in Firestore
    viewed: bool = False
    
    # ==================== Derived Properties ====================
    
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
    
    # ==================== Serialization ====================
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "cameraId": self.camera_id,
            "cameraName": self.camera_name,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "userId": self.user_id,
            "type": self.event_type,
            "status": self.status,
            "videoUrl": self.video_url,
            "thumbnailUrl": self.thumbnail_url,
            "viewed": self.viewed,
            # Derived (for ML features)
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "day_name": self.day_name,
            "is_weekend": self.is_weekend,
            "time_period": self.time_period,
            "severity": self.severity,
        }
    
    @classmethod
    def from_dict(cls, data: dict, event_id: str = "") -> "ViolenceEvent":
        """
        Create ViolenceEvent from dictionary.
        
        Supports both snake_case (internal) and camelCase (Firestore) keys.
        """
        # Handle timestamp
        timestamp = data.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif hasattr(timestamp, 'timestamp'):
            # Firestore Timestamp object
            timestamp = datetime.fromtimestamp(timestamp.timestamp())
        
        return cls(
            event_id=event_id or data.get("event_id", ""),
            camera_id=data.get("cameraId") or data.get("camera_id", ""),
            camera_name=data.get("cameraName") or data.get("camera_name", ""),
            timestamp=timestamp,
            confidence=data.get("confidence", 0.0),
            user_id=data.get("userId") or data.get("user_id", ""),
            event_type=data.get("type") or data.get("event_type", "violence"),
            status=data.get("status", "new"),
            video_url=data.get("videoUrl") or data.get("video_url", ""),
            thumbnail_url=data.get("thumbnailUrl") or data.get("thumbnail_url", ""),
            viewed=data.get("viewed", False),
        )
    
    @classmethod
    def from_firestore(cls, doc_id: str, doc_data: Dict[str, Any]) -> "ViolenceEvent":
        """
        Create ViolenceEvent from Firestore document.
        
        Args:
            doc_id: Firestore document ID
            doc_data: Document data dictionary
            
        Returns:
            ViolenceEvent instance
            
        Example:
            # From Firestore query
            docs = db.collection('events').get()
            events = [ViolenceEvent.from_firestore(doc.id, doc.to_dict()) for doc in docs]
        """
        return cls.from_dict(doc_data, event_id=doc_id)
    
    def __repr__(self) -> str:
        return f"ViolenceEvent({self.camera_name}, {self.timestamp}, conf={self.confidence:.2f})"

