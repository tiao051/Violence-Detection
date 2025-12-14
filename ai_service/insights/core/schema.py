"""
Violence Event Schema

Defines the data structure for violence/nonviolence detection events.
This schema matches the event_persistence.py structure in backend.

Fields (matching event_persistence):
- userId: Owner of the camera
- cameraId: Camera identifier  
- cameraName: Human-readable camera name
- cameraDescription: Location description (intersection/area)
- timestamp: When the event occurred (format: HH:mm:ss DD/MM/YYYY)
- videoUrl: URL to the recorded video clip
- thumbnailUrl: URL to thumbnail image
- confidence: Model confidence score (0.0 to 1.0)
- label: Event type ("violence" or "nonviolence")
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict


@dataclass
class ViolenceEvent:
    """
    Represents a single violence/nonviolence detection event.
    
    Matches event_persistence.py structure for consistency.
    
    Core attributes (matching event_persistence):
        user_id: Owner of the camera (userId)
        camera_id: ID of the camera (cameraId)
        camera_name: Human-readable name (cameraName)
        camera_description: Location description (e.g., "Lê Trọng Tấn giao Tân Kỳ Tân Quý")
        timestamp: When the event occurred
        video_url: URL to the recorded video clip (videoUrl)
        thumbnail_url: URL to thumbnail (thumbnailUrl)
        confidence: Model confidence score (0.0 to 1.0)
        label: Event label ("violence" or "nonviolence")
        
    Derived attributes (computed from timestamp/confidence):
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        day_name: Day name (Monday, Tuesday, etc.)
        is_weekend: Whether event occurred on weekend
        time_period: Morning/Afternoon/Evening/Night
        severity: Low/Medium/High based on confidence
    """
    
    # Core fields (match event_persistence)
    user_id: str  # userId in Firestore
    camera_id: str  # cameraId in Firestore
    camera_name: str  # cameraName in Firestore
    camera_description: str  # Location description (NEW)
    timestamp: datetime
    confidence: float
    video_url: str = ""  # videoUrl in Firestore
    thumbnail_url: str = ""  # thumbnailUrl in Firestore
    label: str = "violence"  # "violence" or "nonviolence" (NEW)
    
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
        """Convert to dictionary for serialization (matches event_persistence format)."""
        return {
            "userId": self.user_id,
            "cameraId": self.camera_id,
            "cameraName": self.camera_name,
            "cameraDescription": self.camera_description,
            "timestamp": self.timestamp.strftime("%H:%M:%S %d/%m/%Y"),  # HH:mm:ss DD/MM/YYYY
            "videoUrl": self.video_url,
            "thumbnailUrl": self.thumbnail_url,
            "confidence": self.confidence,
            "label": self.label,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ViolenceEvent":
        """
        Create ViolenceEvent from dictionary.
        
        Supports both snake_case (internal) and camelCase (Firestore) keys.
        """
        # Handle timestamp
        timestamp = data.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            # Try HH:MM:SS DD/MM/YYYY format first
            try:
                timestamp = datetime.strptime(timestamp, "%H:%M:%S %d/%m/%Y")
            except ValueError:
                # Fall back to ISO format
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif hasattr(timestamp, 'timestamp'):
            # Firestore Timestamp object
            timestamp = datetime.fromtimestamp(timestamp.timestamp())
        
        return cls(
            user_id=data.get("userId") or data.get("user_id", ""),
            camera_id=data.get("cameraId") or data.get("camera_id", ""),
            camera_name=data.get("cameraName") or data.get("camera_name", ""),
            camera_description=data.get("cameraDescription") or data.get("camera_description", ""),
            timestamp=timestamp,
            confidence=data.get("confidence", 0.0),
            video_url=data.get("videoUrl") or data.get("video_url", ""),
            thumbnail_url=data.get("thumbnailUrl") or data.get("thumbnail_url", ""),
            label=data.get("label", "violence"),
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
        return cls.from_dict(doc_data)
    
    def __repr__(self) -> str:
        return f"ViolenceEvent({self.camera_name}, {self.timestamp.strftime('%H:%M:%S %d/%m/%Y')}, label={self.label}, conf={self.confidence:.2f})"

