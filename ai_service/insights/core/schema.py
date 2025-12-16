"""
Violence Event Schema

Defines the data structure for violence/nonviolence detection events.
This schema matches the event_persistence.py structure in backend.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
from ..time_utils import get_time_period, get_day_name, is_weekend


@dataclass
class ViolenceEvent:
    """
    Represents a single violence/nonviolence detection event.
    """
    
    user_id: str 
    camera_id: str 
    camera_name: str 
    camera_description: str 
    timestamp: datetime
    confidence: float
    video_url: str = "" 
    thumbnail_url: str = ""  
    label: str = "violence"  

    @property
    def hour(self) -> int:
        return self.timestamp.hour
    
    @property
    def day_of_week(self) -> int:
        return self.timestamp.weekday()
    
    @property
    def day_name(self) -> str:
        return get_day_name(self.day_of_week)
    
    @property
    def is_weekend(self) -> bool:
        return is_weekend(self.day_of_week)
    
    @property
    def time_period(self) -> str:
        return get_time_period(self.hour)
    
    @property
    def severity(self) -> str:
        if self.confidence < 0.6:
            return "Low"
        elif self.confidence < 0.8:
            return "Medium"
        else:
            return "High"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (matches event_persistence format)."""
        return {
            "userId": self.user_id,
            "cameraId": self.camera_id,
            "cameraName": self.camera_name,
            "cameraDescription": self.camera_description,
            "timestamp": self.timestamp.strftime("%H:%M:%S %d/%m/%Y"),
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
        timestamp = data.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%H:%M:%S %d/%m/%Y")
        elif hasattr(timestamp, 'timestamp'):
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
        """
        return cls.from_dict(doc_data)
    
    def __repr__(self) -> str:
        return f"ViolenceEvent({self.camera_name}, {self.timestamp.strftime('%H:%M:%S %d/%m/%Y')}, label={self.label}, conf={self.confidence:.2f})"

