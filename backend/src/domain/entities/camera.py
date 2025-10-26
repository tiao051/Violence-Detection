"""Camera Entity - Business Logic"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class CameraStatus(str, Enum):
    """Camera status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class Camera:
    """
    Camera Entity - Represents a surveillance camera
    
    This is a pure business entity with no dependencies on frameworks.
    Contains only business logic and validation rules.
    """
    id: Optional[str]
    name: str
    rtsp_url: str
    location: str
    status: CameraStatus
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate camera data"""
        self.validate()
    
    def validate(self) -> None:
        """Validate camera business rules"""
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Camera name cannot be empty")
        
        if not self.rtsp_url or not self.rtsp_url.startswith("rtsp://"):
            raise ValueError("Invalid RTSP URL format")
        
        if not self.location or len(self.location.strip()) == 0:
            raise ValueError("Camera location cannot be empty")
    
    def activate(self) -> None:
        """Activate camera"""
        self.status = CameraStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate camera"""
        self.status = CameraStatus.INACTIVE
        self.updated_at = datetime.utcnow()
    
    def mark_error(self) -> None:
        """Mark camera as error state"""
        self.status = CameraStatus.ERROR
        self.updated_at = datetime.utcnow()
    
    def is_operational(self) -> bool:
        """Check if camera is operational"""
        return self.status == CameraStatus.ACTIVE
