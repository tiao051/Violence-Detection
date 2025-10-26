"""Camera Repository Interface"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities import Camera


class ICameraRepository(ABC):
    """
    Camera Repository Interface
    
    Abstract contract for camera data access.
    Infrastructure layer will implement this interface.
    """
    
    @abstractmethod
    async def add(self, camera: Camera) -> Camera:
        """Add new camera"""
        pass
    
    @abstractmethod
    async def get_by_id(self, camera_id: str) -> Optional[Camera]:
        """Get camera by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[Camera]:
        """Get all cameras with pagination"""
        pass
    
    @abstractmethod
    async def update(self, camera: Camera) -> Camera:
        """Update existing camera"""
        pass
    
    @abstractmethod
    async def delete(self, camera_id: str) -> bool:
        """Delete camera by ID"""
        pass
    
    @abstractmethod
    async def exists(self, camera_id: str) -> bool:
        """Check if camera exists"""
        pass
