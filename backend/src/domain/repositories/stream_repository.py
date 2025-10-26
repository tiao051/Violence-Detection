"""Stream Repository Interface"""
from abc import ABC, abstractmethod
from typing import Optional, List
from ..entities import StreamSession


class IStreamRepository(ABC):
    """
    Stream Repository Interface
    
    Abstract contract for stream session data access.
    """
    
    @abstractmethod
    async def create_session(self, session: StreamSession) -> StreamSession:
        """Create new streaming session"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[StreamSession]:
        """Get session by ID"""
        pass
    
    @abstractmethod
    async def get_active_session_by_camera(self, camera_id: str) -> Optional[StreamSession]:
        """Get active session for a camera"""
        pass
    
    @abstractmethod
    async def update_session(self, session: StreamSession) -> StreamSession:
        """Update session"""
        pass
    
    @abstractmethod
    async def get_sessions_by_camera(self, camera_id: str) -> List[StreamSession]:
        """Get all sessions for a camera"""
        pass
