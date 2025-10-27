"""Camera Use Cases - Application business logic

NOTE: These use cases are not currently connected to any API endpoints.
They are kept for potential future use or reference.
If you need to use them, implement the corresponding API routers in src/presentation/api/

To enable these use cases, uncomment the classes below and create the corresponding API endpoints.
"""
from typing import List, Optional
from uuid import uuid4
from datetime import datetime

from ...domain.entities import Camera, CameraStatus
from ...domain.repositories import ICameraRepository


"""
class AddCameraUseCase:
    """Use case for adding a new camera"""
    
    def __init__(self, camera_repository: ICameraRepository):
        self.camera_repository = camera_repository
    
    async def execute(self, name: str, rtsp_url: str, location: str) -> Camera:
        """Execute add camera use case"""
        # Create domain entity
        camera = Camera(
            id=str(uuid4()),
            name=name,
            rtsp_url=rtsp_url,
            location=location,
            status=CameraStatus.INACTIVE,
            created_at=datetime.utcnow()
        )
        
        # Persist to repository
        saved_camera = await self.camera_repository.add(camera)
        
        return saved_camera


class GetCameraUseCase:
    """Use case for getting a camera by ID"""
    
    def __init__(self, camera_repository: ICameraRepository):
        self.camera_repository = camera_repository
    
    async def execute(self, camera_id: str) -> Optional[Camera]:
        """Execute get camera use case"""
        return await self.camera_repository.get_by_id(camera_id)


class ListCamerasUseCase:
    """Use case for listing all cameras"""
    
    def __init__(self, camera_repository: ICameraRepository):
        self.camera_repository = camera_repository
    
    async def execute(self, skip: int = 0, limit: int = 100) -> List[Camera]:
        """Execute list cameras use case"""
        return await self.camera_repository.get_all(skip, limit)


class UpdateCameraUseCase:
    """Use case for updating a camera"""
    
    def __init__(self, camera_repository: ICameraRepository):
        self.camera_repository = camera_repository
    
    async def execute(
        self,
        camera_id: str,
        name: Optional[str] = None,
        rtsp_url: Optional[str] = None,
        location: Optional[str] = None,
        status: Optional[str] = None
    ) -> Optional[Camera]:
        """Execute update camera use case"""
        camera = await self.camera_repository.get_by_id(camera_id)
        
        if not camera:
            return None
        
        # Update fields if provided
        if name is not None:
            camera.name = name
        if rtsp_url is not None:
            camera.rtsp_url = rtsp_url
        if location is not None:
            camera.location = location
        if status is not None:
            camera.status = CameraStatus(status)
        
        camera.updated_at = datetime.utcnow()
        camera.validate()
        
        return await self.camera_repository.update(camera)


class DeleteCameraUseCase:
    """Use case for deleting a camera"""
    
    def __init__(self, camera_repository: ICameraRepository):
        self.camera_repository = camera_repository
    
    async def execute(self, camera_id: str) -> bool:
        """Execute delete camera use case"""
        return await self.camera_repository.delete(camera_id)
"""
