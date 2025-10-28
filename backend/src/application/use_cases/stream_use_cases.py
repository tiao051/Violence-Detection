"""Stream Use Cases - Streaming business logic

NOTE: These use cases are not currently connected to any API endpoints.
They are kept for potential future use or reference.
If you need to use them, implement the corresponding API routers in src/presentation/api/

To enable these use cases, uncomment the classes below and create the corresponding API endpoints.
"""
from typing import Optional
from uuid import uuid4

from ...domain.entities import StreamSession, StreamStatus
from ...domain.repositories import ICameraRepository, IStreamRepository


"""
class StartStreamUseCase:
    """Use case for starting a camera stream"""
    
    def __init__(
        self,
        camera_repository: ICameraRepository,
        stream_repository: IStreamRepository,
        stream_consumer
    ):
        self.camera_repository = camera_repository
        self.stream_repository = stream_repository
        self.stream_consumer = stream_consumer
    
    async def execute(self, camera_id: str) -> StreamSession:
        """Execute start stream use case"""
        # Verify camera exists
        camera = await self.camera_repository.get_by_id(camera_id)
        if not camera:
            raise ValueError(f"Camera {camera_id} not found")
        
        # Check if already streaming
        active_session = await self.stream_repository.get_active_session_by_camera(camera_id)
        if active_session:
            raise ValueError(f"Camera {camera_id} is already streaming")
        
        # Create new session
        session = StreamSession(
            id=str(uuid4()),
            camera_id=camera_id,
            status=StreamStatus.STARTING
        )
        session.start()
        
        # Save session
        saved_session = await self.stream_repository.create_session(session)
        
        # Start stream consumer (infrastructure layer)
        await self.stream_consumer.start(camera_id, camera.rtsp_url, session.id)
        
        return saved_session


class StopStreamUseCase:
    """Use case for stopping a camera stream"""
    
    def __init__(
        self,
        stream_repository: IStreamRepository,
        stream_consumer
    ):
        self.stream_repository = stream_repository
        self.stream_consumer = stream_consumer
    
    async def execute(self, camera_id: str) -> bool:
        """Execute stop stream use case"""
        # Get active session
        session = await self.stream_repository.get_active_session_by_camera(camera_id)
        if not session:
            raise ValueError(f"No active stream for camera {camera_id}")
        
        # Stop stream consumer
        await self.stream_consumer.stop(camera_id)
        
        # Update session
        session.stop()
        await self.stream_repository.update_session(session)
        
        return True


class GetStreamStatusUseCase:
    """Use case for getting stream status"""
    
    def __init__(self, stream_repository: IStreamRepository):
        self.stream_repository = stream_repository
    
    async def execute(self, camera_id: str) -> Optional[StreamSession]:
        """Execute get stream status use case"""
        return await self.stream_repository.get_active_session_by_camera(camera_id)
"""
