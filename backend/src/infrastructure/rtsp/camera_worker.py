"""Camera worker for continuous RTSP stream processing."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

import cv2

from .rtsp_client import RTSPClient
from .frame_processor import FrameProcessor
from ..memory import get_frame_buffer

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Background worker for processing RTSP camera streams.
    
    Features:
    - Pull frames from RTSP stream
    - Sample frames (e.g., 30 FPS → 6 FPS)
    - Encode to JPEG
    - Push to Redis Streams
    - Auto-reconnect on disconnect
    - Health monitoring
    """
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        redis_producer,
        sample_rate: int = 6,
        frame_width: int = 640,
        frame_height: int = 480,
        jpeg_quality: int = 80,
    ):
        """
        Initialize camera worker.
        
        Args:
            camera_id: Unique camera identifier (e.g., "cam1")
            rtsp_url: RTSP stream URL
            redis_producer: Redis streams producer instance
            sample_rate: Target FPS for frame sampling (default: 6)
            frame_width: Resized frame width
            frame_height: Resized frame height
            jpeg_quality: JPEG encoding quality (1-100)
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.redis_producer = redis_producer
        self.sample_rate = sample_rate
        
        # Get frame buffer (shared in-memory storage)
        self.frame_buffer = get_frame_buffer()
        self.frame_buffer.register_camera(camera_id)
        
        # Store target dimensions for resizing
        self.target_width = frame_width
        self.target_height = frame_height
        
        self.rtsp_client = RTSPClient(
            rtsp_url=rtsp_url,
            camera_id=camera_id,
        )
        
        # Worker state
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        
        # Metrics
        self.frames_input = 0  # Total frames from RTSP
        self.frames_sampled = 0  # Frames after sampling
        self.frames_sent_to_redis = 0  # Successfully sent to Redis
        self.start_time: Optional[datetime] = None
        self.last_sample_time: float = 0  # Track last sampled frame time
        self.sample_interval: float = 1.0 / sample_rate  # seconds between samples
    
    async def start(self) -> None:
        """Start the camera worker background task."""
        if self.is_running:
            logger.warning(f"[{self.camera_id}] Worker already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"[{self.camera_id}] Starting worker...")
        
        # Create background task
        self.task = asyncio.create_task(self._run())
        
        logger.info(f"[{self.camera_id}] Worker started")
    
    async def stop(self) -> None:
        """Stop the camera worker gracefully."""
        if not self.is_running:
            logger.warning(f"[{self.camera_id}] Worker not running")
            return
        
        logger.info(f"[{self.camera_id}] Stopping worker...")
        
        self.is_running = False
        
        # Cancel task
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # Close RTSP connection
        self.rtsp_client.close()
        
        logger.info(f"[{self.camera_id}] Worker stopped")
    
    async def _run(self) -> None:
        """Main worker loop (runs as background task)."""
        logger.info(f"[{self.camera_id}] Worker loop started")
        
        try:
            while self.is_running:
                # Connect if not already connected
                if not self.rtsp_client.is_connected:
                    connected = await self.rtsp_client.connect()
                    
                    if not connected:
                        # Try to reconnect
                        reconnected = await self.rtsp_client.reconnect()
                        if not reconnected:
                            logger.error(f"[{self.camera_id}] Failed to reconnect, stopping worker")
                            self.is_running = False
                            break
                        continue
                
                # Read frame
                success, frame = await self.rtsp_client.read_frame()
                
                if not success:
                    # Connection lost, will reconnect in next loop
                    await asyncio.sleep(0.1)
                    continue
                
                self.frames_input += 1
                
                # Check if should sample this frame (time-based)
                current_time = time.time()
                if (current_time - self.last_sample_time) < self.sample_interval:
                    continue
                
                self.last_sample_time = current_time
                self.frames_sampled += 1
                
                # Resize frame directly (no encoding/decoding)
                resized_frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                if resized_frame is None:
                    logger.warning(f"[{self.camera_id}] Frame resize failed")
                    continue
                
                # Store frame directly in shared memory (keep-last-only)
                frame_id = str(uuid.uuid4())
                self.frame_buffer.put(
                    camera_id=self.camera_id,
                    frame=resized_frame,
                    frame_id=frame_id,
                    timestamp=current_time,
                    frame_seq=self.frames_sampled,
                )
                
                # Push metadata to Redis (only reference to RAM buffer)
                try:
                    await self.redis_producer.add_frame_metadata(
                        camera_id=self.camera_id,
                        frame_id=frame_id,
                        timestamp=current_time,
                        frame_seq=self.frames_sampled,
                    )
                    self.frames_sent_to_redis += 1
                    
                    # Cleanup old frames periodically
                    if self.frames_sampled % 100 == 0:
                        await self.redis_producer.cleanup_old_frames(self.camera_id)
                except Exception as e:
                    logger.error(f"[{self.camera_id}] Redis metadata push error: {str(e)}")
        
        except asyncio.CancelledError:
            logger.info(f"[{self.camera_id}] Worker cancelled")
            self.rtsp_client.close()
        except Exception as e:
            logger.error(f"[{self.camera_id}] Worker error: {str(e)}")
            self.is_running = False
            self.rtsp_client.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics.
        
        Returns:
            Dictionary with worker stats
        """
        elapsed_time = 0
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        input_fps = self.frames_input / elapsed_time if elapsed_time > 0 else 0
        output_fps = self.frames_sent_to_redis / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "camera_id": self.camera_id,
            "is_running": self.is_running,
            "is_connected": self.rtsp_client.is_connected,
            "frames_input": self.frames_input,
            "frames_sampled": self.frames_sampled,
            "frames_sent_to_redis": self.frames_sent_to_redis,
            "input_fps": round(input_fps, 2),
            "output_fps": round(output_fps, 2),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "rtsp_client_stats": self.rtsp_client.get_stats(),
        }
