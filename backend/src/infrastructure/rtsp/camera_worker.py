"""Camera worker for continuous RTSP stream processing."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from .rtsp_client import RTSPClient
from ..kafka import get_kafka_producer

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Background worker for processing RTSP camera streams.
    
    Responsibilities:
    - Pull frames from RTSP stream
    - Sample frames (30 FPS → 5 FPS)
    - Resize to model input size (224x224)
    - Send to Kafka for centralized inference
    - Auto-reconnect on disconnect
    - Health monitoring
    
    Note: This is a PRODUCER ONLY. Inference is done by inference_consumer
    from Kafka messages. Do not run inference here.
    """
    
    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        kafka_producer=None,
        sample_rate: int = 5,
    ):
        """
        Initialize camera worker.
        
        Args:
            camera_id: Unique camera identifier (e.g., "cam1")
            stream_url: RTSP stream URL
            kafka_producer: Kafka producer instance (will use singleton if not provided)
            sample_rate: Target FPS for frame sampling (default: 5)
        """
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.kafka_producer = kafka_producer or get_kafka_producer()
        self.sample_rate = sample_rate
        
        # Create RTSP client
        self.client = RTSPClient(
            rtsp_url=stream_url,
            camera_id=camera_id,
        )
        
        # Worker state
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        
        # Metrics
        self.frames_input = 0  # Total frames from stream
        self.frames_sampled = 0  # Frames after sampling
        self.frames_sent = 0  # Successfully sent to Kafka
        self.frames_failed = 0  # Failed to send to Kafka
        self.start_time: Optional[datetime] = None
        self.last_sample_time: float = 0
        self.sample_interval: float = 1.0 / sample_rate
    
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
        self.client.close()
        
        logger.info(f"[{self.camera_id}] Worker stopped")
    
    async def _run(self) -> None:
        """Main worker loop (runs as background task)."""
        logger.info(f"[{self.camera_id}] Worker loop started")
        
        frames_to_skip = 5  # Skip first frames for codec warmup
        first_processed_frame = False

        try:
            while self.is_running:
                # Connect if not already connected
                if not self.client.is_connected:
                    connected = await self.client.connect()
                    
                    if not connected:
                        logger.warning(f"[{self.camera_id}] Cannot connect. Disabling worker.")
                        self.is_running = False
                        break

                    self.client.is_connected = True
                
                # Read frame
                success, frame = await self.client.read_frame()
                
                if not success:
                    # Connection lost, will reconnect in next loop
                    await asyncio.sleep(0.1)
                    continue
                
                self.frames_input += 1
                
                # Skip first frames (codec warmup)
                if self.frames_input <= frames_to_skip:
                    continue

                # Check if should sample this frame (time-based)
                current_time = time.time()
                if (current_time - self.last_sample_time) < self.sample_interval:
                    # Avoid busy-waiting: yield control if not time to sample yet
                    await asyncio.sleep(0.001)  # 1ms to prevent CPU spin
                    continue
                
                self.last_sample_time = current_time
                self.frames_sampled += 1
                
                frame_id = str(uuid.uuid4())
                
                # Send raw frame to Kafka producer
                # Producer will handle resize + JPEG compression (optimization point)
                kafka_result = await self.kafka_producer.send_frame(
                    camera_id=self.camera_id,
                    frame=frame,  # Raw frame - let producer resize
                    frame_id=frame_id,
                    timestamp=current_time,
                    frame_seq=self.frames_sampled,
                )
                
                if kafka_result:
                    self.frames_sent += 1
                    logger.debug(
                        f"[{self.camera_id}] Frame sent to Kafka: "
                        f"frame_id={frame_id}, seq={self.frames_sampled}"
                    )
                else:
                    self.frames_failed += 1
                    logger.error(
                        f"[{self.camera_id}] Failed to send frame to Kafka: {frame_id}"
                    )
        
        except asyncio.CancelledError:
            logger.info(f"[{self.camera_id}] Worker cancelled")
            self.client.close()
        except Exception as e:
            logger.error(f"[{self.camera_id}] Worker error: {str(e)}")
            self.is_running = False
            self.client.close()
    
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
        output_fps = self.frames_sent / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "camera_id": self.camera_id,
            "is_running": self.is_running,
            "is_connected": self.client.is_connected,
            "frames_input": self.frames_input,
            "frames_sampled": self.frames_sampled,
            "frames_sent": self.frames_sent,
            "frames_failed": self.frames_failed,
            "input_fps": round(input_fps, 2),
            "output_fps": round(output_fps, 2),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "kafka_producer_stats": self.kafka_producer.get_stats(),
            "client_stats": self.client.get_stats(),
        }
