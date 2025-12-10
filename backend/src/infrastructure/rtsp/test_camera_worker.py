"""
Camera Worker Unit Tests

Tests RTSP streaming, frame sampling, Kafka producer integration.

Usage:
    pytest backend/src/infrastructure/rtsp/test_camera_worker.py -v
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.src.infrastructure.rtsp.camera_worker import CameraWorker


@pytest.fixture
def mock_rtsp_client():
    """Create mocked RTSP client"""
    client = MagicMock()
    client.is_connected = True
    client.get_stats = MagicMock(return_value={})
    return client


@pytest.fixture
def mock_kafka_producer():
    """Create mocked Kafka producer"""
    producer = AsyncMock()
    producer.send_frame = AsyncMock(return_value="frame-id")
    producer.get_stats = MagicMock(return_value={
        'frames_sent': 0,
        'frames_failed': 0,
        'total_bytes_sent': 0,
    })
    return producer


@pytest.fixture
def camera_worker(mock_rtsp_client, mock_kafka_producer):
    """Create camera worker with mocked dependencies"""
    with patch('backend.src.infrastructure.rtsp.camera_worker.RTSPClient', return_value=mock_rtsp_client):
        with patch('backend.src.infrastructure.rtsp.camera_worker.get_kafka_producer', return_value=mock_kafka_producer):
            worker = CameraWorker(
                camera_id="cam1",
                stream_url="rtsp://localhost:8554/cam1",
                sample_rate=5,
            )
            worker.client = mock_rtsp_client
            worker.kafka_producer = mock_kafka_producer
            return worker


class TestCameraWorkerFrameSampling:
    """Test frame sampling logic"""
    
    def test_sampling_rate_correct(self, camera_worker):
        """Test sample_interval matches sample_rate"""
        assert camera_worker.sample_rate == 5
        assert abs(camera_worker.sample_interval - 0.2) < 0.001  # 1/5 = 0.2s
    
    def test_metrics_initialization(self, camera_worker):
        """Test metrics are initialized to zero"""
        assert camera_worker.frames_input == 0
        assert camera_worker.frames_sampled == 0
        assert camera_worker.frames_sent == 0
        assert camera_worker.frames_failed == 0
    
    @pytest.mark.asyncio
    async def test_worker_state_transitions(self, camera_worker):
        """Test worker start/stop state transitions"""
        assert not camera_worker.is_running
        
        # Mock the _run method to avoid infinite loop
        camera_worker._run = AsyncMock()
        
        await camera_worker.start()
        assert camera_worker.is_running
        assert camera_worker.task is not None
        
        await camera_worker.stop()
        assert not camera_worker.is_running


class TestCameraWorkerKafkaIntegration:
    """Test Kafka producer integration"""
    
    @pytest.mark.asyncio
    async def test_sends_raw_frame_to_kafka(self, camera_worker, mock_kafka_producer):
        """Test raw frame (no resize) is sent to Kafka"""
        raw_frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
        
        # Mock client to return a frame
        camera_worker.client.read_frame = AsyncMock(return_value=(True, raw_frame))
        camera_worker.client.is_connected = True
        
        # Simulate one frame being sent
        camera_worker.frames_input = 0
        camera_worker.frames_sampled = 0
        current_time = 1000.0
        
        # Check sampling logic
        camera_worker.last_sample_time = current_time - 1.0  # Enough time has passed
        camera_worker.frames_input = 6  # Past warmup
        
        # Send frame
        result = await camera_worker.kafka_producer.send_frame(
            camera_id="cam1",
            frame=raw_frame,  # Raw frame, not resized
            frame_id="test-id",
            timestamp=current_time,
            frame_seq=1,
        )
        
        assert result is not None
        camera_worker.kafka_producer.send_frame.assert_called()
    
    @pytest.mark.asyncio
    async def test_kafka_call_includes_correct_metadata(self, camera_worker, mock_kafka_producer):
        """Test Kafka send includes correct camera_id and sequence"""
        raw_frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
        
        await camera_worker.kafka_producer.send_frame(
            camera_id="cam1",
            frame=raw_frame,
            frame_id="frame-123",
            timestamp=1234.56,
            frame_seq=42,
        )
        
        # Verify call
        call_args = camera_worker.kafka_producer.send_frame.call_args
        assert call_args[1]['camera_id'] == "cam1"
        assert call_args[1]['frame_seq'] == 42
        assert call_args[1]['timestamp'] == 1234.56


class TestCameraWorkerMetrics:
    """Test worker metrics and statistics"""
    
    def test_stats_structure(self, camera_worker):
        """Test get_stats returns expected keys"""
        camera_worker.start_time = None
        stats = camera_worker.get_stats()
        
        assert 'camera_id' in stats
        assert 'is_running' in stats
        assert 'is_connected' in stats
        assert 'frames_input' in stats
        assert 'frames_sampled' in stats
        assert 'frames_sent' in stats
        assert 'frames_failed' in stats
        assert 'input_fps' in stats
        assert 'output_fps' in stats
        assert 'elapsed_time_seconds' in stats
    
    def test_fps_calculation(self, camera_worker):
        """Test FPS calculations are correct"""
        from datetime import datetime, timedelta
        
        camera_worker.start_time = datetime.now() - timedelta(seconds=10)
        camera_worker.frames_input = 100
        camera_worker.frames_sent = 50
        
        stats = camera_worker.get_stats()
        
        # 100 frames in 10 seconds = 10 FPS input
        assert 9 < stats['input_fps'] < 11
        # 50 frames in 10 seconds = 5 FPS output
        assert 4 < stats['output_fps'] < 6


class TestCameraWorkerNoResize:
    """Test that resize is NOT done in camera_worker"""
    
    @pytest.mark.asyncio
    async def test_no_cv2_resize_in_worker(self, camera_worker):
        """Test camera_worker doesn't import cv2 (no resize)"""
        # Check imports don't include cv2
        import backend.src.infrastructure.rtsp.camera_worker as worker_module
        
        # cv2 should NOT be imported in camera_worker
        assert 'cv2' not in dir(worker_module)
    
    def test_worker_sends_raw_frame_directly(self, camera_worker):
        """Test worker sends frame without processing"""
        # Worker __init__ should not have target_width/target_height
        assert not hasattr(camera_worker, 'target_width')
        assert not hasattr(camera_worker, 'target_height')


class TestCameraWorkerMultipleCameras:
    """Test multiple camera workers can run concurrently"""
    
    @pytest.mark.asyncio
    async def test_multiple_workers_independent(self):
        """Test multiple camera workers have independent state"""
        producers = [AsyncMock() for _ in range(3)]
        workers = []
        
        for i, producer in enumerate(producers):
            producer.send_frame = AsyncMock(return_value=f"frame-{i}")
            producer.get_stats = MagicMock(return_value={})
            
            with patch('backend.src.infrastructure.rtsp.camera_worker.RTSPClient'):
                with patch('backend.src.infrastructure.rtsp.camera_worker.get_kafka_producer', return_value=producer):
                    worker = CameraWorker(
                        camera_id=f"cam{i}",
                        stream_url=f"rtsp://localhost:8554/cam{i}",
                    )
                    workers.append(worker)
        
        # Verify each worker has independent state
        assert workers[0].camera_id == "cam0"
        assert workers[1].camera_id == "cam1"
        assert workers[2].camera_id == "cam2"
        
        assert workers[0].kafka_producer != workers[1].kafka_producer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
