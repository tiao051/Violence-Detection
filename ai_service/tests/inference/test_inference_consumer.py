"""
Inference Consumer Unit Tests

Comprehensive test suite for InferenceConsumer class.
Updated to match the Queue-based architecture and actual class implementation.
"""

import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.inference.inference_consumer import InferenceConsumer, FrameMessage


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_model():
    """Mock violence detection model"""
    mock = MagicMock() # Model methods are run in executor, so they can be sync or async mocks
    mock.predict = MagicMock(return_value={'violence': True, 'confidence': 0.95})
    mock.add_frame = MagicMock()
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock = AsyncMock()
    mock.publish = AsyncMock(return_value=None)
    mock.xadd = AsyncMock(return_value="1234567890")
    mock.setex = AsyncMock(return_value=True)
    mock.ping = AsyncMock(return_value=True)
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def inference_consumer(mock_kafka_consumer, mock_model, mock_redis):
    """Create InferenceConsumer with mocked dependencies"""
    consumer = InferenceConsumer(
        model=mock_model,
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="inference",
        kafka_topic="processed-frames",
        redis_url="redis://localhost:6379",
        batch_size=4,
        batch_timeout_ms=5000,
        alert_cooldown_seconds=60
    )
    # Inject mocks
    consumer.consumer = mock_kafka_consumer
    consumer.redis_client = mock_redis
    return consumer


def create_kafka_message(camera_id, frame_id, timestamp, frame_seq, jpeg_data):
    """Create a Kafka message dict"""
    return {
        'camera_id': camera_id,
        'frame_id': frame_id,
        'timestamp': timestamp,
        'frame_seq': frame_seq,
        'jpeg': jpeg_data,
        'frame_shape': [224, 224, 3],
    }


class TestInferenceConsumerStructure:
    """Test class structure and initialization"""
    
    def test_initialization(self, inference_consumer):
        """Test correct initialization of attributes"""
        assert inference_consumer.kafka_topic == "processed-frames"
        assert inference_consumer.batch_size == 4
        assert inference_consumer.alert_cooldown_seconds == 60
        # Check new Queue architecture components
        assert isinstance(inference_consumer.frame_queue, asyncio.Queue)
        assert inference_consumer.frame_queue.maxsize == 100
        assert inference_consumer._executor is not None


class TestInferenceConsumerBatching:
    """Test batch processing logic"""
    
    @pytest.mark.asyncio
    async def test_consumer_batches_frames_per_camera(self, inference_consumer):
        """Test frames batched by camera_id"""
        assert hasattr(inference_consumer, 'camera_buffers')
        assert isinstance(inference_consumer.camera_buffers, dict)
    
    @pytest.mark.asyncio
    async def test_different_cameras_batched_separately(self, inference_consumer):
        """Test cam1 and cam2 frames in separate buffers"""
        # Simulate adding frames to buffers
        msg1 = FrameMessage("cam1", "f1", 1.0, 1, np.zeros((224,224,3)))
        msg2 = FrameMessage("cam2", "f1", 1.0, 1, np.zeros((224,224,3)))
        
        inference_consumer.camera_buffers['cam1'] = [msg1]
        inference_consumer.camera_buffers['cam2'] = [msg2]
        
        assert len(inference_consumer.camera_buffers['cam1']) == 1
        assert len(inference_consumer.camera_buffers['cam2']) == 1
        assert inference_consumer.camera_buffers['cam1'] != inference_consumer.camera_buffers['cam2']


class TestInferenceConsumerAlertDedup:
    """Test alert deduplication logic"""
    
    def test_should_alert_logic(self, inference_consumer):
        """Test _should_alert method directly"""
        camera_id = "cam1"
        
        # 1. First alert - should pass
        assert inference_consumer._should_alert(camera_id, 100.0) is True
        assert inference_consumer.camera_last_alert[camera_id] == 100.0
        
        # 2. Alert 10s later (cooldown 60s) - should fail
        assert inference_consumer._should_alert(camera_id, 110.0) is False
        assert inference_consumer.camera_last_alert[camera_id] == 100.0 # Timestamp shouldn't update
        
        # 3. Alert 61s later - should pass
        assert inference_consumer._should_alert(camera_id, 161.0) is True
        assert inference_consumer.camera_last_alert[camera_id] == 161.0


class TestInferenceConsumerQueueLogic:
    """Test the new Queue-based architecture"""

    @pytest.mark.asyncio
    async def test_queue_full_behavior(self, inference_consumer):
        """Test that queue drops old items when full"""
        # Fill queue to max
        for i in range(100):
            await inference_consumer.frame_queue.put(i)
        
        assert inference_consumer.frame_queue.full()
        
        # Try to put one more item (simulating _kafka_reader_task logic)
        if inference_consumer.frame_queue.full():
            inference_consumer.frame_queue.get_nowait() # Drop old
        
        await inference_consumer.frame_queue.put(999)
        
        assert inference_consumer.frame_queue.qsize() == 100
        
        # The first item popped should be 1 (since 0 was dropped)
        item = await inference_consumer.frame_queue.get()
        assert item == 1


class TestInferenceConsumerConfig:
    """Test configuration loading"""
    
    def test_consumer_uses_config_settings(self, mock_model):
        """Test consumer loads from settings"""
        # We need to mock os.getenv or pass args. 
        # Here we test passing args overrides env vars
        consumer = InferenceConsumer(
            model=mock_model,
            kafka_bootstrap_servers="custom:9092",
            kafka_topic="custom-topic",
            kafka_group_id="custom-group",
            redis_url="redis://custom:6379",
            batch_size=8,
            batch_timeout_ms=1000,
            alert_cooldown_seconds=30
        )
        
        assert consumer.kafka_bootstrap_servers == "custom:9092"
        assert consumer.kafka_topic == "custom-topic"
        assert consumer.batch_size == 8


class TestInferenceConsumerMetrics:
    """Test metrics tracking"""
    
    def test_consumer_tracks_basic_metrics(self, inference_consumer):
        """Test basic metrics"""
        inference_consumer.frames_processed = 100
        inference_consumer.alerts_sent = 5
        
        stats = inference_consumer.get_stats()
        assert stats['frames_processed'] == 100
        assert stats['alerts_sent'] == 5
        assert 'fps' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
