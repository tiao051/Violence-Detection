"""
Inference Consumer Unit Tests

Tests Kafka message consuming, batch processing, alert deduplication, Redis publishing.

Usage:
    pytest ai_service/inference/test_inference_consumer.py -v
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.inference.inference_consumer import InferenceConsumer, FrameMessage


@pytest.fixture
def mock_model():
    """Create mocked violence detection model"""
    model = MagicMock()
    model.add_frame = MagicMock()
    model.predict = MagicMock(return_value=None)  # Returns None until buffer full
    return model


@pytest.fixture
def mock_redis_client():
    """Create mocked Redis client"""
    redis_client = AsyncMock()
    redis_client.ping = AsyncMock(return_value=True)
    redis_client.publish = AsyncMock(return_value=1)
    redis_client.setex = AsyncMock(return_value=True)
    redis_client.xadd = AsyncMock(return_value="1234567-0")
    redis_client.close = AsyncMock()
    return redis_client


@pytest.fixture
def inference_consumer(mock_model, mock_redis_client):
    """Create inference consumer with mocked dependencies"""
    with patch('ai_service.inference.inference_consumer.redis.from_url', return_value=mock_redis_client):
        consumer = InferenceConsumer(
            model=mock_model,
            kafka_bootstrap_servers="localhost:9092",
            kafka_topic="processed-frames",
            kafka_group_id="test-group",
            redis_url="redis://localhost:6379/0",
            batch_size=4,
            batch_timeout_ms=100,
            alert_cooldown_seconds=60,
        )
        consumer.redis_client = mock_redis_client
        return consumer


class TestInferenceConsumerKafkaIntegration:
    """Test Kafka message consumption"""
    
    def test_parse_kafka_message_valid(self, inference_consumer):
        """Test parsing valid Kafka message"""
        # Create a test frame
        test_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        success, jpeg_data = __import__('cv2').imencode('.jpg', test_frame)
        jpeg_hex = jpeg_data.tobytes().hex()
        
        message = {
            'camera_id': 'cam1',
            'frame_id': 'frame-123',
            'timestamp': 1234.56,
            'frame_seq': 42,
            'jpeg': jpeg_hex,
            'frame_shape': [224, 224, 3],
        }
        
        frame_msg = inference_consumer._parse_message('cam1', message)
        
        assert frame_msg is not None
        assert frame_msg.camera_id == 'cam1'
        assert frame_msg.frame_id == 'frame-123'
        assert frame_msg.timestamp == 1234.56
        assert frame_msg.frame_seq == 42
        assert frame_msg.frame.shape == (224, 224, 3)
    
    def test_parse_message_missing_jpeg(self, inference_consumer):
        """Test parsing fails on missing JPEG"""
        message = {
            'camera_id': 'cam1',
            'frame_id': 'frame-123',
            'timestamp': 1234.56,
            'frame_seq': 42,
            # Missing 'jpeg'
        }
        
        frame_msg = inference_consumer._parse_message('cam1', message)
        assert frame_msg is None
    
    def test_parse_message_invalid_jpeg(self, inference_consumer):
        """Test parsing fails on invalid JPEG"""
        message = {
            'camera_id': 'cam1',
            'frame_id': 'frame-123',
            'timestamp': 1234.56,
            'frame_seq': 42,
            'jpeg': 'invalid_hex_data',
        }
        
        frame_msg = inference_consumer._parse_message('cam1', message)
        assert frame_msg is None


class TestInferenceConsumerBatching:
    """Test batch processing logic"""
    
    def test_buffer_accumulates_frames(self, inference_consumer):
        """Test frames accumulate in per-camera buffers"""
        # Create 3 frame messages
        frames = []
        for i in range(3):
            test_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            success, jpeg_data = __import__('cv2').imencode('.jpg', test_frame)
            jpeg_hex = jpeg_data.tobytes().hex()
            
            message = {
                'camera_id': 'cam1',
                'frame_id': f'frame-{i}',
                'timestamp': 1234.56 + i,
                'frame_seq': i,
                'jpeg': jpeg_hex,
                'frame_shape': [224, 224, 3],
            }
            
            frame_msg = inference_consumer._parse_message('cam1', message)
            if frame_msg:
                frames.append(frame_msg)
        
        # Add to buffer
        if 'cam1' not in inference_consumer.camera_buffers:
            inference_consumer.camera_buffers['cam1'] = []
        
        for frame in frames:
            inference_consumer.camera_buffers['cam1'].append(frame)
        
        assert len(inference_consumer.camera_buffers['cam1']) == 3
    
    def test_batch_size_trigger(self, inference_consumer):
        """Test batch is triggered when size reached"""
        # Set batch_size = 4
        assert inference_consumer.batch_size == 4
        
        # Add 3 frames
        frames = []
        for i in range(3):
            test_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            success, jpeg_data = __import__('cv2').imencode('.jpg', test_frame)
            jpeg_hex = jpeg_data.tobytes().hex()
            
            message = {
                'camera_id': 'cam1',
                'frame_id': f'frame-{i}',
                'timestamp': 1234.56 + i,
                'frame_seq': i,
                'jpeg': jpeg_hex,
                'frame_shape': [224, 224, 3],
            }
            
            frame_msg = inference_consumer._parse_message('cam1', message)
            if frame_msg:
                frames.append(frame_msg)
        
        inference_consumer.camera_buffers['cam1'] = frames
        
        # Should not trigger yet (3 < 4)
        assert len(frames) < inference_consumer.batch_size


class TestInferenceConsumerAlertDedup:
    """Test alert deduplication logic"""
    
    def test_first_alert_allowed(self, inference_consumer):
        """Test first alert is always sent"""
        timestamp = 1000.0
        
        should_alert = inference_consumer._should_alert('cam1', timestamp)
        assert should_alert is True
    
    def test_alert_within_cooldown_deduped(self, inference_consumer):
        """Test alert within cooldown period is deduped"""
        # First alert at t=1000
        inference_consumer._should_alert('cam1', 1000.0)
        
        # Second alert at t=1030 (30s later, within 60s cooldown)
        should_alert = inference_consumer._should_alert('cam1', 1030.0)
        assert should_alert is False
    
    def test_alert_after_cooldown_allowed(self, inference_consumer):
        """Test alert after cooldown expires is allowed"""
        # First alert at t=1000
        inference_consumer._should_alert('cam1', 1000.0)
        
        # Second alert at t=1070 (70s later, past 60s cooldown)
        should_alert = inference_consumer._should_alert('cam1', 1070.0)
        assert should_alert is True
    
    def test_cooldown_per_camera(self, inference_consumer):
        """Test cooldown is per-camera independent"""
        # Alert for cam1 at t=1000
        inference_consumer._should_alert('cam1', 1000.0)
        
        # Alert for cam2 at t=1030 should still be allowed
        should_alert_cam2 = inference_consumer._should_alert('cam2', 1030.0)
        assert should_alert_cam2 is True


class TestInferenceConsumerRedisPublishing:
    """Test Redis alert publishing"""
    
    @pytest.mark.asyncio
    async def test_publish_detection_to_redis(self, inference_consumer, mock_redis_client):
        """Test detection is published to Redis"""
        test_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        success, jpeg_data = __import__('cv2').imencode('.jpg', test_frame)
        jpeg_hex = jpeg_data.tobytes().hex()
        
        message = {
            'camera_id': 'cam1',
            'frame_id': 'frame-123',
            'timestamp': 1234.56,
            'frame_seq': 42,
            'jpeg': jpeg_hex,
            'frame_shape': [224, 224, 3],
        }
        
        frame_msg = inference_consumer._parse_message('cam1', message)
        detection = {
            'violence': True,
            'confidence': 0.95,
            'latency_ms': 45.2,
        }
        
        await inference_consumer._publish_detection('cam1', frame_msg, detection)
        
        # Verify Redis calls were made
        assert mock_redis_client.publish.called
        assert mock_redis_client.setex.called
        assert mock_redis_client.xadd.called


class TestInferenceConsumerMetrics:
    """Test metrics tracking"""
    
    def test_metrics_initialization(self, inference_consumer):
        """Test metrics are initialized to zero"""
        assert inference_consumer.frames_consumed == 0
        assert inference_consumer.frames_processed == 0
        assert inference_consumer.detections_made == 0
        assert inference_consumer.alerts_sent == 0
        assert inference_consumer.alerts_deduped == 0
    
    def test_stats_structure(self, inference_consumer):
        """Test get_stats returns expected keys"""
        stats = inference_consumer.get_stats()
        
        assert 'frames_consumed' in stats
        assert 'frames_processed' in stats
        assert 'detections_made' in stats
        assert 'alerts_sent' in stats
        assert 'alerts_deduped' in stats
        assert 'fps' in stats
        assert 'elapsed_seconds' in stats


class TestInferenceConsumerConfig:
    """Test configuration"""
    
    def test_batch_size_from_init(self, inference_consumer):
        """Test batch_size is set correctly"""
        assert inference_consumer.batch_size == 4
    
    def test_cooldown_from_init(self, inference_consumer):
        """Test alert_cooldown_seconds is set correctly"""
        assert inference_consumer.alert_cooldown_seconds == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
