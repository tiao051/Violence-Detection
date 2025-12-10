"""
Kafka Producer Unit Tests

Comprehensive test suite for KafkaFrameProducer class.

Tests:
    - Frame resizing optimization (single point in producer, not in camera_worker)
    - JPEG compression (10-50x size reduction)
    - Kafka message format and JSON structure
    - Partitioning by camera_id for temporal ordering
    - Metrics tracking (frames_sent, frames_failed, compression_ratio)
    - Configuration loading from settings

Usage:
    pytest backend/tests/infrastructure/kafka/test_producer.py -v
    pytest backend/tests/infrastructure/kafka/test_producer.py::TestKafkaProducerResize -v
    pytest backend/tests/infrastructure/kafka/test_producer.py::TestKafkaProducerResize::test_resize_high_to_target_resolution -v
    pytest backend/tests/infrastructure/kafka/test_producer.py -v --cov=backend/src/infrastructure/kafka
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[5]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.src.infrastructure.kafka.producer import KafkaFrameProducer
from backend.src.core.config import settings


@pytest.fixture
def producer():
    """Create producer instance with mocked Kafka"""
    with patch('backend.src.infrastructure.kafka.producer.AIOKafkaProducer'):
        return KafkaFrameProducer(
            bootstrap_servers="localhost:9092",
            topic="test-frames",
            jpeg_quality=80,
            target_width=224,
            target_height=224,
        )


@pytest.fixture
def raw_frame():
    """Create random raw frame (high resolution)"""
    return np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)


class TestKafkaProducerResize:
    """Test frame resizing optimization"""
    
    def test_resize_high_to_target_resolution(self, producer, raw_frame):
        """Test resize from 1080p to 224x224"""
        jpeg_bytes = producer._resize_and_compress(raw_frame)
        
        assert jpeg_bytes is not None
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
    
    def test_resize_skipped_when_already_target(self, producer):
        """Test no resize when frame already 224x224"""
        frame_224 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Should still compress without unnecessary resize
        jpeg_bytes = producer._resize_and_compress(frame_224)
        
        assert jpeg_bytes is not None
        assert isinstance(jpeg_bytes, bytes)
    
    def test_resize_various_input_sizes(self, producer):
        """Test resize works with various input resolutions"""
        test_sizes = [(640, 480), (1280, 720), (1920, 1080), (224, 224), (112, 112)]
        
        for height, width in test_sizes:
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            jpeg_bytes = producer._resize_and_compress(frame)
            
            assert jpeg_bytes is not None
            assert isinstance(jpeg_bytes, bytes)
    
    def test_single_resize_point(self, producer, raw_frame):
        """Test resize happens only in _resize_and_compress"""
        # This test verifies the optimization: resize is only in producer
        # not in camera_worker
        jpeg_bytes = producer._resize_and_compress(raw_frame)
        
        # Decode to verify correct size
        jpeg_array = np.frombuffer(jpeg_bytes, np.uint8)
        decoded = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        
        assert decoded.shape == (224, 224, 3)


class TestKafkaProducerCompression:
    """Test JPEG compression"""
    
    def test_jpeg_compression_reduces_size(self, producer, raw_frame):
        """Test JPEG compression significantly reduces frame size"""
        raw_size = raw_frame.nbytes
        
        jpeg_bytes = producer._resize_and_compress(raw_frame)
        compressed_size = len(jpeg_bytes)
        
        # Should be 10x-50x smaller than raw
        compression_ratio = raw_size / compressed_size
        assert compression_ratio > 10
    
    def test_quality_80_produces_valid_jpeg(self, producer):
        """Test JPEG quality 80 produces valid, decodable JPEG"""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        jpeg_bytes = producer._resize_and_compress(frame)
        
        # Verify it's decodable
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (224, 224, 3)
    
    def test_quality_parameter_affects_size(self):
        """Test higher quality produces larger files"""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        producer_low = KafkaFrameProducer(jpeg_quality=60)
        producer_high = KafkaFrameProducer(jpeg_quality=95)
        
        jpeg_low = producer_low._resize_and_compress(frame)
        jpeg_high = producer_high._resize_and_compress(frame)
        
        assert len(jpeg_low) < len(jpeg_high)


class TestKafkaProducerMessage:
    """Test Kafka message format"""
    
    @pytest.mark.asyncio
    async def test_message_format_correct(self, producer):
        """Test message JSON structure"""
        producer.is_connected = True
        
        # Mock Kafka producer
        producer.producer = AsyncMock()
        producer.producer.send_and_wait = AsyncMock(return_value=None)
        
        frame = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
        
        await producer.send_frame(
            camera_id="cam1",
            frame=frame,
            frame_id="test-id",
            timestamp=123.45,
            frame_seq=42,
        )
        
        # Verify Kafka was called
        assert producer.producer.send_and_wait.called
        
        # Get the message that was sent
        call_args = producer.producer.send_and_wait.call_args
        message_json = call_args[1]['value'].decode('utf-8')
        
        import json
        message = json.loads(message_json)
        
        assert message['camera_id'] == "cam1"
        assert message['frame_id'] == "test-id"
        assert message['timestamp'] == 123.45
        assert message['frame_seq'] == 42
        assert 'jpeg' in message
        assert message['frame_shape'] == [224, 224, 3]
    
    @pytest.mark.asyncio
    async def test_kafka_key_is_camera_id(self, producer):
        """Test Kafka partitioning key is camera_id"""
        producer.is_connected = True
        producer.producer = AsyncMock()
        producer.producer.send_and_wait = AsyncMock(return_value=None)
        
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        await producer.send_frame(
            camera_id="cam1",
            frame=frame,
            frame_id="id",
            timestamp=0,
            frame_seq=0,
        )
        
        call_args = producer.producer.send_and_wait.call_args
        key = call_args[1]['key']
        
        assert key == b"cam1"


class TestKafkaProducerMetrics:
    """Test producer metrics tracking"""
    
    @pytest.mark.asyncio
    async def test_metrics_track_sent_frames(self, producer):
        """Test frames_sent counter increments"""
        producer.is_connected = True
        producer.producer = AsyncMock()
        producer.producer.send_and_wait = AsyncMock(return_value=None)
        
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        for i in range(5):
            await producer.send_frame(
                camera_id=f"cam{i}",
                frame=frame,
                frame_id=f"id-{i}",
                timestamp=i,
                frame_seq=i,
            )
        
        stats = producer.get_stats()
        assert stats['frames_sent'] == 5
    
    @pytest.mark.asyncio
    async def test_metrics_track_failed_frames(self, producer):
        """Test frames_failed counter increments on error"""
        producer.is_connected = True
        producer.producer = AsyncMock()
        producer.producer.send_and_wait = AsyncMock(side_effect=Exception("Kafka error"))
        
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        result = await producer.send_frame(
            camera_id="cam1",
            frame=frame,
            frame_id="id",
            timestamp=0,
            frame_seq=0,
        )
        
        assert result is None
        stats = producer.get_stats()
        assert stats['frames_failed'] == 1
    
    def test_stats_compression_ratio(self, producer):
        """Test compression ratio calculation in stats"""
        # Simulate sending frames
        producer.frames_sent = 10
        producer.total_bytes_sent = 100_000  # 100KB total
        
        stats = producer.get_stats()
        avg_size_kb = stats['avg_frame_size_kb']
        
        assert abs(avg_size_kb - 10.0) < 0.1  # 100KB / 10 frames = 10KB/frame


class TestKafkaProducerConfig:
    """Test configuration loading"""
    
    def test_producer_uses_config_settings(self):
        """Test producer loads from settings"""
        producer = KafkaFrameProducer()
        
        assert producer.bootstrap_servers == settings.kafka_bootstrap_servers
        assert producer.topic == settings.kafka_frame_topic
        assert producer.jpeg_quality == settings.kafka_jpeg_quality
        assert producer.compression_type == settings.kafka_compression_type
    
    def test_producer_override_settings(self):
        """Test producer can override settings"""
        producer = KafkaFrameProducer(
            bootstrap_servers="custom:9092",
            topic="custom-topic",
            jpeg_quality=95,
        )
        
        assert producer.bootstrap_servers == "custom:9092"
        assert producer.topic == "custom-topic"
        assert producer.jpeg_quality == 95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
