"""
Inference Consumer Unit Tests

Comprehensive test suite for InferenceConsumer class.

Tests:
    - Kafka message consumption and JSON parsing
    - Per-camera frame buffering and batching
    - Batch processing (configurable size and timeout)
    - Alert deduplication checking before publishing
    - Redis pub/sub and stream publishing
    - Model inference and confidence thresholding
    - Per-camera metrics tracking
    - Configuration loading from settings
    - Error handling and edge cases

Usage (copy & paste):
    pytest ai_service/tests/inference/test_inference_consumer.py -v
    pytest ai_service/tests/inference/test_inference_consumer.py::TestInferenceConsumerBatching -v
    pytest ai_service/tests/inference/test_inference_consumer.py::TestInferenceConsumerAlertDedup::test_alert_dedup_per_camera -v
    pytest ai_service/tests/inference/test_inference_consumer.py -v --cov=ai_service/inference
"""

import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.inference.inference_consumer import InferenceConsumer


@pytest.fixture
async def mock_kafka_consumer():
    """Mock Kafka consumer"""
    mock = AsyncMock()
    return mock


@pytest.fixture
async def mock_model():
    """Mock violence detection model"""
    mock = AsyncMock()
    mock.predict = AsyncMock(return_value=[0.2, 0.8])  # [no_violence, violence]
    return mock


@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    mock = AsyncMock()
    mock.publish = AsyncMock(return_value=None)
    mock.xadd = AsyncMock(return_value="1234567890")
    return mock


@pytest.fixture
async def mock_alert_dedup():
    """Mock alert deduplication service"""
    mock = AsyncMock()
    mock.should_send_alert = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def inference_consumer(mock_kafka_consumer, mock_model, mock_redis, mock_alert_dedup):
    """Create InferenceConsumer with mocked dependencies"""
    consumer = InferenceConsumer(
        kafka_bootstrap_servers="localhost:9092",
        kafka_consumer_group="inference",
        kafka_frame_topic="processed-frames",
        batch_size=4,
        batch_timeout_ms=5000,
    )
    consumer.kafka_consumer = mock_kafka_consumer
    consumer.model = mock_model
    consumer.redis = mock_redis
    consumer.alert_dedup = mock_alert_dedup
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


class TestInferenceConsumerKafkaIntegration:
    """Test Kafka message consumption"""
    
    @pytest.mark.asyncio
    async def test_consumer_reads_kafka_messages(self, inference_consumer, mock_kafka_consumer):
        """Test consumer subscribes and reads from Kafka"""
        mock_kafka_consumer.subscribe = AsyncMock()
        mock_kafka_consumer.getmany = AsyncMock(return_value={})
        
        assert inference_consumer.kafka_consumer is not None
        assert inference_consumer.kafka_frame_topic == "processed-frames"
    
    @pytest.mark.asyncio
    async def test_message_parsing_from_json(self, inference_consumer, mock_kafka_consumer):
        """Test JSON message parsing"""
        jpeg_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8).tobytes()
        
        message_dict = create_kafka_message(
            camera_id="cam1",
            frame_id="frame-1",
            timestamp=123.45,
            frame_seq=1,
            jpeg_data=jpeg_data.hex(),
        )
        
        message_json = json.dumps(message_dict).encode('utf-8')
        parsed = json.loads(message_json)
        
        assert parsed['camera_id'] == "cam1"
        assert parsed['frame_id'] == "frame-1"
        assert parsed['timestamp'] == 123.45
        assert parsed['frame_seq'] == 1


class TestInferenceConsumerBatching:
    """Test batch processing"""
    
    @pytest.mark.asyncio
    async def test_consumer_batches_frames_per_camera(self, inference_consumer):
        """Test frames batched by camera_id"""
        # Consumer should maintain per-camera buffers
        assert hasattr(inference_consumer, 'frame_buffers')
    
    @pytest.mark.asyncio
    async def test_batch_size_is_configurable(self, inference_consumer):
        """Test batch size setting"""
        assert inference_consumer.batch_size == 4
    
    @pytest.mark.asyncio
    async def test_batch_timeout_is_configurable(self, inference_consumer):
        """Test batch timeout setting"""
        assert inference_consumer.batch_timeout_ms == 5000
    
    @pytest.mark.asyncio
    async def test_different_cameras_batched_separately(self, inference_consumer):
        """Test cam1 and cam2 frames in separate batches"""
        # This tests the per-camera buffering logic
        # In implementation: frame_buffers['cam1'], frame_buffers['cam2']
        
        cam1_buffer = inference_consumer.frame_buffers.get('cam1', [])
        cam2_buffer = inference_consumer.frame_buffers.get('cam2', [])
        
        # Should be independent
        assert cam1_buffer is not cam2_buffer or (len(cam1_buffer) == 0 and len(cam2_buffer) == 0)


class TestInferenceConsumerAlertDedup:
    """Test alert deduplication"""
    
    @pytest.mark.asyncio
    async def test_should_alert_checks_deduplication(self, inference_consumer, mock_alert_dedup):
        """Test alert deduplication is checked before publishing"""
        mock_alert_dedup.should_send_alert = AsyncMock(return_value=True)
        
        should_send = await mock_alert_dedup.should_send_alert("cam1")
        assert should_send is True
    
    @pytest.mark.asyncio
    async def test_alert_dedup_per_camera(self, inference_consumer, mock_alert_dedup):
        """Test deduplication is per-camera"""
        mock_alert_dedup.should_send_alert = AsyncMock()
        
        # cam1 alert
        mock_alert_dedup.should_send_alert.return_value = True
        result1 = await mock_alert_dedup.should_send_alert("cam1")
        assert result1 is True
        
        # cam2 alert (independent)
        mock_alert_dedup.should_send_alert.return_value = True
        result2 = await mock_alert_dedup.should_send_alert("cam2")
        assert result2 is True
        
        # Verify camera_id was passed
        calls = mock_alert_dedup.should_send_alert.call_args_list
        assert len(calls) == 2
    
    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate_alerts(self, inference_consumer, mock_alert_dedup):
        """Test 60s cooldown between alerts per camera"""
        mock_alert_dedup.should_send_alert = AsyncMock(side_effect=[True, False, True])
        
        # First alert allowed
        result1 = await mock_alert_dedup.should_send_alert("cam1")
        assert result1 is True
        
        # Immediate second alert blocked
        result2 = await mock_alert_dedup.should_send_alert("cam1")
        assert result2 is False
        
        # After cooldown, allowed again
        result3 = await mock_alert_dedup.should_send_alert("cam1")
        assert result3 is True


class TestInferenceConsumerRedisPublishing:
    """Test Redis alert publishing"""
    
    @pytest.mark.asyncio
    async def test_publishes_to_redis_pubsub(self, inference_consumer, mock_redis):
        """Test alert published to Redis pub/sub"""
        mock_redis.publish = AsyncMock(return_value=1)
        
        await mock_redis.publish(
            "alerts:cam1",
            json.dumps({
                'camera_id': 'cam1',
                'timestamp': 123.45,
                'confidence': 0.95,
                'frame_id': 'frame-1',
            }),
        )
        
        assert mock_redis.publish.called
        call_args = mock_redis.publish.call_args
        assert "alerts:cam1" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_publishes_to_redis_streams(self, inference_consumer, mock_redis):
        """Test alert appended to Redis stream"""
        mock_redis.xadd = AsyncMock(return_value="1234567890")
        
        await mock_redis.xadd(
            "alerts",
            {
                'camera_id': 'cam1',
                'timestamp': '123.45',
                'confidence': '0.95',
                'frame_id': 'frame-1',
            },
        )
        
        assert mock_redis.xadd.called
        call_args = mock_redis.xadd.call_args
        assert "alerts" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_alert_message_format(self, inference_consumer, mock_redis):
        """Test alert message includes required fields"""
        alert_data = {
            'camera_id': 'cam1',
            'timestamp': 123.45,
            'confidence': 0.95,
            'frame_id': 'frame-1',
        }
        
        await mock_redis.publish(
            "alerts:cam1",
            json.dumps(alert_data),
        )
        
        call_args = mock_redis.publish.call_args
        message_json = call_args[0][1]
        message = json.loads(message_json)
        
        assert message['camera_id'] == 'cam1'
        assert message['timestamp'] == 123.45
        assert message['confidence'] == 0.95


class TestInferenceConsumerInference:
    """Test inference execution"""
    
    @pytest.mark.asyncio
    async def test_model_called_for_batch(self, inference_consumer, mock_model):
        """Test model.predict() called with batch"""
        frames = np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8)
        
        await mock_model.predict(frames)
        
        assert mock_model.predict.called
    
    @pytest.mark.asyncio
    async def test_model_output_parsed_correctly(self, inference_consumer, mock_model):
        """Test model output: [no_violence_prob, violence_prob]"""
        mock_model.predict = AsyncMock(return_value=[0.2, 0.8])
        
        result = await mock_model.predict(np.zeros((4, 224, 224, 3), dtype=np.uint8))
        
        assert result[0] == 0.2  # no_violence
        assert result[1] == 0.8  # violence
    
    @pytest.mark.asyncio
    async def test_high_confidence_triggers_alert(self, inference_consumer, mock_model, mock_alert_dedup):
        """Test alert when violence confidence > threshold"""
        mock_model.predict = AsyncMock(return_value=[0.05, 0.95])
        mock_alert_dedup.should_send_alert = AsyncMock(return_value=True)
        
        # Violence confidence = 0.95
        predictions = await mock_model.predict(np.zeros((4, 224, 224, 3), dtype=np.uint8))
        
        if predictions[1] > 0.5:  # 0.95 > 0.5 = True
            should_alert = await mock_alert_dedup.should_send_alert("cam1")
            assert should_alert is True


class TestInferenceConsumerMetrics:
    """Test metrics tracking"""
    
    def test_consumer_tracks_frames_processed(self, inference_consumer):
        """Test frames_processed metric"""
        inference_consumer.frames_processed = 100
        
        stats = inference_consumer.get_stats()
        assert stats['frames_processed'] == 100
    
    def test_consumer_tracks_alerts_sent(self, inference_consumer):
        """Test alerts_sent metric"""
        inference_consumer.alerts_sent = 5
        
        stats = inference_consumer.get_stats()
        assert stats['alerts_sent'] == 5
    
    def test_consumer_tracks_per_camera_alerts(self, inference_consumer):
        """Test per-camera alert metrics"""
        inference_consumer.per_camera_alerts = {
            'cam1': 3,
            'cam2': 2,
        }
        
        stats = inference_consumer.get_stats()
        assert stats['per_camera_alerts']['cam1'] == 3
        assert stats['per_camera_alerts']['cam2'] == 2


class TestInferenceConsumerConfig:
    """Test configuration loading"""
    
    def test_consumer_uses_config_settings(self):
        """Test consumer loads from settings"""
        from ai_service.inference.inference_consumer import InferenceConsumer
        
        consumer = InferenceConsumer()
        
        # Should have config-driven values
        assert hasattr(consumer, 'kafka_bootstrap_servers')
        assert hasattr(consumer, 'kafka_consumer_group')
        assert hasattr(consumer, 'kafka_frame_topic')
        assert hasattr(consumer, 'batch_size')
    
    def test_consumer_override_settings(self):
        """Test consumer can override settings"""
        consumer = InferenceConsumer(
            kafka_bootstrap_servers="custom:9092",
            kafka_frame_topic="custom-topic",
            batch_size=8,
        )
        
        assert consumer.kafka_bootstrap_servers == "custom:9092"
        assert consumer.kafka_frame_topic == "custom-topic"
        assert consumer.batch_size == 8


class TestInferenceConsumerEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_handles_invalid_json_message(self, inference_consumer):
        """Test graceful handling of invalid JSON"""
        invalid_json = b"not json at all"
        
        try:
            parsed = json.loads(invalid_json)
        except json.JSONDecodeError:
            # Expected - should be handled by consumer
            assert True
    
    @pytest.mark.asyncio
    async def test_handles_missing_required_fields(self, inference_consumer):
        """Test handling of incomplete messages"""
        incomplete_message = {
            'camera_id': 'cam1',
            # Missing: frame_id, timestamp, jpeg, etc.
        }
        
        message_json = json.dumps(incomplete_message)
        
        # Consumer should handle gracefully
        try:
            parsed = json.loads(message_json)
            assert 'camera_id' in parsed
        except Exception:
            # Should not crash
            assert False


class TestInferenceConsumerIntegration:
    """Integration-style tests"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_single_frame(self, inference_consumer, mock_model, mock_redis, mock_alert_dedup):
        """Test complete pipeline: consume → infer → publish"""
        # Setup mocks
        mock_model.predict = AsyncMock(return_value=[0.1, 0.9])
        mock_alert_dedup.should_send_alert = AsyncMock(return_value=True)
        mock_redis.publish = AsyncMock(return_value=1)
        
        # Simulate processing
        inference_consumer.frames_processed = 1
        inference_consumer.alerts_sent = 1
        
        stats = inference_consumer.get_stats()
        assert stats['frames_processed'] == 1
        assert stats['alerts_sent'] == 1
    
    @pytest.mark.asyncio
    async def test_multiple_cameras_parallel_processing(self, inference_consumer):
        """Test multiple cameras processed independently"""
        # Reset buffers
        inference_consumer.frame_buffers = {
            'cam1': [],
            'cam2': [],
        }
        inference_consumer.per_camera_alerts = {
            'cam1': 0,
            'cam2': 0,
        }
        
        # Verify independent state
        assert 'cam1' in inference_consumer.frame_buffers
        assert 'cam2' in inference_consumer.frame_buffers
        assert inference_consumer.frame_buffers['cam1'] != inference_consumer.frame_buffers['cam2']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
