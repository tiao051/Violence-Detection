"""
Redis Alert Deduplication Unit Tests

Comprehensive test suite for AlertDeduplication class.

Tests:
    - TTL-based alert rate limiting (60-second cooldown per camera)
    - Per-camera independence (cameras don't share cooldown state)
    - Redis operations (EXISTS, SETEX, DELETE)
    - Reset functionality (clear_alert() resets cooldown)
    - Full alert lifecycle (send → cooldown → clear → send again)
    - Edge cases (empty camera_id, special characters)

Usage:
    pytest backend/tests/infrastructure/redis/test_alert_deduplication.py -v
    pytest backend/tests/infrastructure/redis/test_alert_deduplication.py::TestAlertDeduplicationTTL -v
    pytest backend/tests/infrastructure/redis/test_alert_deduplication.py::TestAlertDeduplicationTTL::test_cooldown_duration_is_60_seconds -v
    pytest backend/tests/infrastructure/redis/test_alert_deduplication.py -v --cov=backend/src/infrastructure/redis
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.src.infrastructure.redis.alert_deduplication import AlertDeduplication


@pytest.fixture
def redis_mock():
    """Mock Redis client"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def alert_dedup(redis_mock):
    """Create AlertDeduplication with mocked Redis"""
    dedup = AlertDeduplication(redis_client=redis_mock, ttl_seconds=60)
    return dedup


class TestAlertDeduplicationTTL:
    """Test TTL-based alert rate limiting"""
    
    @pytest.mark.asyncio
    async def test_should_send_alert_returns_true_first_time(self, alert_dedup, redis_mock):
        """Test first alert for camera is allowed"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        result = await alert_dedup.should_send_alert("cam1")
        
        assert result is True
        assert redis_mock.setex.called
    
    @pytest.mark.asyncio
    async def test_should_send_alert_returns_false_within_cooldown(self, alert_dedup, redis_mock):
        """Test duplicate alert within 60s is blocked"""
        redis_mock.exists = AsyncMock(return_value=True)
        
        result = await alert_dedup.should_send_alert("cam1")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cooldown_duration_is_60_seconds(self, alert_dedup, redis_mock):
        """Test TTL is set to 60 seconds"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        await alert_dedup.should_send_alert("cam1")
        
        call_args = redis_mock.setex.call_args
        # setex(key, timeout, value)
        timeout = call_args[0][1]
        assert timeout == 60
    
    @pytest.mark.asyncio
    async def test_redis_key_format_includes_camera_id(self, alert_dedup, redis_mock):
        """Test Redis key includes camera_id"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        await alert_dedup.should_send_alert("cam1")
        
        call_args = redis_mock.setex.call_args
        key = call_args[0][0]
        
        assert "cam1" in key


class TestAlertDeduplicationPerCamera:
    """Test per-camera independence"""
    
    @pytest.mark.asyncio
    async def test_different_cameras_independent(self, alert_dedup, redis_mock):
        """Test cameras don't share cooldown"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        # First alert from cam1
        result1 = await alert_dedup.should_send_alert("cam1")
        assert result1 is True
        
        # First alert from cam2 should also pass
        result2 = await alert_dedup.should_send_alert("cam2")
        assert result2 is True
        
        # Both should have set Redis keys
        assert redis_mock.setex.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cam1_cooldown_doesnt_affect_cam2(self, alert_dedup, redis_mock):
        """Test cam2 alerts work while cam1 is on cooldown"""
        redis_mock.exists = AsyncMock(side_effect=[
            False,  # cam1 first alert
            True,   # cam1 second alert (blocked)
            False,  # cam2 first alert (allowed)
        ])
        redis_mock.setex = AsyncMock(return_value=None)
        
        # cam1 first alert
        result1a = await alert_dedup.should_send_alert("cam1")
        assert result1a is True
        
        # cam1 second alert (should be blocked)
        result1b = await alert_dedup.should_send_alert("cam1")
        assert result1b is False
        
        # cam2 first alert (should be allowed)
        result2a = await alert_dedup.should_send_alert("cam2")
        assert result2a is True


class TestAlertDeduplicationReset:
    """Test alert reset functionality"""
    
    @pytest.mark.asyncio
    async def test_clear_alert_resets_cooldown(self, alert_dedup, redis_mock):
        """Test clear_alert() resets the cooldown"""
        redis_mock.delete = AsyncMock(return_value=None)
        
        await alert_dedup.clear_alert("cam1")
        
        assert redis_mock.delete.called
        call_args = redis_mock.delete.call_args
        key = call_args[0][0]
        assert "cam1" in key
    
    @pytest.mark.asyncio
    async def test_after_clear_alert_can_send_again(self, alert_dedup, redis_mock):
        """Test alert can be sent after clear_alert()"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        redis_mock.delete = AsyncMock(return_value=None)
        
        # First alert
        result1 = await alert_dedup.should_send_alert("cam1")
        assert result1 is True
        
        # Clear cooldown
        await alert_dedup.clear_alert("cam1")
        
        # After clearing, should be able to send again
        redis_mock.exists = AsyncMock(return_value=False)
        result2 = await alert_dedup.should_send_alert("cam1")
        assert result2 is True


class TestAlertDeduplicationRedisOps:
    """Test Redis operations"""
    
    @pytest.mark.asyncio
    async def test_uses_redis_exists_check(self, alert_dedup, redis_mock):
        """Test uses Redis EXIST command to check"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        await alert_dedup.should_send_alert("cam1")
        
        assert redis_mock.exists.called
    
    @pytest.mark.asyncio
    async def test_uses_redis_setex_with_ttl(self, alert_dedup, redis_mock):
        """Test uses Redis SETEX for TTL"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        await alert_dedup.should_send_alert("cam1")
        
        assert redis_mock.setex.called
        call_args = redis_mock.setex.call_args
        # Should be (key, timeout, value)
        assert len(call_args[0]) == 3 or 'timeout' in str(call_args)
    
    @pytest.mark.asyncio
    async def test_uses_redis_delete_for_clear(self, alert_dedup, redis_mock):
        """Test uses Redis DELETE for clear_alert"""
        redis_mock.delete = AsyncMock(return_value=None)
        
        await alert_dedup.clear_alert("cam1")
        
        assert redis_mock.delete.called


class TestAlertDeduplicationEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_camera_id(self, alert_dedup, redis_mock):
        """Test handling of empty camera_id"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        # Should not crash, but key might be malformed
        result = await alert_dedup.should_send_alert("")
        
        # Still should process without exception
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_special_chars_in_camera_id(self, alert_dedup, redis_mock):
        """Test camera_id with special characters"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        
        result = await alert_dedup.should_send_alert("cam-1:rtsp://192.168.1.1")
        
        assert isinstance(result, bool)
        assert redis_mock.setex.called


class TestAlertDeduplicationIntegration:
    """Integration-style tests"""
    
    @pytest.mark.asyncio
    async def test_full_alert_lifecycle(self, alert_dedup, redis_mock):
        """Test complete alert lifecycle: send → cooldown → clear → send again"""
        redis_mock.exists = AsyncMock(return_value=False)
        redis_mock.setex = AsyncMock(return_value=None)
        redis_mock.delete = AsyncMock(return_value=None)
        
        # 1. Send alert
        result1 = await alert_dedup.should_send_alert("cam1")
        assert result1 is True
        
        # 2. Block duplicate within cooldown
        redis_mock.exists = AsyncMock(return_value=True)
        result2 = await alert_dedup.should_send_alert("cam1")
        assert result2 is False
        
        # 3. Clear cooldown
        redis_mock.exists = AsyncMock(return_value=False)
        await alert_dedup.clear_alert("cam1")
        
        # 4. Send alert again
        result3 = await alert_dedup.should_send_alert("cam1")
        assert result3 is True
    
    @pytest.mark.asyncio
    async def test_multiple_cameras_different_states(self, alert_dedup, redis_mock):
        """Test multiple cameras with different alert states"""
        redis_mock.exists = AsyncMock()
        redis_mock.setex = AsyncMock(return_value=None)
        redis_mock.delete = AsyncMock(return_value=None)
        
        # cam1: can send
        redis_mock.exists.return_value = False
        result1 = await alert_dedup.should_send_alert("cam1")
        assert result1 is True
        
        # cam2: can send
        redis_mock.exists.return_value = False
        result2 = await alert_dedup.should_send_alert("cam2")
        assert result2 is True
        
        # cam1: on cooldown
        redis_mock.exists.return_value = True
        result3 = await alert_dedup.should_send_alert("cam1")
        assert result3 is False
        
        # cam2: can still send
        redis_mock.exists.return_value = False
        result4 = await alert_dedup.should_send_alert("cam2")
        assert result4 is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
