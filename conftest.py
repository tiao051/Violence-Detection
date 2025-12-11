"""
Shared pytest fixtures and configuration

Central location for all test mocks and config. Using conftest.py allows:
  1. Reuse across multiple test files (DRY principle)
  2. Load config from .env once instead of in each test
  3. Easy to swap mocks or implementations globally
  4. Pytest auto-discovers and makes fixtures available to all tests

Without conftest.py, you'd duplicate fixture definitions in every test file.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from dotenv import load_dotenv

# Load .env file
ENV_FILE = Path(__file__).resolve().parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))


@pytest.fixture
async def mock_kafka_producer():
    """Mock Kafka producer for testing"""
    mock = AsyncMock()
    mock.send_and_wait = AsyncMock(return_value=None)
    mock.start = AsyncMock(return_value=None)
    mock.stop = AsyncMock(return_value=None)
    return mock


@pytest.fixture
async def mock_kafka_consumer():
    """Mock Kafka consumer for testing"""
    mock = AsyncMock()
    mock.subscribe = AsyncMock(return_value=None)
    mock.getmany = AsyncMock(return_value={})
    mock.start = AsyncMock(return_value=None)
    mock.stop = AsyncMock(return_value=None)
    return mock


@pytest.fixture
async def mock_redis_client():
    """Mock Redis client for testing"""
    mock = AsyncMock()
    mock.exists = AsyncMock(return_value=False)
    mock.setex = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=None)
    mock.publish = AsyncMock(return_value=1)
    mock.xadd = AsyncMock(return_value="1234567890")
    mock.close = AsyncMock(return_value=None)
    return mock


@pytest.fixture
async def mock_violence_model():
    """Mock violence detection model"""
    mock = AsyncMock()
    mock.predict = AsyncMock(return_value=[0.2, 0.8])
    mock.load = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_rtsp_client():
    """Mock RTSP client"""
    mock = MagicMock()
    mock.connect = MagicMock(return_value=True)
    mock.read = MagicMock(return_value=(True, None))
    mock.disconnect = MagicMock(return_value=None)
    return mock


@pytest.fixture
async def mock_firestore():
    """Mock Firestore client"""
    mock = AsyncMock()
    mock.collection = MagicMock()
    mock.close = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def test_config():
    """Test configuration from .env file"""
    return {
        'kafka_bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'kafka_frame_topic': os.getenv('KAFKA_FRAME_TOPIC', 'processed-frames'),
        'kafka_consumer_group': os.getenv('KAFKA_CONSUMER_GROUP', 'inference'),
        'kafka_compression_type': os.getenv('KAFKA_COMPRESSION_TYPE', 'gzip'),
        'kafka_jpeg_quality': int(os.getenv('KAFKA_JPEG_QUALITY', 80)),
        'inference_batch_size': int(os.getenv('INFERENCE_BATCH_SIZE', 4)),
        'inference_batch_timeout_ms': int(os.getenv('INFERENCE_BATCH_TIMEOUT_MS', 5000)),
        'alert_cooldown_seconds': int(os.getenv('ALERT_COOLDOWN_SECONDS', 60)),
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'redis_db': int(os.getenv('REDIS_DB', 0)),
    }


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "asyncio: async tests")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "kafka: Kafka-related tests")
    config.addinivalue_line("markers", "redis: Redis-related tests")
    config.addinivalue_line("markers", "rtsp: RTSP-related tests")
    config.addinivalue_line("markers", "inference: inference-related tests")


@pytest.fixture(autouse=True)
def reset_test_state():
    """Reset test state before each test"""
    yield
