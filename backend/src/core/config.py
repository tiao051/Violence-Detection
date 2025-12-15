"""Application configuration."""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # App
    app_name: str = os.getenv("APP_NAME", "Violence Detection Backend")
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_to_file: bool = os.getenv("LOG_TO_FILE", "False").lower() == "true"
    
    # API
    api_v1_prefix: str = os.getenv("API_V1_PREFIX", "/api/v1")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Redis 
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_url: str = os.getenv("REDIS_URL", f"redis://{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}")
    redis_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "10"))
    
    # Kafka
    kafka_enabled: bool = os.getenv("KAFKA_ENABLED", "False").lower() == "true"
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    kafka_frame_topic: str = os.getenv("KAFKA_FRAME_TOPIC", "frames")
    kafka_jpeg_quality: int = int(os.getenv("KAFKA_JPEG_QUALITY", "80"))
    kafka_consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "inference-group")
    kafka_compression_type: str = os.getenv("KAFKA_COMPRESSION_TYPE", "gzip")
    
    # Inference
    inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "4"))
    inference_batch_timeout_ms: int = int(os.getenv("INFERENCE_BATCH_TIMEOUT_MS", "100"))
    alert_cooldown_seconds: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))
    
    # RTSP
    rtsp_enabled: bool = os.getenv("RTSP_ENABLED", "False").lower() == "true"
    rtsp_base_url: str = os.getenv("RTSP_BASE_URL", "rtsp://rtsp-server:8554")
    rtsp_cameras: List[str] = Field(default=["cam1", "cam2", "cam3", "cam4"])
    rtsp_sample_rate: int = int(os.getenv("RTSP_SAMPLE_RATE", "6"))
    rtsp_frame_width: int = int(os.getenv("RTSP_FRAME_WIDTH", "640"))
    rtsp_frame_height: int = int(os.getenv("RTSP_FRAME_HEIGHT", "480"))
    rtsp_jpeg_quality: int = int(os.getenv("RTSP_JPEG_QUALITY", "80"))
    rtsp_max_reconnect_attempts: int = int(os.getenv("RTSP_MAX_RECONNECT_ATTEMPTS", "5"))
    rtsp_reconnect_delay: int = int(os.getenv("RTSP_RECONNECT_DELAY", "2"))
    
    # Streaming
    max_concurrent_streams: int = int(os.getenv("MAX_CONCURRENT_STREAMS", "10"))
    frame_buffer_size: int = int(os.getenv("FRAME_BUFFER_SIZE", "100"))
    stream_reconnect_attempts: int = int(os.getenv("STREAM_RECONNECT_ATTEMPTS", "3"))
    stream_read_timeout: int = int(os.getenv("STREAM_READ_TIMEOUT", "30"))
    
    # Firebase
    firebase_storage_bucket: str = os.getenv("FIREBASE_STORAGE_BUCKET", "")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
settings = Settings()
