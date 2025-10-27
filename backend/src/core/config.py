"""Application configuration."""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # App
    app_name: str = os.getenv("APP_NAME", "Violence Detection Backend")
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # API
    api_v1_prefix: str = os.getenv("API_V1_PREFIX", "/api/v1")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/violence_detection"
    )
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
    
    # PostgreSQL (from docker-compose)
    postgres_db: str = os.getenv("POSTGRES_DB", "violence_detection")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "10"))
    
    # RTSP
    rtsp_enabled: bool = os.getenv("RTSP_ENABLED", "True").lower() == "true"
    rtsp_base_url: str = os.getenv("RTSP_BASE_URL", "rtsp://rtsp-server:8554")
    rtsp_cameras: List[str] = ["cam1", "cam2", "cam3", "cam4"]  # Camera IDs
    rtsp_sample_rate: int = int(os.getenv("RTSP_SAMPLE_RATE", "6"))  # FPS
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
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
settings = Settings()
