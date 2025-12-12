"""
FastAPI Application Entry Point

Clean Architecture setup with dependency injection
"""
import logging
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.core.logger import setup_logging
from src.infrastructure.rtsp import CameraWorker
from src.infrastructure.kafka import get_kafka_producer
from src.presentation.routes import auth_router
from src.presentation.routes.websocket_routes import router as websocket_router
from src.infrastructure.firebase.setup import initialize_firebase
from src.application.event_processor import get_event_processor

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global state
redis_client: redis.Redis = None
camera_workers: list = []
startup_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""

    # Startup
    logger.info("Starting up application...")
    await startup(app)
    
    yield

    # Shutdown
    logger.info("Shutting down application...")
    await shutdown()


async def startup(app: FastAPI) -> None:
    """Initialize application on startup."""
    global redis_client, camera_workers, startup_time

    startup_time = time.time()

    try:
        # 1. Initialize Firebase
        initialize_firebase()

        # NOTE: InferenceConsumer now runs as separate process/container
        # Do NOT load model here - avoid GIL blocking FastAPI
        # Backend focuses on: API + WebSocket + CameraWorker
        # InferenceConsumer runs independently in ai_service/inference_consumer_service.py
        
        # 2. Connect to Redis
        logger.info("Connecting to Redis...")
        redis_client = await redis.from_url(settings.redis_url)
        await redis_client.ping()
        app.state.redis_client = redis_client
        logger.info("Redis connected")

        # 3. Connect Kafka producer
        logger.info("Connecting to Kafka...")
        kafka_producer = get_kafka_producer()
        await kafka_producer.connect()
        logger.info("Kafka producer connected")

        # 4. Start Event Processor (Background Worker)
        event_processor = get_event_processor(redis_client)
        await event_processor.start()

        # 5. Start Camera Workers
        if settings.rtsp_enabled:
            logger.info(f"Starting camera workers for: {settings.rtsp_cameras}")
            for cam_id in settings.rtsp_cameras:
                # Construct stream URL
                # If using a simulator, it might be rtsp://localhost:8554/cam1
                stream_url = f"{settings.rtsp_base_url}/{cam_id}"
                
                worker = CameraWorker(
                    camera_id=cam_id,
                    stream_url=stream_url,
                    kafka_producer=None,  # Will use singleton
                    sample_rate=settings.rtsp_sample_rate
                )
                await worker.start()
                camera_workers.append(worker)
        else:
            logger.info("RTSP disabled in settings")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise


async def shutdown() -> None:
    """Clean up on shutdown."""
    global redis_client, camera_workers

    try:
        # Stop Event Processor
        event_processor = get_event_processor()
        if event_processor:
            await event_processor.stop()

        # Stop camera workers
        if camera_workers:
            logger.info(f"Stopping {len(camera_workers)} camera workers...")

            for worker in camera_workers:
                await worker.stop()
                stats = worker.get_stats()
                logger.info(f"Worker stats: {worker.camera_id} - {stats}")

            logger.info("All camera workers stopped")

        # Close Redis connection
        if redis_client:
            logger.info("Closing Redis connection...")
            await redis_client.close()
            logger.info("Redis closed")

    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="Violence Detection Backend",
    description="Backend API for Violence Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Set redis_producer in app state after app creation (for WebSocket routes)
app.state.redis_producer = None  # Will be set in startup

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register authentication routes
app.include_router(auth_router)
# Register WebSocket routes
app.include_router(websocket_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Violence Detection Backend API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    global redis_client

    redis_status = "connected"
    try:
        if redis_client:
            await redis_client.ping()
        else:
            redis_status = "not_initialized"
    except Exception as e:
        redis_status = f"error: {str(e)}"

    return {
        "message": "Violence Detection Backend API",
        "version": "1.0.0",
        "status": "running",
        "redis_status": redis_status
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics about camera workers."""
    global redis_client, camera_workers, startup_time

    workers_data = {}

    # Get worker stats
    for worker in camera_workers:
        stats = worker.get_stats()
        workers_data[worker.camera_id] = {
            "camera_id": worker.camera_id,
            "is_running": stats.get("is_running", False),
            "is_connected": stats.get("is_connected", False),
            "frames_input": stats.get("frames_input", 0),
            "frames_sampled": stats.get("frames_sampled", 0),
            "frames_sent_to_redis": stats.get("frames_sent_to_redis", 0),
            "input_fps": stats.get("input_fps", 0.0),
            "output_fps": stats.get("output_fps", 0.0),
            "elapsed_time_seconds": stats.get("elapsed_time_seconds", 0)
        }

    redis_frames = {}

    # Get Redis stream stats
    if redis_client:
        for camera_id in settings.rtsp_cameras:
            stream_key = f"frames:{camera_id}"
            try:
                stream_len = await redis_client.xlen(stream_key)
                redis_frames[stream_key] = stream_len
            except Exception as e:
                logger.error(f"Failed to get stream length for {stream_key}: {e}")

    return {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - startup_time,
        "total_workers": len(camera_workers),
        "active_workers": sum(1 for w in camera_workers if w.is_running),
        "workers": workers_data,
        "redis_frames": redis_frames
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )