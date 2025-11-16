"""
FastAPI Application Entry Point

Clean Architecture setup with dependency injection
"""
import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logger import setup_logging
from src.infrastructure.rtsp import CameraWorker
from src.infrastructure.redis.streams import RedisStreamProducer
from src.infrastructure.inference import get_inference_service

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global state
redis_client: redis.Redis = None
redis_producer: RedisStreamProducer = None
camera_workers: list = []
startup_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""

    # Startup
    logger.info("Starting up application...")
    await startup()
    yield

    # Shutdown
    logger.info("Shutting down application...")
    await shutdown()


async def startup() -> None:
    """Initialize application on startup."""
    global redis_client, redis_producer, camera_workers, startup_time

    startup_time = time.time()

    try:
        # Load violence detection model
        logger.info("Loading violence detection model...")
        inference_service = get_inference_service()
        model_path = os.getenv('MODEL_PATH')
        inference_device = os.getenv('INFERENCE_DEVICE', 'cuda')
        confidence_threshold = float(os.getenv('VIOLENCE_CONFIDENCE_THRESHOLD', 0.5))
        
        if not model_path:
            logger.warning("MODEL_PATH not set, inference disabled")
        else:
            try:
                inference_service.initialize(model_path, inference_device, confidence_threshold)
                logger.info(f"✅ Violence detection model loaded successfully (device: {inference_device})")
            except Exception as e:
                logger.error(f"Failed to load violence detection model: {e}", exc_info=True)
                # Continue anyway, inference will fail gracefully
        
        # Connect to Redis
        logger.info("Connecting to Redis...")
        redis_client = await redis.from_url(settings.redis_url)
        await redis_client.ping()
        logger.info("Redis connected")

        # Create Redis producer
        redis_producer = RedisStreamProducer(redis_client)

        # Start RTSP camera workers if enabled
        if settings.rtsp_enabled:
            logger.info(f"Starting {len(settings.rtsp_cameras)} RTSP camera workers...")

            # Create all workers
            workers_to_start = []
            for camera_id in settings.rtsp_cameras:
                rtsp_url = f"{settings.rtsp_base_url}/{camera_id}"

                worker = CameraWorker(
                    camera_id=camera_id,
                    stream_url=rtsp_url,
                    redis_producer=redis_producer,
                    sample_rate=settings.rtsp_sample_rate,
                    frame_width=settings.rtsp_frame_width,
                    frame_height=settings.rtsp_frame_height,
                    jpeg_quality=settings.rtsp_jpeg_quality,
                )

                workers_to_start.append(worker)
                camera_workers.append(worker)
                
            # Start all workers in parallel
            start_tasks = [worker.start() for worker in workers_to_start]
            await asyncio.gather(*start_tasks)
            
            logger.info(f"All {len(camera_workers)} RTSP camera workers started")
        else:
            logger.warning("RTSP is disabled")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise


async def shutdown() -> None:
    """Clean up on shutdown."""
    global redis_client, camera_workers

    try:
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# TODO: Register API routers
# from src.presentation.api import camera_router, stream_router
# from src.presentation.routes import threat_router
# app.include_router(camera_router, prefix="/api/v1")
# app.include_router(stream_router, prefix="/api/v1")
# app.include_router(threat_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )