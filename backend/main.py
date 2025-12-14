"""
FastAPI Application Entry Point

Clean Architecture setup with dependency injection
"""
import logging
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    await startup(app)
    
    yield

    # Shutdown
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
        redis_client = await redis.from_url(settings.redis_url)
        await redis_client.ping()
        app.state.redis_client = redis_client

        # 3. Connect Kafka producer
        kafka_producer = get_kafka_producer()
        await kafka_producer.connect()

        # 4. Start Event Processor (Background Worker)
        event_processor = get_event_processor(redis_client)
        await event_processor.start()

        # 5. Start Camera Workers
        if settings.rtsp_enabled:
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

        # Print startup complete message (always shown)
        elapsed = time.time() - startup_time
        
        if len(settings.rtsp_cameras) == 1 and settings.rtsp_sample_rate <= 2:
             msg = f"DEV MODE ACTIVE: Reduced load (1 Camera, {settings.rtsp_sample_rate} FPS)"
             print(f"\n{msg}")
             logger.info(msg)
             
        msg = f"Backend started in {elapsed:.1f}s | Cameras: {len(camera_workers)}"
        print(f"\n{msg}\n")
        logger.info(msg)

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
            for worker in camera_workers:
                await worker.stop()

        # Close Redis connection
        if redis_client:
            await redis_client.close()

        print("Backend stopped")

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

# Mount static files for video playback
import os
outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(outputs_dir, exist_ok=True)
app.mount("/videos", StaticFiles(directory=outputs_dir), name="videos")


@app.get("/api/events/{event_id}")
async def get_event(event_id: str):
    """Get event details including video URL if available."""
    global redis_client
    
    try:
        if not redis_client:
            return {"error": "Redis unavailable"}, 503
        
        # Try to get event data from Redis (stored during save_event)
        event_data = await redis_client.get(f"event:{event_id}")
        if event_data:
            return json.loads(event_data)
        
        # Fallback: Return minimal response
        return {"id": event_id, "video_url": None}
    except Exception as e:
        logger.error(f"Error fetching event {event_id}: {e}")
        return {"error": str(e)}, 500


@app.get("/api/events/lookup")
async def lookup_event(camera_id: str, timestamp: int):
    """
    Lookup event by camera_id and timestamp (ms).
    Useful when frontend ID differs from backend ID.
    """
    global redis_client
    
    try:
        if not redis_client:
            return {"error": "Redis unavailable"}, 503
            
        # Convert ms to seconds
        ts_sec = timestamp / 1000.0
        
        # Search window: +/- 60 seconds
        min_ts = ts_sec - 60
        max_ts = ts_sec + 60
        
        timeline_key = f"events:timeline:{camera_id}"
        # zrangebyscore returns list of members (bytes)
        results = await redis_client.zrangebyscore(timeline_key, min_ts, max_ts)
        
        if results:
            # Return the most recent match in the window
            # results are sorted by score (timestamp), so last one is newest
            data = json.loads(results[-1])
            return data
            
        return {"video_url": None}
    except Exception as e:
        logger.error(f"Error looking up event: {e}")
        return {"error": str(e)}, 500


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