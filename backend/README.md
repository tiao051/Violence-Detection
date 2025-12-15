  # Violence Detection System - Backend Service

FastAPI-based backend service with Clean Architecture for real-time violence detection across multiple camera feeds.

## Architecture

### System Overview
```
RTSP Cameras (RTSP Server)
    ↓
Backend Service (FastAPI)
    ├─ Reads frames from RTSP
    ├─ Resizes & compresses frames
    └─ Publishes to Kafka
    
Kafka (Message Broker)
    ├─ Topic: "frames" (raw video frames)
    └─ Topic: "detections" (inference results)
    
Inference Service (AI Model)
    ├─ Consumes frames from Kafka
    ├─ Runs violence detection model
    └─ Publishes results to Redis
    
Redis (Cache & Results Storage)
    └─ Stores detection alerts & metadata
    
Frontend Dashboard (React/TypeScript)
    └─ Displays real-time alerts
```

### Code Structure
```
src/
├── core/                    # Configuration & logging
├── domain/                  # Business entities
├── application/             # Use cases & business logic
├── infrastructure/          # External interfaces
│   ├── rtsp/               # RTSP camera streaming
│   ├── kafka/              # Kafka producer (frame publishing)
│   └── redis/              # Redis client (alerts & metadata)
└── presentation/           # API endpoints & HTTP handlers
```

## Quick Start

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View backend logs
docker-compose logs -f backend

# Stop all services
docker-compose down
```

## Test Data

Test videos available in: `../ai_service/utils/test_inputs/`

Test datasets:
- `violence_*.mp4` - Violence scene samples
- `non_violence_*.mp4` - Normal activity samples

These are streamed via FFmpeg containers as simulated RTSP cameras.

## API Endpoints

Health check:
```bash
curl http://localhost:8000/health
```

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| Backend | 8000 | FastAPI application - RTSP reader & Kafka publisher |
| Inference | - | AI model service - Kafka consumer, processes frames |
| Kafka | 9092 | Message broker - distributes frames to inference |
| Zookeeper | 2181 | Kafka coordination |
| Redis | 6379 | Results cache & detection alerts |
| RTSP Server | 8554 | MediaMTX - receives & serves RTSP streams |
| Frontend | 3000 | React dashboard |
| Camera 1-4 | - | FFmpeg containers - simulate IP cameras |

## Data Flow

### Frame Processing Pipeline

```
1. RTSP Cameras (MediaMTX)
   ↓
2. Backend CameraWorker
   - Reads frame from RTSP stream
   - Resizes to 640x480
   - Encodes as JPEG (quality 80)
   - Packs as msgpack
   ↓
3. Kafka Topic: "frames"
   - Partitioned by camera_id
   - Retention: 24 hours
   ↓
4. Inference Consumer
   - Consumes frames from Kafka
   - Batch processing (default 4 frames)
   - Runs PyTorch model inference
   ↓
5. Detection Results
   - Publishes to Redis
   - Stores alerts in cache
   - Sends to frontend via WebSocket
```

**Key Environment Variables:**
```
RTSP_BASE_URL=rtsp://rtsp-server:8554
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
REDIS_URL=redis://redis:6379
```
