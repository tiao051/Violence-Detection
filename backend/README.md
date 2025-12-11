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

### Docker Compose (All Services)

```bash
# Start all services (Backend, Inference, Kafka, Redis, RTSP, Cameras, Frontend)
docker-compose up -d

# View backend logs
docker-compose logs -f backend

# View inference service logs
docker-compose logs -f inference

# View Kafka stream activity
docker exec violence-detection-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic frames --from-beginning --max-messages 10

# Stop all services
docker-compose down

# Clean restart (remove volumes)
docker-compose down -v
```

### Monitor Kafka Streaming

View real-time frame data being streamed:

```bash
# View recent frames (last 10 messages)
docker exec violence-detection-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic frames --from-beginning --max-messages 10 --timeout-ms 5000

# View detection results
docker exec violence-detection-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic detections --from-beginning --max-messages 10 --timeout-ms 5000

# Monitor with Python script
python backend/tests/infrastructure/kafka/view_kafka.py
```

## Test Data

Test videos available in: `../ai_service/utils/test_inputs/`

Test datasets:
- `violence_*.mp4` - Violence scene samples
- `non_violence_*.mp4` - Normal activity samples

These are streamed via FFmpeg containers as simulated RTSP cameras.

## API Endpoints

### Health & Status
```bash
# Health check
curl http://localhost:8000/health

# System statistics (uptime, worker status, Kafka metrics)
curl http://localhost:8000/stats
```

Response includes:
- Backend service status
- Active camera workers (RTSP connections)
- Frame throughput (FPS)
- Kafka broker health

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

### Key Configurations

**Backend Environment Variables:**
```
RTSP_BASE_URL=rtsp://rtsp-server:8554
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_FRAME_TOPIC=frames
REDIS_URL=redis://redis:6379/0
VIOLENCE_CONFIDENCE_THRESHOLD=0.5
```

**Inference Environment Variables:**
```
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_FRAME_TOPIC=frames
KAFKA_CONSUMER_GROUP=inference
INFERENCE_BATCH_SIZE=4
INFERENCE_BATCH_TIMEOUT_MS=5000
ALERT_COOLDOWN_SECONDS=60
```

## RTSP Streaming Details

### Camera Configuration

**Simulated Cameras (via FFmpeg containers):**
- **cam1-cam4**: Stream from MP4 test videos using FFmpeg
- **Configuration**: 
  - Codec: H.264
  - Bitrate: 2500k
  - Sampling: 6 FPS
  - Resolution: 640x480

**MediaMTX RTSP Server:**
- Acts as RTSP relay
- Receives streams from FFmpeg containers
- Backend connects and reads frames

### Backend Frame Processing

```
RTSP Stream → Backend Worker
    ├─ Reads raw frame
    ├─ Resize to 640x480
    ├─ Encode as JPEG (quality 80)
    ├─ Pack as msgpack binary
    └─ Publish to Kafka
       └─ Key: camera_id (for partitioning)
```

### Testing RTSP Streams

```bash
# Test camera RTSP streams (requires ffplay)
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam1
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam2
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam3
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam4

# Check stream with ffprobe
ffprobe rtsp://localhost:8554/cam1
```

## Testing & Debugging

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# View system statistics
curl http://localhost:8000/stats

# Check Kafka broker
docker exec violence-detection-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

### Monitor Data Flowing Through Kafka

```bash
# View frame topics (binary msgpack data)
docker exec violence-detection-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic frames \
  --max-messages 5

# View detection results topic
docker exec violence-detection-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic detections \
  --max-messages 5

# Check topic offsets
docker exec violence-detection-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --list
```

### View Backend Logs

```bash
# Real-time backend logs
docker-compose logs -f backend

# Inference service logs
docker-compose logs -f inference

# See last 50 lines
docker-compose logs --tail 50 backend
```

### Check Redis Cache

```bash
# Connect to Redis
docker exec -it violence-detection-redis redis-cli

# Inside redis-cli:
> KEYS *                              # List all keys
> GET violations:latest               # Get latest violation
> HGETALL camera:stats                # Camera statistics
> exit
```

## Troubleshooting

### No frames appearing in Kafka

1. Check backend is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Verify RTSP streams are accessible:
   ```bash
   ffprobe rtsp://localhost:8554/cam1
   ```

3. Check backend logs for connection errors:
   ```bash
   docker-compose logs backend | grep -i "error\|exception"
   ```

### Inference not consuming frames

1. Check inference service is running:
   ```bash
   docker-compose logs inference | tail -20
   ```

2. Verify Kafka connectivity:
   ```bash
   docker exec violence-detection-kafka kafka-consumer-groups \
     --bootstrap-server localhost:9092 \
     --group inference \
     --describe
   ```

3. Check for model path issues:
   ```bash
   docker exec violence-detection-inference ls -la /app/training/two-stage/checkpoints/
   ```

### High memory usage

- Adjust `INFERENCE_BATCH_SIZE` (lower = less memory)
- Check frame resolution is being resized properly
- Monitor with: `docker stats`
