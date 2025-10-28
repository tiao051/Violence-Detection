# Backend Service

Violence Detection System - Backend API with Clean Architecture

## Architecture

```
src/
├── core/                    # Configuration & logging
├── domain/                  # Business entities
├── application/             # Use cases
├── infrastructure/          # External interfaces
│   ├── rtsp/               # RTSP streaming
│   └── redis/              # Message queue
└── presentation/           # API endpoints
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start all services (PostgreSQL + Redis + Backend + RTSP Simulator)
docker-compose up -d

# View logs
docker-compose logs -f backend

# View specific service logs
docker-compose logs -f camera-1

# Stop all services
docker-compose down

# Stop and remove volumes (clean restart)
docker-compose down -v
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL, Redis, and RTSP services (via Docker)
docker-compose up -d postgres redis rtsp-server camera-1 camera-2 camera-3 camera-4

# Run backend locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Test Data

Test videos are located in: `./tests/rtsp-simulator/test-videos/`

- `violence_1.mp4` - Violence scene 1
- `violence_2.mp4` - Violence scene 2
- `violence_3.mp4` - Violence scene 3
- `non_violence_1.mp4` - Normal activity

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/

curl http://localhost:8000/health
```

### Statistics
```bash
curl http://localhost:8000/stats
```

Response includes:
- Worker statistics (frames, FPS, connection status)
- Redis stream statistics
- Uptime information

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| Backend | 8000 | FastAPI application |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Message queue |
| RTSP Server | 8554 | MediaMTX RTSP server |
| Cameras 1-4 | - | FFmpeg RTSP streams |

## RTSP Streaming Details

- **Input**: 4 FFmpeg cameras streaming MP4 files via H.264 codec
- **Processing**: Stream copy (no re-encoding) for optimal performance
- **Sampling**: Time-based 6 FPS output
- **Encoding**: JPEG with quality 80
- **Storage**: Redis Streams with base64-encoded frames

## Testing

### Test RTSP Streams

```bash
# Test camera streams with ffplay (requires ffmpeg)
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam1
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam2
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam3
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam4
```

### Test Redis Frames

```bash
# Connect to Redis container
docker exec -it violence-detection-redis redis-cli

# Inside redis-cli:
> XLEN frames:cam1                    # Total frames in stream
> XRANGE frames:cam1 - + COUNT 1      # Get first frame
> XLEN frames:cam2
> XLEN frames:cam3
> XLEN frames:cam4
> exit
```
