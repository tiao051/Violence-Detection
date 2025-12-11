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

### Using Docker Compose with USB Camera (Recommended)

**Option 1: With USB Camera**
```bash
# Terminal 1: Start USB Camera Manager (detects & streams USB camera)
python scripts/usb_camera_manager.py

# Terminal 2 (after 2-3 seconds): Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# View specific service logs
docker-compose logs -f camera-1
```

**Option 2: Without USB Camera (Simulated cameras only)**
```bash
# Just start docker-compose (no script needed)
docker-compose up -d

# Backend will auto-skip usb-cam after 5 connection attempts
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean restart)
docker-compose down -v

# Stop USB Camera Manager (in its terminal)
# Press CTRL+C (gracefully kills FFmpeg)
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

Test videos are located in: `../utils/test_inputs/`

The following test videos are available:
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
| Cameras 1-4 | - | FFmpeg RTSP streams (simulated) |
| USB Camera | - | FFmpeg USB device capture (optional) |

## USB Camera Setup

### Prerequisites
- USB webcam connected to host
- FFmpeg installed on host: `ffmpeg -version`

### Auto-Detection
- USB camera must be named `"Web Camera"` in FFmpeg
- Script auto-detects on startup
- If not found: backend skips usb-cam gracefully

### How It Works
```
Host (Windows):
  python scripts/usb_camera_manager.py
    └─ FFmpeg captures USB device
       └─ Streams RTSP to localhost:8554/usb-cam

Docker (Container):
  rtsp-server (MediaMTX)
    └─ Receives RTSP stream from host
       └─ Backend reads as virtual IP camera

Backend Flow:
  rtsp://rtsp-server:8554/usb-cam  (like any IP camera)
    ├─ CameraWorker reads frames
    ├─ Samples 6 FPS
    ├─ Resizes to 640x480
    └─ Pushes to Redis Streams
```

### Troubleshooting USB Camera

**Check if camera is detected:**
```bash
ffmpeg -list_devices true -f dshow -i dummy
```
Look for `"Web Camera" (video)` in output.

**Check RTSP stream:**
```bash
ffplay rtsp://localhost:8554/usb-cam
```

**If camera not detected:**
- Ensure USB camera is plugged in
- Close any other app using the camera
- Rename camera to "Web Camera" in Device Manager (Windows)

### Stats Endpoint
```bash
curl http://localhost:8000/stats
```
Will show:
- `usb-cam` worker status (if available)
- Frame counts & FPS for all cameras
- Redis stream statistics

## RTSP Streaming Details

### Cameras
- **cam1-4**: FFmpeg streams from MP4 files (simulated)
- **usb-cam**: FFmpeg captures USB device (if available, auto-detected)

### Stream Configuration
- **Codec**: H.264
- **Sampling**: Time-based 6 FPS output
- **Resolution**: 640x480
- **JPEG Quality**: 80
- **Storage**: Redis Streams with base64-encoded frames

### Architecture
```
FFmpeg (Host/Container)
  └─ Encode video
     └─ Stream RTSP (libx264, veryfast preset)
        └─ MediaMTX RTSP Server (Container)
           └─ Backend CameraWorker reads frames
              ├─ Sample 6 FPS
              ├─ Store in-memory frame buffer
              └─ Push metadata to Redis Streams
```

## Testing

### Test RTSP Streams

```bash
# Test simulated cameras
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam1
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam2
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam3
ffplay -rtsp_transport tcp rtsp://localhost:8554/cam4

# Test USB camera (if available)
ffplay -rtsp_transport tcp rtsp://localhost:8554/usb-cam
```

### Test Backend Health

```bash
# Health check
curl http://localhost:8000/health

# Get detailed stats
curl http://localhost:8000/stats
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
> XLEN frames:usb-cam                 # USB camera stream
> exit
```
