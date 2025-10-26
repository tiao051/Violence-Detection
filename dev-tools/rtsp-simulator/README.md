# RTSP Simulator

RTSP server with 4 specialized cameras for comprehensive violence detection testing.

## Camera Setup

| Camera | Purpose | Input Source | RTSP URL |
|--------|---------|--------------|----------|
| **Camera 1** | Violence Detection | Video file (dataset sample) | `rtsp://localhost:8554/camera1` |
| **Camera 2** | False Positive Test | Video file (normal activity) | `rtsp://localhost:8554/camera2` |
| **Camera 3** | Live Stream Test | Webcam (continuous data) | `rtsp://localhost:8554/camera3` |
| **Camera 4** | Custom Dataset Test | Video file (your dataset) | `rtsp://localhost:8554/camera4` |

## Setup Instructions

### Step 1: Prepare Video Files

Place your videos in `test-videos/` directory:
- `violence_sample.mp4` - Violence detection dataset sample
- `normal_activity.mp4` - Normal human activities  
- `custom_dataset.mp4` - Your custom dataset

### Step 2: Start RTSP Server and Cameras

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all
docker-compose down
```

**Note**: Camera 3 (webcam) currently shows test pattern. For webcam streaming on Windows, you'll need to implement your own solution as Docker Desktop doesn't support direct webcam access.

## Testing

### Test with VLC

```bash
vlc rtsp://localhost:8554/camera1
```

### Test with FFplay

```bash
ffplay rtsp://localhost:8554/camera1
```

### Test with Python Script

```bash
python test-cameras.py
```

This will test all 4 cameras and show the results.

## Alternative Protocols

MediaMTX also provides:
- **HLS**: `http://localhost:8888/camera1/index.m3u8`
- **RTMP**: `rtmp://localhost:1935/camera1`

## Troubleshooting

### Cameras not streaming

```bash
# Check logs
docker-compose logs

# Restart specific camera
docker-compose restart camera-1
```

### Port already in use

Edit `docker-compose.yml` and change port 8554 to another port:

```yaml
ports:
  - "8555:8554"  # Use 8555 instead
```

Then use: `rtsp://localhost:8555/camera1`

### High CPU usage

Use real video files instead of test patterns - they're less CPU intensive when looping.

## Specifications

- **Resolution**: 1280x720
- **Frame rate**: 25 fps
- **Bitrate**: 2000 kbps
- **Codec**: H.264

