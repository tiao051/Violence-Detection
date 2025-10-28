# MediaMTX USB Webcam Setup

Stream USB webcam as RTSP (like IP camera).

## How It Works

```
USB Camera → MediaMTX (FFmpeg) → RTSP Stream (port 8554) → Backend
```

Backend treats USB cam like any other IP camera (no special code needed).

## Setup

### 1. Find Your USB Camera Device

**Windows:**
```bash
ffmpeg -list_devices true -f dshow -i dummy 2>&1 | grep "video="
# Note: exact camera name (e.g., "USB Camera", "Logitech Webcam", etc.)
```

**Linux:**
```bash
ls /dev/video*
# Your device: /dev/video0 or /dev/video1
```

**Mac:**
```bash
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep video
```

### 2. Update `mediamtx.yml`

**Windows (DirectShow):**
```yaml
paths:
  usb-cam:
    runOnInit: ffmpeg -f dshow -i video="YOUR_EXACT_CAMERA_NAME" -c:v libx264 -preset veryfast -g 30 -b:v 2000k -an -f rtsp rtsp://localhost:8554/usb-cam
    runOnInitRestart: yes
```

**Linux (Video4Linux):**
```yaml
paths:
  usb-cam:
    runOnInit: ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset veryfast -g 30 -b:v 2000k -an -f rtsp rtsp://localhost:8554/usb-cam
    runOnInitRestart: yes
```

**Mac (AVFoundation):**
```yaml
paths:
  usb-cam:
    runOnInit: ffmpeg -f avfoundation -i ":0" -c:v libx264 -preset veryfast -g 30 -b:v 2000k -an -f rtsp rtsp://localhost:8554/usb-cam
    runOnInitRestart: yes
```

### 3. Update `docker-compose.yml`

The `devices` section enables USB access:
```yaml
rtsp-server:
  devices:
    - /dev/video0:/dev/video0  # For Linux
  # Windows/Mac: no device mapping needed
```

### 4. Start Services

```bash
cd backend
docker-compose up -d
```

### 5. Verify

**Check logs:**
```bash
docker logs rtsp-server | head -20
# Should see: "[path/usb-cam] source ready" or FFmpeg starting
```

**Test RTSP stream:**
```bash
ffplay rtsp://localhost:8554/usb-cam
# Should show live video from USB camera
```

**Check backend:**
```bash
curl http://localhost:8000/stats | jq '.workers'
# Should show frames being captured
```

## Configuration

### Change video size
```yaml
usb-cam:
  runOnInit: ffmpeg -f dshow -i video="Camera" -video_size 1280x720 -c:v libx264 -f rtsp rtsp://localhost:8554/usb-cam
```

### Change bitrate (Lower = less CPU, lower quality)
```yaml
runOnInit: ffmpeg ... -b:v 1000k ...  # 1 Mbps
# or
runOnInit: ffmpeg ... -b:v 4000k ...  # 4 Mbps
```

### Change framerate
```yaml
runOnInit: ffmpeg -f dshow -i video="Camera" -framerate 15 ... # 15 FPS instead of 30
```

## Troubleshooting

### Camera Not Found
```bash
# Check exact name (Windows)
ffmpeg -list_devices true -f dshow -i dummy 2>&1 | grep video=

# Update mediamtx.yml with exact name
```

### "Device not found" (Linux)
```bash
# Check permissions
ls -la /dev/video0

# Add to video group
sudo usermod -aG video $USER
# Then logout and login
```

### No video in ffplay
```bash
# Check MediaMTX logs
docker logs rtsp-server

# Make sure FFmpeg command is correct
ffmpeg -f dshow -i video="YOUR_CAMERA_NAME" -t 5 output.mp4
```

### Backend can't connect
```bash
# Test from inside backend container
docker exec -it violence-detection-backend bash
ffplay rtsp://rtsp-server:8554/usb-cam
```

## Backend Configuration

No changes needed! Backend `config.py` already has:
```python
rtsp_cameras: List[str] = ["cam1", "cam2", "cam3", "cam4"]
# Add usb-cam to this list if needed
```

Update if you want to add USB cam to processing:
```python
rtsp_cameras: List[str] = ["cam1", "cam2", "cam3", "cam4", "usb-cam"]
```

## Environment Variables

Backend uses these (no USB-specific config needed):
```
RTSP_ENABLED=true
RTSP_BASE_URL=rtsp://rtsp-server:8554
RTSP_CAMERAS=cam1,cam2,cam3,cam4,usb-cam
RTSP_SAMPLE_RATE=6
```

## Testing Commands

```bash
# View MediaMTX logs
docker logs -f rtsp-server

# View backend logs
docker logs -f violence-detection-backend

# Test RTSP
ffplay rtsp://localhost:8554/usb-cam

# Check Redis
docker exec violence-detection-redis redis-cli XLEN frames:usb-cam

# Get backend stats
curl http://localhost:8000/stats | jq '.workers."usb-cam"'
```

## Architecture

```
┌────────────────┐
│ USB Camera     │
└────────┬───────┘
         │ /dev/video0 (Linux)
         │ or DirectShow (Windows)
         ▼
┌──────────────────────────────────┐
│ MediaMTX Container               │
│  └─ FFmpeg process               │
│     └─ Capture & Encode H.264    │
└────────┬─────────────────────────┘
         │ RTSP stream port 8554
         ▼
┌──────────────────────────────────┐
│ Backend                          │
│  └─ RTSPClient reads RTSP stream │
│     └─ Processes frames          │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Frame Buffer (in-memory)         │
│ + Redis Streams (metadata)       │
└──────────────────────────────────┘
```

## What MediaMTX Does

- ✅ Starts FFmpeg process
- ✅ Manages USB camera capture
- ✅ Encodes to H.264
- ✅ Streams RTSP to port 8554
- ✅ Auto-restarts on crash
- ✅ Handles multiple cameras
- ✅ Provides metrics & health checks

## Next Steps

1. ✅ USB camera set up
2. Verify frames in backend stats
3. Implement AI inference on frames
4. Add database storage
5. Create alert system

## Reference

- **MediaMTX:** https://github.com/bluenviron/mediamtx
- **FFmpeg:** https://ffmpeg.org/
- **RTSP Protocol:** RFC 7826

---

**Ready?** Start with Step 1 and work through! 🎥
