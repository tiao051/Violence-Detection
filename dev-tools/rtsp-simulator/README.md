# RTSP Simulator

Simple RTSP server with 4 virtual cameras for testing the violence detection system.

## Quick Start

```bash
# Start RTSP server and 4 cameras
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all
docker-compose down
```

## Camera URLs

| Camera | Location | RTSP URL |
|--------|----------|----------|
| Camera 1 | Entrance | `rtsp://localhost:8554/camera1` |
| Camera 2 | Parking Lot | `rtsp://localhost:8554/camera2` |
| Camera 3 | Hallway | `rtsp://localhost:8554/camera3` |
| Camera 4 | Storage Room | `rtsp://localhost:8554/camera4` |

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

## Using Custom Video Files

To use real video files instead of test patterns:

1. Create a `test-videos` folder
2. Place your video files there (e.g., `entrance.mp4`)
3. Modify `docker-compose.yml`:

```yaml
camera-1:
  image: linuxserver/ffmpeg:latest
  command: >
    -re -stream_loop -1 -i /videos/entrance.mp4
    -c:v libx264 -preset ultrafast -f rtsp rtsp://rtsp-server:8554/camera1
  volumes:
    - ./test-videos:/videos
```

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

The test pattern encoding can use significant CPU. To reduce:

1. Lower the resolution in `docker-compose.yml`
2. Reduce frame rate
3. Or use real video files which are less CPU intensive when looping

## Specifications

- **Resolution**: 1280x720
- **Frame rate**: 25 fps
- **Bitrate**: 2000 kbps
- **Codec**: H.264
- **Container**: RTSP

All settings can be adjusted in `docker-compose.yml` as needed.

