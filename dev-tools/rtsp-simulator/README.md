# RTSP Simulator

RTSP server with 4 cameras for violence detection testing.

## Camera Setup

| Camera | Input File | RTSP URL |
|--------|------------|----------|
| **Camera 1** | `violence_1.gif` | `rtsp://localhost:8554/cam1` |
| **Camera 2** | `violence_2.gif` | `rtsp://localhost:8554/cam2` |
| **Camera 3** | `no_violence_1.gif` | `rtsp://localhost:8554/cam3` |
| **Camera 4** | `violence_3.gif` | `rtsp://localhost:8554/cam4` |

## Quick Start

### 1. Place video files in `test-videos/` folder:
- `violence_1.gif`
- `violence_2.gif`
- `no_violence_1.gif`
- `violence_3.gif`

### 2. Start cameras:
```bash
docker-compose up -d
```

### 3. View cameras:
```bash
# View with ffplay
ffplay rtsp://localhost:8554/cam1 -rtsp_transport tcp
ffplay rtsp://localhost:8554/cam2 -rtsp_transport tcp
ffplay rtsp://localhost:8554/cam3 -rtsp_transport tcp
ffplay rtsp://localhost:8554/cam4 -rtsp_transport tcp

# Or use VLC
vlc rtsp://localhost:8554/cam1
```

### 4. Stop cameras:
```bash
docker-compose down
```

## Troubleshooting

### Check status
```bash
docker-compose ps
```

### View logs
```bash
docker-compose logs -f
```

### Restart specific camera
```bash
docker-compose restart camera-1
```

### Port already in use
Edit `docker-compose.yml`:
```yaml
ports:
  - "8555:8554"  # Change to 8555
```
Then use: `rtsp://localhost:8555/cam1`

## Specifications

- **Resolution**: 640x360
- **Frame rate**: 25 fps
- **Codec**: H.264
- **Transport**: TCP

