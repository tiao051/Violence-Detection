# Dev Tools

Development and testing utilities for the violence detection system.

## RTSP Simulator

Simulates 4 RTSP cameras for testing without real hardware.

**Location**: `rtsp-simulator/`

**Quick Start**:
```bash
cd rtsp-simulator
docker-compose up -d
python test-cameras.py
```

See [rtsp-simulator/README.md](rtsp-simulator/README.md) for details.

## Future Tools

- Load testing tools
- Mock data generators
- Performance benchmarking
- Database seeders
