# Violence Detection System

> AI-powered real-time violence detection system with web dashboard and mobile application

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flutter](https://img.shields.io/badge/flutter-3.0%2B-blue.svg)](https://flutter.dev/)
[![React](https://img.shields.io/badge/react-18.0%2B-blue.svg)](https://reactjs.org/)

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [AI Model](#ai-model)
- [API Documentation](#api-documentation)
- [Security](#security)
- [Monitoring & Analytics](#monitoring--analytics)
- [Documentation](#documentation)
- [License](#license)

## Overview

This project is a comprehensive violence detection system that leverages AI to identify violent behavior in real-time video streams. The system provides:

- **Real-time Detection**: Process video streams with <2 second latency
- **High Accuracy**: >85% precision and recall on violence detection
- **Multi-platform**: Web dashboard and mobile application
- **Scalable**: Handle 10-20 concurrent video streams
- **Smart Alerts**: Automatic notifications when violence is detected

### Key Objectives

- Build an AI system to detect violent behavior in real-time video with high accuracy
- Develop mobile app (Flutter) and web dashboard (React) for monitoring and management
- Implement automatic alert system when violence is detected
- Provide analytics and statistical reporting on violence events

### Technical Requirements

- **Response Time**: < 2 seconds from detection to alert
- **Accuracy**: > 85% (Precision & Recall)
- **Video Sources**: IP cameras, webcams, file uploads
- **Concurrent Streams**: 10-20 simultaneous video streams
- **Storage**: Efficient event history and retrieval

## System Architecture

The system follows a microservices architecture with Docker containerization:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                             │
│  IP Cameras (RTSP/ONVIF) │ Webcams │ File Upload                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   MEDIA PROCESSING LAYER                         │
│  FFmpeg Service │ Streaming Server │ MinIO Storage              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    AI DETECTION SERVICE (GPU)                    │
│  YOLO Fine-tuned Model │ OpenCV │ TensorRT/ONNX Runtime         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                      BACKEND SERVICES                            │
│  Auth │ Kafka/RabbitMQ │ PostgreSQL/MongoDB/Redis               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐          ┌──────────────────┐
│  MOBILE       │          │  WEB             │
│  Flutter App  │          │  React Dashboard │
└───────────────┘          └──────────────────┘
```

### Components

#### 1. Video Input Sources
- **IP Cameras**: RTSP/ONVIF protocol support
- **Webcam**: WebRTC integration
- **File Upload**: MP4, AVI, MOV formats
- **Live Streaming**: RTMP/HLS streams

#### 2. Media Processing Layer
- **FFmpeg Service**: Video preprocessing and frame extraction
- **Streaming Server**: Node Media Server / Ant Media Server
- **Video Storage**: MinIO S3-compatible storage

#### 3. AI Detection Service
- **Framework**: PyTorch with ONNX Runtime for inference
- **Model**: Fine-tuned YOLO for violence detection
- **Preprocessing**: OpenCV for frame processing
- **GPU Support**: CUDA-enabled Docker container
- **Inference Engine**: TensorRT/ONNX Runtime for optimized performance

#### 4. Backend Services
- **API Gateway**: FastAPI/Kong for routing and load balancing
- **Authentication**: JWT + OAuth 2.0
- **Event Processing**: Apache Kafka/RabbitMQ + Redis
- **Databases**:
  - PostgreSQL: User data, configurations, event logs
  - MongoDB: Video metadata, analysis results
  - InfluxDB: Time-series analytics data
- **Notification Service**: FCM + WebSocket + Email

#### 5. Frontend Applications
- **Mobile App**: Flutter-based iOS/Android application
- **Web Dashboard**: React + TypeScript admin panel

#### 6. DevOps & Monitoring
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana + ELK Stack
- **Load Balancing**: NGINX + HAProxy

## Features

### AI Detection

- **Real-time Processing**: Stream processing with low latency (<2s)
- **Violence Categories**:
  - Physical Violence (fighting, assault)
  - Weapon-based Violence (knife, gun detection)
  - Group Violence (riots, brawls)
  - Vandalism (property destruction)
  - Aggressive Behavior (threatening gestures)
- **Adaptive Threshold**: Auto-adjust sensitivity based on context
- **False Positive Reduction**: Multi-stage validation
- **Continuous Learning**: Improve from user feedback

### Mobile Application (Flutter)

- **Authentication**: Login/Register with biometric support
- **Live Monitoring**: Multi-camera grid view
- **Smart Alerts**: Violence alerts with severity classification
- **Quick Response**: Emergency buttons (call police/security)
- **Evidence Collection**: Auto-save video clips when violence detected
- **Offline Mode**: Cache data when connection lost
- **Multi-language**: Vietnamese + English

### Web Dashboard (React)

- **Real-time Dashboard**: Grid view for multiple cameras
- **Alert Management**: Process and categorize alerts
- **Analytics**: Statistics charts by time/location
- **User Management**: Role-based access control
- **System Configuration**: AI model settings
- **Report Generation**: Export PDF/Excel reports
- **Map Integration**: Display camera locations on map

### Advanced Features

- **Crowd Analysis**: Detect crowds and potential riots
- **Weapon Detection**: Identify weapons in video
- **Face Blurring**: Automatic privacy protection
- **Audio Analysis**: Detect screams, gunshots
- **Integration APIs**: Webhook, REST API for third-party systems

## Technology Stack

### AI Service
- **Language**: Python 3.8+
- **Framework**: PyTorch
- **Inference**: ONNX Runtime, TensorRT
- **Computer Vision**: OpenCV, MediaPipe
- **Model**: Fine-tuned YOLO (YOLOv5/YOLOv8)

### Backend
- **Framework**: FastAPI (Python)
- **Databases**: PostgreSQL, MongoDB, Redis, InfluxDB
- **Message Queue**: Apache Kafka / RabbitMQ
- **Storage**: MinIO (S3-compatible)
- **Authentication**: JWT, OAuth 2.0

### Frontend

#### Mobile App
- **Framework**: Flutter 3.0+
- **State Management**: Riverpod / Bloc
- **Networking**: Dio
- **Real-time**: flutter_webrtc
- **Notifications**: firebase_messaging
- **Local Storage**: sqflite

#### Web Dashboard
- **Framework**: React 18 + TypeScript
- **UI Library**: Ant Design Pro
- **State Management**: Redux Toolkit
- **Real-time**: Socket.io
- **Charts**: Recharts
- **Video**: WebRTC

### DevOps
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana, ELK
- **Web Server**: NGINX

## AI Model

### Architecture

The system uses a **unified YOLO-based approach** for violence detection, replacing the previous multi-phase pipeline (detection → recognition → classification) for better performance.

**Previous Architecture (Deprecated):**
- Phase 1: Object Detection
- Phase 2: Feature Recognition
- Phase 3: Violence Classification
- **Issues**: 3x latency, 3x memory, complex pipeline

**Current Architecture (Unified):**
- **Single fine-tuned YOLO model** handles detection and classification in one pass
- **Benefits**: 
  - 3x faster processing
  - 66% less memory usage
  - Simpler deployment
  - Easier maintenance

### Pipeline

```
Input Video → Frame Extraction → YOLO Inference → Postprocessing → Alert
```

### Training Datasets

- **RWF-2000**: Real World Fighting dataset
- **HMDB-51**: Violence subset
- **UCF-Crime**: Crime and violence dataset
- **Violence in Movies**: Movie violence scenes
- **Custom Dataset**: Vietnam-specific context data

### Performance Metrics

- **Accuracy**: >85% (target)
- **Precision**: High precision to reduce false alarms
- **Recall**: High recall to catch all violence incidents
- **Inference Time**: <100ms per frame
- **FPS**: 10-30 fps depending on hardware

### Model Files

See [`ai-service/README.md`](ai-service/README.md) for detailed AI service documentation.

## API Documentation

### REST API

Base URL: `http://localhost:8000/api/v1`

#### Authentication
```http
POST /auth/login
POST /auth/register
POST /auth/refresh
```

#### Video Streams
```http
GET    /streams              # List all streams
POST   /streams              # Add new stream
GET    /streams/{id}         # Get stream details
PUT    /streams/{id}         # Update stream
DELETE /streams/{id}         # Delete stream
```

#### Violence Events
```http
GET    /events               # List violence events
GET    /events/{id}          # Get event details
PUT    /events/{id}/status   # Update event status
GET    /events/stats         # Get statistics
```

#### Alerts
```http
GET    /alerts               # List alerts
POST   /alerts/acknowledge   # Acknowledge alert
GET    /alerts/active        # Get active alerts
```

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Listen for violence alerts
ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('Violence detected:', alert);
};
```

### Webhook Integration

Configure webhooks to receive violence alerts:

```json
POST /webhooks/configure
{
  "url": "https://your-server.com/violence-alert",
  "events": ["violence_detected", "weapon_detected"],
  "secret": "your-secret-key"
}
```

## Security

- **Authentication**: JWT-based authentication with refresh tokens
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: SSL/TLS for all communications
- **Privacy**: Automatic face blurring in recorded videos
- **API Security**: Rate limiting, CORS, input validation
- **Data Protection**: Encrypted database, secure storage

## Monitoring & Analytics

- **System Metrics**: CPU, GPU, memory, network usage
- **AI Metrics**: Inference time, accuracy, false positive rate
- **Business Metrics**: Total alerts, response time, resolution rate
- **Dashboards**: Grafana dashboards for real-time monitoring
- **Alerts**: Prometheus alerting for system issues

## Testing

```bash
# AI Service tests
cd ai-service
pytest tests/

# Backend tests
cd backend
pytest tests/

# Frontend tests
cd admin-dashboard
npm test

# Mobile app tests
cd flutter-app
flutter test
```

## Documentation

- [Architecture Documentation](docs/architecture.md)
- [AI Pipeline Details](docs/ai-pipeline.md)
- [Detection Pipeline](docs/detection-pipeline.md)
- [AI Service README](ai-service/README.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **tiao051** - *Initial work* - [GitHub](https://github.com/tiao051)

## Acknowledgments

- YOLOv5/YOLOv8 by Ultralytics
- RWF-2000 dataset
- OpenCV community
- Flutter and React communities

## Support

For issues and questions:
- Create an issue: [GitHub Issues](https://github.com/tiao051/violence-detection/issues)
- Email: [your-email@example.com]

---

**Note**: This system is designed for security and safety purposes. Please ensure compliance with local privacy laws and regulations when deploying surveillance systems.
