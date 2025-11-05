# Violence Detection System

<div align="center">

### AI-powered Real-time Violence Detection for Intelligent Surveillance

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## Executive Summary

A comprehensive AI-driven violence detection system designed for intelligent video surveillance. The system leverages YOLOv8 neural networks combined with advanced preprocessing techniques to identify violent behavior in real-time video streams with minimal latency and high accuracy.

**Target Use Cases:**
- Surveillance in public spaces (airports, transit hubs, retail)
- Critical infrastructure protection
- Venue safety monitoring
- Emergency response systems

---

## Key Features

| Feature | Capability |
|---------|-----------|
| **Detection Performance** | Real-time (<2s latency), 85%+ accuracy |
| **Processing** | Optimized inference (PyTorch & ONNX) |
| **Scalability** | Support for 10-20 concurrent video streams |
| **Deployment** | Docker containerization for easy deployment |
| **Flexibility** | Multiple inference backends (PyTorch/ONNX) for different hardware |

---

## System Architecture

The system is built on a modular microservices architecture:

```
┌─────────────────────────────────────────────────────┐
│          VIDEO INPUT SOURCES                        │
│  IP Cameras (RTSP) │ Webcams │ Video Files          │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│          MEDIA PROCESSING LAYER                     │
│  Frame Extraction │ Preprocessing │ Streaming       │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│       AI DETECTION SERVICE                          │
│  YOLOv8 (PyTorch/ONNX) │ OpenCV │ GPU/CPU Inference │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│         BACKEND SERVICES                            │
│  FastAPI │ PostgreSQL │ Redis │ Authentication      │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│      USER-FACING INTERFACES                         │
│  Web Dashboard (React) │ Mobile App (Flutter)       │
└─────────────────────────────────────────────────────┘
```

---

## Technical Specifications

### AI Model

- **Base Architecture**: YOLOv8 Nano (lightweight, optimized)
- **Training Data**: RWF-2000 violence detection dataset
- **Inference Modes**:
  - **PyTorch**: Full precision, best accuracy
  - **ONNX**: Quantized, optimized for CPU/edge devices
- **Input Formats**: 320×320, 384×384, 512×512 (configurable)

### Performance Characteristics

| Metric | PyTorch | ONNX (CPU) |
|--------|---------|-----------|
| Latency (GPU) | 1-3ms | N/A |
| Latency (CPU) | 10-30ms | 2-5ms |
| Model Size | 12MB | 4MB |
| Memory Usage | 300MB+ | 50MB+ |
| Accuracy | 100% baseline | ~98% vs PyTorch |

### Infrastructure

- **Backend Framework**: FastAPI (async, high-performance)
- **Database**: PostgreSQL (structured data), Redis (caching)
- **Containerization**: Docker + Docker Compose
- **Streaming**: RTSP protocol via MediaMTX
- **Security**: JWT authentication, encrypted communications

---

## Project Structure

```
violence-detection/
├── ai_service/                    # AI Detection Engine
│   ├── detection/                 # Detection implementations
│   │   ├── pytorch_detector.py    # YOLOv8 PyTorch inference
│   │   └── onnx_inference.py      # YOLOv8 ONNX inference
│   ├── common/preprocessing/      # Data preprocessing
│   ├── models/weights/            # Model weights storage
│   ├── config/                    # Configuration management
│   ├── utils/                     # Utilities & helpers
│   └── tests/                     # Comprehensive testing suite
│
├── backend/                       # FastAPI Backend Service
│   ├── main.py                    # Application entry point
│   ├── src/
│   │   ├── core/                  # Configuration & logging
│   │   ├── domain/                # Business entities
│   │   ├── application/           # Use cases & business logic
│   │   ├── infrastructure/        # External integrations
│   │   └── presentation/          # API endpoints
│   ├── docker-compose.yml         # Service orchestration
│   └── Dockerfile                 # Container definition
│
├── admin-dashboard/               # React Web Dashboard (UI Layer)
├── flutter-app/                   # Flutter Mobile App (UI Layer)
├── utils/                         # Utility scripts
└── docs/                          # Documentation
```

---

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary development language
- **PyTorch**: Deep learning framework
- **YOLOv8**: Object detection model
- **ONNX Runtime**: Cross-platform inference
- **FastAPI**: Async web framework
- **OpenCV**: Computer vision preprocessing

### Infrastructure & DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **PostgreSQL**: Relational database
- **Redis**: In-memory caching & queuing
- **MediaMTX**: RTSP streaming server

### Frontend
- **React**: Web dashboard
- **Flutter**: Cross-platform mobile app

---

## System Requirements

### Development
- Python 3.11 or higher
- 8GB RAM minimum
- CUDA 11.8+ (optional, for GPU acceleration)

### Production
- Docker & Docker Compose
- GPU (recommended) or CPU (supported)
- 16GB+ RAM
- 50GB+ storage (for video logs)

---

## Quick Overview

### Detection Pipeline

1. **Frame Acquisition**: Frames from RTSP streams, webcams, or files
2. **Preprocessing**: Resize, normalize, prepare for neural network
3. **Inference**: YOLOv8 forward pass (GPU/CPU)
4. **Post-processing**: NMS filtering, confidence thresholding
5. **Alert Generation**: Trigger notifications if violence detected
6. **Logging**: Store results in database for analytics

### Quality Assurance

The project includes comprehensive testing:
- **Unit Tests**: test cases for core components
- **Integration Tests**: End-to-end detection workflows
- **Performance Tests**: Latency & throughput benchmarks
- **Accuracy Tests**: Model performance validation

---

## Notable Features

### Multi-Backend Support
Deploy the same trained model across different platforms:
- **PyTorch backend**: Full precision inference (GPU-optimized)
- **ONNX backend**: Quantized inference (CPU-optimized)
- **Automatic selection**: Based on available hardware

### Modular Architecture
Each component is independently deployable:
- Detection service can run standalone
- Backend API is independent of frontend
- Easy to swap inference engines

### Comprehensive Logging
- Frame-by-frame detection results
- Performance metrics tracking
- Event history & analytics
- Audit trails for security

---

## Development Team

**Lead Developer**: tiao051

The project is actively maintained and welcomes community contributions.

---

## Acknowledgments

### Technologies & Resources
- **YOLOv8**: Ultralytics for state-of-the-art object detection
- **PyTorch**: Meta for deep learning framework
- **OpenCV**: Computer vision community
- **FastAPI**: Starlette team for async web framework

### Datasets
- **RWF-2000**: Real World Fight Dataset for model training

---

## License

This project is provided for educational and research purposes.

---

## Contributing

We welcome contributions! Please refer to the individual component repositories:
- [Backend Service](backend/README.md)
- [AI Service](ai_service/README.md)

---

## Acknowledgments

**🤝 Special Thanks To**

| Category | Resource |
|----------|----------|
| 🧠 **AI & ML** | YOLOv8 (Ultralytics), PyTorch, ONNX Runtime |
| 📊 **Datasets** | RWF-2000 Violence Detection Dataset |
| 🛠️ **Tools** | OpenCV, Docker, FastAPI |

---

<div align="center">

### 📖 Documentation

For detailed implementation guides, visit:
- **[Backend Documentation](backend/README.md)** - API, services, deployment
- **[AI Service Documentation](ai_service/README.md)** - Models, inference, testing

---

### 👨‍💻 Development Team

**tiao051** - Project Creator & Lead Developer

---

### 💖 Built with Passion for a Safer World

**Status**: 🔄 In Active Development | **Last Updated**: November 2025

[![GitHub](https://img.shields.io/badge/GitHub-tiao051-181717?style=for-the-badge&logo=github)](https://github.com/tiao051)
[![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Research-blue?style=flat-square)](LICENSE)

</div>

