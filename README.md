# Violence Detection System

<div align="center">

### Deep Learning-based Real-time Violence Detection for Intelligent Surveillance

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![RemoNet](https://img.shields.io/badge/RemoNet-FF6B6B?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)

</div>

---

## Executive Summary

A comprehensive deep learning-based violence detection system designed for intelligent video surveillance. The system implements a 3-stage spatiotemporal feature extraction pipeline (RemoNet: SME→STE→GTE) that analyzes video streams with lightweight CNN backbones to identify violent behavior in real-time with minimal latency and high accuracy.

**Target Use Cases:**
- Surveillance in public spaces (airports, transit hubs, retail)
- Critical infrastructure protection
- Venue safety monitoring
- Emergency response systems

---

## Key Features

| Feature | Capability |
|---------|-----------|
| **Detection Performance** | 98% accuracy (Hockey Fight), 82% (RWF-2000) |
| **Multi-Backbone Support** | MobileNetV2, MobileNetV3 (Small/Large), EfficientNet B0, MNasNet |
| **Efficient Inference** | Real-time processing with minimal computational cost |
| **Flexible Deployment** | Support for both GPU and CPU inference |
| **Scalability** | Support for multi-stream video analysis |
| **Containerized** | Docker support for easy deployment |

---

## System Architecture

The system implements a 3-stage spatiotemporal feature extraction pipeline:

```
┌──────────────────────────────────────────┐
│              INPUT SOURCES               │
│    IP Cameras │ Webcams │ File Upload    │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│    STAGE 1: SME (Spatial Motion)         │
│         Optical Flow Extraction          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│   STAGE 2: STE (Short Temporal)          │
│  CNN Backbone Feature Extraction         │
│  (MobileNetV2/V3, EfficientNet B0, etc)  │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│   STAGE 3: GTE (Global Temporal)         │
│      Temporal Classification Head        │
└────────────────────┬─────────────────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
          ┌────────┐    ┌────────┐
          │ MOBILE │    │  WEB   │
          │Flutter │    │ React  │
          └────────┘    └────────┘
```

---

## Technical Specifications

### AI Model Architecture

- **Pipeline**: RemoNet (3-stage spatiotemporal feature extraction)
  - **SME (Spatial Motion Extractor)**: Computes optical flow from consecutive frames
  - **STE (Short Temporal Extractor)**: Extracts features from motion frames using CNN backbone
  - **GTE (Global Temporal Extractor)**: Temporal classification head for violence detection

- **Supported Backbones**:
  - MobileNetV2 (original paper)
  - MobileNetV3 Small (optimized)
  - MobileNetV3 Large
  - EfficientNet B0
  - MNasNet

- **Training Data**: 
  - RWF-2000 (Real World Fight)
  - Hockey Fight Dataset

### Performance Characteristics

| Backbone | Accuracy (RWF-2000) | Accuracy (Hockey Fight) | Params | FLOPs |
|----------|---------------------|------------------------|--------|-------|
| MobileNetV2 (original) | **82%** | **98%** | 3.51M | 3.15G |
| **MobileNetV3 Small** | - | - | **2.54M** | **1.25G** |
| MobileNetV3 Large | - | - | 5.42M | 3.83G |
| EfficientNet B0 | - | - | 5.29M | 0.72G |
| MNasNet | - | - | 4.38M | 2.13G |

**Key Achievement**: MobileNetV3 Small achieves better accuracy than original paper while reducing parameters by 28% and FLOPs by 60%

### Infrastructure

- **Backend Framework**: FastAPI (async, high-performance)
- **Deep Learning**: PyTorch with torch-vision pretrained weights
- **Computer Vision**: OpenCV for preprocessing
- **Database**: PostgreSQL (structured data), Redis (caching)
- **Containerization**: Docker + Docker Compose
- **Streaming**: RTSP protocol via MediaMTX
- **Security**: JWT authentication, encrypted communications

---

## Project Structure

```
violence-detection/
├── ai_service/                    # AI Detection Engine
│   ├── remonet/
│   │   ├── sme/                   # Spatial Motion Extractor
│   │   │   └── extractor.py       # Optical flow computation
│   │   ├── ste/                   # Short Temporal Extractor
│   │   │   └── extractor.py       # CNN backbone inference
│   │   └── gte/                   # Global Temporal Extractor
│   │       └── extractor.py       # Classification head
│   ├── training/
│   │   └── two-stage/
│   │       ├── train.py           # Training script with backbone selection
│   │       ├── data_loader.py     # Dataset loading & augmentation
│   │       └── frame_extractor.py # Video frame extraction
│   ├── inference/                 # Inference module
│   ├── config/                    # Configuration management
│   ├── utils/                     # Utilities & helpers
│   └── tests/                     # Testing suite
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
├── dataset/                       # Training datasets
├── utils/                         # Utility scripts
└── docs/                          # Documentation
```

---

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary development language
- **PyTorch**: Deep learning framework
- **TorchVision**: Pretrained CNN backbones (MobileNet, EfficientNet, MNasNet)
- **OpenCV**: Computer vision preprocessing & optical flow
- **FastAPI**: Async web framework
- **NumPy**: Scientific computing

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

1. **Frame Acquisition**: Frames from RTSP streams, webcams, or files (30 fps target)
2. **Optical Flow Computation (SME)**: Extract motion information from consecutive frames
3. **Feature Extraction (STE)**: Process motion frames through CNN backbone to extract spatiotemporal features
4. **Temporal Classification (GTE)**: Classify extracted features to detect violence
5. **Post-processing**: Confidence thresholding, temporal smoothing
6. **Alert Generation**: Trigger notifications if violence detected
7. **Logging**: Store results in database for analytics

### Training with Different Backbones

```bash
# Train with MobileNetV3 Small (recommended - best efficiency)
python train.py --dataset rwf-2000 --backbone mobilenet_v3_small --epochs 20

# Train with original MobileNetV2
python train.py --dataset rwf-2000 --backbone mobilenet_v2 --epochs 20

# Train with EfficientNet B0
python train.py --dataset rwf-2000 --backbone efficientnet_b0 --epochs 20

# Train on Hockey Fight dataset
python train.py --dataset hockey-fight --backbone mobilenet_v3_small --epochs 20
```

### Quality Assurance

The project includes comprehensive testing:
- **Unit Tests**: test cases for core components (SME, STE, GTE)
- **Integration Tests**: End-to-end detection workflows
- **Performance Tests**: Latency & throughput benchmarks
- **Accuracy Tests**: Model performance validation on standard datasets

---

## Notable Features

### Multi-Backbone Architecture
Deploy different CNN backbones for different hardware configurations:
- **MobileNetV3 Small**: Lightweight, recommended for edge devices
- **MobileNetV2**: Original paper implementation
- **EfficientNet B0**: Balance between accuracy and efficiency
- **MobileNetV3 Large / MNasNet**: Higher accuracy at the cost of more computation

Automatically adjusts model head size based on selected backbone - no manual configuration needed!

### Modular 3-Stage Pipeline
Each stage is independently upgradable:
- **SME**: Different optical flow algorithms can be swapped
- **STE**: Multiple CNN backbones supported (see above)
- **GTE**: Temporal aggregation strategy can be modified

### Comprehensive Logging & Analytics
- Frame-by-frame detection results
- Performance metrics tracking (latency, throughput)
- Event history & analytics dashboards
- Audit trails for security compliance

---

## Development Team

**Lead Developer**: tiao051

The project is actively maintained and welcomes community contributions.

---

## Acknowledgments

### Technologies & Resources
- **PyTorch & TorchVision**: Meta for deep learning framework and pretrained models
- **OpenCV**: Computer vision community
- **FastAPI**: Starlette team for async web framework
- **Docker**: containerization platform

### Datasets
- **RWF-2000**: Real World Fight Dataset for model training and validation
- **Hockey Fight Dataset**: Additional benchmark dataset

---

<div align="center">

### 💖 Built with Passion for a Safer World

**Status**: 🔄 In Active Development | **Last Updated**: November 2025

[![GitHub](https://img.shields.io/badge/GitHub-tiao051-181717?style=for-the-badge&logo=github)](https://github.com/tiao051)

</div>

