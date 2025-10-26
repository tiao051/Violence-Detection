# Backend Service

Violence Detection System - Backend API with Clean Architecture

## Architecture

```
backend/
├── src/
│   ├── domain/              # Enterprise Business Rules
│   │   ├── entities/        # Business entities
│   │   └── repositories/    # Repository interfaces
│   ├── application/         # Application Business Rules
│   │   ├── use_cases/       # Use cases (interactors)
│   │   └── dtos/            # Data Transfer Objects
│   ├── infrastructure/      # Frameworks & Drivers
│   │   ├── database/        # Database implementations
│   │   ├── streaming/       # RTSP stream consumers
│   │   └── messaging/       # Redis, RabbitMQ
│   └── presentation/        # Interface Adapters
│       ├── api/             # FastAPI routes
│       └── schemas/         # Request/Response schemas
├── tests/
├── main.py
├── requirements.txt
└── docker-compose.yml
```

## Clean Architecture Layers

### 1. Domain Layer (Innermost)
- **Entities**: Camera, StreamSession, Frame
- **Repository Interfaces**: ICameraRepository, IStreamRepository
- **Domain Services**: Pure business logic
- **No dependencies** on outer layers

### 2. Application Layer
- **Use Cases**: AddCamera, StartStream, StopStream, GetCameraStatus
- **DTOs**: CameraDTO, StreamStatusDTO
- **Depends only** on Domain layer

### 3. Infrastructure Layer
- **Database**: PostgreSQL, Redis implementations
- **External Services**: RTSP stream consumer, Message queue
- **Implements** repository interfaces from Domain

### 4. Presentation Layer (Outermost)
- **API Controllers**: FastAPI routes
- **Schemas**: Pydantic models for API
- **Depends on** Application layer (Use Cases)

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (SQLAlchemy), Redis
- **Streaming**: OpenCV (cv2)
- **Message Queue**: Redis Streams / RabbitMQ
- **DI Container**: dependency-injector

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Docker
docker-compose up -d

# Run locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

```
POST   /api/v1/cameras              # Add camera
GET    /api/v1/cameras              # List cameras
GET    /api/v1/cameras/{id}         # Get camera
PUT    /api/v1/cameras/{id}         # Update camera
DELETE /api/v1/cameras/{id}         # Delete camera

POST   /api/v1/cameras/{id}/stream/start   # Start streaming
POST   /api/v1/cameras/{id}/stream/stop    # Stop streaming
GET    /api/v1/cameras/{id}/stream/status  # Stream status
```
