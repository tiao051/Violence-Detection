# Backend Service

Violence Detection System - Backend API with Clean Architecture

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services (PostgreSQL + Redis + Backend)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Stop and remove volumes (clean restart)
docker-compose down -v
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

# Start PostgreSQL and Redis (via Docker)
docker-compose up -d postgres redis

# Run backend locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
