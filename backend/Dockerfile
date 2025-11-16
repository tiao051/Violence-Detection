# Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

# Create directory for ai_service (for model imports)
# Note: ai_service will be mounted via docker-compose
RUN mkdir -p /app/ai_service

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

