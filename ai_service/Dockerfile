FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Java for PySpark)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Spark (auto-detected from default-jre)

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (with longer timeout for large packages)
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY . .

# Set PYTHONPATH to include ai_service
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import redis.asyncio as redis; import asyncio; asyncio.run(redis.from_url('${REDIS_URL}').ping())" || exit 1

# Run inference consumer
CMD ["python", "inference/inference_consumer_service.py"]
