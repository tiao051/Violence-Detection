# Backend Pipeline Architecture

## Overview

The backend service orchestrates real-time violence detection by managing:
1. **RTSP Stream Capture** - Multiple camera streams
2. **Kafka-based Frame Processing** - Distributed frame pipeline
3. **Inference & Alert Generation** - Real-time threat detection
4. **Alert Deduplication & Notification** - Smart alert routing
5. **Event Persistence** - Video recording and analytics

This architecture ensures **scalability**, **reliability**, and **real-time performance** for video surveillance systems.

---

## Architecture Components

The pipeline consists of five main modules working sequentially and in parallel:

### 1. RTSP Camera Worker (Frame Source)
**Purpose**: Capture raw video frames from RTSP camera streams

**Input**: RTSP URLs for multiple cameras (cam1, cam2, cam3, cam4)

**Process**:
1. **Stream Connection**: Connect to RTSP server at configured URL
   - Example: `rtsp://rtsp-server:8554/cam1`
   - Retry logic with configurable attempts (3 retries)

2. **Frame Sampling**: Extract frames at target FPS (5 FPS default)
   - Calculate frame interval: `interval = source_fps / target_fps`
   - Skip frames between samples for efficiency
   - Capture raw RGB frames without processing

3. **Metadata Attachment**: Add frame metadata
   - `camera_id`: Source camera identifier
   - `timestamp`: Frame capture time (epoch seconds)
   - `frame_number`: Sequential counter
   - `width`, `height`: Original frame dimensions

4. **Kafka Publishing**: Send raw frames to Kafka producer

**Key Advantages**:
- **Single Responsibility**: Only captures frames, no preprocessing
- **Minimal Computation**: Sampling at 5 FPS reduces bandwidth (~150 MB/hour per camera)
- **Clean Separation**: Leaves resize/compression for dedicated producer

**Output**: Raw RGB frames with metadata → Kafka topic `processed-frames`

**Configuration** (from `.env`):
```
RTSP_BASE_URL=rtsp://rtsp-server:8554
RTSP_SAMPLE_RATE=6  # FPS (actually 5 after conversion)
RTSP_ENABLED=true
RTSP_CAMERAS=["cam1", "cam2", "cam3", "cam4"]
```

---

### 2. Kafka Frame Producer (Preprocessing)
**Purpose**: Optimize frames for inference through compression and resizing

**Input**: Raw frames from camera workers via Kafka

**Process**:
1. **Frame Resizing**: Single optimization point
   - Target resolution: **224×224** (matches AI model input)
   - Method: Bilinear interpolation
   - Applied once per frame (no duplication in camera_worker)

2. **JPEG Compression**: Reduce bandwidth and storage
   - Quality level: **80** (balances quality and size)
   - Size reduction: **10-50x** compared to raw RGB
   - Example: 640×480 RGB (~900 KB) → 224×224 JPEG (~15-20 KB)

3. **Message Formatting**: Create structured Kafka message
   ```json
   {
     "camera_id": "cam1",
     "timestamp": 1702150000.123,
     "frame_number": 42,
     "jpeg_data": "<base64-encoded JPEG>",
     "width": 224,
     "height": 224,
     "original_width": 640,
     "original_height": 480
   }
   ```

4. **Partitioning**: Distribute by camera for temporal ordering
   - Key: `camera_id`
   - Ensures all frames from same camera go to same partition
   - Maintains frame sequence per camera

5. **Metrics Tracking**:
   - `frames_sent`: Count of successful sends
   - `frames_failed`: Count of failed sends
   - `compression_ratio`: Average size reduction ratio
   - `avg_latency_ms`: Processing time per frame

**Key Advantages**:
- **Single Resize Point**: Avoids duplication and ensures consistency
- **Efficient Compression**: JPEG reduces Kafka broker load significantly
- **Partitioned by Camera**: Enables per-camera batching in consumer
- **Metrics Enabled**: Monitor pipeline health

**Output**: Compressed, resized frames → Kafka topic `processed-frames`
- Per-camera partitions maintain temporal ordering
- Ready for inference

---

### 3. Inference Consumer (AI Processing)
**Purpose**: Process frames through violence detection model and generate alerts

**Input**: Compressed frames from Kafka topic

**Process**:
1. **Kafka Consumption**: Read frames from Kafka
   - Consumer group: `inference-group`
   - Automatic offset management
   - Batch polling for efficiency

2. **Frame Batching**: Group frames by camera for efficient inference
   - Batch size: **4 frames per camera**
   - Batch timeout: **100 ms** (max wait time)
   - Separate batches per camera (temporal ordering)
   - Batching reduces model overhead, improves throughput

3. **JPEG Decoding**: Convert JPEG data back to frames
   - Decompress using OpenCV
   - Validate dimensions match expected (224×224)

4. **Model Inference**: Run violence detection
   - Input: Batch of frames (4, 3, 224, 224)
   - Model: ReMotion (SME + STE + GTE modules)
   - Output: Probability scores [violence_prob, non_violence_prob]

5. **Alert Deduplication**: Smart alert rate limiting
   - Per-camera TTL cooldown: **60 seconds**
   - Redis-based state tracking
   - Check: Should send alert for this camera?
   - If YES: Set cooldown, publish alert
   - If NO: Suppress alert (already alerted recently)

6. **Alert Publishing**: Broadcast threats via Redis
   - **PubSub**: Real-time notifications to WebSocket clients
     ```json
     {
       "threat": "violence_detected",
       "camera_id": "cam1",
       "confidence": 0.95,
       "timestamp": 1702150000.123
     }
     ```
   - **Streams**: Persistent event log for replay/analytics
   - Topic: `camera:{camera_id}:alerts`

7. **Metrics Tracking**:
   - `frames_processed`: Total frames consumed
   - `alerts_sent`: Total alerts published
   - `alerts_per_camera`: Per-camera alert counts
   - `inference_latency_ms`: Model execution time
   - `batching_ratio`: Average frames per batch

**Key Advantages**:
- **Batched Inference**: 4x throughput improvement over single-frame processing
- **Per-Camera Batching**: Maintains temporal ordering for same camera
- **Smart Deduplication**: 60-second cooldown prevents alert spam
- **Dual Publishing**: Both real-time (PubSub) and persistent (Streams)

**Output**: 
- Real-time alerts → Redis PubSub
- Event log → Redis Streams
- Metrics → Internal counters

---

### 4. Alert Deduplication (Smart Throttling)
**Purpose**: Prevent alert spam while maintaining responsiveness

**Input**: Violence detection results from inference consumer

**Process**:
1. **State Check**: Query Redis for alert cooldown
   ```python
   KEY = f"alert:cooldown:{camera_id}"
   IS_COOLING_DOWN = redis.exists(KEY)
   ```

2. **Decision Logic**:
   ```
   IF camera is cooling down:
     → SUPPRESS alert (return False)
   ELSE:
     → ALLOW alert (return True)
     → SET cooldown: SETEX(KEY, 60 seconds, True)
   ```

3. **Per-Camera Isolation**: Each camera has independent cooldown
   - cam1 alert cooldown does NOT affect cam2
   - Enables simultaneous alerts on different cameras
   - Example flow:
     ```
     t=0: cam1 violence → ALERT, start 60s cooldown
     t=10: cam2 violence → ALERT (cam1 still cooling, but cam2 is fresh)
     t=30: cam1 violence again → SUPPRESS (still in cooldown)
     t=60: cam1 violence again → ALERT (cooldown expired)
     ```

4. **Reset Functionality**: Manual cooldown clear
   - `clear_alert(camera_id)` → Delete cooldown key
   - Allows immediate alert after clear (for testing/manual override)

5. **Redis Operations**:
   - `EXISTS`: Check if key exists
   - `SETEX`: Set key with TTL
   - `DELETE`: Clear cooldown

**Key Advantages**:
- **Prevents Alert Fatigue**: Reduces spam when violence persists
- **Per-Camera Isolation**: Simultaneous multi-camera alerts work correctly
- **Fast Implementation**: O(1) Redis operations
- **TTL-Based**: Automatic cleanup (no stale state)

**Output**: Boolean decision (should send alert?)

---

### 5. Event Persistence & Notification (Storage & Users)
**Purpose**: Record detected events and notify users via push notifications

**Input**: Alerted threat events from inference consumer

**Process**:
1. **Video Generation**: Capture surrounding video context
   - Retrieve last N frames from frame buffer (e.g., last 5 seconds)
   - Encode as MP4 video
   - Store temporarily

2. **Firebase Upload**: Persist video and metadata
   - Upload video file to Firebase Storage
   - Example path: `gs://bucket/alerts/{timestamp}_{camera_id}.mp4`
   - Record event metadata in Firestore:
     ```json
     {
       "camera_id": "cam1",
       "timestamp": 1702150000,
       "confidence": 0.95,
       "video_url": "gs://bucket/alerts/1702150000_cam1.mp4",
       "owner_uid": "user_123",
       "status": "new",
       "reviewed": false
     }
     ```

3. **User Notification**: Push alert to device
   - Query user's FCM tokens from Firestore
   - Send FCM push notification:
     ```json
     {
       "title": "Violence Detected!",
       "body": "Front Gate - 95% confidence",
       "data": {
         "camera_id": "cam1",
         "event_id": "evt_xyz"
       }
     }
     ```

4. **Event Logging**: Audit trail for compliance
   - Log alert details for analytics
   - Track notification delivery status

**Key Advantages**:
- **Video Context**: Alert includes recorded evidence
- **Async Processing**: Non-blocking user notifications
- **Firestore**: Queryable event history
- **Firebase Storage**: Scalable video hosting
- **FCM Tokens**: Multi-device support per user

**Output**: 
- Video file → Firebase Storage
- Event record → Firestore
- Push notification → User's devices

---

## Complete Pipeline Summary

```
Input: RTSP Camera Streams (4 cameras at source FPS)
    ↓
[RTSP Camera Worker] - Frame Capture
    → Sample at 5 FPS
    → Attach metadata (camera_id, timestamp)
    → Stream to Kafka
    ↓
[Kafka Topic: processed-frames] - Transport Layer
    → Partitioned by camera_id for ordering
    ↓
[Kafka Frame Producer] - Preprocessing
    → Resize: original resolution → 224×224
    → Compress: RGB → JPEG (quality 80)
    → Size reduction: 10-50x
    → Format as JSON message
    ↓
[Kafka Topic: processed-frames] - Optimized Frames
    → Ready for inference
    ↓
[Inference Consumer] - AI Processing
    → Batch 4 frames per camera (100 ms timeout)
    → Decode JPEG frames
    → Run violence detection model
    → Get confidence scores
    ↓
[Alert Deduplication Check]
    → Per-camera 60-second cooldown
    → Should send alert?
    ↓
    IF HIGH CONFIDENCE & NOT COOLING DOWN:
        ↓
    [Alert Publishing] - Real-time Distribution
        ├→ Redis PubSub (for WebSocket clients)
        └→ Redis Streams (for persistent log)
        ↓
    [Event Persistence] - Storage & Notification
        ├→ Capture video context
        ├→ Upload to Firebase Storage
        ├→ Save event to Firestore
        └→ Send FCM push to user devices
        ↓
    Output: User notification on smartphone
    
    ELSE:
        ↓
    → Suppress alert (already alerted recently)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      RTSP Camera Streams                         │
│  cam1 (640×480)  cam2 (640×480)  cam3 (640×480)  cam4 (640×480) │
└────────────┬────────────┬────────────┬────────────┬─────────────┘
             │            │            │            │
        ┌────▼──────┬────▼──────┬────▼──────┬────▼──────┐
        │  Camera   │  Camera   │  Camera   │  Camera   │
        │  Worker   │  Worker   │  Worker   │  Worker   │
        │   (5 FPS) │   (5 FPS) │   (5 FPS) │   (5 FPS) │
        └────┬──────┴────┬──────┴────┬──────┴────┬──────┘
             │           │           │           │
          (Raw frames with metadata)
             │           │           │           │
        ┌────▼───────────▼───────────▼───────────▼────┐
        │  Kafka Topic: processed-frames (Raw)        │
        │  (Partitioned by camera_id)                 │
        └────┬──────────────────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │   Kafka Frame Producer             │
        │  - Resize: 224×224                 │
        │  - Compress: JPEG (quality 80)     │
        │  - Create JSON message             │
        └────┬──────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  Kafka Topic: processed-frames     │
        │  (Optimized, compressed)           │
        │  (Partitioned by camera_id)        │
        └────┬──────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  Inference Consumer                │
        │  - Batch 4 frames per camera       │
        │  - Decode JPEG                     │
        │  - Run AI model (SME+STE+GTE)      │
        │  - Output: confidence scores       │
        └────┬──────────────────────────────┘
             │
        ┌────▼──────────────────────────────┐
        │  Alert Deduplication               │
        │  - Per-camera 60s cooldown (Redis) │
        │  - Decision: send alert?           │
        └────┬──────────────────────────────┘
             │
        ┌────▼────────────────────┐
        │  IF HIGH CONFIDENCE:     │
        │  ┌──────────────────┐    │
        │  │ Alert Publishing │    │
        │  │ ├─ PubSub        │    │
        │  │ └─ Streams       │    │
        │  └──────┬───────────┘    │
        │         │                │
        │  ┌──────▼────────────┐   │
        │  │ Event Persistence │   │
        │  │ ├─ Video capture  │   │
        │  │ ├─ Firebase Store │   │
        │  │ ├─ Firestore DB   │   │
        │  │ └─ FCM Notify     │   │
        │  └──────┬────────────┘   │
        │         │                │
        │  ┌──────▼────────────┐   │
        │  │  User Smartphone   │   │
        │  │  (Push Alert)      │   │
        │  └────────────────────┘   │
        └────────────────────────────┘
        
        ELSE (cooling down):
           → Suppress alert
```

---

## Key Design Principles

### 1. **Separation of Concerns**
- **Camera Worker**: Only captures, no preprocessing
- **Kafka Producer**: Handles resize/compress (single point)
- **Inference Consumer**: Only runs AI, publishes results
- **Deduplication**: Isolated smart throttling
- **Persistence**: Handles storage and notifications

### 2. **Scalability**
- **Kafka Partitioning**: Each camera → separate partition (parallel processing)
- **Multi-Camera Support**: Up to 10 concurrent streams (configurable)
- **Async Publishing**: Non-blocking alert distribution
- **Distributed Architecture**: Kafka brokers, Redis cluster ready

### 3. **Efficiency**
- **Frame Sampling**: 5 FPS reduces bandwidth 6x (30→5 FPS)
- **JPEG Compression**: 10-50x size reduction
- **Batch Inference**: 4x throughput improvement
- **Per-Camera Batching**: Maintains temporal ordering without sacrificing speed

### 4. **Reliability**
- **Alert Deduplication**: Prevents false positive storms
- **Redis Cooldown**: Automatic TTL cleanup (no stale state)
- **Metrics Tracking**: Monitor pipeline health
- **Event Logging**: Audit trail for compliance

### 5. **Real-Time Responsiveness**
- **PubSub Distribution**: <100ms latency to WebSocket clients
- **Persistent Streams**: Replay capability for missed alerts
- **FCM Notifications**: Immediate push to user devices
- **Batch Timeout**: 100ms max wait ensures freshness

---

## Performance Characteristics

| Stage | Throughput | Latency | Notes |
|-------|-----------|---------|-------|
| **RTSP Capture** | 20 frames/s (4 cameras × 5 FPS) | ~33 ms (frame time) | Single sample per camera |
| **Kafka Producer** | 20 frames/s | ~5 ms (resize+compress) | Per-frame overhead minimal |
| **Inference Consumer** | 5 batches/s (4 fps each) | ~100 ms (batch timeout) | Batching improves throughput |
| **Inference Model** | 20 fps (4-frame batches) | ~50 ms (model latency) | Depends on hardware (GPU) |
| **Alert Dedup** | 20 alerts/s possible | <1 ms (Redis check) | Per-camera decisions |
| **Event Persistence** | 4 events max/camera (cooldown) | ~500 ms (Firebase upload) | Async, non-blocking |
| **End-to-End** | 4 alerts/min/camera (max) | ~750 ms (capture→notify) | Real-time with dedup |

---

## Monitoring & Metrics

Each stage exposes metrics for observability:

### RTSP Camera Worker
```python
{
    "frames_captured": 1200,      # Total frames
    "frames_failed": 2,            # Connection errors
    "avg_latency_ms": 33.5,        # Per-frame latency
    "connection_status": "active"  # Health check
}
```

### Kafka Frame Producer
```python
{
    "frames_sent": 1198,           # Successfully sent
    "frames_failed": 2,            # Send errors
    "compression_ratio": 25.5,     # Avg size reduction (10-50x)
    "avg_latency_ms": 4.8          # Resize + compress time
}
```

### Inference Consumer
```python
{
    "frames_processed": 1198,      # Total frames
    "alerts_sent": 12,             # Total alerts
    "alerts_per_camera": {         # Per-camera breakdown
        "cam1": 3,
        "cam2": 4,
        "cam3": 3,
        "cam4": 2
    },
    "avg_batch_size": 3.8,         # Frames per batch
    "inference_latency_ms": 52.3   # Model execution
}
```

### Alert Deduplication
```python
{
    "checks_performed": 1198,      # Total checks
    "alerts_allowed": 12,          # Passed dedup
    "alerts_suppressed": 1186,     # Blocked by cooldown
    "suppression_ratio": 0.99      # 99% reduction
}
```

---

## Deployment Notes

### Local Development
```bash
# Start all services
docker-compose up

# Access backend at http://localhost:8000
# Kafka at localhost:9092
# Redis at localhost:6379
```

### Production Considerations
1. **Scale Kafka**: Multiple brokers for fault tolerance
2. **Redis Clustering**: Sentinel for high availability
3. **GPU Allocation**: Assign NVIDIA GPU to inference container
4. **Monitoring**: Prometheus + Grafana for metrics
5. **Logging**: ELK stack for centralized logs
6. **Alerting**: PagerDuty integration for system failures

---

## Related Documentation
- **AI Model Details**: See `docs/README.md` (SME, STE, GTE modules)
- **API Endpoints**: See `backend/README.md`
- **WebSocket Events**: Real-time alerts to frontend
- **Firebase Integration**: Event logging and user notifications
