"""
Spark-based Distributed Inference Worker.

Distributes violence detection inference across Spark workers for parallel processing.
- Distributed frame processing using PySpark RDD
- Per-partition model loading (lazy initialization)
- Automatic result aggregation with deduplication
- Performance metrics collection
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql import SparkSession
from pyspark import RDD

logger = logging.getLogger(__name__)


@dataclass
class FrameBatch:
    """Batch of frames for processing."""
    camera_id: str
    frames: List[np.ndarray]
    timestamps: List[float]
    frame_ids: List[str]
    batch_id: str
    created_at: float


@dataclass
class InferenceResult:
    """Result from single frame inference."""
    frame_id: str
    camera_id: str
    timestamp: float
    is_violence: bool
    confidence: float
    processing_time_ms: float
    worker_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "is_violence": self.is_violence,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "worker_id": self.worker_id,
        }


class SparkInferenceWorker:
    """Distributed inference worker using PySpark."""
    
    def __init__(
        self,
        n_workers: int = 4,
        batch_size: int = 32,
        model_path: str = None,
        device: str = "cpu",
        kafka_servers: str = None,
        kafka_frame_topic: str = "frames",
        kafka_result_topic: str = "inference-results",
        redis_url: str = None,
        alert_confidence_threshold: float = 0.85,
        inference_timeout_sec: int = 30,
    ):
        """Initialize Spark inference worker."""
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.model_path = model_path
        self.device = device
        self.kafka_servers = kafka_servers
        self.kafka_frame_topic = kafka_frame_topic
        self.kafka_result_topic = kafka_result_topic
        self.redis_url = redis_url
        self.alert_confidence_threshold = alert_confidence_threshold
        self.inference_timeout_sec = inference_timeout_sec
        
        self.is_running = False
        self.spark: Optional[SparkSession] = None
        self.sc = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        self.metrics = {
            "frames_processed": 0,
            "total_processing_time_ms": 0,
            "alerts_sent": 0,
        }
    
    def start(self) -> None:
        """Start Spark session."""
        logger.info(f"Starting SparkInferenceWorker with {self.n_workers} workers")
        
        try:
            # Get event log directory from environment or use default
            event_log_dir = os.environ.get("SPARK_EVENT_LOG_DIR", "/tmp/spark-events")
            
            self.spark = SparkSession.builder \
                .appName("ViolenceDetectionInference") \
                .master(f"local[{self.n_workers}]") \
                .config("spark.dynamicAllocation.enabled", "false") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "4g") \
                .config("spark.python.worker.memory", "2g") \
                .config("spark.ui.enabled", "true") \
                .config("spark.ui.port", "4040") \
                .config("spark.driver.bindAddress", "0.0.0.0") \
                .config("spark.driver.host", "0.0.0.0") \
                .config("spark.eventLog.enabled", "true") \
                .config("spark.eventLog.dir", event_log_dir) \
                .config("spark.history.fs.logDirectory", event_log_dir) \
                .getOrCreate()
            
            self.sc = self.spark.sparkContext
            self.sc.setLogLevel("WARN")
            
            # Configure parallelism (read-only properties in newer Spark versions)
            # self.sc.defaultParallelism = self.n_workers
            # self.sc.defaultMinPartitions = self.n_workers
            
            self.is_running = True
            logger.info("SparkInferenceWorker started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start SparkInferenceWorker: {e}", exc_info=True)
            raise
    
    def stop(self) -> None:
        """Stop Spark session."""
        logger.info("Stopping SparkInferenceWorker")
        
        if self.spark:
            self.spark.stop()
        
        if self._executor:
            self._executor.shutdown()
        
        self.is_running = False
        logger.info("SparkInferenceWorker stopped")
    
    def validate_frames(self, frames: List[np.ndarray]) -> None:
        """
        Validate frames format.
        
        Args:
            frames: List of frame arrays
            
        Raises:
            ValueError: If any frame is invalid
        """
        for i, frame in enumerate(frames):
            if frame is None:
                raise ValueError(f"Frame {i} is None")
            if not isinstance(frame, np.ndarray):
                raise ValueError(f"Frame {i} is not numpy array")
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Frame {i} has invalid shape {frame.shape}, "
                    f"expected (height, width, 3)"
                )
    
    def distribute_frames(
        self,
        frames: List[np.ndarray],
        frame_ids: List[str],
        camera_id: str,
        timestamps: List[float] = None
    ) -> "RDD":
        """
        Distribute frames across Spark partitions.
        
        Args:
            frames: List of frame arrays
            frame_ids: List of frame IDs
            camera_id: Camera identifier
            timestamps: Optional timestamp for each frame
            
        Returns:
            Spark RDD with distributed frames
        """
        self.validate_frames(frames)
        
        if len(frames) != len(frame_ids):
            raise ValueError("frames and frame_ids must have same length")
        
        if timestamps is None:
            timestamps = [time.time()] * len(frames)
        
        # Create list of (frame, frame_id, camera_id, timestamp) tuples
        frame_data = [
            (frame, frame_id, camera_id, timestamp)
            for frame, frame_id, timestamp in zip(frames, frame_ids, timestamps)
        ]
        
        # Distribute across workers
        rdd = self.sc.parallelize(frame_data, numSlices=self.n_workers)
        
        logger.info(
            f"Distributed {len(frames)} frames across {self.n_workers} workers "
            f"from camera {camera_id}"
        )
        
        return rdd
    
    def infer_multi_batch(self, inputs: List[Dict]) -> List[InferenceResult]:
        """
        Run inference on multiple batches (one per camera) in parallel.
        
        Args:
            inputs: List of dicts, each containing:
                   - camera_id
                   - frames (context + new)
                   - frame_ids (new only)
                   - timestamps (new only)
                   - context_length (number of frames to skip output for)
        
        Returns:
            List of InferenceResult objects (only for new frames)
        """
        if not self.is_running:
            raise RuntimeError("Worker not started. Call start() first.")
        
        # Create RDD with 1 partition per input (camera batch)
        # This ensures each camera's batch is processed by a single worker sequentially
        rdd = self.sc.parallelize(inputs, numSlices=len(inputs))
        
        # Capture params
        model_path = self.model_path
        device = self.device
        confidence_threshold = self.alert_confidence_threshold
        
        def worker_func(partition):
            # Load model once per partition
            from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig
            
            try:
                config = InferenceConfig(
                    model_path=model_path,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )
                model = ViolenceDetectionModel(config=config)
                worker_id = os.environ.get("SPARK_EXECUTOR_ID", "unknown")
                
                # Partition contains 1 item (the input dict)
                for item in partition:
                    cam_id = item['camera_id']
                    frames = item['frames']
                    target_frame_ids = item['frame_ids']
                    target_timestamps = item['timestamps']
                    context_len = item['context_length']
                    
                    # Process all frames to build state
                    for i, frame in enumerate(frames):
                        try:
                            start_time = time.time()
                            
                            # Add frame to model buffer
                            model.add_frame(frame)
                            
                            # Only predict and yield results for NEW frames (after context)
                            if i >= context_len:
                                # Calculate index in target arrays
                                target_idx = i - context_len
                                
                                # Run prediction
                                detection = model.predict()
                                elapsed_ms = (time.time() - start_time) * 1000
                                
                                if detection:
                                    yield InferenceResult(
                                        frame_id=target_frame_ids[target_idx],
                                        camera_id=cam_id,
                                        timestamp=target_timestamps[target_idx],
                                        is_violence=detection['violence'],
                                        confidence=detection['confidence'],
                                        processing_time_ms=elapsed_ms,
                                        worker_id=worker_id
                                    )
                                    
                        except Exception as e:
                            logger.error(f"Error processing frame {i} on worker {worker_id}: {e}")
            
            except Exception as e:
                logger.error(f"Error loading model on worker: {e}", exc_info=True)
                raise

        # Run inference
        result_rdd = rdd.mapPartitions(worker_func)
        results = result_rdd.collect()
        
        # Update metrics
        self._update_metrics(results)
        
        return results

    def infer_batch(
        self,
        frames: List[np.ndarray],
        camera_id: str,
        frame_ids: List[str] = None,
        timestamps: List[float] = None
    ) -> List[InferenceResult]:
        """
        Run inference on batch of frames (distributed).
        
        Args:
            frames: List of frame arrays
            camera_id: Camera identifier
            frame_ids: Optional frame IDs
            timestamps: Optional timestamps
            
        Returns:
            List of InferenceResult objects
        """
        if not self.is_running:
            raise RuntimeError("Worker not started. Call start() first.")
        
        if frame_ids is None:
            frame_ids = [f"{camera_id}_{i}" for i in range(len(frames))]
        
        # Distribute frames
        frame_rdd = self.distribute_frames(frames, frame_ids, camera_id, timestamps)
        
        # Capture params for worker
        model_path = self.model_path
        device = self.device
        confidence_threshold = self.alert_confidence_threshold
        
        def worker_func(partition):
            # Load model once per partition (lazy initialization)
            from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig
            
            try:
                config = InferenceConfig(
                    model_path=model_path,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )
                model = ViolenceDetectionModel(config=config)
                worker_id = os.environ.get("SPARK_EXECUTOR_ID", "unknown")
                
                for frame, frame_id, cam_id, timestamp in partition:
                    try:
                        # Run inference
                        start_time = time.time()
                        detection = model.detect(frame)  # Returns (is_violence, confidence)
                        elapsed_ms = (time.time() - start_time) * 1000
                        
                        is_violence, confidence = detection
                        
                        yield InferenceResult(
                            frame_id=frame_id,
                            camera_id=cam_id,
                            timestamp=timestamp,
                            is_violence=is_violence,
                            confidence=confidence,
                            processing_time_ms=elapsed_ms,
                            worker_id=worker_id
                        )
                        
                    except Exception as e:
                        logger.error(
                            f"Error inferring frame {frame_id} on worker {worker_id}: {e}"
                        )
                        # Yield error result
                        yield InferenceResult(
                            frame_id=frame_id,
                            camera_id=cam_id,
                            timestamp=timestamp,
                            is_violence=False,
                            confidence=0.0,
                            processing_time_ms=0.0,
                            worker_id=worker_id
                        )
            
            except Exception as e:
                logger.error(f"Error loading model on worker: {e}", exc_info=True)
                raise

        # Run inference on each partition
        result_rdd = frame_rdd.mapPartitions(worker_func)
        
        # Collect results
        results = result_rdd.collect()
        
        # Aggregate and deduplicate
        aggregated = self.aggregate_results(results)
        
        # Update metrics
        self._update_metrics(aggregated)
        
        return aggregated
    
    @staticmethod
    def run_inference_on_partition(partition, camera_id: str, model_path: str, device: str):
        """
        Inference function for a single partition (runs on worker).
        
        This method runs on each Spark executor.
        Each executor loads the model once and reuses it.
        
        Args:
            partition: Iterator of (frame, frame_id, camera_id, timestamp) tuples
            camera_id: Camera identifier
            model_path: Path to model checkpoint
            device: Device to run inference on
            
        Yields:
            InferenceResult objects
        """
        # Load model once per partition (lazy initialization)
        from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig
        
        try:
            # Keep Spark executor threshold consistent with the service configuration.
            # If not set, fall back to InferenceConfig default.
            threshold_env = os.environ.get("VIOLENCE_CONFIDENCE_THRESHOLD")
            confidence_threshold = float(threshold_env) if threshold_env else None

            config = InferenceConfig(
                model_path=model_path,
                device=device,
                confidence_threshold=confidence_threshold if confidence_threshold is not None else InferenceConfig.confidence_threshold,
            )
            model = ViolenceDetectionModel(config=config)
            worker_id = os.environ.get("SPARK_EXECUTOR_ID", "unknown")
            
            for frame, frame_id, cam_id, timestamp in partition:
                try:
                    # Run inference
                    start_time = time.time()
                    detection = model.detect(frame)  # Returns (is_violence, confidence)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    is_violence, confidence = detection
                    
                    yield InferenceResult(
                        frame_id=frame_id,
                        camera_id=cam_id,
                        timestamp=timestamp,
                        is_violence=is_violence,
                        confidence=confidence,
                        processing_time_ms=elapsed_ms,
                        worker_id=worker_id
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error inferring frame {frame_id} on worker {worker_id}: {e}"
                    )
                    # Yield error result
                    yield InferenceResult(
                        frame_id=frame_id,
                        camera_id=cam_id,
                        timestamp=timestamp,
                        is_violence=False,
                        confidence=0.0,
                        processing_time_ms=0.0,
                        worker_id=worker_id
                    )
        
        except Exception as e:
            logger.error(f"Error loading model on worker: {e}", exc_info=True)
            raise

    def aggregate_results(self, results: List[InferenceResult]) -> List[InferenceResult]:
        """
        Aggregate results from all workers, with deduplication.
        
        If same frame_id appears multiple times, keep highest confidence result.
        
        Args:
            results: List of InferenceResult from workers
            
        Returns:
            Deduplicated and aggregated results
        """
        if not results:
            return []
        
        # Deduplicate by frame_id (keep highest confidence)
        seen = {}
        for result in results:
            if result.frame_id not in seen:
                seen[result.frame_id] = result
            else:
                # Keep result with higher confidence
                if result.confidence > seen[result.frame_id].confidence:
                    seen[result.frame_id] = result
        
        # Sort by frame_id to preserve order
        aggregated = sorted(
            seen.values(),
            key=lambda r: r.frame_id
        )
        
        logger.info(
            f"Aggregated {len(results)} results to {len(aggregated)} "
            f"(removed {len(results) - len(aggregated)} duplicates)"
        )
        
        return aggregated
    
    def _update_metrics(self, results: List[InferenceResult]) -> None:
        """Update metrics from results."""
        self.metrics["frames_processed"] += len(results)
        
        total_time = sum(r.processing_time_ms for r in results)
        self.metrics["total_processing_time_ms"] += total_time
        
        alerts = sum(1 for r in results if r.confidence >= self.alert_confidence_threshold)
        self.metrics["alerts_sent"] += alerts
    
    def get_metrics(
        self,
        results: List[InferenceResult] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            results: Optional list of results to analyze
            
        Returns:
            Dictionary with metrics
        """
        if results is None:
            results = []
        
        if not results:
            return {
                "total_frames": 0,
                "avg_processing_time_ms": 0,
                "min_processing_time_ms": 0,
                "max_processing_time_ms": 0,
                "frames_per_second": 0,
                "worker_distribution": {},
            }
        
        times = [r.processing_time_ms for r in results]
        
        # Worker distribution
        worker_dist = {}
        for result in results:
            worker_dist[result.worker_id] = worker_dist.get(result.worker_id, 0) + 1
        
        total_time = sum(times)
        avg_time = total_time / len(results)
        
        # Calculate FPS (assuming results processed sequentially)
        total_seconds = total_time / 1000
        fps = len(results) / total_seconds if total_seconds > 0 else 0
        
        return {
            "total_frames": len(results),
            "avg_processing_time_ms": round(avg_time, 2),
            "min_processing_time_ms": round(min(times), 2),
            "max_processing_time_ms": round(max(times), 2),
            "frames_per_second": round(fps, 2),
            "worker_distribution": worker_dist,
        }


__all__ = ["SparkInferenceWorker", "InferenceResult", "FrameBatch"]
