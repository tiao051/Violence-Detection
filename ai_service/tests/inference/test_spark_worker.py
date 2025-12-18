"""
Essential tests for Spark-based inference worker.

Only tests critical functionality:
- Worker initialization
- Frame distribution
- Result aggregation
- Error handling
"""

import pytest
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Result from model inference."""
    frame_id: str
    camera_id: str
    timestamp: float
    is_violence: bool
    confidence: float
    processing_time_ms: float
    worker_id: str


class TestSparkWorkerBasics:
    """Basic Spark worker tests."""
    
    def test_worker_initialization(self):
        """Test worker initialization with config."""
        from ai_service.inference.spark_worker import SparkInferenceWorker
        
        worker = SparkInferenceWorker(
            n_workers=4,
            batch_size=32,
            model_path="/path/to/model.pt",
            device="cuda"
        )
        
        assert worker.n_workers == 4
        assert worker.batch_size == 32
        assert worker.model_path == "/path/to/model.pt"
    
    def test_inference_result_schema(self):
        """Test InferenceResult has required fields."""
        result = InferenceResult(
            frame_id="frame_001",
            camera_id="camera_1",
            timestamp=time.time(),
            is_violence=True,
            confidence=0.95,
            processing_time_ms=25.5,
            worker_id="worker_1"
        )
        
        assert result.frame_id == "frame_001"
        assert result.camera_id == "camera_1"
        assert 0 <= result.confidence <= 1
        assert result.processing_time_ms > 0


class TestFrameProcessing:
    """Test frame processing."""
    
    def test_distribute_frames_preserves_count(self):
        """Test frame distribution preserves frame count."""
        from ai_service.inference.spark_worker import SparkInferenceWorker
        
        worker = SparkInferenceWorker(n_workers=4, batch_size=16)
        
        frames = [np.random.rand(224, 224, 3).astype(np.uint8) for _ in range(100)]
        frame_ids = [f"frame_{i}" for i in range(100)]
        
        rdd = worker.distribute_frames(frames, frame_ids, "camera_1")
        assert rdd.count() == 100
    
    def test_aggregate_results_deduplication(self):
        """Test aggregation removes duplicate results."""
        from ai_service.inference.spark_worker import SparkInferenceWorker
        
        worker = SparkInferenceWorker(n_workers=4, batch_size=8)
        
        # Same frame processed by multiple workers
        results = [
            InferenceResult(
                frame_id="frame_1",
                camera_id="camera_1",
                timestamp=1.0,
                is_violence=True,
                confidence=0.9,
                processing_time_ms=20.0,
                worker_id="worker_1"
            ),
            InferenceResult(
                frame_id="frame_1",
                camera_id="camera_1",
                timestamp=1.0,
                is_violence=False,
                confidence=0.8,
                processing_time_ms=20.0,
                worker_id="worker_2"
            ),
        ]
        
        aggregated = worker.aggregate_results(results)
        assert len(aggregated) == 1
        assert aggregated[0].confidence == 0.9  # Higher confidence wins


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_frame_raises_error(self):
        """Test invalid frames are rejected."""
        from ai_service.inference.spark_worker import SparkInferenceWorker
        
        worker = SparkInferenceWorker(n_workers=2, batch_size=4)
        
        frames = [
            np.random.rand(224, 224, 3).astype(np.uint8),
            None,  # Invalid
        ]
        
        with pytest.raises((ValueError, TypeError)):
            worker.validate_frames(frames)


class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_calculation(self):
        """Test metrics are calculated correctly."""
        from ai_service.inference.spark_worker import SparkInferenceWorker
        
        worker = SparkInferenceWorker(n_workers=4, batch_size=8)
        
        results = [
            InferenceResult(
                frame_id=f"frame_{i}",
                camera_id="camera_1",
                timestamp=float(i),
                is_violence=False,
                confidence=0.5,
                processing_time_ms=20.0 + i,
                worker_id=f"worker_{i % 4}"
            )
            for i in range(100)
        ]
        
        metrics = worker.get_metrics(results)
        
        assert metrics["total_frames"] == 100
        assert metrics["avg_processing_time_ms"] > 0
        assert metrics["frames_per_second"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
