"""
Inference Speed Benchmark Test

Tests real-time inference performance using actual test frames.
Measures end-to-end latency of violence detection pipeline.

Usage:
    pytest ai_service/tests/remonet/test_inference_speed.py -v -s
    python -m pytest ai_service/tests/remonet/test_inference_speed.py -v -s
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import time
import pytest

# Path resolution: ai_service/tests/remonet/test_inference_speed.py -> parents[3] = violence-detection root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig


@pytest.fixture
def test_frames_dir():
    """Get test frames directory."""
    frames_dir = PROJECT_ROOT / "ai_service" / "utils" / "test_data" / "inputs" / "frames"
    if not frames_dir.exists():
        pytest.skip(f"Test frames directory not found: {frames_dir}")
    return frames_dir


@pytest.fixture
def model_checkpoint():
    """Get trained model checkpoint path."""
    model_path = AI_SERVICE_DIR / "training" / "two-stage" / "checkpoints" / "best_model.pt"
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found: {model_path}")
    return str(model_path)


@pytest.fixture
def inference_model(model_checkpoint):
    """Initialize violence detection model (outside timing)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = InferenceConfig(
        model_path=model_checkpoint,
        device=device,
        confidence_threshold=0.5,
        num_frames=30,
        frame_size=(224, 224)
    )
    return ViolenceDetectionModel(config)


class TestInferenceSpeed:
    """Benchmark inference speed with real test frames."""
    
    def test_load_test_frames(self, test_frames_dir):
        """Verify test frames can be loaded."""
        frames = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        assert len(frames) >= 30, f"Need at least 30 frames, found {len(frames)}"
    
    def test_inference_single_batch(self, inference_model, test_frames_dir):
        """Test single inference with 30 frames - measure ONLY inference time."""
        
        # Load 30 frames
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            assert frame is not None, f"Failed to load frame: {frame_file}"
            frames.append(frame)
        
        # Add frames to buffer
        for frame in frames:
            inference_model.add_frame(frame)
        
        assert len(inference_model.frame_buffer) == 30, "Buffer should contain 30 frames"
        
        # Run inference and measure time
        start_time = time.time()
        result = inference_model.predict()
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'violence' in result and 'confidence' in result
        assert 'latency_ms' in result and 'class_id' in result
        
        # Verify result values
        assert isinstance(result['violence'], bool)
        assert 0 <= result['confidence'] <= 1
        assert result['class_id'] in [0, 1]
        assert result['latency_ms'] > 0
        
        print(f"\nInference Time: {inference_time_ms:.2f}ms")
        print(f"Model Latency: {result['latency_ms']:.2f}ms")
        print(f"Prediction: {'Violence' if result['violence'] else 'Non-Violence'} ({result['confidence']:.4f})")
    
    def test_inference_multiple_batches(self, inference_model, test_frames_dir):
        """Test multiple consecutive inferences - measure inference time only."""
        
        # Load frames
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            assert frame is not None, f"Failed to load frame: {frame_file}"
            frames.append(frame)
        
        latencies = []
        
        for batch_idx in range(3):
            # Reset buffer
            inference_model.reset_buffer()
            
            # Add frames
            for frame in frames:
                inference_model.add_frame(frame)
            
            # Run inference
            result = inference_model.predict()
            latencies.append(result['latency_ms'])
        
        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        print(f"\nLatency Statistics (3 batches × 30 frames):")
        print(f"  Mean:    {avg_latency:.2f}ms")
        print(f"  Min:     {min_latency:.2f}ms")
        print(f"  Max:     {max_latency:.2f}ms")
        print(f"  Std Dev: {std_latency:.2f}ms")
        print(f"  FPS:     {fps:.1f}")
        
        assert avg_latency < 5000, f"Average latency {avg_latency:.2f}ms is too high"
    
    def test_inference_throughput(self, inference_model, test_frames_dir):
        """Test inference throughput (frames per second)."""
        
        # Load frames
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            assert frame is not None, f"Failed to load frame: {frame_file}"
            frames.append(frame)
        
        # Reset buffer and add frames
        inference_model.reset_buffer()
        for frame in frames:
            inference_model.add_frame(frame)
        
        # Run inference
        result = inference_model.predict()
        
        latency_ms = result['latency_ms']
        fps = 1000 / latency_ms if latency_ms > 0 else 0
        
        print(f"\nThroughput Analysis (30 frames):")
        print(f"  Inference Time: {latency_ms:.2f}ms")
        print(f"  Throughput:     {fps:.2f} FPS")
        print(f"  Prediction:     {'Violence' if result['violence'] else 'Non-Violence'} ({result['confidence']:.4f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
