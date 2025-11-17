"""
Inference Speed Benchmark Test

Tests real-time inference performance using actual test frames.
Measures end-to-end latency of violence detection pipeline.

Usage:
    pytest ai_service/tests/test_inference_speed.py -v -s
    python ai_service/tests/test_inference_speed.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import time
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# violence-detection root directory
# parents[2] from ai_service/tests/test_inference_speed.py = violence-detection root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ai_service directory
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig


@pytest.fixture
def test_frames_dir():
    """Get test frames directory."""
    frames_dir = PROJECT_ROOT / "utils" / "test_data" / "inputs" / "frames"
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
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = InferenceConfig(
        model_path=model_checkpoint,
        device=device,
        confidence_threshold=0.5,
        num_frames=30,
        frame_size=(224, 224)
    )
    # Initialize model BEFORE timing
    model = ViolenceDetectionModel(config)
    print(f"\n✅ Model initialized on {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    return model


class TestInferenceSpeed:
    """Benchmark inference speed with real test frames."""
    
    def test_load_test_frames(self, test_frames_dir):
        """Verify test frames can be loaded."""
        frames = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        
        assert len(frames) >= 30, f"Need at least 30 frames, found {len(frames)}"
        print(f"\n✅ Found {len(frames)} test frames in {test_frames_dir}")
    
    def test_inference_single_batch(self, inference_model, test_frames_dir):
        """Test single inference with 30 frames - measure ONLY inference time."""
        
        # Load 30 frames (outside timing)
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            assert frame is not None, f"Failed to load frame: {frame_file}"
            frames.append(frame)
        
        print(f"\n📸 Loaded {len(frames)} frames (outside timing)")
        print(f"   Frame shape: {frames[0].shape}")
        
        # Add frames to buffer (outside timing)
        for frame in frames:
            inference_model.add_frame(frame)
        
        assert len(inference_model.frame_buffer) == 30, "Buffer should contain 30 frames"
        print(f"✅ Frame buffer filled: {len(inference_model.frame_buffer)} frames")
        
        # ⏱️ START TIMING - INFERENCE ONLY ⏱️
        start_time = time.time()
        result = inference_model.predict()
        inference_time_ms = (time.time() - start_time) * 1000
        # ⏱️ END TIMING ⏱️
        
        # Verify result
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'violence' in result, "Result should contain 'violence' key"
        assert 'confidence' in result, "Result should contain 'confidence' key"
        assert 'latency_ms' in result, "Result should contain 'latency_ms' key"
        assert 'class_id' in result, "Result should contain 'class_id' key"
        
        # Check values
        assert isinstance(result['violence'], bool), "violence should be bool"
        assert 0 <= result['confidence'] <= 1, "confidence should be in [0, 1]"
        assert result['class_id'] in [0, 1], "class_id should be 0 or 1"
        assert result['latency_ms'] > 0, "latency_ms should be > 0"
        
        print(f"\n{'='*60}")
        print(f"📊 INFERENCE SPEED BENCHMARK (30 frames)")
        print(f"{'='*60}")
        print(f"Total Inference Time: {inference_time_ms:.2f}ms")
        print(f"Model Measured Time: {result['latency_ms']:.2f}ms")
        print(f"\n📈 RESULTS:")
        print(f"  Violence Detected: {result['violence']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Class: {'Violence' if result['class_id'] == 0 else 'Non-Violence'}")
        print(f"{'='*60}")
    
    def test_inference_multiple_batches(self, inference_model, test_frames_dir):
        """Test multiple consecutive inferences - measure inference time only."""
        
        # Load frames (outside timing)
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            assert frame is not None, f"Failed to load frame: {frame_file}"
            frames.append(frame)
        
        print(f"\n📸 Running {3} consecutive inference batches (30 frames each)...\n")
        
        latencies = []
        results = []
        
        for batch_idx in range(3):
            # Reset buffer (outside timing)
            inference_model.reset_buffer()
            
            # Add frames (outside timing)
            for frame in frames:
                inference_model.add_frame(frame)
            
            # ⏱️ INFERENCE ONLY ⏱️
            result = inference_model.predict()
            latencies.append(result['latency_ms'])
            results.append(result)
            
            print(f"Batch {batch_idx + 1}:")
            print(f"  Inference Time: {result['latency_ms']:.2f}ms")
            print(f"  Violence: {result['violence']} | Confidence: {result['confidence']:.4f}")
        
        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 INFERENCE SPEED STATISTICS (3 batches × 30 frames)")
        print(f"{'='*60}")
        print(f"Mean Latency:  {avg_latency:.2f}ms")
        print(f"Min Latency:   {min_latency:.2f}ms")
        print(f"Max Latency:   {max_latency:.2f}ms")
        print(f"Std Dev:       {std_latency:.2f}ms")
        print(f"FPS Capacity:  {fps:.1f} FPS")
        print(f"{'='*60}")
        
        # Performance check
        assert avg_latency < 5000, f"Average latency {avg_latency:.2f}ms is too high (should be < 5000ms)"
        print(f"✅ Inference speed is acceptable")
    
    def test_inference_throughput(self, inference_model, test_frames_dir):
        """Test inference throughput (frames per second)."""
        
        # Load frames (outside timing)
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            assert frame is not None, f"Failed to load frame: {frame_file}"
            frames.append(frame)
        
        print(f"\n📸 Testing inference throughput (30 frames)...\n")
        
        # Reset buffer (outside timing)
        inference_model.reset_buffer()
        
        # Add frames (outside timing)
        for frame in frames:
            inference_model.add_frame(frame)
        
        # ⏱️ INFERENCE ONLY ⏱️
        result = inference_model.predict()
        
        # Calculate throughput
        latency_ms = result['latency_ms']
        fps = 1000 / latency_ms if latency_ms > 0 else 0
        ms_per_batch = latency_ms
        frames_per_second = fps
        
        print(f"{'='*60}")
        print(f"📊 THROUGHPUT ANALYSIS (30 frames)")
        print(f"{'='*60}")
        print(f"Inference Time:  {latency_ms:.2f}ms")
        print(f"Throughput:      {frames_per_second:.1f} FPS")
        print(f"Result:          {result['violence']} (confidence: {result['confidence']:.4f})")
        print(f"{'='*60}")
        
        print(f"✅ Throughput test complete")


class TestInferenceDevice:
    """Test inference on different devices."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inference_cuda(self, test_frames_dir):
        """Test inference on CUDA device."""
        
        model_path = AI_SERVICE_DIR / "training" / "two-stage" / "checkpoints" / "best_model.pt"
        
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        
        config = InferenceConfig(
            model_path=str(model_path),
            device='cuda',
            confidence_threshold=0.5,
            num_frames=30,
            frame_size=(224, 224)
        )
        model = ViolenceDetectionModel(config)
        
        # Load frames
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 30:
            pytest.skip(f"Not enough frames: {len(frames)}")
        
        # Add frames and run inference
        for frame in frames:
            model.add_frame(frame)
        
        result = model.predict()
        
        print(f"\n🚀 CUDA Inference:")
        print(f"   Device: cuda")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Confidence: {result['confidence']:.4f}")
        
        assert result['latency_ms'] > 0, "Inference should measure latency"
        print(f"✅ CUDA inference successful")
    
    def test_inference_cpu(self, test_frames_dir):
        """Test inference on CPU device."""
        
        model_path = AI_SERVICE_DIR / "training" / "two-stage" / "checkpoints" / "best_model.pt"
        
        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")
        
        config = InferenceConfig(
            model_path=str(model_path),
            device='cpu',
            confidence_threshold=0.5,
            num_frames=30,
            frame_size=(224, 224)
        )
        model = ViolenceDetectionModel(config)
        
        # Load frames
        frame_files = sorted(test_frames_dir.glob("*.jpg")) + sorted(test_frames_dir.glob("*.png"))
        frame_files = frame_files[:30]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 30:
            pytest.skip(f"Not enough frames: {len(frames)}")
        
        # Add frames and run inference
        for frame in frames:
            model.add_frame(frame)
        
        result = model.predict()
        
        print(f"\n💻 CPU Inference:")
        print(f"   Device: cpu")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Confidence: {result['confidence']:.4f}")
        
        assert result['latency_ms'] > 0, "Inference should measure latency"
        print(f"✅ CPU inference successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
