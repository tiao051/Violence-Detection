"""
STE (Short Temporal Extractor) Unit Tests

Test suite for STEExtractor covering:
- Embedding shape and dtype validation
- Temporal composite creation (Equation 2 from paper)
- Deterministic inference behavior
- Processing performance (< 100ms per composite on CPU)
- Batch processing correctness

Usage:
    pytest ai_service/tests/remonet/ste/test_ste_module.py -v -s
    python -m pytest ai_service/tests/remonet/ste/test_ste_module.py
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import pytest

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.ste import STEExtractor, STEOutput, create_ste_extractor


class TestStructuralCorrectness:
    """Validate embedding shape, dtype, and output format."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste_mobilenet = create_ste_extractor(backbone='mobilenet_v2', device='cpu')
        self.ste_resnet = create_ste_extractor(backbone='resnet18', device='cpu')
        self.frame_h, self.frame_w = 224, 224

    def create_test_frames(self, seed: int = None) -> List[np.ndarray]:
        """Create 3 random RGB frames for testing."""
        if seed is not None:
            np.random.seed(seed)
        return [np.random.randint(0, 255, (self.frame_h, self.frame_w, 3), dtype=np.uint8) for _ in range(3)]
    
    def test_embedding_shape_and_type(self):
        """Verify embedding has correct shape (1280-dim for MobileNetV2) and dtype."""
        frames = self.create_test_frames(seed=42)
        embedding = self.ste_mobilenet.extract_features(frames)
        
        assert embedding.shape == (1280,), f"Expected embedding shape (1280,), got {embedding.shape}"
        assert isinstance(embedding, np.ndarray), f"Expected numpy.ndarray, got {type(embedding)}"
        assert np.issubdtype(embedding.dtype, np.floating), f"Expected float dtype, got {embedding.dtype}"
        assert not np.any(np.isnan(embedding)), "Embedding contains NaN values"
        assert not np.any(np.isinf(embedding)), "Embedding contains Inf values"
    
    def test_resnet18_embedding_shape(self):
        """Verify ResNet18 produces 512-dim embedding."""
        frames = self.create_test_frames(seed=42)
        embedding = self.ste_resnet.extract_features(frames)
        
        assert embedding.shape == (512,), f"Expected ResNet18 embedding shape (512,), got {embedding.shape}"
    
    def test_ste_output_dataclass_structure(self):
        """Verify STEOutput contains all required fields."""
        frames = self.create_test_frames(seed=42)
        output = self.ste_mobilenet.process(
            frames=frames,
            camera_id="test_cam_01",
            timestamp=1234567890.0
        )
        
        assert isinstance(output, STEOutput), f"Expected STEOutput instance, got {type(output)}"
        assert hasattr(output, 'camera_id'), "Missing camera_id field"
        assert hasattr(output, 'timestamp'), "Missing timestamp field"
        assert hasattr(output, 'embedding'), "Missing embedding field"
        assert hasattr(output, 'latency_ms'), "Missing latency_ms field"
        
        assert output.camera_id == "test_cam_01"
        assert output.timestamp == 1234567890.0
        assert isinstance(output.embedding, np.ndarray)
        assert isinstance(output.latency_ms, float)
        assert output.latency_ms > 0, "Latency should be positive"
    
    def test_json_serialization_structure(self):
        """Verify JSON output contains all necessary fields."""
        frames = self.create_test_frames(seed=42)
        output = self.ste_mobilenet.process(frames)
        json_output = self.ste_mobilenet.to_json(output)
        
        assert isinstance(json_output, dict), "JSON output should be dict"
        
        required_keys = {'camera_id', 'timestamp', 'embedding', 'latency_ms', 'embedding_shape'}
        assert required_keys.issubset(json_output.keys()), \
            f"Missing keys: {required_keys - json_output.keys()}"
        
        assert isinstance(json_output['embedding'], list), "Embedding should be serialized as list"
        assert len(json_output['embedding']) == 1280, f"Embedding list should have 1280 elements"
        
        # Verify JSON serializable
        import json
        try:
            json.dumps(json_output)
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed: {e}")


class TestTemporalFusionLogic:
    """Validate temporal composite creation according to Equation 2."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste = create_ste_extractor(backbone='mobilenet_v2', device='cpu')
        self.frame_h, self.frame_w = 224, 224

    def create_test_frames(self, seed: int = None) -> List[np.ndarray]:
        """Create 3 random RGB frames for testing."""
        if seed is not None:
            np.random.seed(seed)
        return [np.random.randint(0, 255, (self.frame_h, self.frame_w, 3), dtype=np.uint8) for _ in range(3)]
    
    def test_temporal_composite_channel_averaging(self):
        """Verify temporal composite correctly averages RGB channels per frame.
        
        Paper Equation 2: p_t = (1/3) * Σ(M_t^c) for c ∈ {R, G, B}
        """
        # Create frames with known pixel values
        frame_t = np.full((self.frame_h, self.frame_w, 3), [60, 90, 120], dtype=np.uint8)
        frame_t1 = np.full((self.frame_h, self.frame_w, 3), [30, 60, 90], dtype=np.uint8)
        frame_t2 = np.full((self.frame_h, self.frame_w, 3), [90, 120, 150], dtype=np.uint8)
        
        frames = [frame_t, frame_t1, frame_t2]
        
        # Expected grayscale values (average of RGB in uint8 space)
        expected_p_t = np.mean([60, 90, 120])      # = 90
        expected_p_t1 = np.mean([30, 60, 90])      # = 60
        expected_p_t2 = np.mean([90, 120, 150])    # = 120
        
        composite = self.ste.create_temporal_composite(frames)
        
        # Denormalize: inverse of preprocessing
        # Preprocessing: (frame_uint8 / 255.0 - mean) / std
        # Denormalize: (normalized_frame * std + mean) * 255.0
        # However, composite channels contain grayscale (averaged) normalized values
        # We need to denormalize using a uniform std/mean since we averaged across channels
        uniform_mean = np.mean(self.ste.mean)  # Average of [0.485, 0.456, 0.406]
        uniform_std = np.mean(self.ste.std)    # Average of [0.229, 0.224, 0.225]
        
        denorm = composite * uniform_std + uniform_mean
        denorm = denorm * 255.0
        
        # Check center pixel
        pixel = (112, 112)
        actual_p_t = denorm[pixel[0], pixel[1], 0]
        actual_p_t1 = denorm[pixel[0], pixel[1], 1]
        actual_p_t2 = denorm[pixel[0], pixel[1], 2]
        
        assert np.isclose(actual_p_t, expected_p_t, atol=5.0), \
            f"Channel 0: expected ~{expected_p_t}, got {actual_p_t}"
        assert np.isclose(actual_p_t1, expected_p_t1, atol=5.0), \
            f"Channel 1: expected ~{expected_p_t1}, got {actual_p_t1}"
        assert np.isclose(actual_p_t2, expected_p_t2, atol=5.0), \
            f"Channel 2: expected ~{expected_p_t2}, got {actual_p_t2}"
    
    def test_temporal_ordering_preserved(self):
        """Verify 3 frames map to 3 channels in temporal order."""
        # Create frames with increasing brightness
        frame_t = np.full((self.frame_h, self.frame_w, 3), 40, dtype=np.uint8)   # Dark
        frame_t1 = np.full((self.frame_h, self.frame_w, 3), 80, dtype=np.uint8)  # Medium
        frame_t2 = np.full((self.frame_h, self.frame_w, 3), 120, dtype=np.uint8) # Bright
        
        frames = [frame_t, frame_t1, frame_t2]
        composite = self.ste.create_temporal_composite(frames)
        
        # Denormalize: use uniform mean/std since channels are grayscale (averaged)
        uniform_mean = np.mean(self.ste.mean)
        uniform_std = np.mean(self.ste.std)
        denorm = composite * uniform_std + uniform_mean
        denorm = denorm * 255.0
        
        # Verify temporal ordering in channels
        pixel = (100, 100)
        ch0 = denorm[pixel[0], pixel[1], 0]
        ch1 = denorm[pixel[0], pixel[1], 1]
        ch2 = denorm[pixel[0], pixel[1], 2]
        
        assert ch0 < ch1 < ch2, f"Temporal ordering violated: {ch0} < {ch1} < {ch2}"
        assert np.isclose(ch0, 40, atol=5)
        assert np.isclose(ch1, 80, atol=5)
        assert np.isclose(ch2, 120, atol=5)
    
    def test_composite_output_shape(self):
        """Verify composite has shape (224, 224, 3)."""
        frames = self.create_test_frames(seed=42)
        composite = self.ste.create_temporal_composite(frames)
        
        assert composite.shape == (224, 224, 3), \
            f"Expected composite shape (224, 224, 3), got {composite.shape}"
    
    def test_frame_count_validation(self):
        """Verify exactly 3 frames are required."""
        # Too few frames
        with pytest.raises(ValueError, match="Expected 3 frames"):
            frames_too_few = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
            self.ste.create_temporal_composite(frames_too_few)
        
        # Too many frames
        with pytest.raises(ValueError, match="Expected 3 frames"):
            frames_too_many = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
            self.ste.create_temporal_composite(frames_too_many)


class TestDeterminism:
    """Validate deterministic behavior with eval mode."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste = create_ste_extractor(backbone='mobilenet_v2', device='cpu')
        self.frame_h, self.frame_w = 224, 224

    def create_test_frames(self, seed: int = None) -> List[np.ndarray]:
        """Create 3 random RGB frames for testing."""
        if seed is not None:
            np.random.seed(seed)
        return [np.random.randint(0, 255, (self.frame_h, self.frame_w, 3), dtype=np.uint8) for _ in range(3)]
    
    def test_deterministic_output(self):
        """Verify repeated inference produces identical embeddings."""
        frames = self.create_test_frames(seed=42)
        
        # Run inference twice
        embedding_1 = self.ste.extract_features(frames)
        embedding_2 = self.ste.extract_features(frames)
        
        assert np.allclose(embedding_1, embedding_2, rtol=1e-6, atol=1e-6), \
            "Embeddings should be deterministic for identical inputs"
    
    def test_deterministic_across_process_calls(self):
        """Verify determinism through full process() pipeline."""
        frames = self.create_test_frames(seed=123)
        
        output_1 = self.ste.process(frames, camera_id="test", timestamp=1000.0)
        output_2 = self.ste.process(frames, camera_id="test", timestamp=1000.0)
        
        assert np.allclose(output_1.embedding, output_2.embedding, rtol=1e-6), \
            "process() should produce deterministic embeddings"
    
    def test_model_in_eval_mode(self):
        """Verify model is in eval mode for deterministic inference."""
        assert not self.ste.model.training, "Model should be in eval() mode"


class TestPerformanceEfficiency:
    """Validate inference latency and computational efficiency."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste = create_ste_extractor(backbone='mobilenet_v2', device='cpu')
        self.frame_h, self.frame_w = 224, 224

    def create_test_frames(self, seed: int = None) -> List[np.ndarray]:
        """Create 3 random RGB frames for testing."""
        if seed is not None:
            np.random.seed(seed)
        return [np.random.randint(0, 255, (self.frame_h, self.frame_w, 3), dtype=np.uint8) for _ in range(3)]
    
    def test_latency_under_threshold(self):
        """Verify inference latency < 100ms on CPU per composite."""
        frames = self.create_test_frames(seed=42)
        
        start = time.perf_counter()
        _ = self.ste.extract_features(frames)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 100, f"Inference took {elapsed_ms:.2f}ms, expected < 100ms"
    
    def test_process_latency_recorded(self):
        """Verify process() correctly records latency."""
        frames = self.create_test_frames(seed=42)
        output = self.ste.process(frames)
        
        assert output.latency_ms > 0, "Latency should be positive"
        assert output.latency_ms < 200, f"process() latency {output.latency_ms:.2f}ms seems too high"
    
    def test_batch_processing_efficiency(self):
        """Verify batch processing efficiency vs sequential."""
        num_sequences = 5
        sequences = [self.create_test_frames(seed=i) for i in range(num_sequences)]
        
        # Batch processing
        start_batch = time.perf_counter()
        batch_results = self.ste.batch_process(sequences)
        batch_time = time.perf_counter() - start_batch
        
        # Sequential processing
        start_seq = time.perf_counter()
        seq_results = [self.ste.process(seq) for seq in sequences]
        seq_time = time.perf_counter() - start_seq
        
        assert len(batch_results) == num_sequences
        assert len(seq_results) == num_sequences
        
        # Batch should not be significantly slower
        assert batch_time < seq_time * 1.5, \
            f"Batch processing too slow: {batch_time:.3f}s vs {seq_time:.3f}s"


class TestBatchProcessing:
    """Validate batch processing correctness."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste = create_ste_extractor(backbone='mobilenet_v2', device='cpu')
        self.frame_h, self.frame_w = 224, 224

    def create_test_frames(self, seed: int = None) -> List[np.ndarray]:
        """Create 3 random RGB frames for testing."""
        if seed is not None:
            np.random.seed(seed)
        return [np.random.randint(0, 255, (self.frame_h, self.frame_w, 3), dtype=np.uint8) for _ in range(3)]
    
    def test_batch_process_output_count(self):
        """Verify batch processing returns correct number of outputs."""
        num_sequences = 7
        sequences = [self.create_test_frames(seed=i) for i in range(num_sequences)]
        
        results = self.ste.batch_process(sequences)
        
        assert len(results) == num_sequences, \
            f"Expected {num_sequences} outputs, got {len(results)}"
        assert all(isinstance(r, STEOutput) for r in results), \
            "All batch results should be STEOutput instances"
    
    def test_batch_process_with_camera_ids(self):
        """Verify batch processing preserves camera IDs."""
        sequences = [self.create_test_frames(seed=i) for i in range(3)]
        camera_ids = ["cam_lobby", "cam_parking", "cam_entrance"]
        
        results = self.ste.batch_process(sequences, camera_ids=camera_ids)
        
        assert results[0].camera_id == "cam_lobby"
        assert results[1].camera_id == "cam_parking"
        assert results[2].camera_id == "cam_entrance"
    
    def test_batch_process_with_timestamps(self):
        """Verify batch processing preserves timestamps."""
        sequences = [self.create_test_frames(seed=i) for i in range(3)]
        timestamps = [1000.0, 2000.0, 3000.0]
        
        results = self.ste.batch_process(sequences, timestamps=timestamps)
        
        assert results[0].timestamp == 1000.0
        assert results[1].timestamp == 2000.0
        assert results[2].timestamp == 3000.0
    
    def test_batch_default_camera_ids(self):
        """Verify batch processing generates default camera IDs."""
        sequences = [self.create_test_frames(seed=i) for i in range(3)]
        results = self.ste.batch_process(sequences)
        
        assert results[0].camera_id == "cam_00"
        assert results[1].camera_id == "cam_01"
        assert results[2].camera_id == "cam_02"


class TestInputValidation:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste = create_ste_extractor(backbone='mobilenet_v2', device='cpu')

    def test_empty_frame_rejection(self):
        """Verify empty/None frames are rejected."""
        with pytest.raises(ValueError, match="empty"):
            self.ste.preprocess_frame(None)
    
    def test_wrong_channel_count_rejection(self):
        """Verify non-RGB frames are rejected."""
        grayscale = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="3 channels"):
            self.ste.preprocess_frame(grayscale)
    
    def test_auto_resize_validation(self):
        """Verify frames are auto-resized to expected input size."""
        # Large frame
        large_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        preprocessed = self.ste.preprocess_frame(large_frame)
        
        assert preprocessed.shape == (224, 224, 3), \
            f"Frame should be resized to (224, 224, 3), got {preprocessed.shape}"
        
        # Small frame
        small_frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        preprocessed = self.ste.preprocess_frame(small_frame)
        
        assert preprocessed.shape == (224, 224, 3)


class TestEndToEndIntegration:
    """Test complete pipeline from frames to JSON output."""

    def setup_method(self):
        """Initialize STE extractor before each test."""
        self.ste = create_ste_extractor(backbone='mobilenet_v2', device='cpu')
        self.frame_h, self.frame_w = 224, 224

    def create_test_frames(self, seed: int = None) -> List[np.ndarray]:
        """Create 3 random RGB frames for testing."""
        if seed is not None:
            np.random.seed(seed)
        return [np.random.randint(0, 255, (self.frame_h, self.frame_w, 3), dtype=np.uint8) for _ in range(3)]
    
    def test_full_pipeline(self):
        """Verify complete pipeline: frames → embedding → JSON."""
        frames = self.create_test_frames(seed=42)
        
        # Process
        output = self.ste.process(
            frames=frames,
            camera_id="integration_test",
            timestamp=9999.0
        )
        
        # Validate output
        assert output.camera_id == "integration_test"
        assert output.timestamp == 9999.0
        assert output.embedding.shape == (1280,)
        
        # Convert to JSON
        json_output = self.ste.to_json(output)
        
        assert json_output['camera_id'] == "integration_test"
        assert json_output['timestamp'] == 9999.0
        assert len(json_output['embedding']) == 1280


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])
