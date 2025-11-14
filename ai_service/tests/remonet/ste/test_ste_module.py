"""
STE (Short Temporal Extractor) Unit Tests

This module contains unit tests for the STEExtractor component, verifying:
- Temporal composite creation between consecutive frames
- Output format, shapes, data types, and value ranges
- Batch consistency and reproducibility
- Normalization correctness (critical)
- Integration with SME-like RGB frames
- Error handling and input validation

Usage:
    pytest ai_service/tests/remonet/ste/test_ste_module.py -v -s
    python -m pytest ai_service/tests/remonet/ste/test_ste_module.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.ste import STEExtractor, STEOutput

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@pytest.fixture
def ste_extractor():
    """Returns an STEExtractor instance on CPU for testing."""
    return STEExtractor(device='cpu')


def create_random_frames(count: int, seed: int = 42):
    """Generate a list of random 224x224 RGB frames."""
    np.random.seed(seed)
    return [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(count)]


def create_frame_with_rgb(rgb_values, shape=(224, 224, 3)):
    """Create a frame filled with specific RGB values."""
    return np.full(shape, rgb_values, dtype=np.uint8)


def create_frames_with_values(count: int):
    """Create frames where each has distinct uniform pixel value."""
    return [np.full((224, 224, 3), i % 255, dtype=np.uint8) for i in range(count)]


def create_invalid_shape_frame(shape, dtype=np.uint8):
    """Create a frame with invalid shape or dtype."""
    if dtype == np.float32:
        return np.random.rand(*shape).astype(np.float32)
    return np.random.randint(0, (65535 if dtype == np.uint16 else 255), shape, dtype=dtype)


def denormalize_pixels(array, y, x):
    """Get denormalized pixel values at (y, x) position."""
    denorm = (array * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return denorm[y, x, 0], denorm[y, x, 1], denorm[y, x, 2]


class TestSTEExtractorFormat:

    def test_batch_output_shapes(self, ste_extractor):
        """Test: process_batch returns feature maps with correct shape (T/3, C, W, H)"""
        frames = create_random_frames(30)
        features = ste_extractor.process_batch(frames)
        assert features.shape == (10, 1280, 7, 7)
        assert isinstance(features, torch.Tensor)

    def test_process_output_type(self, ste_extractor):
        """Test: process() returns STEOutput with feature maps per paper spec"""
        frames = create_random_frames(30)
        output = ste_extractor.process(frames, camera_id="test_cam", timestamp=123.0)
        assert isinstance(output, STEOutput)
        assert output.features.shape == (10, 1280, 7, 7)
        assert output.camera_id == "test_cam"
        assert output.timestamp == 123.0
        assert output.latency_ms > 0


class TestTemporalCompositeCorrectness:

    def setup_method(self):
        self.ste = STEExtractor(device='cpu')

    def test_rgb_averaging_before_normalization(self):
        frames = [
            create_frame_with_rgb((60, 90, 120)),
            create_frame_with_rgb((30, 60, 90)),
            create_frame_with_rgb((90, 120, 150)),
        ]
        composite = self.ste.create_temporal_composite(frames)
        ch0, ch1, ch2 = denormalize_pixels(composite, 112, 112)
        assert np.isclose(ch0, 90, atol=5)
        assert np.isclose(ch1, 60, atol=5)
        assert np.isclose(ch2, 120, atol=5)

    def test_composite_shape_and_dtype(self):
        frames = create_random_frames(3)
        composite = self.ste.create_temporal_composite(frames)
        assert composite.shape == (224, 224, 3)
        assert composite.dtype == np.float32


class TestNormalization:

    def setup_method(self):
        self.ste = STEExtractor(device='cpu')

    def test_normalization_with_extremes(self):
        for val in [0, 255]:
            frames = [np.full((224, 224, 3), val, dtype=np.uint8) for _ in range(3)]
            composite = self.ste.create_temporal_composite(frames)
            assert not np.any(np.isnan(composite))
            assert not np.any(np.isinf(composite))


class TestBatchConsistency:

    def setup_method(self):
        self.ste = STEExtractor(device='cpu')

    def test_batch_deterministic_processing(self):
        """Test: Batch processing produces consistent results across runs"""
        np.random.seed(42)
        frames_30 = create_random_frames(30)
        
        # Run 1
        features_1 = self.ste.process_batch(frames_30)
        
        # Run 2
        features_2 = self.ste.process_batch(frames_30)
        
        # Should be identical (deterministic)
        assert torch.allclose(features_1, features_2, rtol=1e-5, atol=1e-5)


class TestIntegration:

    def setup_method(self):
        self.ste = STEExtractor(device='cpu')

    def test_batch_processing_accepts_sme_like_rgb_frames(self):
        """Test: process_batch accepts SME-like RGB frames and outputs feature maps"""
        frames = create_random_frames(30)
        features = self.ste.process_batch(frames)
        assert features.shape == (10, 1280, 7, 7)
        assert isinstance(features, torch.Tensor)

    def test_does_not_modify_input_frames(self):
        """Test: Processing does not modify input frames"""
        frames = create_random_frames(30)
        frames_copy = [f.copy() for f in frames]
        _ = self.ste.process_batch(frames)
        for orig, copy in zip(frames, frames_copy):
            assert np.array_equal(orig, copy)


class TestErrorHandling:

    def setup_method(self):
        self.ste = STEExtractor(device='cpu')

    def test_reject_none_frames(self):
        with pytest.raises((ValueError, AttributeError)):
            self.ste.create_temporal_composite([None, None, None])

    def test_reject_empty_frame_list(self):
        with pytest.raises(ValueError):
            self.ste.create_temporal_composite([])

    @pytest.mark.parametrize("count,error_match", [
        (2, "Expected 3 frames"),
        (4, "Expected 3 frames"),
    ])
    def test_reject_wrong_frame_count_for_composite(self, count, error_match):
        frames = create_random_frames(count)
        with pytest.raises(ValueError, match=error_match):
            self.ste.create_temporal_composite(frames)

    @pytest.mark.parametrize("count,error_match", [
        (20, "Expected 30 frames"),
        (40, "Expected 30 frames"),
    ])
    def test_reject_wrong_batch_frame_count(self, count, error_match):
        frames = create_random_frames(count)
        with pytest.raises(ValueError, match=error_match):
            self.ste.process_batch(frames)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
