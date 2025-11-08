"""
SME (Spatial Motion Extractor) Preprocessor Unit Tests

This module contains unit tests for the SMEPreprocessor component, verifying:
- Frame resizing to target dimensions
- Color space conversion (BGR/BGRA/Grayscale → RGB)
- Data type handling and normalization
- Processing performance and efficiency
- Deterministic and reproducible preprocessing results
- Real video frame preprocessing

Usage:
    pytest ai_service/tests/remonet/sme/test_sme_preprocessor.py -v -s
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.sme import SMEPreprocessor


@pytest.fixture
def preprocessor():
    """Returns an SMEPreprocessor instance for 224x224 RGB frames."""
    return SMEPreprocessor(target_size=(224, 224))

# Helper Functions

def create_random_bgr_frame(seed: int = 42, size=(480, 640)) -> np.ndarray:
    """Generate a random BGR frame (typical OpenCV format)."""
    np.random.seed(seed)
    return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)


def create_random_grayscale_frame(seed: int = 42, size=(480, 640)) -> np.ndarray:
    """Generate a random grayscale frame."""
    np.random.seed(seed)
    return np.random.randint(0, 256, size, dtype=np.uint8)


def create_random_bgra_frame(seed: int = 42, size=(480, 640)) -> np.ndarray:
    """Generate a random BGRA frame."""
    np.random.seed(seed)
    return np.random.randint(0, 256, (*size, 4), dtype=np.uint8)


def validate_preprocessor_output(frame: np.ndarray, target_size=(224, 224)):
    """
    Validate SMEPreprocessor output.
    
    Args:
        frame: Output frame to validate
        target_size: Expected output size (width, height)
    """
    # Shape
    assert frame.shape == (*target_size[::-1], 3), f"Expected shape {(*target_size[::-1], 3)}, got {frame.shape}"
    
    # Channels
    assert frame.shape[2] == 3, "Output should have 3 channels (RGB)"
    
    # Data type
    assert frame.dtype == np.uint8, f"Output should be uint8, got {frame.dtype}"
    
    # Value range
    assert np.all((0 <= frame) & (frame <= 255)), "Pixel values should be in [0, 255]"


class TestSMEPreprocessorFormat:

    def test_bgr_frame_conversion(self, preprocessor):
        """Test: BGR frame is converted to RGB"""
        frame = create_random_bgr_frame()
        processed = preprocessor.preprocess(frame)
        validate_preprocessor_output(processed)

    def test_grayscale_frame_conversion(self, preprocessor):
        """Test: Grayscale frame is converted to RGB"""
        frame = create_random_grayscale_frame()
        processed = preprocessor.preprocess(frame)
        validate_preprocessor_output(processed)

    def test_bgra_frame_conversion(self, preprocessor):
        """Test: BGRA frame is converted to RGB (alpha dropped)"""
        frame = create_random_bgra_frame()
        processed = preprocessor.preprocess(frame)
        validate_preprocessor_output(processed)

    def test_output_shape(self, preprocessor):
        """Test: Output has correct shape"""
        frame = create_random_bgr_frame()
        processed = preprocessor.preprocess(frame)
        assert processed.shape == (224, 224, 3)

    def test_output_dtype(self, preprocessor):
        """Test: Output is uint8"""
        frame = create_random_bgr_frame()
        processed = preprocessor.preprocess(frame)
        assert processed.dtype == np.uint8

    def test_output_channels(self, preprocessor):
        """Test: Output has 3 channels (RGB)"""
        frame = create_random_bgr_frame()
        processed = preprocessor.preprocess(frame)
        assert processed.shape[2] == 3


class TestSMEPreprocessorResizing:

    def test_resize_larger_to_target(self, preprocessor):
        """Test: Resizes from larger size to 224x224"""
        frame = create_random_bgr_frame(size=(720, 1280))
        processed = preprocessor.preprocess(frame)
        assert processed.shape == (224, 224, 3)

    def test_resize_smaller_to_target(self, preprocessor):
        """Test: Resizes from smaller size to 224x224"""
        frame = create_random_bgr_frame(size=(112, 112))
        processed = preprocessor.preprocess(frame)
        assert processed.shape == (224, 224, 3)

    def test_resize_already_correct_size(self, preprocessor):
        """Test: Handles frame already at target size"""
        frame = create_random_bgr_frame(size=(224, 224))
        processed = preprocessor.preprocess(frame)
        assert processed.shape == (224, 224, 3)

    def test_resize_maintains_content(self, preprocessor):
        """Test: Resizing maintains approximate content distribution"""
        frame = create_random_bgr_frame(size=(480, 640))
        processed = preprocessor.preprocess(frame)
        # Check that mean pixel value is roughly preserved
        original_mean = frame.mean()
        processed_mean = processed.mean()
        # Allow ~20% deviation due to interpolation
        assert abs(original_mean - processed_mean) < original_mean * 0.2


class TestSMEPreprocessorColorConversion:

    def test_bgr_to_rgb_channel_swap(self, preprocessor):
        """Test: BGR channels are properly swapped to RGB"""
        # Create frame with distinct channel values
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 50   # B channel
        frame[:, :, 1] = 100  # G channel
        frame[:, :, 2] = 150  # R channel
        
        processed = preprocessor.preprocess(frame)
        # After BGR→RGB conversion: R should be first, B last
        assert processed[0, 0, 0] == 150  # R
        assert processed[0, 0, 1] == 100  # G
        assert processed[0, 0, 2] == 50   # B

    def test_grayscale_produces_rgb(self, preprocessor):
        """Test: Grayscale becomes RGB with equal channels"""
        gray_val = 128
        frame = np.full((480, 640), gray_val, dtype=np.uint8)
        processed = preprocessor.preprocess(frame)
        # All channels should have same value
        assert processed[0, 0, 0] == processed[0, 0, 1] == processed[0, 0, 2]

    def test_bgra_alpha_channel_dropped(self, preprocessor):
        """Test: BGRA alpha channel is properly dropped"""
        frame = create_random_bgra_frame()
        processed = preprocessor.preprocess(frame)
        # Should have 3 channels, not 4
        assert processed.shape[2] == 3


class TestSMEPreprocessorPerformance:

    def test_preprocessing_speed_single_frame(self, preprocessor):
        """Test: Single frame preprocessing is fast"""
        frame = create_random_bgr_frame()
        start = time.perf_counter()
        preprocessor.preprocess(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"\n Single frame preprocessing: {elapsed_ms:.2f}ms")
        # Should be < 5ms per frame
        assert elapsed_ms < 5

    def test_preprocessing_speed_batch(self, preprocessor):
        """Test: Batch processing performance"""
        frames = np.random.randint(0, 256, (10, 480, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        preprocessor.preprocess(frames)
        elapsed_ms = (time.perf_counter() - start) * 1000
        elapsed_per_frame = elapsed_ms / 10
        print(f"\n Batch processing (10 frames): {elapsed_ms:.2f}ms total ({elapsed_per_frame:.2f}ms per frame)")
        # Should be < 5ms per frame on average
        assert elapsed_per_frame < 5

    def test_consistent_preprocessing_speed(self, preprocessor):
        """Test: Preprocessing speed is consistent"""
        timings = []
        for _ in range(10):
            frame = create_random_bgr_frame()
            start = time.perf_counter()
            preprocessor.preprocess(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)
        
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)
        print(f"\n Preprocessing speed (10 frames):")
        print(f"   Mean: {mean_time:.2f}ms")
        print(f"   Std:  {std_time:.2f}ms")
        print(f"   Min:  {min_time:.2f}ms")
        print(f"   Max:  {max_time:.2f}ms")
        
        # Mean should be consistent
        assert mean_time < 5
        # Variance should be low
        assert std_time < mean_time * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
