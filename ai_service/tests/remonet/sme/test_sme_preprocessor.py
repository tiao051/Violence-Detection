"""
SME (Spatial Motion Extractor) Preprocessor Unit Tests

Test suite for SMEPreprocessor covering:
- Frame resizing to target dimensions
- Color space conversion (BGR, Grayscale, RGBA to RGB)
- Batch processing efficiency
- Output format validation

Usage:
    pytest ai_service/tests/remonet/sme/test_sme_preprocessor.py -v -s
    python -m pytest ai_service/tests/remonet/sme/test_sme_preprocessor.py
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


class TestSMEPreprocessorColorConversion:
    """Test SMEPreprocessor color space conversion and resizing."""

    def setup_method(self):
        """Initialize SMEPreprocessor before each test."""
        self.preprocessor = SMEPreprocessor(target_size=(224, 224))

    def test_bgr_to_rgb_conversion(self):
        """Test BGR to RGB color space conversion."""
        # Create BGR frame
        bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 255  # B channel
        bgr_frame[:, :, 1] = 0    # G channel
        bgr_frame[:, :, 2] = 0    # R channel
        
        rgb_frame = self.preprocessor.preprocess(bgr_frame)
        
        assert rgb_frame.shape == (224, 224, 3), f"Expected (224, 224, 3), got {rgb_frame.shape}"
        assert rgb_frame.dtype == np.uint8, f"Expected uint8, got {rgb_frame.dtype}"
        # In RGB, blue should be in the last channel
        assert np.mean(rgb_frame[:, :, 2]) > 200, "Blue channel should be high in RGB"

    def test_grayscale_to_rgb_conversion(self):
        """Test grayscale to RGB conversion."""
        gray_frame = np.full((480, 640), 128, dtype=np.uint8)
        
        rgb_frame = self.preprocessor.preprocess(gray_frame)
        
        assert rgb_frame.shape == (224, 224, 3), f"Expected (224, 224, 3), got {rgb_frame.shape}"
        assert rgb_frame.dtype == np.uint8
        # All channels should be similar in grayscale
        assert np.allclose(rgb_frame[:, :, 0], rgb_frame[:, :, 1], atol=5)

    def test_frame_resizing(self):
        """Test frame resizing to target size."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        resized = self.preprocessor.preprocess(frame)
        
        assert resized.shape == (224, 224, 3), f"Expected (224, 224, 3), got {resized.shape}"

    def test_batch_frame_resizing(self):
        """Test batch frame resizing."""
        batch = np.random.randint(0, 256, (30, 480, 640, 3), dtype=np.uint8)
        
        resized_batch = self.preprocessor.preprocess(batch)
        
        assert resized_batch.shape == (30, 224, 224, 3), f"Expected (30, 224, 224, 3), got {resized_batch.shape}"
        assert resized_batch.dtype == np.uint8

    def test_output_value_range(self):
        """Verify output values are in [0, 255]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess(frame)
        
        assert np.all(processed >= 0) and np.all(processed <= 255), "Values should be in [0, 255]"

    def test_single_channel_frame(self):
        """Test single channel frame (H, W, 1)."""
        frame = np.full((480, 640, 1), 100, dtype=np.uint8)
        
        rgb_frame = self.preprocessor.preprocess(frame)
        
        assert rgb_frame.shape == (224, 224, 3), f"Expected (224, 224, 3), got {rgb_frame.shape}"

    def test_rgba_to_rgb_conversion(self):
        """Test RGBA to RGB conversion."""
        rgba_frame = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        
        rgb_frame = self.preprocessor.preprocess(rgba_frame)
        
        assert rgb_frame.shape == (224, 224, 3), f"Expected (224, 224, 3), got {rgb_frame.shape}"
        assert rgb_frame.dtype == np.uint8


class TestSMEPreprocessorPerformance:
    """Test preprocessing performance (resizing + color conversion < 5ms per frame)."""

    def setup_method(self):
        """Initialize SMEPreprocessor before each test."""
        self.preprocessor = SMEPreprocessor(target_size=(224, 224))

    def test_single_frame_preprocessing_speed(self):
        """Verify single frame preprocessing is fast."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        start = time.perf_counter()
        processed = self.preprocessor.preprocess(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 5, f"Preprocessing time {elapsed_ms:.2f}ms should be < 5ms"

    def test_batch_preprocessing_speed(self):
        """Verify batch preprocessing is efficient."""
        batch = np.random.randint(0, 256, (30, 480, 640, 3), dtype=np.uint8)
        
        start = time.perf_counter()
        processed = self.preprocessor.preprocess(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        avg_per_frame = elapsed_ms / 30
        assert avg_per_frame < 3, f"Average per-frame time {avg_per_frame:.2f}ms should be < 3ms"


class TestSMEPreprocessorReproducibility:
    """Test preprocessing result stability and reproducibility."""

    def setup_method(self):
        """Initialize SMEPreprocessor before each test."""
        self.preprocessor = SMEPreprocessor(target_size=(224, 224))

    def test_identical_inputs_identical_outputs(self):
        """Same input frames should produce identical preprocessed outputs."""
        frame = np.random.RandomState(42).randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # First preprocessing
        processed1 = self.preprocessor.preprocess(frame)
        
        # Second preprocessing with identical input
        processed2 = self.preprocessor.preprocess(frame)

        # Outputs should be identical
        assert np.array_equal(processed1, processed2), "Preprocessed frames should be identical for identical inputs"

    def test_multiple_preprocessors_same_results(self):
        """Different SMEPreprocessor instances should produce identical results."""
        frame = np.random.RandomState(42).randint(0, 256, (480, 640, 3), dtype=np.uint8)

        preprocessor1 = SMEPreprocessor(target_size=(224, 224))
        preprocessor2 = SMEPreprocessor(target_size=(224, 224))

        processed1 = preprocessor1.preprocess(frame)
        processed2 = preprocessor2.preprocess(frame)

        assert np.array_equal(processed1, processed2), "Different preprocessors should produce identical results"


class TestSMEPreprocessorRealData:
    """Test preprocessing with real video frames from utils/test_data/inputs/frames."""

    def setup_method(self):
        """Initialize SMEPreprocessor and locate frame files."""
        self.preprocessor = SMEPreprocessor(target_size=(224, 224))
        self.frames_dir = ROOT_DIR / "utils" / "test_data" / "inputs" / "frames"
        self.output_dir = ROOT_DIR / "utils" / "test_data" / "outputs" / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_frame_paths(self) -> list:
        """Get sorted list of frame files."""
        if not self.frames_dir.exists():
            pytest.skip(f"Test frames directory not found: {self.frames_dir}")
        
        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        return frame_files

    def test_real_frames_exist(self):
        """Verify test frames are available."""
        frame_files = self.get_frame_paths()
        assert len(frame_files) >= 1, f"Need at least 1 frame for testing, found {len(frame_files)}"

    def test_real_frame_preprocessing(self):
        """Test preprocessing of real video frames."""
        frame_files = self.get_frame_paths()
        
        if len(frame_files) < 1:
            pytest.skip("Insufficient test frames")
        
        # Load first frame
        frame = cv2.imread(str(frame_files[0]))
        assert frame is not None, f"Failed to load frame: {frame_files[0]}"
        
        # Preprocess
        processed = self.preprocessor.preprocess(frame)
        
        # Verify output
        assert processed.shape == (224, 224, 3), f"Expected (224, 224, 3), got {processed.shape}"
        assert processed.dtype == np.uint8
        assert np.all(processed >= 0) and np.all(processed <= 255), "Values out of range"

    def test_real_batch_preprocessing(self):
        """Test batch preprocessing of real video frames."""
        frame_files = self.get_frame_paths()
        
        if len(frame_files) < 2:
            pytest.skip("Insufficient test frames")
        
        # Load frames
        frames = []
        for frame_file in frame_files[:min(5, len(frame_files))]:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 2:
            pytest.skip("Could not load enough real frames")
        
        batch = np.array(frames, dtype=np.uint8)
        
        # Preprocess batch
        processed_batch = self.preprocessor.preprocess(batch)
        
        # Verify output
        assert processed_batch.shape[0] == len(frames), f"Batch size mismatch"
        assert processed_batch.shape[1:] == (224, 224, 3), f"Expected (224, 224, 3) per frame"
        assert processed_batch.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
