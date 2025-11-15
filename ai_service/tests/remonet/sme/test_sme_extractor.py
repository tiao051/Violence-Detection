"""
SME (Spatial Motion Extractor) Unit Tests

This module contains unit tests for the SMEExtractor component, verifying:
- Motion detection between consecutive frames
- Output format, shapes, data types, and value ranges
- Processing performance and elapsed time constraints (< 5ms per frame pair)
- Deterministic and reproducible results
- Real video frame processing and visualization

Usage:
    pytest ai_service/tests/remonet/sme/test_sme_extractor.py -v -s
    python -m pytest ai_service/tests/remonet/sme/test_sme_extractor.py
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

from ai_service.remonet.sme import SMEExtractor, SMEPreprocessor


@pytest.fixture
def extractor():
    """Returns an SMEExtractor instance with standard configuration."""
    return SMEExtractor(kernel_size=3, iteration=2)


@pytest.fixture
def preprocessor():
    """Returns an SMEPreprocessor instance for 224x224 RGB frames."""
    return SMEPreprocessor(target_size=(224, 224))


def create_random_pair(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pair of random 224x224 RGB frames."""
    np.random.seed(seed)
    frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return frame_t, frame_t1


def create_identical_pair() -> tuple[np.ndarray, np.ndarray]:
    """Generate a pair of identical 224x224 RGB frames (no motion)."""
    frame = np.full((224, 224, 3), 128, dtype=np.uint8)
    return frame, frame.copy()


def create_pair_with_motion_region(value_before: int = 100,
                                   value_after: int = 200) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Generate a pair with a synthetic motion region in the center."""
    frame_t = np.full((224, 224, 3), value_before, dtype=np.uint8)
    frame_t1 = frame_t.copy()
    y_start, y_end, x_start, x_end = 56, 168, 56, 168
    frame_t1[y_start:y_end, x_start:x_end] = value_after
    return frame_t, frame_t1, (y_start, y_end, x_start, x_end)


def create_pair_with_corner_motion(value_before: int = 50, value_after: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pair with motion in the top-left corner."""
    frame_t = np.full((224, 224, 3), value_before, dtype=np.uint8)
    frame_t1 = frame_t.copy()
    frame_t1[0:56, 0:56] = value_after
    return frame_t, frame_t1


def validate_sme_output(roi: np.ndarray, mask: np.ndarray, diff: np.ndarray, 
                       elapsed_ms: float = None, max_elapsed_ms: float = None):
    """
    Validate SMEExtractor outputs.

    Args:
        roi: Region of interest frame (RGB, float32 [0, 1])
        mask: Motion mask (grayscale, uint8)
        diff: Difference frame (grayscale, uint8)
        elapsed_ms: Actual processing time in milliseconds
        max_elapsed_ms: Optional maximum allowed processing time
    """
    # Shapes
    assert roi.shape[:2] == (224, 224)
    assert mask.shape[:2] == (224, 224)
    assert diff.shape[:2] == (224, 224)
    
    # Channels
    assert roi.shape[2] == 3
    assert len(mask.shape) == 2
    assert len(diff.shape) == 2
    
    # Data type: ROI is now float32 [0, 1], mask/diff still uint8
    assert roi.dtype == np.float32
    assert mask.dtype == np.uint8
    assert diff.dtype == np.uint8
    
    # Value ranges: ROI is [0, 1], mask/diff are [0, 255]
    assert np.all((0 <= roi) & (roi <= 1))
    assert np.all((0 <= mask) & (mask <= 255))
    assert np.all((0 <= diff) & (diff <= 255))
    
    # Optional elapsed time
    if max_elapsed_ms is not None and elapsed_ms is not None:
        assert elapsed_ms <= max_elapsed_ms
def load_first_frame_pair(frame_files: list) -> tuple[np.ndarray, np.ndarray]:
    """Load the first two frames from a list of file paths."""
    frame_t = cv2.imread(str(frame_files[0]))
    frame_t1 = cv2.imread(str(frame_files[1]))
    assert frame_t is not None and frame_t1 is not None
    return frame_t, frame_t1


class TestSMEExtractorFormat:

    def test_output_shapes(self, extractor):
        frame_t, frame_t1 = create_random_pair()
        roi, mask, diff, elapsed_ms = extractor.process(frame_t, frame_t1)
        validate_sme_output(roi, mask, diff)

    def test_output_channels_and_types(self, extractor):
        frame_t, frame_t1 = create_random_pair()
        roi, mask, diff, elapsed_ms = extractor.process(frame_t, frame_t1)
        validate_sme_output(roi, mask, diff)

    def test_elapsed_time_numeric(self, extractor):
        frame_t, frame_t1 = create_random_pair()
        roi, mask, diff, elapsed_ms = extractor.process(frame_t, frame_t1)
        assert isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0


class TestSMEMotionDetection:

    def test_no_motion_for_identical_frames(self, extractor):
        frame_t, frame_t1 = create_identical_pair()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        assert np.all(diff == 0)
        assert np.count_nonzero(mask) / mask.size < 0.01

    def test_detects_synthetic_center_motion(self, extractor):
        frame_t, frame_t1, (y0, y1, x0, x1) = create_pair_with_motion_region()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        # diff is still uint8, test it
        motion_mean = np.mean(diff[y0:y1, x0:x1])
        non_motion_mean = np.mean(np.concatenate([diff[:y0, :].flatten(), diff[y1:, :].flatten()]))
        assert motion_mean > non_motion_mean * 1.5

    def test_mask_highlights_corner_motion(self, extractor):
        frame_t, frame_t1 = create_pair_with_corner_motion()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        top_left_mask = mask[0:56, 0:56]
        assert np.count_nonzero(top_left_mask) > 100

    def test_roi_preserves_motion_regions(self, extractor):
        frame_t, frame_t1, (y0, y1, x0, x1) = create_pair_with_motion_region()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        # ROI is now float [0, 1]: multiply by mask and divide by 255
        # Motion region should have higher mean than static region
        motion_roi = roi[y0:y1, x0:x1]
        static_roi = np.concatenate([roi[:y0, :], roi[y1:, :]])
        
        motion_mean = np.mean(motion_roi[motion_roi > 0])  # Only compare non-zero regions
        static_mean = np.mean(static_roi[static_roi > 0])
        
        # Motion region should have preserved higher values
        assert motion_mean > static_mean if static_mean > 0 else True


class TestSMEExtractorPerformance:

    def test_processing_speed_under_5ms(self, extractor):
        frame_t, frame_t1 = create_random_pair()
        roi, mask, diff, elapsed_ms = extractor.process(frame_t, frame_t1)
        assert elapsed_ms < 5

    def test_consistent_performance(self, extractor):
        timings = []
        for _ in range(10):
            frame_t, frame_t1 = create_random_pair()
            _, _, _, elapsed_ms = extractor.process(frame_t, frame_t1)
            timings.append(elapsed_ms)
        assert np.mean(timings) < 5
        assert np.max(timings) < 8

    def test_timing_stability(self, extractor):
        timings = []
        for _ in range(20):
            frame_t, frame_t1 = create_random_pair()
            _, _, _, elapsed_ms = extractor.process(frame_t, frame_t1)
            timings.append(elapsed_ms)
        std_dev = np.std(timings)
        mean_time = np.mean(timings)
        assert std_dev < mean_time * 0.3


class TestSMEExtractorReproducibility:

    def test_identical_inputs_produce_identical_outputs(self, extractor):
        frame_t, frame_t1 = create_random_pair()
        roi1, mask1, diff1, _ = extractor.process(frame_t, frame_t1)
        roi2, mask2, diff2, _ = extractor.process(frame_t, frame_t1)
        assert np.array_equal(roi1, roi2)
        assert np.array_equal(mask1, mask2)
        assert np.array_equal(diff1, diff2)

    def test_multiple_extractors_same_results(self):
        frame_t, frame_t1 = create_random_pair()
        ext1 = SMEExtractor(kernel_size=3, iteration=2)
        ext2 = SMEExtractor(kernel_size=3, iteration=2)
        roi1, mask1, diff1, _ = ext1.process(frame_t, frame_t1)
        roi2, mask2, diff2, _ = ext2.process(frame_t, frame_t1)
        assert np.array_equal(roi2, roi2)
        assert np.array_equal(mask1, mask2)
        assert np.array_equal(diff1, diff2)


class TestSMEExtractorBatch:
    """Test batch processing of multiple frames."""

    def test_batch_output_shape(self):
        extractor = SMEExtractor(kernel_size=3, iteration=2)
        # Create 30 random frames
        frames = np.random.randint(0, 256, (30, 224, 224, 3), dtype=np.uint8)
        
        motion_frames = extractor.process_batch(frames)
        
        # Should output same number of frames as input (30)
        assert motion_frames.shape == (30, 224, 224, 3)
        assert motion_frames.dtype == np.float32

    def test_batch_30_frames(self):
        extractor = SMEExtractor(kernel_size=3, iteration=2)
        frames = np.random.randint(0, 256, (30, 224, 224, 3), dtype=np.uint8)
        
        motion_frames = extractor.process_batch(frames)
        
        # 30 input → 30 output (29 from pairs + 1 duplicated)
        assert len(motion_frames) == 30
        # Last motion frame should be duplicate of 29th
        assert np.array_equal(motion_frames[-1], motion_frames[-2])

    def test_batch_arbitrary_length(self):
        extractor = SMEExtractor(kernel_size=3, iteration=2)
        for num_frames in [2, 10, 15, 30, 50]:
            frames = np.random.randint(0, 256, (num_frames, 224, 224, 3), dtype=np.uint8)
            motion_frames = extractor.process_batch(frames)
            
            # Output should match input count
            assert len(motion_frames) == num_frames
            assert motion_frames.dtype == np.float32
            # All values in [0, 1]
            assert np.all((0 <= motion_frames) & (motion_frames <= 1))

    def test_batch_value_range(self):
        extractor = SMEExtractor(kernel_size=3, iteration=2)
        frames = np.random.randint(0, 256, (30, 224, 224, 3), dtype=np.uint8)
        
        motion_frames = extractor.process_batch(frames)
        
        # All values should be in [0, 1]
        assert np.all(motion_frames >= 0)
        assert np.all(motion_frames <= 1)

    def test_batch_insufficient_frames(self):
        extractor = SMEExtractor(kernel_size=3, iteration=2)
        frames = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Expected at least 2 frames"):
            extractor.process_batch(frames)

    def test_batch_consistency_with_process(self):
        """Verify batch processing produces same results as individual processing."""
        extractor = SMEExtractor(kernel_size=3, iteration=2)
        frames = np.random.randint(0, 256, (5, 224, 224, 3), dtype=np.uint8)
        
        # Process individually
        individual_rois = []
        for i in range(len(frames) - 1):
            roi, _, _, _ = extractor.process(frames[i], frames[i + 1])
            individual_rois.append(roi)
        individual_rois.append(individual_rois[-1].copy())  # Duplicate last
        individual_array = np.array(individual_rois, dtype=np.float32)
        
        # Process batch
        batch_array = extractor.process_batch(frames)
        
        # Should be identical
        assert np.allclose(batch_array, individual_array, rtol=1e-5, atol=1e-7)


class TestSMEExtractorRealData:

    def setup_method(self):
        self.preprocessor = SMEPreprocessor(target_size=(224, 224))
        self.extractor = SMEExtractor(kernel_size=3, iteration=2)
        self.frames_dir = ROOT_DIR / "utils" / "test_data" / "inputs" / "frames"
        self.output_dir = ROOT_DIR / "utils" / "test_data" / "outputs" / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_frame_paths(self) -> list:
        if not self.frames_dir.exists():
            pytest.skip(f"No test frames found at {self.frames_dir}")
        return sorted(self.frames_dir.glob("frame_*.jpg"))

    def save_visualization(self, frame_t, frame_t1, roi, mask, diff):
        """Save SMEExtractor processing results as a single visualization image."""
        processed_t = self.preprocessor.preprocess(frame_t)
        processed_t1 = self.preprocessor.preprocess(frame_t1)
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        diff_3ch = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        grid = np.hstack([processed_t, processed_t1, diff_3ch, mask_3ch, roi_bgr])
        cv2.imwrite(str(self.output_dir / "sme_output_visualization.jpg"), grid)

    def test_real_frames_processing(self):
        frame_files = self.get_frame_paths()
        if len(frame_files) < 2:
            pytest.skip("Insufficient frames")

        frame_t, frame_t1 = load_first_frame_pair(frame_files)
        frame_t_processed = self.preprocessor.preprocess(frame_t)
        frame_t1_processed = self.preprocessor.preprocess(frame_t1)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t_processed, frame_t1_processed)
        validate_sme_output(roi, mask, diff, elapsed_ms=elapsed_ms, max_elapsed_ms=5)
        self.save_visualization(frame_t, frame_t1, roi, mask, diff)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

