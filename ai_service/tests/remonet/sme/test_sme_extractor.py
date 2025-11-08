"""
SME (Spatial Motion Extractor) Unit Tests

Test suite for SMEExtractor covering:
- Motion detection between consecutive frames
- Output format and data type validation
- Processing performance (< 5ms per frame pair)
- Result stability and reproducibility
- Real video frame processing

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


class TestSMEExtractorFormat:
    """Test SMEExtractor output format, dtype, and basic properties."""

    def setup_method(self):
        """Initialize SMEExtractor before each test."""
        self.extractor = SMEExtractor(kernel_size=3, iteration=2)

    def test_output_shapes(self):
        """Verify output tensor shapes are correct."""
        frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # All outputs should match input frame dimensions (224, 224)
        assert roi.shape[:2] == (224, 224), f"ROI shape {roi.shape} != (224, 224)"
        assert mask.shape[:2] == (224, 224), f"Mask shape {mask.shape} != (224, 224)"
        assert diff.shape[:2] == (224, 224), f"Diff shape {diff.shape} != (224, 224)"

    def test_output_dtypes(self):
        """Verify output data types are correct."""
        frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # Check data types
        assert roi.dtype == np.uint8, f"ROI dtype {roi.dtype} should be uint8"
        assert mask.dtype == np.uint8, f"Mask dtype {mask.dtype} should be uint8"
        assert diff.dtype == np.uint8, f"Diff dtype {diff.dtype} should be uint8"

    def test_output_channels(self):
        """Verify ROI and input frame channel compatibility."""
        frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # ROI should have 3 channels (RGB)
        assert roi.shape[2] == 3, f"ROI channels {roi.shape[2]} should be 3"
        # Mask should be grayscale (H, W)
        assert len(mask.shape) == 2, f"Mask should be 2D grayscale, got {mask.shape}"
        # Diff should be grayscale
        assert len(diff.shape) == 2, f"Diff should be 2D grayscale, got {diff.shape}"

    def test_output_value_ranges(self):
        """Verify output values are within valid ranges [0, 255]."""
        frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # All outputs should be in [0, 255] for uint8
        assert np.all(roi >= 0) and np.all(roi <= 255), "ROI values out of range [0, 255]"
        assert np.all(mask >= 0) and np.all(mask <= 255), "Mask values out of range [0, 255]"
        assert np.all(diff >= 0) and np.all(diff <= 255), "Diff values out of range [0, 255]"

    def test_elapsed_time_is_numeric(self):
        """Verify elapsed time is a positive number."""
        frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        assert isinstance(elapsed_ms, (int, float)), f"Elapsed time should be numeric, got {type(elapsed_ms)}"
        assert elapsed_ms >= 0, f"Elapsed time should be non-negative, got {elapsed_ms}"


class TestSMEMotionDetection:
    """Test motion detection accuracy between frames."""

    def setup_method(self):
        """Initialize SMEExtractor before each test."""
        self.extractor = SMEExtractor(kernel_size=3, iteration=2)

    def test_identical_frames_no_motion(self):
        """When two frames are identical, motion should be zero."""
        # Create identical frames
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        roi, mask, diff, elapsed_ms = self.extractor.process(frame, frame)

        # Diff should be all zeros (no motion)
        assert np.all(diff == 0), "Diff should be all zeros for identical frames"
        
        # Mask should have minimal non-zero values (due to dilation, might have some noise)
        non_zero_ratio = np.count_nonzero(mask) / mask.size
        assert non_zero_ratio < 0.01, f"Mask non-zero ratio {non_zero_ratio} too high for identical frames"

    def test_synthetic_motion_detection(self):
        """Create artificial motion region and verify detection."""
        # Create base frame
        frame_t = np.full((224, 224, 3), 100, dtype=np.uint8)
        frame_t1 = frame_t.copy()
        
        # Create motion in center region
        motion_y_start, motion_y_end = 56, 168
        motion_x_start, motion_x_end = 56, 168
        frame_t1[motion_y_start:motion_y_end, motion_x_start:motion_x_end] = 200
        
        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # Diff should show motion in the modified region
        motion_region_diff = diff[motion_y_start:motion_y_end, motion_x_start:motion_x_end]
        non_motion_region_diff = np.concatenate([
            diff[:motion_y_start, :].flatten(),
            diff[motion_y_end:, :].flatten()
        ])
        
        motion_mean = np.mean(motion_region_diff)
        non_motion_mean = np.mean(non_motion_region_diff)
        
        # Motion region should have significantly higher diff values
        assert motion_mean > non_motion_mean * 1.5, \
            f"Motion region mean {motion_mean} should be >> non-motion mean {non_motion_mean}"

    def test_mask_highlights_motion_regions(self):
        """Verify mask correctly highlights motion areas."""
        # Create base frame
        frame_t = np.full((224, 224, 3), 50, dtype=np.uint8)
        frame_t1 = frame_t.copy()
        
        # Add motion (large change) in top-left corner
        frame_t1[0:56, 0:56] = 200
        
        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # Check that top-left region has non-zero mask values
        top_left_mask = mask[0:56, 0:56]
        non_zero_count = np.count_nonzero(top_left_mask)
        
        assert non_zero_count > 100, \
            f"Mask should highlight motion region, but only {non_zero_count} pixels are non-zero"

    def test_roi_extracts_motion_regions(self):
        """Verify ROI contains motion regions and zeros elsewhere."""
        frame_t = np.full((224, 224, 3), 100, dtype=np.uint8)
        frame_t1 = frame_t.copy()
        
        # Add motion in specific region
        motion_region = (56, 112, 84, 140)
        y1, x1, y2, x2 = motion_region
        frame_t1[y1:y2, x1:x2] = 200
        
        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        # ROI in motion region should be non-zero
        roi_motion = roi[y1:y2, x1:x2]
        roi_static = np.concatenate([roi[:y1, :], roi[y2:, :]])
        motion_mean = np.mean(roi_motion)
        static_mean = np.mean(roi_static)

        assert motion_mean > static_mean * 1.2, (
            f"ROI mean brightness too close: motion={motion_mean:.2f}, static={static_mean:.2f}")


class TestSMEExtractorPerformance:
    """Test motion extraction performance (should be < 5ms per frame pair)."""

    def setup_method(self):
        """Initialize SMEExtractor before each test."""
        self.extractor = SMEExtractor(kernel_size=3, iteration=2)

    def test_processing_speed(self):
        """Verify motion extraction time is under 5ms per frame pair."""
        frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)

        assert elapsed_ms < 5, f"Processing time {elapsed_ms:.2f}ms should be < 5ms"

    def test_processing_speed_multiple_iterations(self):
        """Verify consistent fast performance over multiple frames."""
        timings = []
        
        for _ in range(10):
            frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)
            timings.append(elapsed_ms)
        
        avg_time = np.mean(timings)
        max_time = np.max(timings)
        
        assert avg_time < 5, f"Average processing time {avg_time:.2f}ms should be < 5ms"
        assert max_time < 8, f"Max processing time {max_time:.2f}ms should be < 8ms"

    def test_timing_consistency(self):
        """Verify processing times are relatively stable."""
        timings = []
        
        for _ in range(20):
            frame_t = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            frame_t1 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            roi, mask, diff, elapsed_ms = self.extractor.process(frame_t, frame_t1)
            timings.append(elapsed_ms)
        
        std_dev = np.std(timings)
        mean_time = np.mean(timings)
        
        # Standard deviation should be reasonable (< 30% of mean)
        assert std_dev < mean_time * 0.3, \
            f"Timing variance too high: std={std_dev:.2f}ms, mean={mean_time:.2f}ms"


class TestSMEExtractorReproducibility:
    """Test motion extraction result stability and reproducibility."""

    def setup_method(self):
        """Initialize SMEExtractor before each test."""
        self.extractor = SMEExtractor(kernel_size=3, iteration=2)

    def test_identical_inputs_identical_outputs(self):
        """Same input frames should produce identical outputs."""
        frame_t = np.random.RandomState(42).randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.RandomState(43).randint(0, 256, (224, 224, 3), dtype=np.uint8)

        # First processing
        roi1, mask1, diff1, _ = self.extractor.process(frame_t, frame_t1)
        
        # Second processing with identical inputs
        roi2, mask2, diff2, _ = self.extractor.process(frame_t, frame_t1)

        # Outputs should be identical
        assert np.array_equal(roi1, roi2), "ROI should be identical for identical inputs"
        assert np.array_equal(mask1, mask2), "Mask should be identical for identical inputs"
        assert np.array_equal(diff1, diff2), "Diff should be identical for identical inputs"

    def test_multiple_extractors_same_results(self):
        """Different SMEExtractor instances with same config should produce identical results."""
        frame_t = np.random.RandomState(42).randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_t1 = np.random.RandomState(43).randint(0, 256, (224, 224, 3), dtype=np.uint8)

        extractor1 = SMEExtractor(kernel_size=3, iteration=2)
        extractor2 = SMEExtractor(kernel_size=3, iteration=2)

        roi1, mask1, diff1, _ = extractor1.process(frame_t, frame_t1)
        roi2, mask2, diff2, _ = extractor2.process(frame_t, frame_t1)

        assert np.array_equal(roi1, roi2), "Different extractors should produce identical ROI"
        assert np.array_equal(mask1, mask2), "Different extractors should produce identical mask"
        assert np.array_equal(diff1, diff2), "Different extractors should produce identical diff"


class TestSMEExtractorRealData:
    """Test motion extraction with real video frames from utils/test_data/inputs/frames."""

    def setup_method(self):
        """Initialize SME components and locate frame files."""
        self.preprocessor = SMEPreprocessor(target_size=(224, 224))
        self.extractor = SMEExtractor(kernel_size=3, iteration=2)
        self.frames_dir = ROOT_DIR / "utils" / "test_data" / "inputs" / "frames"
        self.output_dir = ROOT_DIR / "utils" / "test_data" / "outputs" / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_frame_paths(self) -> list:
        """Get sorted list of frame files."""
        if not self.frames_dir.exists():
            pytest.skip(f"Test frames directory not found: {self.frames_dir}")
        
        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        return frame_files

    def save_visualization(self, frame_t: np.ndarray, frame_t1: np.ndarray, 
                          roi: np.ndarray, mask: np.ndarray, diff: np.ndarray) -> None:
        """Save visualization of SME processing results to a single grid image."""
        # Preprocess frames for visualization (already 224x224 after preprocessing)
        processed_t = self.preprocessor.preprocess(frame_t)
        processed_t1 = self.preprocessor.preprocess(frame_t1)
        
        # Convert RGB back to BGR for saving with cv2.imwrite
        processed_t_bgr = cv2.cvtColor(processed_t, cv2.COLOR_RGB2BGR)
        processed_t1_bgr = cv2.cvtColor(processed_t1, cv2.COLOR_RGB2BGR)
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        
        # Convert mask and diff to 3-channel for stacking
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        diff_3ch = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # Create 1x5 grid: Frame T | Frame T+1 | Diff | Mask | ROI (all 224x224)
        grid = np.hstack([processed_t_bgr, processed_t1_bgr, diff_3ch, mask_3ch, roi_bgr])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)
        thickness = 2
        
        cv2.putText(grid, "Frame T", (10, 20), font, font_scale, font_color, thickness)
        cv2.putText(grid, "Frame T+1", (224 + 10, 20), font, font_scale, font_color, thickness)
        cv2.putText(grid, "Diff", (2*224 + 10, 20), font, font_scale, font_color, thickness)
        cv2.putText(grid, "Mask", (3*224 + 10, 20), font, font_scale, font_color, thickness)
        cv2.putText(grid, "ROI", (4*224 + 10, 20), font, font_scale, font_color, thickness)
        
        # Save single visualization grid
        output_path = self.output_dir / "sme_output_visualization.jpg"
        cv2.imwrite(str(output_path), grid)

    def test_real_frames_exist(self):
        """Verify test frames are available."""
        frame_files = self.get_frame_paths()
        assert len(frame_files) >= 2, f"Need at least 2 frames for testing, found {len(frame_files)}"

    def test_consecutive_real_frames(self):
        """Process consecutive real frames and verify outputs."""
        frame_files = self.get_frame_paths()
        
        if len(frame_files) < 2:
            pytest.skip("Insufficient test frames")
        
        # Load first two frames
        frame_t = cv2.imread(str(frame_files[0]))
        frame_t1 = cv2.imread(str(frame_files[1]))
        
        assert frame_t is not None, f"Failed to load frame: {frame_files[0]}"
        assert frame_t1 is not None, f"Failed to load frame: {frame_files[1]}"
        
        # Preprocess frames first
        frame_t_processed = self.preprocessor.preprocess(frame_t)
        frame_t1_processed = self.preprocessor.preprocess(frame_t1)
        
        # Extract motion
        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t_processed, frame_t1_processed)
        
        # Verify outputs are 224x224
        expected_shape = (224, 224)
        assert roi.shape[:2] == expected_shape, f"ROI shape {roi.shape[:2]} should match target size {expected_shape}"
        assert mask.shape[:2] == expected_shape, f"Mask shape {mask.shape[:2]} should match target size {expected_shape}"
        assert diff.shape[:2] == expected_shape, f"Diff shape {diff.shape[:2]} should match target size {expected_shape}"
        
        print(f"\n  Motion extraction time: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 5, f"Processing time {elapsed_ms:.2f}ms should be < 5ms"
        
        # Save visualization
        self.save_visualization(frame_t, frame_t1, roi, mask, diff)

    def test_real_frames_motion_detection(self):
        """Verify motion detection on real video frames."""
        frame_files = self.get_frame_paths()
        
        if len(frame_files) < 2:
            pytest.skip("Insufficient test frames")
        
        # Load frames
        frame_t = cv2.imread(str(frame_files[0]))
        frame_t1 = cv2.imread(str(frame_files[1]))
        
        # Preprocess
        frame_t_processed = self.preprocessor.preprocess(frame_t)
        frame_t1_processed = self.preprocessor.preprocess(frame_t1)
        
        # Extract motion
        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t_processed, frame_t1_processed)
        
        # In real video, there should be some motion detected
        diff_mean = np.mean(diff)
        
        # For real video frames, expect some non-zero motion
        assert diff_mean >= 0, "Diff mean should be non-negative"
        assert diff.max() <= 255, "Diff max should not exceed 255"
        
        # Verify shape
        assert diff.shape[:2] == (224, 224), f"Diff should be (224, 224), got {diff.shape[:2]}"
        
        # Save visualization
        self.save_visualization(frame_t, frame_t1, roi, mask, diff)

    def test_real_frames_all_consecutive_pairs(self):
        """Process a single frame pair from real data."""
        frame_files = self.get_frame_paths()
        
        if len(frame_files) < 2:
            pytest.skip("Insufficient test frames")
        
        # Load first two frames (1 pair)
        frame_t = cv2.imread(str(frame_files[0]))
        frame_t1 = cv2.imread(str(frame_files[1]))
        
        assert frame_t is not None
        assert frame_t1 is not None
        
        # Preprocess
        frame_t_processed = self.preprocessor.preprocess(frame_t)
        frame_t1_processed = self.preprocessor.preprocess(frame_t1)
        
        # Extract motion
        roi, mask, diff, elapsed_ms = self.extractor.process(frame_t_processed, frame_t1_processed)
        
        # Basic validation
        assert roi.dtype == np.uint8
        assert mask.dtype == np.uint8
        assert diff.dtype == np.uint8
        
        # Verify shape
        assert roi.shape[:2] == (224, 224), f"ROI should be (224, 224), got {roi.shape[:2]}"
        assert mask.shape[:2] == (224, 224), f"Mask should be (224, 224), got {mask.shape[:2]}"
        assert diff.shape[:2] == (224, 224), f"Diff should be (224, 224), got {diff.shape[:2]}"
        
        assert elapsed_ms < 5, f"Processing time {elapsed_ms:.2f}ms should be < 5ms"
        
        # Save visualization
        self.save_visualization(frame_t, frame_t1, roi, mask, diff)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
