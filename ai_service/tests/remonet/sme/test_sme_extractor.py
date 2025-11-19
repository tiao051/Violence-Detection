"""
SME (Spatial Motion Extractor) Unit Tests

Tests motion detection between consecutive frames, output format/shapes, performance, reproducibility.

Usage:
    pytest ai_service/tests/remonet/sme/test_sme_extractor.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.remonet.sme import SMEExtractor


@pytest.fixture
def extractor():
    return SMEExtractor(kernel_size=3, iteration=2)


def create_identical_pair():
    """Create pair of identical frames (no motion)"""
    frame = np.full((224, 224, 3), 128, dtype=np.uint8)
    return frame, frame.copy()


def create_pair_with_motion_region(value_before=100, value_after=200):
    """Create pair with synthetic motion in center region"""
    frame_t = np.full((224, 224, 3), value_before, dtype=np.uint8)
    frame_t1 = frame_t.copy()
    y_start, y_end, x_start, x_end = 56, 168, 56, 168
    frame_t1[y_start:y_end, x_start:x_end] = value_after
    return frame_t, frame_t1, (y_start, y_end, x_start, x_end)


def create_random_pair():
    """Create pair of random frames"""
    return (np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))


class TestSMEExtractor:
    """Test SME motion detection pipeline"""
    
    def test_output_shapes_and_types(self, extractor):
        """Test correct output shapes and data types"""
        frame_t, frame_t1 = create_random_pair()
        roi, mask, diff, elapsed_ms = extractor.process(frame_t, frame_t1)
        
        assert roi.shape == (224, 224, 3)
        assert mask.shape == (224, 224)
        assert diff.shape == (224, 224)
        assert roi.dtype == np.float32
        assert mask.dtype == np.uint8
        assert diff.dtype == np.uint8
        assert isinstance(elapsed_ms, (int, float)) and elapsed_ms >= 0
    
    def test_output_value_ranges(self, extractor):
        """Test output values are in valid ranges"""
        frame_t, frame_t1 = create_random_pair()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        
        assert np.all((0 <= roi) & (roi <= 1))
        assert np.all((0 <= mask) & (mask <= 255))
        assert np.all((0 <= diff) & (diff <= 255))
    
    def test_no_motion_for_identical_frames(self, extractor):
        """Test zero diff when frames are identical"""
        frame_t, frame_t1 = create_identical_pair()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        
        assert np.all(diff == 0)
        # Motion mask should be nearly empty
        assert np.count_nonzero(mask) / mask.size < 0.01
    
    def test_detects_synthetic_motion(self, extractor):
        """Test motion detection in center region"""
        frame_t, frame_t1, (y0, y1, x0, x1) = create_pair_with_motion_region()
        roi, mask, diff, _ = extractor.process(frame_t, frame_t1)
        
        motion_mean = np.mean(diff[y0:y1, x0:x1])
        non_motion_mean = np.mean(np.concatenate([diff[:y0, :].flatten(), diff[y1:, :].flatten()]))
        assert motion_mean > non_motion_mean * 1.5
    
    def test_processing_speed(self, extractor):
        """Test processing completes within 5ms per frame pair"""
        frame_t, frame_t1 = create_random_pair()
        _, _, _, elapsed_ms = extractor.process(frame_t, frame_t1)
        
        assert elapsed_ms < 5
    
    def test_reproducibility_same_inputs(self, extractor):
        """Test identical inputs produce identical outputs"""
        frame_t, frame_t1 = create_random_pair()
        
        roi1, mask1, diff1, _ = extractor.process(frame_t, frame_t1)
        roi2, mask2, diff2, _ = extractor.process(frame_t, frame_t1)
        
        assert np.array_equal(roi1, roi2)
        assert np.array_equal(mask1, mask2)
        assert np.array_equal(diff1, diff2)
    
    def test_batch_processing_output_shape(self, extractor):
        """Test batch processing returns correct number of frames"""
        frames = np.random.randint(0, 256, (30, 224, 224, 3), dtype=np.uint8)
        motion_frames = extractor.process_batch(frames)
        
        assert motion_frames.shape == (30, 224, 224, 3)
        assert motion_frames.dtype == np.float32
    
    def test_batch_processing_value_range(self, extractor):
        """Test batch output values in [0, 1]"""
        frames = np.random.randint(0, 256, (30, 224, 224, 3), dtype=np.uint8)
        motion_frames = extractor.process_batch(frames)
        
        assert np.all(motion_frames >= 0)
        assert np.all(motion_frames <= 1)
    
    def test_batch_processing_various_lengths(self, extractor):
        """Test batch processing with different frame counts"""
        for num_frames in [2, 10, 30, 50]:
            frames = np.random.randint(0, 256, (num_frames, 224, 224, 3), dtype=np.uint8)
            motion_frames = extractor.process_batch(frames)
            
            assert len(motion_frames) == num_frames
            assert motion_frames.dtype == np.float32
    
    def test_batch_processing_consistency(self, extractor):
        """Test batch processing matches individual frame processing"""
        frames = np.random.randint(0, 256, (5, 224, 224, 3), dtype=np.uint8)
        
        # Individual processing
        individual_rois = []
        for i in range(len(frames) - 1):
            roi, _, _, _ = extractor.process(frames[i], frames[i + 1])
            individual_rois.append(roi)
        individual_rois.append(individual_rois[-1].copy())
        individual_array = np.array(individual_rois, dtype=np.float32)
        
        # Batch processing
        batch_array = extractor.process_batch(frames)
        
        assert np.allclose(batch_array, individual_array, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

