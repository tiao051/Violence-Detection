"""
SME Preprocessor Unit Tests

Tests SMEPreprocessor component: frame resizing, color space conversion, data type handling.

Usage:
    pytest ai_service/tests/remonet/sme/test_sme_preprocessor.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.remonet.sme import SMEPreprocessor


@pytest.fixture
def preprocessor():
    return SMEPreprocessor(target_size=(224, 224))


def create_bgr_frame(size=(480, 640)):
    """Generate random BGR frame"""
    return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)


def create_grayscale_frame(size=(480, 640)):
    """Generate random grayscale frame"""
    return np.random.randint(0, 256, size, dtype=np.uint8)


def create_bgra_frame(size=(480, 640)):
    """Generate random BGRA frame"""
    return np.random.randint(0, 256, (*size, 4), dtype=np.uint8)


class TestSMEPreprocessor:
    """Test SME preprocessing pipeline"""
    
    def test_bgr_to_rgb_conversion(self, preprocessor):
        """Test BGR→RGB conversion with shape, dtype, channels"""
        frame = create_bgr_frame()
        processed = preprocessor.preprocess(frame)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
        assert np.all((0 <= processed) & (processed <= 255))
    
    def test_grayscale_to_rgb_conversion(self, preprocessor):
        """Test grayscale→RGB conversion (replicate channels)"""
        frame = create_grayscale_frame()
        processed = preprocessor.preprocess(frame)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
        # All channels should be equal for grayscale input
        assert processed[0, 0, 0] == processed[0, 0, 1] == processed[0, 0, 2]
    
    def test_bgra_to_rgb_conversion(self, preprocessor):
        """Test BGRA→RGB conversion (drop alpha channel)"""
        frame = create_bgra_frame()
        processed = preprocessor.preprocess(frame)
        
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
    
    def test_resize_larger_to_target(self, preprocessor):
        """Test resizing from larger (720×1280) to target (224×224)"""
        frame = create_bgr_frame(size=(720, 1280))
        processed = preprocessor.preprocess(frame)
        
        assert processed.shape == (224, 224, 3)
    
    def test_resize_smaller_to_target(self, preprocessor):
        """Test resizing from smaller (112×112) to target (224×224)"""
        frame = create_bgr_frame(size=(112, 112))
        processed = preprocessor.preprocess(frame)
        
        assert processed.shape == (224, 224, 3)
    
    def test_resize_already_correct_size(self, preprocessor):
        """Test when frame already at target size"""
        frame = create_bgr_frame(size=(224, 224))
        processed = preprocessor.preprocess(frame)
        
        assert processed.shape == (224, 224, 3)
    
    def test_channel_swap_bgr_to_rgb(self, preprocessor):
        """Test BGR channels are correctly swapped to RGB"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 50   # B
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 150  # R
        
        processed = preprocessor.preprocess(frame)
        
        # After BGR→RGB: R should be first, B last
        assert processed[0, 0, 0] == 150  # R
        assert processed[0, 0, 1] == 100  # G
        assert processed[0, 0, 2] == 50   # B


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
