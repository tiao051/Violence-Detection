"""
STE (Short Temporal Extractor) Unit Tests

This module contains unit tests for the STEExtractor component, verifying:
- Temporal composite creation between consecutive frames
- Output format, shapes, data types, and value ranges
- Batch consistency and reproducibility
- Normalization correctness (critical)
- Integration with SME-like RGB frames
- Multiple backbone support (MobileNetV2, V3, EfficientNet, MNasNet)
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

from ai_service.remonet.ste import STEExtractor, STEOutput, BackboneType, BACKBONE_CONFIG


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
        
        # Expected values after averaging RGB channels, normalizing to [0, 1], 
        # then applying ImageNet normalization
        # Frame 0: mean(60, 90, 120) = 90 → 90/255 ≈ 0.353 → ImageNet norm
        # Frame 1: mean(30, 60, 90) = 60 → 60/255 ≈ 0.235 → ImageNet norm
        # Frame 2: mean(90, 120, 150) = 120 → 120/255 ≈ 0.471 → ImageNet norm
        
        # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Each channel uses its corresponding mean and std
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        ch0 = composite[112, 112, 0]  # Frame 0 averaged + normalized
        ch1 = composite[112, 112, 1]  # Frame 1 averaged + normalized
        ch2 = composite[112, 112, 2]  # Frame 2 averaged + normalized
        
        expected_ch0 = (90.0 / 255.0 - imagenet_mean[0]) / imagenet_std[0]
        expected_ch1 = (60.0 / 255.0 - imagenet_mean[1]) / imagenet_std[1]
        expected_ch2 = (120.0 / 255.0 - imagenet_mean[2]) / imagenet_std[2]
        
        assert np.isclose(ch0, expected_ch0, atol=1e-5)
        assert np.isclose(ch1, expected_ch1, atol=1e-5)
        assert np.isclose(ch2, expected_ch2, atol=1e-5)

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


class TestBackboneSupport:
    """Test support for different CNN backbones."""

    @pytest.mark.parametrize("backbone", [
        BackboneType.MOBILENET_V2,
        BackboneType.MOBILENET_V3_SMALL,
        BackboneType.MOBILENET_V3_LARGE,
        BackboneType.EFFICIENTNET_B0,
        BackboneType.MNASNET,
    ])
    def test_all_backbones_available(self, backbone):
        """Test: All backbone types can be instantiated."""
        ste = STEExtractor(device='cpu', backbone=backbone)
        assert ste.backbone_type == backbone

    @pytest.mark.parametrize("backbone", [
        BackboneType.MOBILENET_V2,
        BackboneType.MOBILENET_V3_SMALL,
        BackboneType.MOBILENET_V3_LARGE,
        BackboneType.EFFICIENTNET_B0,
        BackboneType.MNASNET,
    ])
    def test_backbone_processing(self, backbone):
        """Test: Each backbone can process frames correctly."""
        ste = STEExtractor(device='cpu', backbone=backbone)
        frames = create_random_frames(30)
        features = ste.process_batch(frames)
        
        # Check output shape
        config = BACKBONE_CONFIG[backbone]
        expected_channels = config['out_channels']
        expected_spatial = config['spatial_size']
        
        assert features.shape == (10, expected_channels, expected_spatial, expected_spatial)
        assert features.dtype == torch.float32

    @pytest.mark.parametrize("backbone_str", [
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "mnasnet",
    ])
    def test_backbone_string_init(self, backbone_str):
        """Test: Can initialize with backbone name as string."""
        ste = STEExtractor(device='cpu', backbone=backbone_str)
        assert str(ste.backbone_type) == backbone_str

    def test_get_backbone_info(self):
        """Test: get_backbone_info returns correct configuration."""
        ste = STEExtractor(device='cpu', backbone=BackboneType.MOBILENET_V2)
        info = ste.get_backbone_info()
        
        assert info['backbone'] == 'mobilenet_v2'
        assert info['out_channels'] == 1280
        assert info['spatial_size'] == 7
        assert info['feature_shape'] == (10, 1280, 7, 7)

    def test_available_backbones(self):
        """Test: get_available_backbones returns all supported backbones."""
        backbones = STEExtractor.get_available_backbones()
        expected = ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 
                   'efficientnet_b0', 'mnasnet']
        assert set(backbones) == set(expected)

    def test_ste_output_includes_backbone_info(self):
        """Test: STEOutput includes backbone information."""
        ste = STEExtractor(device='cpu', backbone=BackboneType.MOBILENET_V3_SMALL)
        frames = create_random_frames(30)
        output = ste.process(frames, camera_id="test", timestamp=100.0)
        
        assert output.backbone == 'mobilenet_v3_small'
        assert output.camera_id == "test"
        assert output.timestamp == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
