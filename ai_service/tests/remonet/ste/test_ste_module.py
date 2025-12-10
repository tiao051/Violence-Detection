"""
STE (Short Temporal Extractor) Unit Tests

Tests temporal composite creation, feature extraction, normalization, backbone support.

Usage:
    pytest ai_service/tests/remonet/ste/test_ste_module.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.remonet.ste import STEExtractor, STEOutput, BackboneType, BACKBONE_CONFIG


@pytest.fixture
def ste_extractor():
    return STEExtractor(device='cpu')


def create_random_frames(count, seed=42):
    """Generate random uint8 RGB frames"""
    np.random.seed(seed)
    return [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(count)]


class TestSTEPipeline:
    """Test STE feature extraction pipeline"""
    
    def test_batch_processing_output_shape(self, ste_extractor):
        """Test process_batch output shape matches spec (T/3, C, H, W)"""
        frames = create_random_frames(30)
        features = ste_extractor.process_batch(frames)
        
        assert features.shape == (10, 1280, 7, 7)
        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
    
    def test_process_returns_valid_ste_output(self, ste_extractor):
        """Test process() returns STEOutput with correct metadata"""
        frames = create_random_frames(30)
        output = ste_extractor.process(frames, camera_id="test_cam", timestamp=123.0)
        
        assert isinstance(output, STEOutput)
        assert output.features.shape == (10, 1280, 7, 7)
        assert output.camera_id == "test_cam"
        assert output.timestamp == 123.0
        assert output.latency_ms > 0
    
    def test_temporal_composite_normalization(self, ste_extractor):
        """Test temporal composite creation with correct ImageNet normalization"""
        frames = [np.full((224, 224, 3), val, dtype=np.uint8) for val in [60, 90, 120]]
        composite = ste_extractor.create_temporal_composite(frames)
        
        assert composite.shape == (224, 224, 3)
        assert composite.dtype == np.float32
        assert not np.any(np.isnan(composite))
        assert not np.any(np.isinf(composite))
    
    def test_does_not_modify_input_frames(self, ste_extractor):
        """Test processing doesn't modify input frames"""
        frames = create_random_frames(30)
        frames_copy = [f.copy() for f in frames]
        
        _ = ste_extractor.process_batch(frames)
        
        for orig, copy in zip(frames, frames_copy):
            assert np.array_equal(orig, copy)
    
    def test_deterministic_processing(self, ste_extractor):
        """Test batch processing is deterministic"""
        frames = create_random_frames(30)
        
        features_1 = ste_extractor.process_batch(frames)
        features_2 = ste_extractor.process_batch(frames)
        
        assert torch.allclose(features_1, features_2, rtol=1e-5, atol=1e-5)


class TestSTEBackbones:
    """Test different CNN backbone support"""
    
    @pytest.mark.parametrize("backbone", [
        BackboneType.MOBILENET_V2,
        BackboneType.MOBILENET_V3_SMALL,
        BackboneType.MOBILENET_V3_LARGE,
        BackboneType.EFFICIENTNET_B0,
        BackboneType.MNASNET,
    ])
    def test_all_backbones_instantiate(self, backbone):
        """Test all backbone types can be instantiated"""
        ste = STEExtractor(device='cpu', backbone=backbone)
        assert ste.backbone_type == backbone
    
    @pytest.mark.parametrize("backbone", [
        BackboneType.MOBILENET_V2,
        BackboneType.MOBILENET_V3_SMALL,
        BackboneType.MOBILENET_V3_LARGE,
        BackboneType.EFFICIENTNET_B0,
        BackboneType.MNASNET,
    ])
    def test_backbone_feature_extraction(self, backbone):
        """Test each backbone processes frames correctly"""
        ste = STEExtractor(device='cpu', backbone=backbone)
        frames = create_random_frames(30)
        features = ste.process_batch(frames)
        
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
    def test_backbone_string_initialization(self, backbone_str):
        """Test backbone initialization with string names"""
        ste = STEExtractor(device='cpu', backbone=backbone_str)
        assert str(ste.backbone_type) == backbone_str
    
    def test_backbone_info_retrieval(self):
        """Test get_backbone_info returns correct configuration"""
        ste = STEExtractor(device='cpu', backbone=BackboneType.MOBILENET_V2)
        info = ste.get_backbone_info()
        
        assert info['backbone'] == 'mobilenet_v2'
        assert info['out_channels'] == 1280
        assert info['spatial_size'] == 7
        assert info['feature_shape'] == (10, 1280, 7, 7)
    
    def test_available_backbones_list(self):
        """Test get_available_backbones returns all supported backbones"""
        backbones = STEExtractor.get_available_backbones()
        expected = ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 
                   'efficientnet_b0', 'mnasnet']
        assert set(backbones) == set(expected)


class TestSTEInputValidation:
    """Test error handling and input validation"""
    
    def test_reject_wrong_frame_count_for_composite(self):
        """Test composite creation rejects wrong frame count"""
        ste = STEExtractor(device='cpu')
        
        for count in [2, 4]:
            frames = create_random_frames(count)
            with pytest.raises(ValueError, match="Expected 3 frames"):
                ste.create_temporal_composite(frames)
    
    def test_reject_wrong_batch_frame_count(self):
        """Test batch processing rejects wrong frame count"""
        ste = STEExtractor(device='cpu')
        
        for count in [20, 40]:
            frames = create_random_frames(count)
            with pytest.raises(ValueError, match="Expected 30 frames"):
                ste.process_batch(frames)
    
    def test_reject_none_frames(self):
        """Test processing rejects None frames"""
        ste = STEExtractor(device='cpu')
        
        with pytest.raises((ValueError, AttributeError)):
            ste.create_temporal_composite([None, None, None])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
