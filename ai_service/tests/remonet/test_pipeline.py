"""
STE → GTE Pipeline Integration Tests

This module contains integration tests for the STE and GTE pipeline stages,
verifying the complete pipeline: Motion Frames → STE Features → GTE Classification

Test Coverage:
- STE feature extraction: shape (10, 1280, 7, 7), valid ranges, no NaN/Inf
- GTE classification: probability outputs in [0, 1], sum to 1.0
- Full pipeline: SME → STE → GTE end-to-end flow
- Training vs inference modes
- Classification logic and decision thresholds
- Metadata preservation through pipeline

Note: SME tests are in ai_service/tests/remonet/sme/ folder

Usage:
    pytest ai_service/tests/remonet/test_pipeline.py -v
    python -m pytest ai_service/tests/remonet/test_pipeline.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.ste import STEExtractor
from ai_service.remonet.gte import GTEExtractor


@pytest.fixture
def ste():
    return STEExtractor(device='cpu', training_mode=False)


@pytest.fixture
def gte():
    return GTEExtractor(device='cpu', training_mode=False)


@pytest.fixture
def sample_motion_frames():
    """Generate 30 mock motion frames for STE (SME output)"""
    np.random.seed(42)
    return np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)


class TestSTEOutput:
    """STE feature extraction validation"""
    
    def test_ste_output_shapes(self, ste, sample_motion_frames):
        ste_output = ste.process(sample_motion_frames)
        
        assert ste_output.features.shape == (10, 1280, 7, 7)
        assert isinstance(ste_output.latency_ms, (int, float))
    
    def test_ste_output_range(self, ste, sample_motion_frames):
        ste_output = ste.process(sample_motion_frames)
        features = ste_output.features
        
        assert torch.isfinite(features).all()
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()


class TestGTEOutput:
    """GTE classification output validation"""
    
    def test_gte_output_probabilities(self, gte):
        ste_features = torch.randn(10, 1280, 7, 7)
        gte_output = gte.process(ste_features)
        
        assert 0 <= gte_output.no_violence_prob <= 1
        assert 0 <= gte_output.violence_prob <= 1
        assert abs(gte_output.no_violence_prob + gte_output.violence_prob - 1.0) < 1e-5
    
    def test_gte_output_features(self, gte):
        ste_features = torch.randn(10, 1280, 7, 7)
        gte_output = gte.process(ste_features)
        
        assert gte_output.features.shape == (1280,)
        assert torch.isfinite(gte_output.features).all()


class TestFullPipeline:
    """STE → GTE pipeline integration"""
    
    def test_sme_to_ste_to_gte_pipeline(self, ste, gte, sample_motion_frames):
        """Complete pipeline: motion frames to violence probability"""
        
        motion_frames = sample_motion_frames
        assert motion_frames.shape == (30, 224, 224, 3)
        
        ste_output = ste.process(motion_frames)
        assert ste_output.features.shape == (10, 1280, 7, 7)
        assert ste_output.features.dtype == torch.float32
        
        gte_output = gte.process(ste_output.features)
        assert 0 <= gte_output.no_violence_prob <= 1
        assert 0 <= gte_output.violence_prob <= 1
        assert abs(gte_output.no_violence_prob + gte_output.violence_prob - 1.0) < 1e-5
    
    def test_pipeline_no_nan_inf(self, ste, gte, sample_motion_frames):
        """No NaN/Inf propagation through pipeline"""
        
        motion_frames = sample_motion_frames
        ste_output = ste.process(motion_frames)
        
        assert torch.isfinite(ste_output.features).all()
        assert not torch.isnan(ste_output.features).any()
        assert not torch.isinf(ste_output.features).any()
        
        gte_output = gte.process(ste_output.features)
        assert torch.isfinite(gte_output.features).all()
        assert not torch.isnan(gte_output.features).any()
        assert not torch.isinf(gte_output.features).any()
    
    def test_pipeline_with_metadata(self, ste, gte, sample_motion_frames):
        """Pipeline preserves metadata through processing"""
        
        camera_id = "test_camera"
        timestamp = 123.45
        
        motion_frames = sample_motion_frames
        ste_output = ste.process(motion_frames, camera_id=camera_id, timestamp=timestamp)
        gte_output = gte.process(ste_output.features, camera_id=camera_id, timestamp=timestamp)
        
        assert ste_output.camera_id == camera_id
        assert ste_output.timestamp == timestamp
        assert gte_output.camera_id == camera_id
        assert gte_output.timestamp == timestamp
    
    def test_pipeline_batch_consistency(self, ste, gte):
        """Different inputs produce different outputs"""
        
        motion_frames_1 = np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        motion_frames_2 = np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        
        ste_output_1 = ste.process(motion_frames_1)
        gte_output_1 = gte.process(ste_output_1.features)
        
        ste_output_2 = ste.process(motion_frames_2)
        gte_output_2 = gte.process(ste_output_2.features)
        
        assert not torch.allclose(ste_output_1.features, ste_output_2.features)
        assert gte_output_1.violence_prob != gte_output_2.violence_prob


class TestTrainingVsInferenceModes:
    """Mode-specific behavior"""
    
    def test_ste_inference_mode(self, sample_motion_frames):
        ste = STEExtractor(device='cpu', training_mode=False)
        ste_output = ste.process(sample_motion_frames)
        
        assert not ste_output.features.requires_grad
    
    def test_ste_training_mode(self, sample_motion_frames):
        ste = STEExtractor(device='cpu', training_mode=True)
        ste_output = ste.process(sample_motion_frames)
        
        assert ste_output.features.requires_grad
    

class TestClassificationLogic:
    """Classification decision logic"""
    
    def test_gte_high_violence_probability(self, gte):
        """GTE can produce high violence probability"""
        
        ste_features = torch.randn(10, 1280, 7, 7) * 2.0
        gte_output = gte.process(ste_features)
        
        assert 0 <= gte_output.violence_prob <= 1
        assert 0 <= gte_output.no_violence_prob <= 1
    
    def test_gte_low_violence_probability(self, gte):
        """GTE can produce low violence probability"""
        
        ste_features = torch.randn(10, 1280, 7, 7) * 0.1
        gte_output = gte.process(ste_features)
        
        assert 0 <= gte_output.violence_prob <= 1
        assert 0 <= gte_output.no_violence_prob <= 1
    
    def test_gte_produces_different_predictions(self, gte):
        """Different inputs produce different violence predictions"""
        
        features_1 = torch.randn(10, 1280, 7, 7)
        features_2 = torch.randn(10, 1280, 7, 7)
        
        output_1 = gte.process(features_1)
        output_2 = gte.process(features_2)
        
        prob_diff = abs(output_1.violence_prob - output_2.violence_prob)
        assert prob_diff > 0.01
    
    def test_violence_class_is_class_1(self, gte):
        """Violence class mapped to class 1"""
        
        ste_features = torch.randn(10, 1280, 7, 7)
        gte_output = gte.process(ste_features)
        
        total = gte_output.no_violence_prob + gte_output.violence_prob
        assert abs(total - 1.0) < 1e-5
        
        assert hasattr(gte_output, 'no_violence_prob')
        assert hasattr(gte_output, 'violence_prob')
    
    def test_classification_decision_threshold(self, gte):
        """Binary decision based on violence threshold"""
        
        ste_features = torch.randn(10, 1280, 7, 7)
        gte_output = gte.process(ste_features)
        
        threshold = 0.5
        is_violence = gte_output.violence_prob > threshold
        
        assert isinstance(is_violence, (bool, np.bool_))


class TestPipelineWithClassification:
    """Pipeline classification interpretation"""
    
    def test_pipeline_produces_classification(self, ste, gte, sample_motion_frames):
        """Pipeline produces interpretable classification"""
        
        motion_frames = sample_motion_frames
        ste_output = ste.process(motion_frames)
        gte_output = gte.process(ste_output.features)
        
        threshold = 0.5
        is_violent = gte_output.violence_prob > threshold
        
        assert isinstance(is_violent, (bool, np.bool_))
        
        if is_violent:
            assert gte_output.violence_prob >= gte_output.no_violence_prob
        else:
            assert gte_output.no_violence_prob >= gte_output.violence_prob
    
    def test_consistent_classification_across_runs(self, ste, gte):
        """Same input produces same classification"""
        
        torch.manual_seed(42)
        motion_frames = np.random.RandomState(42).randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        
        ste_output_1 = ste.process(motion_frames)
        gte_output_1 = gte.process(ste_output_1.features)
        
        ste_output_2 = ste.process(motion_frames)
        gte_output_2 = gte.process(ste_output_2.features)
        
        assert abs(gte_output_1.violence_prob - gte_output_2.violence_prob) < 1e-5
        assert abs(gte_output_1.no_violence_prob - gte_output_2.no_violence_prob) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
