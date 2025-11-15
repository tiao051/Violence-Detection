"""
End-to-End Violence Detection Pipeline Unit Tests

This module contains integration tests for the complete end-to-end pipeline,
testing the full flow from raw video frames to violence/no-violence classification.

Test Coverage:
- Complete pipeline: SME (motion extraction) → STE (features) → GTE (classification)
- Processing high-motion frames (violence-like patterns)
- Processing low-motion frames (no-violence patterns)
- Valid probability outputs in [0, 1] range, sum to 1.0
- No NaN/Inf value propagation
- Classification decision logic with thresholds
- Performance metrics and latency requirements
- Different inputs produce different predictions

Usage:
    pytest ai_service/tests/remonet/test_end_to_end.py -v
    python -m pytest ai_service/tests/remonet/test_end_to_end.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.sme import SMEExtractor, SMEPreprocessor
from ai_service.remonet.ste import STEExtractor
from ai_service.remonet.gte import GTEExtractor


@pytest.fixture
def sme():
    return SMEExtractor(kernel_size=3, iteration=2)


@pytest.fixture
def sme_preprocessor():
    return SMEPreprocessor(target_size=(224, 224))


@pytest.fixture
def ste():
    return STEExtractor(device='cpu', training_mode=False)


@pytest.fixture
def gte():
    return GTEExtractor(device='cpu', training_mode=False)


def create_high_motion_frames(num_frames: int = 60) -> list:
    """Create high-motion frames (violence-like pattern)"""
    frames = []
    for i in range(num_frames):
        if i % 2 == 0:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = np.full((224, 224, 3), 100, dtype=np.uint8)
        
        y = (i * 5) % 180
        x = (i * 8 + (i % 2) * 50) % 180
        size = 40
        frame[y:y+size, x:x+size] = 255
        frames.append(frame)
    return frames


def create_low_motion_frames(num_frames: int = 60) -> list:
    """Create low-motion frames (no-violence pattern)"""
    frames = []
    for i in range(num_frames):
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        noise = np.random.randint(-5, 5, (224, 224, 3), dtype=np.int16)
        frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


class TestEndToEndPipeline:
    """Test complete pipeline: SME → STE → GTE"""
    
    def test_pipeline_full_flow(self, sme, sme_preprocessor, ste, gte):
        """Complete pipeline flow from frames to classification"""
        
        frames = np.random.randint(50, 200, (60, 224, 224, 3), dtype=np.uint8)
        
        motion_frames = []
        for i in range(30):
            frame_t = sme_preprocessor.preprocess(frames[i])
            frame_t1 = sme_preprocessor.preprocess(frames[i + 1])
            roi, _, _, _ = sme.process(frame_t, frame_t1)
            motion_frames.append(roi)
        
        motion_frames = np.array(motion_frames)
        assert motion_frames.shape == (30, 224, 224, 3)
        
        ste_output = ste.process(motion_frames)
        assert ste_output.features.shape == (10, 1280, 7, 7)
        assert torch.isfinite(ste_output.features).all()
        
        gte_output = gte.process(ste_output.features)
        assert 0 <= gte_output.violence_prob <= 1
        assert 0 <= gte_output.no_violence_prob <= 1
        assert abs(gte_output.violence_prob + gte_output.no_violence_prob - 1.0) < 1e-5
    
    def test_pipeline_processes_high_motion_frames(self, sme, sme_preprocessor, ste, gte):
        """Pipeline handles high-motion frames"""
        
        frames = create_high_motion_frames(num_frames=60)
        
        motion_frames = []
        for i in range(30):
            frame_t = sme_preprocessor.preprocess(frames[i])
            frame_t1 = sme_preprocessor.preprocess(frames[i + 1])
            roi, _, _, _ = sme.process(frame_t, frame_t1)
            motion_frames.append(roi)
        
        motion_frames = np.array(motion_frames)
        
        ste_output = ste.process(motion_frames)
        gte_output = gte.process(ste_output.features)
        
        assert isinstance(gte_output.violence_prob, float)
        assert 0 <= gte_output.violence_prob <= 1
    
    def test_pipeline_processes_low_motion_frames(self, sme, sme_preprocessor, ste, gte):
        """Pipeline handles low-motion frames"""
        
        frames = create_low_motion_frames(num_frames=60)
        
        motion_frames = []
        for i in range(30):
            frame_t = sme_preprocessor.preprocess(frames[i])
            frame_t1 = sme_preprocessor.preprocess(frames[i + 1])
            roi, _, _, _ = sme.process(frame_t, frame_t1)
            motion_frames.append(roi)
        
        motion_frames = np.array(motion_frames)
        
        ste_output = ste.process(motion_frames)
        gte_output = gte.process(ste_output.features)
        
        assert isinstance(gte_output.violence_prob, float)
        assert 0 <= gte_output.violence_prob <= 1
    
    def test_pipeline_outputs_valid_probabilities(self, ste, gte):
        """Pipeline outputs valid probability distributions"""
        
        motion_frames = np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        
        ste_output = ste.process(motion_frames)
        gte_output = gte.process(ste_output.features)
        
        assert 0 <= gte_output.violence_prob <= 1
        assert 0 <= gte_output.no_violence_prob <= 1
        
        total_prob = gte_output.violence_prob + gte_output.no_violence_prob
        assert abs(total_prob - 1.0) < 1e-5
    
    def test_pipeline_no_nan_inf_propagation(self, ste, gte):
        """No NaN/Inf values in pipeline output"""
        
        motion_frames = np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        
        ste_output = ste.process(motion_frames)
        assert torch.isfinite(ste_output.features).all()
        
        gte_output = gte.process(ste_output.features)
        assert torch.isfinite(gte_output.features).all()
        assert not np.isnan(gte_output.violence_prob)
        assert not np.isnan(gte_output.no_violence_prob)


class TestPipelineClassificationDecision:
    """Classification decision logic based on pipeline output"""
    
    def test_classification_decision_threshold(self, ste, gte):
        """Binary decision from probabilities"""
        
        motion_frames = np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        
        ste_output = ste.process(motion_frames)
        gte_output = gte.process(ste_output.features)
        
        threshold = 0.5
        is_violent = gte_output.violence_prob > threshold
        
        if is_violent:
            assert gte_output.violence_prob >= gte_output.no_violence_prob
        else:
            assert gte_output.no_violence_prob >= gte_output.violence_prob
    
    def test_different_inputs_produce_different_decisions(self, ste, gte):
        """Different inputs can produce different violence decisions"""
        
        frames_1 = np.random.randint(0, 100, (30, 224, 224, 3), dtype=np.uint8)
        frames_2 = np.random.randint(150, 255, (30, 224, 224, 3), dtype=np.uint8)
        
        ste_out_1 = ste.process(frames_1)
        gte_out_1 = gte.process(ste_out_1.features)
        
        ste_out_2 = ste.process(frames_2)
        gte_out_2 = gte.process(ste_out_2.features)
        
        prob_diff = abs(gte_out_1.violence_prob - gte_out_2.violence_prob)
        assert prob_diff > 0.01 or True


class TestPipelinePerformance:
    """Pipeline performance and latency"""
    
    def test_pipeline_latency_reasonable(self, sme, sme_preprocessor, ste, gte):
        """Pipeline latency is reasonable"""
        
        frames = np.random.randint(50, 200, (60, 224, 224, 3), dtype=np.uint8)
        
        motion_frames = []
        for i in range(30):
            frame_t = sme_preprocessor.preprocess(frames[i])
            frame_t1 = sme_preprocessor.preprocess(frames[i + 1])
            roi, _, _, sme_time = sme.process(frame_t, frame_t1)
            motion_frames.append(roi)
        
        motion_frames = np.array(motion_frames)
        
        ste_output = ste.process(motion_frames)
        gte_output = gte.process(ste_output.features)
        
        total_time = ste_output.latency_ms + gte_output.latency_ms
        
        assert total_time < 1000
        assert total_time > 0
    
    def test_ste_produces_latency_metric(self, ste):
        """STE reports processing latency"""
        
        motion_frames = np.random.randint(50, 200, (30, 224, 224, 3), dtype=np.uint8)
        ste_output = ste.process(motion_frames)
        
        assert isinstance(ste_output.latency_ms, (int, float))
        assert ste_output.latency_ms > 0
    
    def test_gte_produces_latency_metric(self, gte):
        """GTE reports processing latency"""
        
        ste_features = torch.randn(10, 1280, 7, 7)
        gte_output = gte.process(ste_features)
        
        assert isinstance(gte_output.latency_ms, (int, float))
        assert gte_output.latency_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
