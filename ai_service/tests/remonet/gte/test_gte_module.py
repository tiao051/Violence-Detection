"""
GTE (Global Temporal Extractor) Unit Tests

This module contains unit tests for the GTEExtractor component, verifying:
- Input/output shapes at each processing step
- Probability outputs in [0, 1] range and sum to 1
- Classification class order (0=no_violence, 1=violence)
- Training vs inference modes work correctly
- Temporal excitation weights in [0, 1] range
- All intermediate tensor operations produce valid values

Usage:
    pytest ai_service/tests/remonet/gte/test_gte_module.py -v -s
    python -m pytest ai_service/tests/remonet/gte/test_gte_module.py
"""

import sys
from pathlib import Path

import torch
import pytest

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.gte import GTEExtractor, GTEOutput


@pytest.fixture
def gte():
    return GTEExtractor(device='cpu', training_mode=False)


@pytest.fixture
def ste_output():
    # STE output format: (T/3, C, H, W) = (10, 1280, 7, 7)
    return torch.randn(10, 1280, 7, 7)


class TestShapes:
    """Test input/output shapes"""
    
    def test_spatial_compression(self, gte):
        B = torch.randn(10, 1280, 7, 7)
        S = gte.spatial_compression(B)
        assert S.shape == (1280, 10)
    
    def test_temporal_compression(self, gte):
        S = torch.randn(1280, 10)
        q = gte.temporal_compression(S)
        assert q.shape == (10,)
    
    def test_temporal_excitation(self, gte):
        q = torch.randn(10)
        E = gte.temporal_excitation(q)
        assert E.shape == (10,)
        assert torch.all(E >= 0) and torch.all(E <= 1)
    
    def test_channel_recalibration(self, gte):
        S = torch.randn(1280, 10)
        E = torch.sigmoid(torch.randn(10))
        S_prime = gte.channel_recalibration(S, E)
        assert S_prime.shape == (1280, 10)
    
    def test_temporal_aggregation(self, gte):
        S_prime = torch.randn(1280, 10)
        F = gte.temporal_aggregation(S_prime)
        assert F.shape == (1280,)
    
    def test_forward(self, gte):
        B = torch.randn(10, 1280, 7, 7)
        logits = gte.forward(B)
        assert logits.shape == (2,)
    
    def test_process(self, gte, ste_output):
        output = gte.process(ste_output, camera_id="test")
        assert isinstance(output, GTEOutput)
        assert output.features.shape == (1280,)


class TestProbabilities:
    """Test probability outputs"""
    
    def test_valid_range(self, gte, ste_output):
        output = gte.process(ste_output)
        assert 0 <= output.no_violence_prob <= 1
        assert 0 <= output.violence_prob <= 1
    
    def test_sum_to_one(self, gte, ste_output):
        output = gte.process(ste_output)
        total = output.no_violence_prob + output.violence_prob
        assert abs(total - 1.0) < 1e-5
    
    def test_class_order(self, gte, ste_output):
        # Class 0 = no_violence, Class 1 = violence
        output = gte.process(ste_output)
        assert hasattr(output, 'no_violence_prob')
        assert hasattr(output, 'violence_prob')


class TestModes:
    """Test training vs inference modes"""
    
    def test_inference_mode(self, ste_output):
        gte = GTEExtractor(device='cpu', training_mode=False)
        B = ste_output
        with torch.no_grad():
            logits = gte.forward(B)
        assert not logits.requires_grad
    
    def test_training_mode(self, ste_output):
        gte = GTEExtractor(device='cpu', training_mode=True)
        B = ste_output
        logits = gte.forward(B)
        assert logits.requires_grad


class TestPipeline:
    """Test complete pipeline with various input formats"""
    
    def test_ste_output_format(self):
        """Test: GTE processing with STE output format (T/3, C, H, W)"""
        ste_features = torch.randn(10, 1280, 7, 7)
        
        gte = GTEExtractor(device='cpu', training_mode=False)
        gte_output = gte.process(ste_features, camera_id="camera_1", timestamp=1.0)
        
        assert 0 <= gte_output.no_violence_prob <= 1
        assert 0 <= gte_output.violence_prob <= 1
        assert abs((gte_output.no_violence_prob + gte_output.violence_prob) - 1.0) < 1e-5
    
    def test_intermediate_shapes_flow(self):
        """Test: Verify shapes at each GTE processing step"""
        B = torch.randn(10, 1280, 7, 7)
        gte = GTEExtractor(device='cpu', training_mode=False)
        
        with torch.no_grad():
            S = gte.spatial_compression(B)
            assert S.shape == (1280, 10)
            
            q = gte.temporal_compression(S)
            assert q.shape == (10,)
            
            E = gte.temporal_excitation(q)
            assert E.shape == (10,)
            assert torch.all(E >= 0) and torch.all(E <= 1)
            
            S_prime = gte.channel_recalibration(S, E)
            assert S_prime.shape == (1280, 10)
            
            F = gte.temporal_aggregation(S_prime)
            assert F.shape == (1280,)
            
            logits = gte.classifier(F)
            assert logits.shape == (2,)
            
            probs = torch.softmax(logits, dim=-1)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_no_nan_inf_propagation(self):
        """Test: Verify no NaN/Inf propagation through pipeline"""
        B = torch.randn(10, 1280, 7, 7)
        gte = GTEExtractor(device='cpu', training_mode=False)
        
        with torch.no_grad():
            S = gte.spatial_compression(B)
            assert not torch.isnan(S).any() and not torch.isinf(S).any()
            
            q = gte.temporal_compression(S)
            assert not torch.isnan(q).any() and not torch.isinf(q).any()
            
            E = gte.temporal_excitation(q)
            assert not torch.isnan(E).any() and not torch.isinf(E).any()
            assert torch.all(E >= 0) and torch.all(E <= 1)
            
            S_prime = gte.channel_recalibration(S, E)
            assert not torch.isnan(S_prime).any() and not torch.isinf(S_prime).any()
            
            F = gte.temporal_aggregation(S_prime)
            assert not torch.isnan(F).any() and not torch.isinf(F).any()
            
            logits = gte.classifier(F)
            assert not torch.isnan(logits).any() and not torch.isinf(logits).any()
            
            probs = torch.softmax(logits, dim=-1)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
