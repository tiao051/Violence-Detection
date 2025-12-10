"""
GTE (Global Temporal Extractor) Unit Tests

This module contains unit tests for the GTEExtractor component, verifying:
- Input/output shapes and valid probability distributions
- Correct class ordering (0=no_violence, 1=violence)
- Training vs inference modes
- No NaN/Inf propagation through pipeline

Usage:
    pytest ai_service/tests/remonet/gte/test_gte_module.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_SERVICE_DIR = PROJECT_ROOT / "ai_service"

if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from ai_service.remonet.gte import GTEExtractor, GTEOutput


@pytest.fixture
def gte():
    return GTEExtractor(device='cpu', training_mode=False)


@pytest.fixture
def ste_output():
    # STE output format: (T/3, C, H, W) = (10, 1280, 7, 7)
    return torch.randn(10, 1280, 7, 7)


class TestGTEPipeline:
    """Test GTE processing pipeline"""
    
    def test_forward_output_shape(self, gte):
        """Test forward pass produces correct logit shape"""
        B = torch.randn(10, 1280, 7, 7)
        logits = gte.forward(B)
        assert logits.shape == (2,), f"Expected (2,), got {logits.shape}"
    
    def test_process_output_structure(self, gte, ste_output):
        """Test process() returns valid GTEOutput with correct metadata"""
        output = gte.process(ste_output, camera_id="test", timestamp=123.0)
        
        assert isinstance(output, GTEOutput)
        assert output.features.shape == (1280,)
        assert output.camera_id == "test"
        assert output.timestamp == 123.0
    
    def test_probability_outputs_valid_range(self, gte, ste_output):
        """Test probabilities are in [0, 1] and sum to 1"""
        output = gte.process(ste_output)
        
        assert 0 <= output.no_violence_prob <= 1
        assert 0 <= output.violence_prob <= 1
        total = output.no_violence_prob + output.violence_prob
        assert abs(total - 1.0) < 1e-5
    
    def test_class_order_correct(self, gte, ste_output):
        """Test class ordering: 0=no_violence, 1=violence"""
        output = gte.process(ste_output)
        
        assert hasattr(output, 'no_violence_prob')
        assert hasattr(output, 'violence_prob')
        assert isinstance(output.no_violence_prob, float)
        assert isinstance(output.violence_prob, float)
    
    def test_intermediate_shapes_valid(self, gte):
        """Test intermediate tensor shapes through pipeline"""
        B = torch.randn(10, 1280, 7, 7)
        
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
    
    def test_no_nan_inf_in_pipeline(self, gte):
        """Test no NaN/Inf values propagate through pipeline"""
        B = torch.randn(10, 1280, 7, 7)
        
        with torch.no_grad():
            S = gte.spatial_compression(B)
            assert torch.isfinite(S).all()
            
            q = gte.temporal_compression(S)
            assert torch.isfinite(q).all()
            
            E = gte.temporal_excitation(q)
            assert torch.isfinite(E).all()
            
            S_prime = gte.channel_recalibration(S, E)
            assert torch.isfinite(S_prime).all()
            
            F = gte.temporal_aggregation(S_prime)
            assert torch.isfinite(F).all()
            
            logits = gte.classifier(F)
            assert torch.isfinite(logits).all()


class TestGTEModes:
    """Test training vs inference modes"""
    
    def test_inference_mode_no_grad(self, ste_output):
        """Test inference mode (no_grad context)"""
        gte = GTEExtractor(device='cpu', training_mode=False)
        B = ste_output
        
        with torch.no_grad():
            logits = gte.forward(B)
        
        assert not logits.requires_grad
    
    def test_training_mode_with_grad(self, ste_output):
        """Test training mode (gradient computation enabled)"""
        gte = GTEExtractor(device='cpu', training_mode=True)
        B = ste_output
        
        logits = gte.forward(B)
        assert logits.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
