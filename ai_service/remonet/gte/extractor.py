"""
Global Temporal Extractor (GTE) - Extract long-term temporal relationships.

Processes spatiotemporal feature maps from STE to learn global temporal patterns
and recalibrate features based on temporal importance. Outputs classification
for violence detection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import time
from dataclasses import dataclass


@dataclass
class GTEOutput:
    """GTE processing output with classification results."""
    camera_id: str
    timestamp: float
    no_violence_prob: float  # Class 0: Probability of no violence [0, 1]
    violence_prob: float     # Class 1: Probability of violence [0, 1]
    features: torch.Tensor   # Shape: (1280,) - final feature vector before classification
    latency_ms: float


class TemporalExcitation(nn.Module):
    """
    Temporal Excitation Module - Learn temporal importance weights.
    
    Applies 2 FC layers with ReLU and Sigmoid to learn which temporal frames
    are most important for violence detection.
    
    Architecture:
        FC(T/3) -> ReLU -> FC(T/3) -> Sigmoid
    """
    
    def __init__(self, temporal_dim: int = 10, reduction_factor: int = 2):
        """
        Initialize temporal excitation module.
        
        Args:
            temporal_dim: Number of temporal frames (T/3, default: 10)
            reduction_factor: Factor to reduce dimension in hidden layer
        """
        super().__init__()
        self.temporal_dim = temporal_dim
        hidden_dim = max(temporal_dim // reduction_factor, 1)
        
        self.fc1 = nn.Linear(temporal_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, temporal_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal excitation weights.
        
        Args:
            q: Temporal summary vector, shape (batch, temporal_dim) or (temporal_dim,)
            
        Returns:
            E: Excitation weights, shape same as input, values in [0, 1]
        """
        # Handle both batched and unbatched input
        original_shape = q.shape
        if q.dim() == 1:
            q = q.unsqueeze(0)  # (temporal_dim,) -> (1, temporal_dim)
        
        # FC -> ReLU -> FC -> Sigmoid
        h = self.relu(self.fc1(q))
        E = self.sigmoid(self.fc2(h))
        
        # Reshape back to original shape
        if len(original_shape) == 1:
            E = E.squeeze(0)  # (1, temporal_dim) -> (temporal_dim,)
        
        return E


class GTEExtractor(nn.Module):
    """
    Global Temporal Extractor (GTE) - Extract global temporal characteristics.
    
    Takes spatiotemporal feature maps B from STE module (T/3, C, H, W) and:
    1. Applies spatial compression (Global Average Pooling) → S (C, T/3)
    2. Applies temporal compression (Global Average Pooling) → q (T/3,)
    3. Learns temporal excitation weights via 2 FC layers → E (T/3,)
    4. Recalibrates features with learned weights → S' (C, T/3)
    5. Aggregates temporal dimension → F (C,)
    6. Classifies as Violence/No Violence
    
    Per paper: improves effectiveness by learning relationships between temporal frames.
    """
    
    def __init__(
        self,
        num_channels: int = 1280,
        temporal_dim: int = 10,
        num_classes: int = 2,
        device: str = 'cuda',
        training_mode: bool = False
    ):
        """
        Initialize GTE extractor.
        
        Args:
            num_channels: Number of feature channels from STE (default: 1280 from MobileNetV2)
            temporal_dim: Number of temporal frames (T/3, default: 10 from STE)
            num_classes: Number of output classes (default: 2 for binary classification)
            device: 'cuda' or 'cpu'
            training_mode: If True, model is in training mode. If False (default), inference mode.
        """
        super().__init__()
        self.num_channels = num_channels
        self.temporal_dim = temporal_dim
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.training_mode = training_mode
        
        # Temporal Excitation Module - learns which frames are important
        self.temporal_excitation = TemporalExcitation(
            temporal_dim=temporal_dim,
            reduction_factor=2
        )
        
        # Classification head - FC layer for Violence/No Violence
        self.classifier = nn.Linear(num_channels, num_classes)
        
        # Move to device
        self.to(self.device)
        
        # Set training mode
        if self.training_mode:
            self.train()
        else:
            self.eval()
    
    def spatial_compression(self, B: torch.Tensor) -> torch.Tensor:
        """
        Apply Global Average Pooling on spatial dimensions.
        
        Compresses spatial information (H×W) while preserving channel and temporal dims.
        Accepts STE output format: (T/3, C, H, W)
        
        Formula:
            S_c = (1 / (H×W)) * Σ Σ B_c(i, j)
        
        Args:
            B: Feature map from STE, shape (T/3, C, H, W) or (batch, T/3, C, H, W)
               Example: (10, 1280, 7, 7) from STE
            
        Returns:
            S: Spatially compressed features, shape (C, T/3)
        """
        # Handle both 4D and 5D input
        if B.dim() == 4:
            # Unbatched: (T/3, C, H, W)
            # Reorder to (C, T/3, H, W) for processing
            B = B.permute(1, 0, 2, 3)  # (C, T/3, H, W)
            squeeze_output = True
        elif B.dim() == 5:
            # Batched: (batch, T/3, C, H, W)
            # Reorder to (batch, C, T/3, H, W)
            B = B.permute(0, 2, 1, 3, 4)  # (batch, C, T/3, H, W)
            squeeze_output = False
        else:
            raise ValueError(f"Expected 4D or 5D input, got {B.dim()}D")
        
        # Global Average Pooling over spatial dimensions (H, W)
        S = torch.mean(B, dim=(-2, -1))  # (C, T/3) or (batch, C, T/3)
        
        return S
    
    def temporal_compression(self, S: torch.Tensor) -> torch.Tensor:
        """
        Apply Global Average Pooling on channel dimension.
        
        Compresses channel information to get temporal summary.
        
        Formula:
            q = (1 / C) * Σ S_c
        
        Args:
            S: Spatially compressed features, shape (C, T/3) or (batch, C, T/3)
               Example: (1280, 10)
            
        Returns:
            q: Temporal summary vector, shape (T/3,) or (batch, T/3)
               Single value per temporal frame representing all channels
        """
        # Mean over channel dimension (dim 0 for unbatched, dim 1 for batched)
        if S.dim() == 3:
            # Batched: (batch, C, T/3) -> mean over dim 1 (channel)
            q = torch.mean(S, dim=1)  # (batch, T/3)
        else:
            # Unbatched: (C, T/3) -> mean over dim 0 (channel)
            q = torch.mean(S, dim=0)  # (T/3,)
        
        return q
    
    def channel_recalibration(self, S: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Recalibrate channel features using temporal excitation weights.
        
        Applies element-wise multiplication between S and E to reweight features
        based on learned temporal importance.
        
        Formula:
            S'_c = S_c ⊙ E  (element-wise multiplication with broadcasting)
        
        Args:
            S: Spatially compressed features, shape (C, T/3) or (batch, C, T/3)
            E: Temporal excitation weights, shape (T/3,) or (batch, T/3)
               Values in [0, 1] representing importance of each temporal frame
            
        Returns:
            S_prime: Recalibrated features, same shape as S
        """
        # Determine if batched
        if S.dim() == 3:
            # Batched: (batch, C, T/3) * (batch, T/3) -> broadcast and multiply
            # Need to reshape E for broadcasting: (batch, T/3) -> (batch, 1, T/3)
            E_expanded = E.unsqueeze(1)  # (batch, 1, T/3)
            S_prime = S * E_expanded  # (batch, C, T/3)
        else:
            # Unbatched: (C, T/3) * (T/3,) -> broadcast and multiply
            # E will automatically broadcast: (C, T/3) * (T/3,) -> (C, T/3)
            S_prime = S * E.unsqueeze(0)  # (C, 1, T/3) * (T/3,) -> (C, T/3)
        
        return S_prime
    
    def temporal_aggregation(self, S_prime: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal dimension to create final feature vector.
        
        Sums all temporal frames to get global temporal characteristics.
        
        Formula:
            F = Σ S'_{:,t} for t=1 to T/3
        
        Args:
            S_prime: Recalibrated features, shape (C, T/3) or (batch, C, T/3)
            
        Returns:
            F: Aggregated feature vector, shape (C,) or (batch, C)
               Final representation combining all temporal frames
        """
        # Sum over temporal dimension
        if S_prime.dim() == 3:
            # Batched: (batch, C, T/3) -> sum over dim 2 (T/3)
            F = torch.sum(S_prime, dim=2)  # (batch, C)
        else:
            # Unbatched: (C, T/3) -> sum over dim 1 (T/3)
            F = torch.sum(S_prime, dim=1)  # (C,)
        
        return F
    
    def forward(self, B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GTE module.
        
        Takes STE feature maps and outputs classification logits.
        
        Args:
            B: Feature map from STE
               Shape: (T/3, C, H, W) = (10, 1280, 7, 7) for unbatched
                      (batch, T/3, C, H, W) for batched
            
        Returns:
            logits: Classification logits, shape (num_classes,) or (batch, num_classes)
        """
        # Step 1: Spatial Compression
        S = self.spatial_compression(B)  # (C, T/3)
        
        # Step 2: Temporal Compression
        q = self.temporal_compression(S)  # (T/3,)
        
        # Step 3: Learn Temporal Excitation
        E = self.temporal_excitation(q)  # (T/3,)
        
        # Step 4: Recalibrate Channel Features
        S_prime = self.channel_recalibration(S, E)  # (C, T/3)
        
        # Step 5: Temporal Aggregation
        F = self.temporal_aggregation(S_prime)  # (C,)
        
        # Step 6: Classification
        logits = self.classifier(F)  # (num_classes,)
        
        return logits
    
    def process(
        self,
        features: torch.Tensor,
        camera_id: str = "unknown",
        timestamp: Optional[float] = None
    ) -> GTEOutput:
        """
        Process STE feature maps and generate violence classification.
        
        Args:
            features: Feature map tensor from STE
                     Shape: (C, T/3, H, W) = (1280, 10, 7, 7)
            camera_id: Camera identifier
            timestamp: Timestamp for this batch (auto-generated if None)
            
        Returns:
            GTEOutput with classification probabilities and features
        """
        start = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure features are on correct device
        features = features.to(self.device)
        
        # Forward pass
        if self.training_mode:
            logits = self.forward(features)
        else:
            with torch.no_grad():
                logits = self.forward(features)
        
        # Get intermediate features (before classification)
        S = self.spatial_compression(features)
        q = self.temporal_compression(S)
        E = self.temporal_excitation(q)
        S_prime = self.channel_recalibration(S, E)
        final_features = self.temporal_aggregation(S_prime)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Extract individual probabilities
        # Class 0: no_violence, Class 1: violence
        if probs.dim() == 1:
            # Unbatched case
            no_violence_prob = float(probs[0].item())
            violence_prob = float(probs[1].item()) if self.num_classes == 2 else 1.0 - no_violence_prob
        else:
            # Batched case - take first sample
            no_violence_prob = float(probs[0, 0].item())
            violence_prob = float(probs[0, 1].item()) if self.num_classes == 2 else 1.0 - no_violence_prob
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        return GTEOutput(
            camera_id=camera_id,
            timestamp=timestamp,
            no_violence_prob=no_violence_prob,
            violence_prob=violence_prob,
            features=final_features.cpu().detach(),
            latency_ms=latency_ms
        )
