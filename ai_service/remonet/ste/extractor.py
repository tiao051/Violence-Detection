"""
Short Temporal Extractor (STE) - Extract short-term spatiotemporal features.

Processes 30 motion frames into 10 temporal composites using MobileNetV2 backbone.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class STEOutput:
    """STE processing output with spatiotemporal feature maps."""
    camera_id: str
    timestamp: float
    features: torch.Tensor  # Shape: (T/3, C, W, H) = (10, 1280, 7, 7) per paper
    latency_ms: float


class STEExtractor:
    """
    Extract short-term temporal features from motion frames using MobileNetV2.
    
    Takes 3 consecutive frames, creates temporal composite by averaging RGB channels,
    then extracts spatial feature maps. Processes 30 frames into 10 temporal features.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 3,
        device: str = 'cuda'
    ):
        """
        Initialize STE extractor with MobileNetV2 backbone.
        
        Args:
            input_size: Expected input frame size (width, height) - default (224, 224)
            num_frames: Number of consecutive frames to process (default: 3 per paper)
            device: 'cuda' or 'cpu'
        """
        self.input_size = input_size
        self.num_frames = num_frames
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Build MobileNetV2 feature extractor
        self.backbone = self._build_backbone()
        self.backbone.to(self.device)
        self.backbone.eval() # comment this when we train the model, rn it's for inference only
        
        # ImageNet normalization parameters (REQUIRED for pretrained MobileNetV2)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer for normalization parameters."""
        setattr(self, name, tensor.to(self.device))

    def _build_backbone(self) -> nn.Module:
        """
        Build MobileNetV2 feature extractor (preserve spatial dimensions).
        
        Returns spatial feature maps (batch, 1280, 7, 7) without pooling or classifier.
        """
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Extract feature extractor only (no avgpool, no classifier)
        # Output is (batch, 1280, 7, 7) preserving spatial dimensions
        feature_extractor = mobilenet.features
        
        return feature_extractor

    def create_temporal_composite(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Create temporal composite from 3 consecutive frames.
        
        Process:
        1. Average RGB channels for each frame independently (grayscale)
        2. Stack 3 grayscale frames as pseudo-RGB image
        3. Normalize using ImageNet statistics
        
        Args:
            frames: List of 3 RGB frames (each: 224×224×3, uint8, [0-255])
                   (Validation is done upstream by SME)
            
        Returns:
            composite_normalized: Shape (224, 224, 3), float32, ImageNet-normalized
        """
        if len(frames) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frames)}")
        
        # Average RGB channels for each frame to get grayscale
        p_t = np.mean(frames[0].astype(np.float32), axis=2)    # (224, 224)
        p_t1 = np.mean(frames[1].astype(np.float32), axis=2)   # (224, 224)
        p_t2 = np.mean(frames[2].astype(np.float32), axis=2)   # (224, 224)
        
        # Stack as 3-channel pseudo-RGB image (224, 224, 3)
        composite = np.stack([p_t, p_t1, p_t2], axis=2)
        
        # Normalize to [0, 1] and apply ImageNet normalization
        composite_normalized = composite / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        composite_normalized = (composite_normalized - mean) / std
        
        return composite_normalized.astype(np.float32)

    def process_batch(self, frames: np.ndarray) -> torch.Tensor:
        """
        Process batch of 30 frames to extract spatiotemporal features.
        
        Converts 30 frames into 10 temporal composites (T/3).
        Extracts feature maps for all 10 composites in batch using MobileNetV2.
        
        Args:
            frames: numpy array (30, 224, 224, 3), RGB, uint8, [0-255]
            
        Returns:
            features: Tensor (T/3, C, W, H) = (10, 1280, 7, 7)
                     Feature maps preserving spatial dimensions as per paper
        """
        if len(frames) != 30:
            raise ValueError(f"Expected 30 frames, got {len(frames)}")
        
        # Create 10 temporal composites from 30 frames (T/3)
        composites = []
        for i in range(0, 30, 3):
            if i + 2 < len(frames):
                composite = self.create_temporal_composite([frames[i], frames[i+1], frames[i+2]])
                composites.append(composite)
        
        # Stack composites: (10, 224, 224, 3)
        composites_batch = np.array(composites)
        
        # Convert to tensor: (10, 224, 224, 3) -> (10, 3, 224, 224)
        composites_tensor = torch.from_numpy(composites_batch).permute(0, 3, 1, 2)
        composites_tensor = composites_tensor.to(self.device)
        
        # Extract feature maps in batch
        # Output: B ∈ ℝ^(T/3 × C × W × H) = (10, 1280, 7, 7)
        with torch.no_grad(): # if trainning e2e, remove no_grad()
            features = self.backbone(composites_tensor)  # (10, 1280, 7, 7)
        
        return features

    def process(
        self,
        frames: np.ndarray,
        camera_id: str = "unknown",
        timestamp: Optional[float] = None
    ) -> STEOutput:
        """
        Process 30 motion frames and generate spatiotemporal feature maps.
        
        According to paper: output is feature map B ∈ ℝ^(T/3 × C × W × H)
        where T=30, so output shape is (10, 1280, 7, 7) from MobileNetV2.
        
        Args:
            frames: numpy array (30, 224, 224, 3), RGB, uint8
            camera_id: Camera identifier
            timestamp: Timestamp for this batch (auto-generated if None)
            
        Returns:
            STEOutput with spatiotemporal feature maps (T/3, C, W, H)
        """
        start = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        features = self.process_batch(frames)
        latency_ms = (time.perf_counter() - start) * 1000
        
        return STEOutput(
            camera_id=camera_id,
            timestamp=timestamp,
            features=features,      # (10, 1280, 7, 7) - spatial feature maps for GTE
            latency_ms=latency_ms
        )