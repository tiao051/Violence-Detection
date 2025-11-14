"""
Short Temporal Extractor (STE) - Extract short-term spatiotemporal features.

Processes 30 motion frames into 10 temporal composites using various CNN backbones.
Supports: MobileNetV2, MobileNetV3, EfficientNet B0, MNasNet
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights, MNASNet1_0_Weights
)
from typing import List, Dict, Tuple, Optional, Literal
import time
from dataclasses import dataclass
from enum import Enum


class BackboneType(str, Enum):
    """Supported CNN backbones for feature extraction."""
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    EFFICIENTNET_B0 = "efficientnet_b0"
    MNASNET = "mnasnet"
    
    def __str__(self):
        return self.value


# Backbone output channel configuration
BACKBONE_CONFIG = {
    BackboneType.MOBILENET_V2: {
        'out_channels': 1280,
        'spatial_size': 7,  # 224 / 32 = 7
        'weights_class': MobileNet_V2_Weights.IMAGENET1K_V1,
    },
    BackboneType.MOBILENET_V3_SMALL: {
        'out_channels': 576,
        'spatial_size': 7,
        'weights_class': MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    },
    BackboneType.MOBILENET_V3_LARGE: {
        'out_channels': 960,
        'spatial_size': 7,
        'weights_class': MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    },
    BackboneType.EFFICIENTNET_B0: {
        'out_channels': 1280,
        'spatial_size': 7,
        'weights_class': EfficientNet_B0_Weights.IMAGENET1K_V1,
    },
    BackboneType.MNASNET: {
        'out_channels': 1280,
        'spatial_size': 7,
        'weights_class': MNASNet1_0_Weights.IMAGENET1K_V1,
    },
}


@dataclass
class STEOutput:
    """STE processing output with spatiotemporal feature maps."""
    camera_id: str
    timestamp: float
    features: torch.Tensor  # Shape: (T/3, C, W, H) - varies by backbone
    latency_ms: float
    backbone: str = "mobilenet_v2"  # Which backbone was used


class STEExtractor:
    """
    Extract short-term temporal features from motion frames using various CNN backbones.
    
    Takes 3 consecutive frames, creates temporal composite by averaging RGB channels,
    then extracts spatial feature maps. Processes 30 frames into 10 temporal features.
    
    Supported backbones:
    - MobileNetV2 (1280 channels, default)
    - MobileNetV3 Small (576 channels)
    - MobileNetV3 Large (960 channels)
    - EfficientNet B0 (1280 channels)
    - MNasNet (1280 channels)
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 3,
        device: str = 'cuda',
        training_mode: bool = False,
        backbone: BackboneType = BackboneType.MOBILENET_V2
    ):
        """
        Initialize STE extractor with specified backbone.
        
        Args:
            input_size: Expected input frame size (width, height) - default (224, 224)
            num_frames: Number of consecutive frames to process (default: 3 per paper)
            device: 'cuda' or 'cpu'
            training_mode: If True, enables gradient computation for training.
                          If False (default), inference mode with no_grad.
            backbone: CNN backbone to use for feature extraction.
                     Options: 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                             'efficientnet_b0', 'mnasnet'
        """
        self.input_size = input_size
        self.num_frames = num_frames
        # Respect user's device choice, but validate if 'cuda' is requested
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.training_mode = training_mode
        
        # Convert string to BackboneType if needed
        if isinstance(backbone, str):
            backbone = BackboneType(backbone)
        self.backbone_type = backbone
        self.backbone_config = BACKBONE_CONFIG[backbone]

        # Build backbone feature extractor
        self.backbone = self._build_backbone()
        self.backbone.to(self.device)
        
        # Set model mode based on training_mode flag
        if self.training_mode:
            self.backbone.train()
        else:
            self.backbone.eval()

    def _build_backbone(self) -> nn.Module:
        """
        Build feature extractor based on selected backbone.
        
        Returns spatial feature maps preserving spatial dimensions.
        Output channels and spatial size depend on chosen backbone.
        """
        backbone_type = self.backbone_type
        config = self.backbone_config
        
        if backbone_type == BackboneType.MOBILENET_V2:
            mobilenet = models.mobilenet_v2(weights=config['weights_class'])
            feature_extractor = mobilenet.features
            
        elif backbone_type == BackboneType.MOBILENET_V3_SMALL:
            mobilenet = models.mobilenet_v3_small(weights=config['weights_class'])
            feature_extractor = mobilenet.features
            
        elif backbone_type == BackboneType.MOBILENET_V3_LARGE:
            mobilenet = models.mobilenet_v3_large(weights=config['weights_class'])
            feature_extractor = mobilenet.features
            
        elif backbone_type == BackboneType.EFFICIENTNET_B0:
            efficientnet = models.efficientnet_b0(weights=config['weights_class'])
            feature_extractor = efficientnet.features
            
        elif backbone_type == BackboneType.MNASNET:
            mnasnet = models.mnasnet1_0(weights=config['weights_class'])
            # MNASNet uses layers attribute instead of features
            feature_extractor = mnasnet.layers
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        return feature_extractor

    def create_temporal_composite(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Create temporal composite from 3 consecutive frames.
        
        Process (per paper):
        1. Average RGB channels for each frame independently (grayscale)
        2. Stack 3 grayscale frames as pseudo-RGB image P_t,t+1,t+2
        3. Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        Args:
            frames: List of 3 RGB motion frames (each: 224×224×3, uint8, [0-255])
            
        Returns:
            composite_normalized: Shape (224, 224, 3), float32, ImageNet normalized
        """
        if len(frames) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frames)}")
        
        # Average RGB channels for each frame to get grayscale
        # M_t (H×W×3) → p_t = mean(RGB channels) → (H×W)
        p_t = np.mean(frames[0].astype(np.float32), axis=2)    # (224, 224)
        p_t1 = np.mean(frames[1].astype(np.float32), axis=2)   # (224, 224)
        p_t2 = np.mean(frames[2].astype(np.float32), axis=2)   # (224, 224)
        
        # Stack 3 temporal frames as channels to create composite P_t,t+1,t+2
        # Temporal information is encoded in the 3 channels
        composite = np.stack([p_t, p_t1, p_t2], axis=2)  # (224, 224, 3)
        
        # Normalize to [0, 1] range first
        composite = composite / 255.0
        
        # Apply ImageNet normalization (mean and std per channel)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        composite_normalized = (composite - imagenet_mean) / imagenet_std
        
        return composite_normalized.astype(np.float32)

    def process_batch(self, frames: np.ndarray) -> torch.Tensor:
        """
        Process batch of 30 frames to extract spatiotemporal features.
        
        Converts 30 frames into 10 temporal composites (T/3).
        Extracts feature maps for all 10 composites in batch using selected backbone.
        
        Args:
            frames: numpy array (30, 224, 224, 3), RGB, uint8, [0-255]
            
        Returns:
            features: Tensor (T/3, C, W, H) where:
                     - T/3 = 10 (30 frames / 3)
                     - C = depends on backbone (1280 for V2/B0/MNasNet, 576/960 for V3)
                     - W, H = spatial dimensions (typically 7x7)
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
        if self.training_mode:
            # Training: enable gradient computation
            features = self.backbone(composites_tensor)
        else:
            # Inference: no gradient computation
            with torch.no_grad():
                features = self.backbone(composites_tensor)
        
        return features

    def process(
        self,
        frames: np.ndarray,
        camera_id: str = "unknown",
        timestamp: Optional[float] = None
    ) -> STEOutput:
        """
        Process 30 motion frames and generate spatiotemporal feature maps.
        
        Converts 30 frames to 10 temporal composites and extracts features
        using selected backbone. Output dimensions vary by backbone:
        
        - MobileNetV2: (10, 1280, 7, 7)
        - MobileNetV3 Small: (10, 576, 7, 7)
        - MobileNetV3 Large: (10, 960, 7, 7)
        - EfficientNet B0: (10, 1280, 7, 7)
        - MNasNet: (10, 1280, 7, 7)
        
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
            features=features,
            latency_ms=latency_ms,
            backbone=str(self.backbone_type)
        )

    def get_backbone_info(self) -> Dict[str, any]:
        """
        Get configuration information about the current backbone.
        
        Returns:
            Dictionary with backbone name, output channels, and spatial size
        """
        return {
            'backbone': str(self.backbone_type),
            'out_channels': self.backbone_config['out_channels'],
            'spatial_size': self.backbone_config['spatial_size'],
            'feature_shape': (10, self.backbone_config['out_channels'], 
                            self.backbone_config['spatial_size'], 
                            self.backbone_config['spatial_size']),
        }

    @staticmethod
    def get_available_backbones() -> List[str]:
        """Get list of available backbone types."""
        return [b.value for b in BackboneType]