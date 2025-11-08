"""
Short Temporal Extractor (STE) - Extract short-term motion patterns from video frames.

Transforms consecutive motion ROIs into compact feature embeddings representing
temporal dynamics over small time windows (typically 3 frames).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class STEOutput:
    """STE processing output with spatiotemporal feature embedding."""
    camera_id: str
    timestamp: float
    embedding: np.ndarray
    latency_ms: float


class STEExtractor:
    """
    Extract short-term temporal features from motion frames using pretrained CNNs.
    
    Pipeline: Motion ROIs (from SME) → Temporal Composite → CNN Features → Embedding
    
    Paper: "Violence Detection Using Short and Global Temporal Extractors" (Sensors 2024)
    Section 3.3: STE extracts spatiotemporal features, outputs feature map B ∈ ℝ^(T/3×C×W×H)
    """

    def __init__(
        self,
        backbone: str = 'mobilenet_v2',
        input_size: Tuple[int, int] = (224, 224),
        embedding_dim: int = 1280,
        num_frames: int = 3,
        device: str = 'cuda'
    ):
        """
        Initialize STE extractor with pretrained backbone.
        
        Args:
            backbone: 'mobilenet_v2' (1280-d, lightweight) or 'resnet18' (512-d, accurate)
            input_size: Expected input frame size (width, height) - default (224, 224)
            embedding_dim: Output embedding dimension (auto-set for ResNet18)
            num_frames: Number of consecutive frames to process (default: 3 per paper)
            device: 'cuda' or 'cpu'
        """
        self.backbone_name = backbone
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.num_frames = num_frames
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Build feature extraction model
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Normalization parameters (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _build_model(self) -> nn.Module:
        """
        Build feature extraction model (no classifier).
        
        Supports two backbones:
        - MobileNetV2: 1280-dim embedding, ~3.5M params (lightweight for edge devices)
        - ResNet18: 512-dim embedding, ~11M params (better accuracy)
        
        Returns:
            Feature extractor only
        """
        if self.backbone_name == 'mobilenet_v2':
            # Why pretrained: Transfer learning from ImageNet provides better initial features
            # than random initialization, especially critical for small violence datasets
            backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            feature_extractor = backbone.features
            
            # Why AdaptiveAvgPool2d: Makes model input-size agnostic and reduces 
            # spatial dimensions (7x7) to single values while preserving channel semantics
            model = nn.Sequential(
                feature_extractor,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
                
        elif self.backbone_name == 'resnet18':
            # Why ResNet18: Skip connections enable deeper networks without vanishing gradients,
            # better for capturing complex temporal patterns in violence detection
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
   
            # Why [:-1]: Remove ImageNet classifier, keep feature extraction layers
            modules = list(backbone.children())[:-1]
            model = nn.Sequential(*modules, nn.Flatten())
            
            self.embedding_dim = 512
                
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        return model

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame: BGR->RGB, resize to expected size, normalize with ImageNet stats.
        
        Validates frame dimensions and resizes if needed to ensure consistent input size.
        """
        # Validate input is not empty
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")
        
        # Validate frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame with 3 channels (BGR), got shape {frame.shape}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size if not already correct
        if frame_rgb.shape[:2] != (self.input_size[1], self.input_size[0]):
            frame_rgb = cv2.resize(frame_rgb, self.input_size)
        
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_normalized = (frame_normalized - self.mean) / self.std
        
        return frame_normalized

    def _preprocess_frames_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Batch preprocess multiple frames efficiently (pre-allocated memory).
        
        Returns:
            preprocessed_frames: Shape (N, H, W, 3) where N is number of frames
        """
        n_frames = len(frames)
        h, w = self.input_size[1], self.input_size[0]
        
        # Pre-allocate output array (avoids copy overhead)
        preprocessed = np.empty((n_frames, h, w, 3), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            # Validate input is not empty
            if frame is None or frame.size == 0:
                raise ValueError("Input frame is empty or None")
            
            # Validate frame has 3 channels (BGR)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected frame with 3 channels (BGR), got shape {frame.shape}")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to expected input size if not already correct
            if frame_rgb.shape[:2] != (h, w):
                frame_rgb = cv2.resize(frame_rgb, self.input_size)
            
            # Normalize in-place to pre-allocated array
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            preprocessed[i] = (frame_normalized - self.mean) / self.std
        
        return preprocessed

    def create_temporal_composite(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Create temporal composite by averaging channels and stacking across time.
        
        Strategy: For each frame Mₜ (H×W×3), average over 3 RGB channels to get 
        grayscale pₜ (H×W×1). Then stack [pₜ, pₜ₊₁, pₜ₊₂] as 3 channels of composite P.
        
        Optimized: Uses contiguous memory layout for PyTorch.
        """
        if len(frames) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frames)}")
        
        # Batch preprocess all frames efficiently (vectorized, pre-allocated)
        preprocessed_frames = self._preprocess_frames_batch(frames)  # Shape: (3, H, W, 3)
        
        # Vectorized average over RGB channels
        # Shape: (3, H, W, 3) -> (3, H, W) using mean over last axis
        grayscale_frames = np.mean(preprocessed_frames, axis=3)  # More efficient than stack of means
        
        # Stack along new channel dimension: (3, H, W) -> (H, W, 3)
        # Using np.moveaxis for clarity and efficiency
        composite = np.moveaxis(grayscale_frames, 0, 2)  # Faster than stack for this case
        
        # Ensure contiguous memory (important for torch.from_numpy)
        return np.ascontiguousarray(composite)

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract spatio-temporal features from consecutive frames.
        
        Optimized: Uses torch.as_tensor with device parameter for zero-copy conversion.
        
        Returns:
            embedding: Feature vector (1280-d for MobileNetV2, 512-d for ResNet18)
        """
        composite = self.create_temporal_composite(frames)  # Already contiguous
        
        # Optimized tensor conversion: (H, W, 3) -> (1, 3, H, W)
        # torch.as_tensor with device=self.device avoids CPU tensor creation
        # Then permute and unsqueeze for model input shape
        composite_tensor = (
            torch.as_tensor(composite, device=self.device)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )
        
        with torch.inference_mode():
            output = self.model(composite_tensor)
            embedding = output.cpu().numpy()[0]
        
        return embedding

    def extract_features_batch(self, frame_sequences: List[List[np.ndarray]]) -> np.ndarray:
        """
        Extract features from multiple frame sequences in batch (optimized).
        
        Processes all composites in a single forward pass with pre-allocated memory.
        
        Args:
            frame_sequences: List of frame sequences, each with 3 frames
            
        Returns:
            embeddings: Shape (N, embedding_dim) where N is number of sequences
        """
        n_sequences = len(frame_sequences)
        h, w = self.input_size[1], self.input_size[0]
        
        # Pre-allocate composites array (N, H, W, 3)
        composites_batch = np.empty((n_sequences, h, w, 3), dtype=np.float32)
        
        # Fill directly instead of list + stack
        for i, frames in enumerate(frame_sequences):
            composite = self.create_temporal_composite(frames)
            composites_batch[i] = composite
        
        # Optimized tensor conversion: (N, H, W, 3) -> (N, 3, H, W)
        # torch.as_tensor with device parameter avoids CPU tensor creation
        composites_tensor = (
            torch.as_tensor(composites_batch, device=self.device)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        
        with torch.inference_mode():
            output = self.model(composites_tensor)
            embeddings = output.cpu().numpy()
        
        return embeddings

    def process(
        self,
        frames: List[np.ndarray],
        camera_id: str = "unknown",
        timestamp: Optional[float] = None
    ) -> STEOutput:
        """Process consecutive motion frames and generate feature embedding."""
        start = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        embedding = self.extract_features(frames)
        latency_ms = (time.perf_counter() - start) * 1000
        
        return STEOutput(
            camera_id=camera_id,
            timestamp=timestamp,
            embedding=embedding,
            latency_ms=latency_ms
        )

    def to_json(self, output: STEOutput) -> Dict:
        """Convert STEOutput to JSON-serializable dictionary."""
        return {
            "camera_id": output.camera_id,
            "timestamp": output.timestamp,
            "embedding": output.embedding.tolist(),
            "latency_ms": round(output.latency_ms, 2),
            "embedding_shape": output.embedding.shape
        }

    def batch_process(
        self,
        frame_sequences: List[List[np.ndarray]],
        camera_ids: Optional[List[str]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[STEOutput]:
        """Process multiple frame sequences efficiently using batch inference."""
        n_sequences = len(frame_sequences)
        
        # Generate IDs efficiently (only if needed)
        if camera_ids is None:
            camera_ids = [f"cam_{i:02d}" for i in range(n_sequences)]
        
        if timestamps is None:
            current_time = time.time()
            timestamps = [current_time] * n_sequences
        
        start = time.perf_counter()
        
        # Single batch inference (massive speedup vs sequential)
        embeddings = self.extract_features_batch(frame_sequences)
        
        latency_ms = (time.perf_counter() - start) * 1000
        latency_per_output = latency_ms / n_sequences if n_sequences > 0 else 0
        
        # Efficient list creation with zip
        return [
            STEOutput(
                camera_id=cam_id,
                timestamp=ts,
                embedding=embedding,
                latency_ms=latency_per_output
            )
            for cam_id, ts, embedding in zip(camera_ids, timestamps, embeddings)
        ]


def create_ste_extractor(
    backbone: str = 'mobilenet_v2',
    device: str = 'cuda'
) -> STEExtractor:
    """
    Factory function to create STE extractor.
    
    Args:
        backbone: CNN architecture for feature extraction
        device: Device for inference ('cuda' or 'cpu')
        
    Returns:
        Configured STEExtractor instance
    """
    return STEExtractor(
        backbone=backbone,
        device=device
    )
