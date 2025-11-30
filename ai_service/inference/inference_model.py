"""
Violence Detection Inference Model.

Loads trained model and performs inference on video frames.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
import logging
from dataclasses import dataclass
import time

from ai_service.remonet.sme.extractor import SMEExtractor
from ai_service.remonet.ste.extractor import STEExtractor, BackboneType, BACKBONE_CONFIG
from ai_service.remonet.gte.extractor import GTEExtractor

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for violence detection inference."""
    model_path: str
    backbone: str = 'mobilenet_v2'  # STE backbone used during training
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    confidence_threshold: float = 0.5
    num_frames: int = 30
    frame_size: tuple = (224, 224)


class ViolenceDetectionModel:
    """
    Violence detection model for real-time inference.
    
    Pipeline:
    1. Load raw frame (224x224)
    2. Apply SME (Spatial Motion Extraction) - optical flow
    3. Apply STE (Spatial Temporal Feature Extraction) - backbone features
    4. Apply GTE (Global Temporal Extraction) - classification
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize violence detection model.
        
        Args:
            config: InferenceConfig with model path and device
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize extractors
        self.sme_extractor = SMEExtractor()
        self.ste_extractor = STEExtractor(device=config.device, training_mode=False, backbone=config.backbone)
        
        # Get backbone config for GTE initialization
        self.backbone_config = BACKBONE_CONFIG[BackboneType(config.backbone)]
        
        # Load trained GTE model
        self.gte_model = self._load_gte_model()
        
        # Frame buffer for temporal aggregation
        self.frame_buffer = []
        self.MAX_BUFFER_SIZE = config.num_frames
        
        # Latency tracking
        self.last_inference_time = 0.0
        self.last_inference_latency = 0.0
        
        logger.info(f"ViolenceDetectionModel initialized on {self.device}")
    
    def _load_gte_model(self) -> GTEExtractor:
        """Load trained GTE model from checkpoint."""
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Check if checkpoint has backbone and num_channels info
        checkpoint_backbone = checkpoint.get('backbone', None)
        checkpoint_num_channels = checkpoint.get('num_channels', None)
        
        if checkpoint_backbone is None or checkpoint_num_channels is None:
            logger.warning(
                f"Model checkpoint config incomplete:"
            )
            if checkpoint_backbone is None:
                logger.warning(f"   - 'backbone' field missing (using config default: {self.config.backbone})")
            if checkpoint_num_channels is None:
                logger.warning(f"   - 'num_channels' field missing (using config backbone's channels: {self.backbone_config['out_channels']})")
            logger.warning(f"   → Skipping backbone validation. Using inference config values.")
            logger.warning(f"   → Model was likely trained with old code. Next training will include this info.\n")
        else:
            # Checkpoint has both fields - validate them
            logger.info(
                f"Model checkpoint config complete:"
            )
            logger.info(f"   - Backbone: {checkpoint_backbone}")
            logger.info(f"   - Num channels: {checkpoint_num_channels}")
            
            # Validate backbone matches
            if checkpoint_backbone != self.config.backbone:
                logger.warning(
                    f"Backbone mismatch: checkpoint trained with '{checkpoint_backbone}' "
                    f"but config specifies '{self.config.backbone}'. "
                    f"Auto-correcting to use checkpoint backbone.\n"
                )
                # Auto-correct to use checkpoint backbone
                self.config.backbone = checkpoint_backbone
                self.backbone_config = BACKBONE_CONFIG[BackboneType(checkpoint_backbone)]
                # Reinitialize STE with correct backbone
                self.ste_extractor = STEExtractor(
                    device=self.config.device,
                    training_mode=False,
                    backbone=checkpoint_backbone
                )
            
            # Verify num_channels matches expected value
            expected_channels = self.backbone_config['out_channels']
            if checkpoint_num_channels != expected_channels:
                raise ValueError(
                    f"Channel mismatch: checkpoint has {checkpoint_num_channels} channels "
                    f"but backbone {self.config.backbone} should have {expected_channels}"
                )
        
        # Get num_channels (from checkpoint if available, else from config)
        num_channels = checkpoint_num_channels or self.backbone_config['out_channels']
        
        # Create model instance
        gte_model = GTEExtractor(
            num_channels=num_channels,
            temporal_dim=10,
            num_classes=2,
            device=self.config.device
        )
        
        # Load weights
        try:
            gte_model.load_state_dict(checkpoint['model_state_dict'])
            gte_model.eval()
            best_val_acc = checkpoint.get('best_val_acc', 'N/A')
            logger.info(
                f"\nModel loaded successfully:"
            )
            logger.info(f"   - Path: {model_path}")
            logger.info(f"   - Backbone: {self.config.backbone}")
            logger.info(f"   - Channels: {num_channels}")
            logger.info(f"   - Best val accuracy: {best_val_acc}\n")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
        
        return gte_model
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add frame to buffer.
        
        Args:
            frame: Input frame (BGR, uint8, 224×224)
                   Should already be resized by camera_worker
        """
        # Ensure frame is uint8 (preprocessing step)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB (frame should already be 224×224)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.frame_buffer.append(frame_rgb)
        
        # Keep buffer at max size (sliding window)
        if len(self.frame_buffer) > self.MAX_BUFFER_SIZE:
            self.frame_buffer.pop(0)
    
    def predict(self) -> Optional[Dict]:
        """
        Perform inference on buffered frames.
        
        Returns:
            Dict with detection result if buffer is full, else None
        """
        # Skip if buffer not full
        if len(self.frame_buffer) < self.MAX_BUFFER_SIZE:
            return None
        
        try:
            inference_start = time.time()
            
            # Convert buffer to numpy array
            frames = np.array(self.frame_buffer, dtype=np.uint8)  # (30, 224, 224, 3)
            
            with torch.no_grad():
                # Step 1: Apply SME (Spatial Motion Extraction)
                motion_frames = self.sme_extractor.process_batch(frames)
                
                # Step 2: Apply STE (Spatial Temporal Feature Extraction)
                ste_output = self.ste_extractor.process(motion_frames)
                features = ste_output.features  # (10, C, 7, 7) where C depends on backbone
                
                # Step 3: Apply GTE (Global Temporal Extraction)
                gte_output = self.gte_model.process(features, timestamp=time.time())
            
            # Extract probabilities from GTE output
            # violence_prob = P(class=1=Violence)
            # no_violence_prob = P(class=0=NonViolence)
            violence_prob = gte_output.violence_prob
            no_violence_prob = gte_output.no_violence_prob
            
            # Determine if violence detected
            is_violence = violence_prob >= self.config.confidence_threshold
            confidence = violence_prob
            
            # Calculate latency
            self.last_inference_latency = (time.time() - inference_start) * 1000  # ms
            self.last_inference_time = time.time()
            
            return {
                'violence': is_violence,
                'confidence': confidence,
                'buffer_size': len(self.frame_buffer),
                'latency_ms': self.last_inference_latency
            }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'violence': False,
                'confidence': 0.0,
                'error': str(e),
                'latency_ms': 0.0
            }
    
    def reset_buffer(self) -> None:
        """Clear frame buffer."""
        self.frame_buffer.clear()


# Lazy-loaded singleton instance
_model_instance: Optional[ViolenceDetectionModel] = None


def get_violence_detection_model(config: Optional[InferenceConfig] = None) -> ViolenceDetectionModel:
    """
    Get or create violence detection model instance.
    
    Args:
        config: InferenceConfig (required on first call)
    
    Returns:
        ViolenceDetectionModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        if config is None:
            raise ValueError("config required on first initialization")
        _model_instance = ViolenceDetectionModel(config)
    
    return _model_instance
