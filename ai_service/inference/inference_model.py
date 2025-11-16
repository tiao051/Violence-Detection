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

from ai_service.remonet.sme.extractor import SMEExtractor
from ai_service.remonet.ste.extractor import STEExtractor
from ai_service.remonet.gte.extractor import GTEExtractor

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for violence detection inference."""
    model_path: str
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
        self.ste_extractor = STEExtractor(device=config.device, training_mode=False)
        
        # Load trained GTE model
        self.gte_model = self._load_gte_model()
        
        # Frame buffer for temporal aggregation
        self.frame_buffer = []
        self.MAX_BUFFER_SIZE = config.num_frames
        
        logger.info(f"ViolenceDetectionModel initialized on {self.device}")
    
    def _load_gte_model(self) -> GTEExtractor:
        """Load trained GTE model from checkpoint."""
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model instance (matches training setup)
        gte_model = GTEExtractor(
            num_channels=1280,  # MobileNetV2 output channels
            temporal_dim=10,     # 30 frames / 3 = 10
            num_classes=2,       # Violence / NonViolence
            device=self.config.device
        )
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            gte_model.load_state_dict(checkpoint['model_state_dict'])
            gte_model.eval()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return gte_model
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add frame to buffer.
        
        Args:
            frame: Input frame (BGR, uint8, any size)
        """
        # Resize to expected size
        if frame.shape[:2] != self.config.frame_size:
            import cv2
            frame = cv2.resize(frame, self.config.frame_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.frame_buffer.append(frame_rgb)
        
        # Keep buffer at max size (sliding window)
        if len(self.frame_buffer) > self.MAX_BUFFER_SIZE:
            self.frame_buffer.pop(0)
    
    def predict(self) -> Dict:
        """
        Perform inference on buffered frames.
        
        Returns:
            Dict with keys:
                - violence: bool (True if violence detected)
                - confidence: float (0-1)
                - class_id: int (0=Violence, 1=NonViolence)
        """
        # Need minimum frames for inference
        if len(self.frame_buffer) < self.MAX_BUFFER_SIZE:
            return {
                'violence': False,
                'confidence': 0.0,
                'class_id': 1,
                'buffer_size': len(self.frame_buffer)
            }
        
        try:
            # Convert buffer to numpy array
            frames = np.array(self.frame_buffer, dtype=np.uint8)  # (30, 224, 224, 3)
            
            with torch.no_grad():
                # Step 1: Apply SME (Spatial Motion Extraction)
                motion_frames = self.sme_extractor.process_batch(frames)
                
                # Step 2: Apply STE (Spatial Temporal Feature Extraction)
                ste_output = self.ste_extractor.process(motion_frames)
                features = ste_output.features  # (10, 1280, 7, 7)
                
                # Step 3: Apply GTE (Global Temporal Extraction)
                logits = self.gte_model.forward(features)  # (num_classes,)
                
                # Get predictions
                probabilities = torch.softmax(logits, dim=0).cpu().numpy()
                class_id = int(torch.argmax(logits).cpu().numpy())
                confidence = float(probabilities[0])  # Violence confidence
            
            # Determine if violence detected
            is_violence = (class_id == 0) and (confidence >= self.config.confidence_threshold)
            
            return {
                'violence': is_violence,
                'confidence': confidence,
                'class_id': class_id,
                'buffer_size': len(self.frame_buffer)
            }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'violence': False,
                'confidence': 0.0,
                'class_id': 1,
                'error': str(e)
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
