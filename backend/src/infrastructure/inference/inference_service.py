"""Inference service for violence detection."""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add ai_service to path
ai_service_path = Path(__file__).parent.parent.parent.parent / 'ai_service'
sys.path.insert(0, str(ai_service_path))

try:
    from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig
except ImportError:
    # Fallback if ai_service not available
    ViolenceDetectionModel = None
    InferenceConfig = None

logger = logging.getLogger(__name__)


def _auto_detect_model() -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect best available model and backbone.
    
    Looks for checkpoints in ai_service/training/two-stage/checkpoints/
    Priority order: best_model_hf_v3.pt → best_model_rwf_v3.pt → best_model_hf.pt → best_model_rwf.pt
    
    Returns:
        Tuple of (model_path, backbone) or (None, None) if no model found
    """
    checkpoints_dir = ai_service_path / 'training' / 'two-stage' / 'checkpoints'
    
    if not checkpoints_dir.exists():
        logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
        return None, None
    
    # Priority order
    priority_models = [
        'best_model_hf_v3.pt',
        'best_model_rwf_v3.pt',
        'best_model_hf.pt',
        'best_model_rwf.pt'
    ]
    
    for model_name in priority_models:
        model_path = checkpoints_dir / model_name
        if model_path.exists():
            # Auto-detect backbone from filename
            backbone = 'mobilenet_v3_small' if 'v3' in model_name else 'mobilenet_v2'
            logger.info(f"Auto-detected model: {model_name} (backbone: {backbone})")
            return str(model_path), backbone
    
    logger.warning(f"No models found in {checkpoints_dir}")
    return None, None


class InferenceService:
    """
    Service for violence detection inference.
    
    Wraps ViolenceDetectionModel with singleton pattern and error handling.
    """
    
    _instance: Optional['InferenceService'] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize inference service."""
        if self._initialized:
            return
        
        self.model: Optional[object] = None
        self._initialized = True
        
        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference_")
    
    def initialize(self, model_path: Optional[str] = None, device: str = 'cuda', confidence_threshold: float = 0.5, backbone: Optional[str] = None) -> None:
        """
        Initialize violence detection model.
        
        If model_path is None, auto-detects best available model.
        If backbone is None, auto-detects from model filename.
        
        Args:
            model_path: Path to trained model checkpoint. If None, auto-detects best model.
            device: Device to use ('cuda' or 'cpu')
            confidence_threshold: Confidence threshold for violence detection
            backbone: Backbone architecture name. If None, auto-detects from model filename.
        """
        try:
            if ViolenceDetectionModel is None:
                logger.warning("ViolenceDetectionModel not available, using dummy model")
                self.model = None
                return
            
            # Auto-detect model if not provided
            if model_path is None:
                model_path, detected_backbone = _auto_detect_model()
                if model_path is None:
                    logger.error("Could not auto-detect model and no model_path provided")
                    self.model = None
                    return
                if backbone is None:
                    backbone = detected_backbone
            
            # Auto-detect backbone from filename if not provided
            if backbone is None:
                backbone = 'mobilenet_v3_small' if 'v3' in model_path else 'mobilenet_v2'
                logger.info(f"Auto-detected backbone from filename: {backbone}")
                
            config = InferenceConfig(
                model_path=model_path,
                backbone=backbone,
                device=device,
                confidence_threshold=confidence_threshold
            )
            self.model = ViolenceDetectionModel(config)
            logger.info(f"InferenceService initialized with model: {model_path} (backbone: {backbone})")
        except Exception as e:
            logger.error(f"Failed to initialize InferenceService: {e}")
            raise
    
    def detect_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Add frame and perform inference if buffer is full.
        
        Args:
            frame: Input frame (BGR, uint8)
        
        Returns:
            Dict with detection results if buffer is full, else None
        
        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        try:
            # Add frame to buffer
            self.model.add_frame(frame)
            
            # Predict only if buffer is full (returns None otherwise)
            return self.model.predict()
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise
    
    async def detect_frame_async(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Add frame and perform inference asynchronously.
        
        Args:
            frame: Input frame (BGR, uint8)
        
        Returns:
            Dict with detection results if buffer is full, else None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.detect_frame, frame)
    
    def reset(self) -> None:
        """Reset frame buffer."""
        if self.model:
            self.model.reset_buffer()
            logger.info("Frame buffer reset")


# Singleton instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
