"""Inference service for violence detection."""

import sys
from pathlib import Path
from typing import Dict, Optional
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
    
    def initialize(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.5) -> None:
        """
        Initialize violence detection model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda' or 'cpu')
            confidence_threshold: Confidence threshold for violence detection
        """
        try:
            if ViolenceDetectionModel is None:
                logger.warning("ViolenceDetectionModel not available, using dummy model")
                self.model = None
                return
                
            config = InferenceConfig(
                model_path=model_path,
                device=device,
                confidence_threshold=confidence_threshold
            )
            self.model = ViolenceDetectionModel(config)
            logger.info(f"InferenceService initialized with model: {model_path}")
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
        """
        if self.model is None:
            logger.warning("Model not initialized")
            return None
        
        try:
            # Add frame to buffer
            self.model.add_frame(frame)
            
            # Predict only if buffer is full (returns None otherwise)
            return self.model.predict()
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None
    
    async def detect_frame_async(self, frame: np.ndarray) -> Dict:
        """
        Add frame and perform inference asynchronously.
        
        Args:
            frame: Input frame (BGR, uint8)
        
        Returns:
            Dict with detection results
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
