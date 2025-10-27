"""Frame processing utilities."""

import cv2
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Process video frames: resize, encode, sampling.
    
    Features:
    - Resize frames to target size
    - Encode to JPEG format
    - Frame sampling (skip frames)
    """
    
    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 480,
        jpeg_quality: int = 80,
    ):
        """
        Initialize frame processor.
        
        Args:
            target_width: Target frame width after resize
            target_height: Target frame height after resize
            jpeg_quality: JPEG encoding quality (1-100)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target dimensions.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            Resized frame
        """
        try:
            resized = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )
            return resized
        except Exception as e:
            logger.error(f"Frame resize error: {str(e)}")
            return frame
    
    def encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Encode frame to JPEG bytes.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            JPEG bytes or None if encoding failed
        """
        try:
            success, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            
            if not success:
                logger.error("JPEG encoding failed")
                return None
            
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"JPEG encoding error: {str(e)}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Complete frame processing: resize → encode JPEG.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            JPEG bytes or None if processing failed
        """
        try:
            # Resize
            resized = self.resize_frame(frame)
            
            # Encode
            jpeg_bytes = self.encode_jpeg(resized)
            
            return jpeg_bytes
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return None
