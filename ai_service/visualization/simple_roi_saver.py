"""
Simple ROI saver - saves only extracted ROI from SME.

This module saves the Region Of Interest extracted by the SME (Spatial Motion Extractor)
module. Each saved frame is the direct ROI that the model uses for violence classification.

"""

import cv2
import numpy as np
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SimpleROISaver:
    """
    Simple ROI saver - saves only extracted ROI from SME.
    
    Each frame is the direct ROI output from SME.process().
    No visualizations, no overlays, just the extracted region.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize ROI saver.
        
        Args:
            output_dir: Directory to save extracted ROI frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        logger.info(f"ROI Saver initialized: {self.output_dir}")
    
    def save_roi_frame(self, roi: np.ndarray, window_idx: int, prediction: str, confidence: float):
        """
        Save extracted ROI frame.
        
        Args:
            roi: Extracted ROI from SME (H×W×3 or H×W)
            window_idx: Window index for numbering
            prediction: "VIOLENCE" or "NO VIOLENCE"
            confidence: Confidence value (0-1)
        """
        if roi is None:
            logger.warning(f"Skipping frame {window_idx}: ROI is None")
            return
        
        # Ensure roi is uint8
        if roi.dtype != np.uint8:
            roi = np.clip(roi * 255, 0, 255).astype(np.uint8) if roi.max() <= 1.0 else np.clip(roi, 0, 255).astype(np.uint8)
        
        # Ensure 3 channels (BGR)
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.shape[2] == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        # Add prediction text overlay
        h, w = roi.shape[:2]
        overlay = roi.copy()
        
        # Draw background for text
        cv2.rectangle(overlay, (5, 5), (w-5, 40), (0, 0, 0), -1)
        
        # Add text: VIOLENCE or NO VIOLENCE
        color = (0, 0, 255) if prediction == "VIOLENCE" else (0, 255, 0)  # Red or Green
        cv2.putText(
            overlay,
            f"{prediction}: {confidence:.1%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Save frame
        frame_num = window_idx
        output_path = self.output_dir / f"roi_{frame_num:04d}.png"
        cv2.imwrite(str(output_path), overlay)
        
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            logger.info(f"Saved {self.frame_count} ROI frames")
    
    def get_summary(self) -> str:
        """Get summary of saved frames."""
        return f"Saved {self.frame_count} extracted ROI frames to {self.output_dir}"
