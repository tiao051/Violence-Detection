"""
Frame Extraction Module.

Extracts frames from videos at specified FPS and resizes to target size.
"""

import cv2
import hashlib
import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from ai_service.remonet.sme.extractor import SMEPreprocessor

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""
    target_size: tuple = (224, 224)
    target_fps: int = 20
    jpeg_quality: int = 85
    log_file: str = "frame_extraction.log"


class FrameExtractor:
    """Extract frames from video files."""
    
    SUPPORTED_FORMATS = ('.avi', '.mp4', '.mov', '.mkv')
    
    def __init__(self, config: ExtractionConfig = None):
        """
        Initialize frame extractor.
        
        Args:
            config: ExtractionConfig object (uses defaults if None)
        """
        self.config = config or ExtractionConfig()
        self.preprocessor = SMEPreprocessor(target_size=self.config.target_size)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup file logging."""
        handler = logging.FileHandler(self.config.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def _get_video_hash(self, video_path: str) -> str:
        """Generate unique hash for video file."""
        return hashlib.md5(video_path.encode()).hexdigest()[:12]
    
    def extract_batch(self, video_items: List, output_base_dir: str) -> Dict:
        """
        Extract frames from multiple videos.
        
        Args:
            video_items: List of VideoItem objects
            output_base_dir: Base output directory
                            Structure: output_base_dir/split/label/video_hash/
        
        Returns:
            Dict with batch extraction results
        """
        total_frames = 0
        failed_count = 0
        
        output_base = Path(output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        
        for idx, video_item in enumerate(video_items):
            # Construct output directory
            video_hash = self._get_video_hash(video_item.path)
            output_dir = output_base / video_item.split / video_item.label / video_hash
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Validate input
            if not Path(video_item.path).exists():
                logger.error(f"Video not found: {video_item.path}")
                failed_count += 1
                continue
            
            # Open video
            cap = cv2.VideoCapture(video_item.path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_item.path}")
                failed_count += 1
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            
            # Calculate frame sampling
            step_t = 1.0 / float(self.config.target_fps)
            next_t = 0.0
            frame_idx = 0
            extracted_count = 0
            
            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    current_t = frame_idx / fps
                    
                    if current_t + 1e-6 >= next_t:
                        try:
                            # Resize using SMEPreprocessor
                            resized_frame = self.preprocessor.preprocess(frame)
                            
                            # Save frame
                            frame_path = output_path / f"frame_{extracted_count:06d}.jpg"
                            cv2.imwrite(
                                str(frame_path), 
                                cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                            )
                            
                            extracted_count += 1
                            next_t += step_t
                        
                        except Exception as e:
                            logger.warning(f"Error processing frame {frame_idx}: {e}")
                            next_t += step_t
                    
                    frame_idx += 1
            
            finally:
                cap.release()
            
            total_frames += extracted_count
            logger.info(f"[{idx+1}/{len(video_items)}] Extracted {extracted_count} frames from {video_item.path}")
        
        logger.info(f"Batch extraction completed: {len(video_items) - failed_count}/{len(video_items)} successful, {total_frames} total frames")
        
        return {
            'total': len(video_items),
            'success': len(video_items) - failed_count,
            'failed': failed_count,
            'total_frames': total_frames
        }
