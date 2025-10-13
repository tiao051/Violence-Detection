"""
Video processing utilities
"""

import cv2
import numpy as np
from typing import Generator, Tuple


class VideoProcessor:
    """
    Process video files for violence detection
    """
    
    def __init__(self, video_path: str, frame_skip: int = 1):
        """
        Initialize video processor
        
        Args:
            video_path (str): Path to video file
            frame_skip (int): Process every Nth frame (1 = all frames)
        """
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = None
        self.fps = None
        self.total_frames = None
    
    def __enter__(self):
        """Context manager entry"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.cap:
            self.cap.release()
    
    def get_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from video
        
        Yields:
            Tuple[int, np.ndarray]: (frame_number, frame)
        """
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num % self.frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_num, frame_rgb
            
            frame_num += 1
    
    def get_video_info(self) -> dict:
        """Get video metadata"""
        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': self.total_frames / self.fps if self.fps > 0 else 0
        }
