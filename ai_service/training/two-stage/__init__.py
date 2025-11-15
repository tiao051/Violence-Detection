"""
Training module for violence detection (two-stage pipeline).

Provides utilities for:
- Loading RWF-2000 dataset
- Extracting frames from videos
"""

from .data_loader import VideoDatasetLoader, VideoItem
from .frame_extractor import FrameExtractor, ExtractionConfig

__all__ = [
    'VideoDatasetLoader',
    'VideoItem',
    'FrameExtractor',
    'ExtractionConfig',
]
