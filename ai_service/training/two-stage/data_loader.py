"""
Video Dataset Loader - RWF-2000 only.

Loads RWF-2000 dataset with its predefined train/val splits.
Integrates frame extraction, SME (motion), and STE (feature extraction).
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

# Fix encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger(__name__)


@dataclass
class VideoItem:
    """Represents a single video item in the dataset."""
    path: str
    label: str
    split: str  # 'train' or 'val'
    
    def __post_init__(self):
        """Validate video item."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        if self.split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {self.split}")


class VideoDatasetLoader:
    """Load RWF-2000 dataset with predefined train/val split."""
    
    LABEL_MAP = {
        'Violence': 0,
        'NonViolence': 1,
    }
    
    SUPPORTED_FORMATS = ('.avi', '.mp4', '.mov', '.mkv')
    
    def __init__(self, dataset_root: str):
        """
        Initialize video dataset loader.
        
        Args:
            dataset_root: Root directory containing RWF-2000 folder
                         Expected structure: dataset_root/RWF-2000/train/Fight, etc.
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root not found: {dataset_root}")
    
    def load_rwf2000(self) -> Dict[str, List[VideoItem]]:
        """
        Load RWF-2000 dataset using its predefined train/val split.
        
        Returns:
            Dict mapping 'train'/'val' -> list of VideoItem objects
        """
        rwf_root = self.dataset_root / 'RWF-2000'
        if not rwf_root.exists():
            raise ValueError(f"RWF-2000 folder not found: {rwf_root}")
        
        items = {'train': [], 'val': []}
        
        for split in ['train', 'val']:
            for cls in ['Fight', 'NonFight']:
                cls_dir = rwf_root / split / cls
                if not cls_dir.exists():
                    continue
                
                label = 'Violence' if cls == 'Fight' else 'NonViolence'
                
                try:
                    video_files = os.listdir(cls_dir)
                except Exception as e:
                    print(f"ERROR: Cannot read directory {cls_dir}: {e}")
                    continue
                
                for video_file in sorted(video_files):
                    video_path = cls_dir / video_file
                    
                    if not video_path.is_file():
                        continue
                    
                    if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                        continue
                    
                    try:
                        item = VideoItem(
                            path=str(video_path),
                            label=label,
                            split=split
                        )
                        items[split].append(item)
                    except FileNotFoundError:
                        print(f"WARNING: Video file not found: {video_path}")
                        continue
        
        return items


class VideoDataLoader(VideoDatasetLoader, Dataset):
    """
    PyTorch Dataset for extracted video frames.
    
    Processes frames through SME (motion extraction) and STE (feature extraction).
    Input: 30 extracted frames per video
    Output: (features, label) where features are spatiotemporal embeddings
    """
    
    def __init__(
        self,
        extracted_frames_dir: str,
        split: str = 'train',
        sme_extractor=None,
        ste_extractor=None,
        target_frames: int = 30,
    ):
        """
        Initialize VideoDataLoader.
        
        Args:
            extracted_frames_dir: Path to directory with extracted frames
                                 Structure: split/label/video_hash/frame_*.jpg
            split: Dataset split ('train' or 'val')
            sme_extractor: SMEExtractor instance for motion extraction
            ste_extractor: STEExtractor instance for feature extraction
            target_frames: Expected number of frames per video (default: 30)
        """
        self.extracted_frames_dir = Path(extracted_frames_dir)
        self.split = split
        self.sme_extractor = sme_extractor
        self.ste_extractor = ste_extractor
        self.target_frames = target_frames
        
        if not self.extracted_frames_dir.exists():
            raise ValueError(f"Extracted frames directory not found: {extracted_frames_dir}")
        
        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {split}")
        
        # Load video metadata
        self.video_items = self._load_video_metadata()
        logger.info(f"Loaded {len(self.video_items)} videos for split '{split}'")
    
    def _load_video_metadata(self) -> List[Dict]:
        """
        Load metadata for all videos in the split.
        
        Returns:
            List of dicts with keys: 'video_id', 'label', 'frames_dir'
        """
        items = []
        split_dir = self.extracted_frames_dir / self.split
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return items
        
        # Iterate through labels
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            label_name = label_dir.name
            if label_name not in self.LABEL_MAP:
                logger.warning(f"Unknown label: {label_name}")
                continue
            
            label_id = self.LABEL_MAP[label_name]
            
            # Iterate through video hashes
            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                
                # Check if this directory has frames
                frames = list(video_dir.glob('frame_*.jpg'))
                if len(frames) == 0:
                    logger.warning(f"No frames found in: {video_dir}")
                    continue
                
                items.append({
                    'video_id': video_dir.name,
                    'label': label_id,
                    'label_name': label_name,
                    'frames_dir': str(video_dir),
                    'frame_count': len(frames),
                })
        
        return items
    
    def _load_frames(self, frames_dir: str) -> np.ndarray:
        """
        Load frames from a directory (up to target_frames).
        
        Args:
            frames_dir: Directory containing frame_*.jpg files
        
        Returns:
            np.ndarray of shape (N, 224, 224, 3) in uint8 RGB where N <= target_frames
        """
        frame_dir = Path(frames_dir)
        frame_files = sorted(frame_dir.glob('frame_*.jpg'))
        
        if len(frame_files) == 0:
            raise FileNotFoundError(f"No frames found in: {frames_dir}")
        
        # Limit to target_frames
        frame_files = frame_files[:self.target_frames]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_file}")
                continue
            
            # Convert BGR (from imread) to RGB for model input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        if len(frames) == 0:
            raise RuntimeError(f"Failed to load any frames from: {frames_dir}")
        
        frames = np.array(frames, dtype=np.uint8)
        
        # Validate shape
        if frames.shape[1:] != (224, 224, 3):
            logger.warning(
                f"Unexpected frame shape {frames.shape} in {frames_dir}, "
                f"expected (..., 224, 224, 3)"
            )
        
        return frames
    
    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.video_items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single training sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (features, label) where:
                - features: torch.Tensor of shape (T/3, C, H, W) from STE
                           e.g., (10, 1280, 7, 7) for MobileNetV2 backbone
                - label: int label (0 for Violence, 1 for NonViolence)
        """
        item = self.video_items[idx]
        
        # Load frames as (30, 224, 224, 3) RGB uint8
        frames = self._load_frames(item['frames_dir'])
        
        # Apply SME (spatial motion extraction)
        # Input: (30, 224, 224, 3) RGB uint8
        # Output: (30, 224, 224, 3) motion uint8
        if self.sme_extractor is not None:
            motion_frames = self.sme_extractor.process_batch(frames)
        else:
            motion_frames = frames
        
        # Apply STE (spatial temporal feature extraction)
        # Input: (30, 224, 224, 3) motion uint8
        # Output: STEOutput with features of shape (T/3, C, H, W) e.g., (10, 1280, 7, 7)
        if self.ste_extractor is not None:
            ste_output = self.ste_extractor.process(motion_frames)
            # Extract the feature tensor from STEOutput dataclass
            features = ste_output.features  # torch.Tensor (T/3, C, H, W)
        else:
            # Fallback: return motion frames as tensor
            features = torch.from_numpy(motion_frames).permute(0, 3, 1, 2).float()  # (30, 3, 224, 224)
        
        return features, item['label']
