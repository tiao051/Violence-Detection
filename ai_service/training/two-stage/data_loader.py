"""
Video Dataset Loader - RWF-2000 only.

Loads RWF-2000 dataset with its predefined train/val splits.
Integrates frame extraction, SME (motion), and STE (feature extraction).
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

# Sửa encoding cho Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Thêm các import này để worker có thể tự khởi tạo
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from remonet.sme.extractor import SMEExtractor
from remonet.ste.extractor import STEExtractor

logger = logging.getLogger(__name__)


class VideoDataLoader(Dataset):
    """
    PyTorch Dataset for extracted video frames.
    
    Processes frames through SME (motion extraction) and STE (feature extraction).
    Input: 30 extracted frames per video
    Output: (features, label) where features are spatiotemporal embeddings
    """
    
    LABEL_MAP = {
        'Violence': 0,
        'NonViolence': 1,
    }
    
    SUPPORTED_FORMATS = ('.avi', '.mp4', '.mov', '.mkv')
    
    @staticmethod
    def load_rwf2000_videos(dataset_root: str) -> Dict[str, List[Dict]]:
        """
        Load RWF-2000 dataset using its predefined train/val split.
        Used by frame extraction pipeline.
        
        Args:
            dataset_root: Root directory containing RWF-2000 folder
        
        Returns:
            Dict mapping 'train'/'val' -> list of dicts with 'path' and 'label' keys
        """
        dataset_root = Path(dataset_root)
        rwf_root = dataset_root / 'RWF-2000'
        if not rwf_root.exists():
            raise ValueError(f"RWF-2000 folder not found: {rwf_root}")
        
        items = {'train': [], 'val': []}
        
        for split in ['train', 'val']:
            for cls in ['Fight', 'NonFight']:
                cls_dir = rwf_root / split / cls
                if not cls_dir.exists():
                    continue
                
                label = 'Violence' if cls == 'Fight' else 'NonViolence'
                
                for video_file in sorted(cls_dir.iterdir()):
                    if not video_file.is_file():
                        continue
                    
                    if video_file.suffix.lower() not in VideoDataLoader.SUPPORTED_FORMATS:
                        continue
                    
                    items[split].append({
                        'path': str(video_file),
                        'label': label,
                        'split': split
                    })
        
        return items
    
    def __init__(
        self,
        extracted_frames_dir: str,
        split: str = 'train',
        target_frames: int = 30,
    ):
        """
        Initialize VideoDataLoader.
        
        Args:
            extracted_frames_dir: Path to directory with extracted frames
                                 Structure: split/label/video_hash/frame_*.jpg
            split: Dataset split ('train' or 'val')
            target_frames: Expected number of frames per video (default: 30)
        """
        self.extracted_frames_dir = Path(extracted_frames_dir)
        self.split = split
        self.target_frames = target_frames
        
        # Trạng thái (None) để worker tự khởi tạo
        self.sme_extractor = None
        self.ste_extractor = None
        # Thiết bị sẽ được sử dụng bởi STEExtractor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        # Khởi tạo extractor bên trong worker (chỉ 1 lần/worker)
        if self.sme_extractor is None:
            self.sme_extractor = SMEExtractor()
            self.ste_extractor = STEExtractor(device=self.device, training_mode=True)

        item = self.video_items[idx]
        
        try:
            # Load frames as (30, 224, 224, 3) RGB uint8
            frames = self._load_frames(item['frames_dir'])
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"Failed to load frames for item {idx} ({item['frames_dir']}): {e}")
            # Trả về tensor rỗng với shape chuẩn để collate không lỗi
            # (10, 1280, 7, 7) là shape giả định của GTE input
            return torch.zeros((10, 1280, 7, 7)), item['label']
            
        
        # Bọc toàn bộ quá trình xử lý model (SME, STE) trong torch.no_grad()
        # để vô hiệu hóa việc tính gradient
        with torch.no_grad():
            # Áp dụng SME (Spatial Motion Extraction)
            if self.sme_extractor is not None:
                motion_frames = self.sme_extractor.process_batch(frames)
            else:
                motion_frames = frames
            
            # Áp dụng STE (Spatial Temporal Feature Extraction)
            if self.ste_extractor is not None:
                ste_output = self.ste_extractor.process(motion_frames)
                features = ste_output.features  # torch.Tensor (T/3, C, H, W)
            else:
                # Fallback: trả về motion frames dưới dạng tensor
                features = torch.from_numpy(motion_frames).permute(0, 3, 1, 2).float()
        
        return features, item['label']