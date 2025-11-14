"""
Video Dataset Loader - RWF-2000 only.

Loads RWF-2000 dataset with its predefined train/val splits.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict


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
                
                for video_file in sorted(cls_dir.iterdir()):
                    if video_file.suffix.lower() not in self.SUPPORTED_FORMATS:
                        continue
                    
                    item = VideoItem(
                        path=str(video_file),
                        label=label,
                        split=split
                    )
                    items[split].append(item)
        
        return items
