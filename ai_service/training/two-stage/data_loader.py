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
cv2.setNumThreads(0)  # Prevent OpenCV from spawning threads to avoid conflicts with PyTorch DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import random

# Fix encoding for Windows compatibility
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add parent directory to path so workers can initialize extractors
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from remonet.sme.extractor import SMEExtractor
from remonet.ste.extractor import STEExtractor

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enable_augmentation: bool = True
    crop_size: int = 224  # Keep 224x224 to match STE input requirement
    flip_prob: float = 0.3  # Horizontal flip probability (30%)
    color_brightness: float = 0.2  # Color brightness jitter (20%)
    color_contrast: float = 0.2  # Color contrast jitter (20%)
    color_saturation: float = 0.2  # Color saturation jitter (20%)
    temporal_jitter_prob: float = 0.15  # Probability to apply temporal jitter (15%)
    temporal_jitter_range: int = 2  # Max frames to shift (+/- range)


class FrameAugmentor:
    """Apply augmentations to frames (as RGB numpy arrays)."""
    
    def __init__(self, config: AugmentationConfig = None, is_training: bool = True):
        """
        Initialize augmentor.
        
        Args:
            config: AugmentationConfig for augmentation parameters
            is_training: If False, skip augmentation even if enabled
        """
        self.config = config or AugmentationConfig()
        self.is_training = is_training
    
    def augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to all frames in batch.
        
        Args:
            frames: numpy array of shape (T, H, W, 3) in uint8 RGB
        
        Returns:
            Augmented frames of same shape
        """
        if not self.is_training or not self.config.enable_augmentation:
            return frames
        
        # Apply same augmentation to all frames in sequence for consistency
        # This preserves temporal coherence (all frames undergo same crop/flip)
        
        # Temporal jitter (apply first to vary temporal patterns)
        if random.random() < self.config.temporal_jitter_prob:
            frames = self._temporal_jitter(frames)
        
        # Random crop (35% = 30-40% range)
        if random.random() < 0.35:
            frames = self._random_crop(frames)
        
        # Random horizontal flip (30%)
        if random.random() < self.config.flip_prob:
            frames = self._horizontal_flip(frames)
        
        # Random color jitter (25% = 20-30% range)
        if random.random() < 0.25:
            frames = self._color_jitter(frames)
        
        return frames
    
    def _random_crop(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply random crop to all frames.
        
        Args:
            frames: (T, H, W, 3)
        
        Returns:
            Cropped frames (T, crop_size, crop_size, 3)
        """
        H, W = frames.shape[1], frames.shape[2]
        crop_size = self.config.crop_size
        
        if H < crop_size or W < crop_size:
            return frames
        
        # Random top-left corner
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        
        return frames[:, top:top+crop_size, left:left+crop_size, :]
    
    def _horizontal_flip(self, frames: np.ndarray) -> np.ndarray:
        """Flip frames horizontally (left-right)."""
        return np.flip(frames, axis=2)  # Flip width dimension
    
    def _color_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Apply color jitter (brightness, contrast, hue)."""
        frames_jittered = frames.astype(np.float32)
        
        # Brightness jitter
        if random.random() > 0.5:
            brightness_factor = 1.0 + random.uniform(-self.config.color_brightness, self.config.color_brightness)
            frames_jittered = frames_jittered * brightness_factor
        
        # Contrast jitter
        if random.random() > 0.5:
            contrast_factor = 1.0 + random.uniform(-self.config.color_contrast, self.config.color_contrast)
            mean_val = frames_jittered.mean(axis=(1, 2), keepdims=True)
            frames_jittered = mean_val + contrast_factor * (frames_jittered - mean_val)
        
        # Hue/saturation jitter (simpler approach without HSV conversion per frame)
        if random.random() > 0.5:
            saturation_factor = 1.0 + random.uniform(-self.config.color_saturation, self.config.color_saturation)
            # Apply to each channel with slight variation
            mean_rgb = frames_jittered.mean(axis=(1, 2), keepdims=True)
            frames_jittered = mean_rgb + saturation_factor * (frames_jittered - mean_rgb)
        
        # Clip to valid range and convert back to uint8
        frames_jittered = np.clip(frames_jittered, 0, 255).astype(np.uint8)
        return frames_jittered
    
    def _temporal_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Apply temporal jitter by randomly shifting frame indices.
        
        This augmentation helps model learn temporal invariance and not overfit
        to specific temporal patterns. Randomly drops or duplicates frames.
        
        Args:
            frames: (T, H, W, 3) numpy array
        
        Returns:
            Temporally jittered frames with same shape (T, H, W, 3)
        """
        T = len(frames)
        jitter_range = self.config.temporal_jitter_range
        
        # Random shift between -jitter_range and +jitter_range
        shift = random.randint(-jitter_range, jitter_range)
        
        if shift == 0:
            return frames
        
        # Create new indices with shift
        new_indices = []
        for i in range(T):
            new_idx = i + shift
            # Clamp to valid range [0, T-1]
            new_idx = max(0, min(T - 1, new_idx))
            new_indices.append(new_idx)
        
        # Gather frames with new indices
        jittered_frames = frames[new_indices]
        
        return jittered_frames


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
        'Fight': 0,
        'NonFight': 1,
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
    
    @staticmethod
    def load_hockey_fight_videos(dataset_root: str) -> Dict[str, List[Dict]]:
        """
        Load HockeyFight dataset using predefined train/val split.
        
        Args:
            dataset_root: Root directory containing HockeyFight folder
        
        Returns:
            Dict mapping 'train'/'val' -> list of dicts with 'path' and 'label' keys
        """
        dataset_root = Path(dataset_root)
        hockey_root = dataset_root / 'HockeyFight'
        if not hockey_root.exists():
            raise ValueError(f"HockeyFight folder not found: {hockey_root}")
        
        items = {'train': [], 'val': []}
        
        for split in ['train', 'val']:
            for cls in ['Fight', 'NonFight']:
                cls_dir = hockey_root / split / cls
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
    
    @staticmethod
    def load_hockey_fight_videos_auto_split(dataset_root: str, train_ratio: float = 0.7, 
                                            val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """
        Load HockeyFight dataset and auto-split into train/val/test if not already split.
        
        Args:
            dataset_root: Root directory containing HockeyFight folder
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
        
        Returns:
            Dict mapping 'train'/'val'/'test' -> list of dicts with 'path' and 'label' keys
        """
        import random
        
        dataset_root = Path(dataset_root)
        hockey_root = dataset_root / 'HockeyFight'
        if not hockey_root.exists():
            raise ValueError(f"HockeyFight folder not found: {hockey_root}")
        
        items = {'train': [], 'val': [], 'test': []}
        
        # Collect all videos by class
        videos_by_class = {'Fight': [], 'NonFight': []}
        
        for cls in ['Fight', 'NonFight']:
            # Check if already split into train/val/test
            split_dir = hockey_root / 'train' / cls
            if split_dir.exists():
                # Already split, use existing structure but adapt for test
                for split in ['train', 'val']:
                    split_path = hockey_root / split / cls
                    if split_path.exists():
                        for video_file in sorted(split_path.iterdir()):
                            if not video_file.is_file():
                                continue
                            if video_file.suffix.lower() not in VideoDataLoader.SUPPORTED_FORMATS:
                                continue
                            
                            label = 'Violence' if cls == 'Fight' else 'NonViolence'
                            items[split].append({
                                'path': str(video_file),
                                'label': label,
                                'split': split
                            })
                
                # No test split yet, return as-is
                items['test'] = []
                return items
            
            # Not split yet, collect all videos from single directory
            videos_dir = hockey_root / cls
            if videos_dir.exists():
                for video_file in sorted(videos_dir.iterdir()):
                    if not video_file.is_file():
                        continue
                    if video_file.suffix.lower() not in VideoDataLoader.SUPPORTED_FORMATS:
                        continue
                    
                    videos_by_class[cls].append(str(video_file))
        
        # If no videos found in class directories, return empty
        if not videos_by_class['Fight'] and not videos_by_class['NonFight']:
            return items
        
        # Auto-split each class
        for cls, video_paths in videos_by_class.items():
            if not video_paths:
                continue
            
            # Shuffle videos
            random.shuffle(video_paths)
            
            # Calculate split indices
            n = len(video_paths)
            train_count = int(n * train_ratio)
            val_count = int(n * val_ratio)
            
            train_videos = video_paths[:train_count]
            val_videos = video_paths[train_count:train_count + val_count]
            test_videos = video_paths[train_count + val_count:]
            
            label = 'Violence' if cls == 'Fight' else 'NonViolence'
            
            # Add to splits
            for video_path in train_videos:
                items['train'].append({
                    'path': video_path,
                    'label': label,
                    'split': 'train'
                })
            
            for video_path in val_videos:
                items['val'].append({
                    'path': video_path,
                    'label': label,
                    'split': 'val'
                })
            
            for video_path in test_videos:
                items['test'].append({
                    'path': video_path,
                    'label': label,
                    'split': 'test'
                })
        
        return items
    
    @staticmethod
    def load_uvd_videos(dataset_root: str) -> Dict[str, List[Dict]]:
        """
        Load UVD dataset using its predefined train/val split.
        
        Args:
            dataset_root: Root directory containing UVD folder
        
        Returns:
            Dict mapping 'train'/'val' -> list of dicts with 'path' and 'label' keys
        """
        dataset_root = Path(dataset_root)
        uvd_root = dataset_root / 'UVD' / 'extracted_frames'
        if not uvd_root.exists():
            raise ValueError(f"UVD folder not found: {uvd_root}")
        
        items = {'train': [], 'val': []}
        
        for split in ['train', 'val']:
            for cls in ['Fight', 'NonFight']:
                cls_dir = uvd_root / split / cls
                if not cls_dir.exists():
                    continue
                
                label = 'Violence' if cls == 'Fight' else 'NonViolence'
                
                for video_dir in sorted(cls_dir.iterdir()):
                    if not video_dir.is_dir():
                        continue
                    
                    # For UVD, we point to the directory containing frames
                    items[split].append({
                        'path': str(video_dir),
                        'label': label,
                        'split': split
                    })
        
        return items

    def __init__(
        self,
        extracted_frames_dir: str,
        split: str = 'train',
        target_frames: int = 30,
        augmentation_config: AugmentationConfig = None,
        dataset: str = 'rwf-2000',
        backbone: str = 'mobilenet_v2'
    ):
        """
        Initialize VideoDataLoader.
        
        Args:
            extracted_frames_dir: Path to directory with extracted frames
                                 Structure: split/label/video_hash/frame_*.jpg
            split: Dataset split ('train' or 'val')
            target_frames: Expected number of frames per video (default: 30)
            augmentation_config: Augmentation configuration (optional)
            dataset: Dataset name ('rwf-2000', 'hockey-fight', 'uvd'), default: 'rwf-2000'
        """
        self.extracted_frames_dir = Path(extracted_frames_dir)
        self.split = split
        self.target_frames = target_frames
        self.dataset = dataset
        self.backbone = backbone

        # Initialize augmentor (only apply to training split, not validation)
        aug_config = augmentation_config or AugmentationConfig()
        self.augmentor = FrameAugmentor(aug_config, is_training=(split == 'train'))
        
        # Extractors initialized to None, will be created lazily per worker
        # This avoids pickling issues with multiprocessing
        self.sme_extractor = None
        self.ste_extractor = None
        
        # Detect GPU availability for feature extraction
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not self.extracted_frames_dir.exists():
            raise ValueError(f"Extracted frames directory not found: {extracted_frames_dir}")
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        if dataset not in ['rwf-2000', 'hockey-fight', 'uvd']:
            raise ValueError(f"Invalid dataset: {dataset}. Must be 'rwf-2000', 'hockey-fight', or 'uvd'")
        
        # Load video metadata
        self.video_items = self._load_video_metadata()
        logger.info(f"Loaded {len(self.video_items)} videos for {dataset} split '{split}' with augmentation={aug_config.enable_augmentation}")

    
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
        
        # Iterate through label subdirectories (Violence/NonViolence)
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            label_name = label_dir.name
            if label_name not in self.LABEL_MAP:
                logger.warning(f"Unknown label: {label_name}")
                continue
            
            label_id = self.LABEL_MAP[label_name]
            
            # Iterate through individual video directories
            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                
                # Check if video directory contains extracted frames
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
        
        # Take only the first target_frames (usually 30 frames per video)
        frame_files = frame_files[:self.target_frames]
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_file}")
                continue
            
            # Convert BGR (OpenCV default) to RGB for consistent color ordering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        if len(frames) == 0:
            raise RuntimeError(f"Failed to load any frames from: {frames_dir}")
        
        # Stack frames into single numpy array
        frames = np.array(frames, dtype=np.uint8)
        
        # Validate frame dimensions match expected size
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
        
        # Initialize feature extractors lazily (once per worker process)
        # Lazy initialization prevents pickling issues in multiprocessing
        if self.sme_extractor is None:
            self.sme_extractor = SMEExtractor()
            # CRITICAL: Keep STE in eval mode to freeze pretrained backbone (prevents overfitting)
            # Only GTE parameters should be trained on small datasets
            self.ste_extractor = STEExtractor(
                device=self.device,
                training_mode=False,
                backbone=self.backbone
            )

        item = self.video_items[idx]
        
        try:
            # Load raw video frames: shape (30, 224, 224, 3) in RGB uint8
            frames = self._load_frames(item['frames_dir'])
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"Failed to load frames for item {idx} ({item['frames_dir']}): {e}")
            # Return zeros with expected GTE input shape on error
            # This allows batch processing to continue without crashes
            channels = self.ste_extractor.backbone_config['out_channels']
            spatial_size = self.ste_extractor.backbone_config['spatial_size']
            return torch.zeros((10, channels, spatial_size, spatial_size)), item['label']
        
        # Wrap feature extraction in no_grad context (no gradients needed for pretrained models)
        with torch.no_grad():
            # Step 1: Apply SME (Spatial Motion Extraction)
            # IMPORTANT: SME computes optical flow between consecutive frames
            # Must run on ORIGINAL frames to compute valid motion vectors
            if self.sme_extractor is not None:
                motion_frames = self.sme_extractor.process_batch(frames)
            else:
                motion_frames = frames
            
            # Step 2: Apply data augmentation AFTER SME
            # Augmenting motion-enhanced frames is safe because:
            # - Optical flow already computed (step 1)
            # - Augmentation adds variation to motion signal to prevent overfitting
            # - Essential for small datasets like RWF-2000 (1,600 training videos)
            motion_frames = self.augmentor.augment_frames(motion_frames)
            
            # Step 3: Apply STE (Spatial Temporal Feature Extraction)
            # Extracts MobileNetV2 backbone features from motion-enhanced frames
            if self.ste_extractor is not None:
                ste_output = self.ste_extractor.process(motion_frames)
                features = ste_output.features  # shape: (T/3, C, H, W) = (10, 1280, 7, 7)
            else:
                # Fallback: convert motion frames to tensor if STE not available
                features = torch.from_numpy(motion_frames).permute(0, 3, 1, 2).float()
        
        return features, item['label']
