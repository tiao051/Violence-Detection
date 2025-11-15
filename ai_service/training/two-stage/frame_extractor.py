"""
Frame Extraction Module.

Extracts frames from videos at specified FPS and resizes to target size.

Usage:
    python frame_extractor.py --dataset rwf-2000 --dataset-root <path> [OPTIONS]

Arguments:
    --dataset {rwf-2000}           Dataset to extract frames from (required)
    --dataset-root PATH            Path to dataset root directory (required)

Options:
    --target-fps FPS               Target FPS for frame extraction (default: 30)
    --target-size W H              Target frame size in pixels (default: 224 224)
    --jpeg-quality QUALITY         JPEG quality 0-100 (default: 85)

Example:
    python frame_extractor.py --dataset rwf-2000 --dataset-root d:/DATN/violence-detection/dataset
    python frame_extractor.py --dataset rwf-2000 --dataset-root d:/DATN/violence-detection/dataset --target-fps 30 --target-size 256 256
"""

import cv2
import argparse
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Add current directory to path for data_loader import
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import VideoDataLoader

# Add ai_service to path for SME import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from remonet.sme.extractor import SMEPreprocessor


# Helper class for frame extraction
class VideoItem:
    """Video item for frame extraction."""
    def __init__(self, path: str, label: str, split: str):
        self.path = path
        self.label = label
        self.split = split


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""
    target_size: tuple = (224, 224)
    target_fps: int = 30
    jpeg_quality: int = 85


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
    
    def extract_batch(self, video_items: List, output_base_dir: str) -> Dict:
        """
        Extract frames from multiple videos.
        
        Args:
            video_items: List of VideoItem objects
            output_base_dir: Base output directory
                            Structure: output_base_dir/split/label/video_index/
        
        Returns:
            Dict with batch extraction results
        """
        total_frames = 0
        failed_count = 0
        
        output_base = Path(output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        
        for idx, video_item in enumerate(video_items):
            # Construct output directory using video index (easy to understand, short path)
            output_dir = output_base / video_item.split / video_item.label / f"{idx:05d}"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Validate input
            if not Path(video_item.path).exists():
                print(f"ERROR: Video not found: {video_item.path}")
                failed_count += 1
                continue
            
            # Open video
            cap = cv2.VideoCapture(video_item.path)
            if not cap.isOpened():
                print(f"ERROR: Cannot open video: {video_item.path}")
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
                            # Preprocess frame: resize to 224x224 and convert to RGB
                            resized_frame = self.preprocessor.preprocess(frame)
                            
                            # Convert RGB back to BGR for cv2.imwrite (OpenCV expects BGR)
                            frame_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                            
                            # Save frame
                            frame_path = output_path / f"frame_{extracted_count:06d}.jpg"
                            cv2.imwrite(
                                str(frame_path), 
                                frame_bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                            )
                            
                            extracted_count += 1
                            next_t += step_t
                        
                        except Exception as e:
                            print(f"WARNING: Error processing frame {frame_idx}: {e}")
                            next_t += step_t
                    
                    frame_idx += 1
            
            finally:
                cap.release()
            
            total_frames += extracted_count
            print(f"[{idx+1}/{len(video_items)}] Extracted {extracted_count} frames from {video_item.path}")
        
        print(f"Batch extraction completed: {len(video_items) - failed_count}/{len(video_items)} successful, {total_frames} total frames")
        
        return {
            'total': len(video_items),
            'success': len(video_items) - failed_count,
            'failed': failed_count,
            'total_frames': total_frames
        }


def main():
    """CLI entry point for frame extraction."""
    parser = argparse.ArgumentParser(
        description='Extract frames from video dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True, choices=['rwf-2000'], help='Dataset to extract frames from')
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--target-fps', type=int, default=30, help='Target FPS for frame extraction (default: 30)')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224], help='Target frame size (width height) (default: 224 224)')
    parser.add_argument('--jpeg-quality', type=int, default=85, help='JPEG quality (0-100) (default: 85)')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root).resolve()
    
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        sys.exit(1)
    
    # Create config and extractor
    config = ExtractionConfig(
        target_size=tuple(args.target_size),
        target_fps=args.target_fps,
        jpeg_quality=args.jpeg_quality
    )

    extractor = FrameExtractor(config=config)
    
    if args.dataset == 'rwf-2000':
        print(f"Loading RWF-2000 dataset from {dataset_root}")
        videos = VideoDataLoader.load_rwf2000_videos(str(dataset_root))
        
        total_videos = len(videos['train']) + len(videos['val'])
        print(f"Found {total_videos} videos (train: {len(videos['train'])}, val: {len(videos['val'])})")
        
        output_dir = dataset_root / 'RWF-2000' / 'extracted_frames'
        
        # Extract frames for each split
        for split in ['train', 'val']:
            split_videos = videos[split]
            if not split_videos:
                print(f"WARNING: No videos for split: {split}")
                continue
            
            # Convert dict items to VideoItem objects
            video_items = [
                VideoItem(
                    path=video_info['path'],
                    label=video_info['label'],
                    split=video_info['split']
                )
                for video_info in split_videos
            ]
        
            extractor.extract_batch(
                video_items=video_items,
                output_base_dir=str(output_dir)
            )
        
        print(f"Frames saved to: {output_dir}")


if __name__ == '__main__':
    main()
