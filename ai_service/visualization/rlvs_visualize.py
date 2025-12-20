"""
Generate RLVS Dataset Sample Frames Grid (5x5)
Similar to RWF-2000 and Hockey Fight visualizations
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
RLVS_FIGHT_PATH = Path("c:/Users/thang/Desktop/Violence-Detection/dataset/RLVS/Fight")
RLVS_NONFIGHT_PATH = Path("c:/Users/thang/Desktop/Violence-Detection/dataset/RLVS/NonFight")
OUTPUT_PATH = Path("c:/Users/thang/Desktop/Violence-Detection/ai_service/visualization")


def extract_middle_frame(video_path: Path) -> np.ndarray:
    """Extract frame from middle of video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def create_sample_grid(folder_path: Path, title: str, n_samples: int = 25, grid_size: tuple = (5, 5)):
    """Create a grid of sample frames from videos."""
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = [f for f in folder_path.iterdir() 
              if f.suffix.lower() in video_extensions]
    
    print(f"Found {len(videos)} videos in {folder_path}")
    
    # Random sample
    random.seed(42)  # For reproducibility
    sampled_videos = random.sample(videos, min(n_samples, len(videos)))
    
    # Extract frames
    frames = []
    for video in sampled_videos:
        frame = extract_middle_frame(video)
        if frame is not None:
            # Resize for consistency
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    print(f"Extracted {len(frames)} frames")
    
    if len(frames) < n_samples:
        print(f"Warning: Only got {len(frames)} frames")
    
    # Create grid
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(frames):
            ax.imshow(frames[idx])
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.02, hspace=0.02)
    
    return fig


def create_combined_grid(fight_folder: Path, nonfight_folder: Path, title: str, 
                         n_samples: int = 25, grid_size: tuple = (5, 5)):
    """Create grid sampling from both Fight and NonFight folders."""
    # Get videos from both folders
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    fight_videos = [f for f in fight_folder.iterdir() 
                    if f.suffix.lower() in video_extensions]
    nonfight_videos = [f for f in nonfight_folder.iterdir() 
                       if f.suffix.lower() in video_extensions]
    
    all_videos = fight_videos + nonfight_videos
    print(f"Total videos: {len(all_videos)} (Fight: {len(fight_videos)}, NonFight: {len(nonfight_videos)})")
    
    # Random sample
    random.seed(42)
    sampled_videos = random.sample(all_videos, min(n_samples, len(all_videos)))
    
    # Extract frames
    frames = []
    for video in sampled_videos:
        frame = extract_middle_frame(video)
        if frame is not None:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    print(f"Extracted {len(frames)} frames")
    
    # Create grid with black background
    rows, cols = grid_size
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98, color='white')
    
    for idx in range(rows * cols):
        ax = fig.add_subplot(rows, cols, idx + 1)
        if idx < len(frames):
            ax.imshow(frames[idx])
        else:
            ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
        ax.axis('off')
        ax.set_facecolor('black')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.02, hspace=0.02)
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("RLVS Dataset Sample Frames Visualization")
    print("=" * 60)
    
    # Create combined grid (mixing Fight and NonFight)
    fig = create_combined_grid(
        RLVS_FIGHT_PATH, 
        RLVS_NONFIGHT_PATH,
        "RLVS - Sample Frames (5×5 Grid)",
        n_samples=25
    )
    
    # Save
    output_file = OUTPUT_PATH / "rlvs_sample_frames.png"
    fig.savefig(output_file, dpi=150, facecolor='black', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.1)
    print(f"\nSaved to: {output_file}")
    
    plt.show()
    plt.close()
    
    print("\n✅ Done!")
