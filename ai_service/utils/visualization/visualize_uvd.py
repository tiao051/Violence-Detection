"""
UVD Dataset Visualization.

Generates visualizations for the Unified Violence Dataset (UVD):
- Class distribution (Violence vs NonViolence)
- Sample frames grid
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import random
import logging

# Configure style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_uvd(dataset_root: str, output_dir: str):
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    uvd_root = dataset_root / 'UVD' / 'extracted_frames'
    if not uvd_root.exists():
        logger.error(f"UVD root not found: {uvd_root}")
        return

    # 1. Class Distribution
    logger.info("Analyzing class distribution...")
    counts = {'Violence': 0, 'NonViolence': 0}
    video_paths = {'Violence': [], 'NonViolence': []}
    
    for split in ['train', 'val']:
        for label in ['Violence', 'NonViolence']:
            label_dir = uvd_root / split / label
            if label_dir.exists():
                videos = [d for d in label_dir.iterdir() if d.is_dir()]
                counts[label] += len(videos)
                video_paths[label].extend(videos)
                logger.info(f"Found {len(videos)} {label} videos in {split}")

    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=['#ff9999', '#66b3ff'])
    plt.title('UVD Class Distribution')
    plt.ylabel('Number of Videos')
    plt.savefig(output_dir / 'uvd_class_distribution.png')
    logger.info(f"Saved class distribution to {output_dir / 'uvd_class_distribution.png'}")
    
    # 2. Sample Frames Grid
    logger.info("Generating sample frames grid...")
    num_samples = 25
    grid_size = (5, 5)
    frame_size = (160, 120)
    
    all_videos = video_paths['Violence'] + video_paths['NonViolence']
    if not all_videos:
        logger.error("No videos found!")
        return
        
    sampled_frames = []
    selected_videos = random.sample(all_videos, min(num_samples, len(all_videos)))
    
    for video_dir in selected_videos:
        frames = list(video_dir.glob('frame_*.jpg'))
        if frames:
            frame_path = random.choice(frames)
            img = cv2.imread(str(frame_path))
            if img is not None:
                img = cv2.resize(img, frame_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sampled_frames.append(img)
    
    # Pad if needed
    while len(sampled_frames) < num_samples:
        sampled_frames.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))
        
    # Create grid
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 8))
    for idx, ax in enumerate(axes.flat):
        if idx < len(sampled_frames):
            ax.imshow(sampled_frames[idx])
        ax.axis('off')
        
    plt.suptitle('UVD Sample Frames')
    plt.tight_layout()
    plt.savefig(output_dir / 'uvd_sample_frames.png')
    logger.info(f"Saved sample frames to {output_dir / 'uvd_sample_frames.png'}")

if __name__ == "__main__":
    # Auto-detect dataset root
    current_dir = Path(__file__).parent
    # Try to find 'dataset' folder
    dataset_root = None
    for candidate in [
        Path.cwd() / 'dataset',
        Path.cwd().parent / 'dataset',
        Path('d:/doantotnghiep/Violence-Detection/dataset')
    ]:
        if candidate.exists():
            dataset_root = candidate
            break
            
    if dataset_root:
        visualize_uvd(str(dataset_root), 'd:/doantotnghiep/Violence-Detection')
    else:
        print("Dataset root not found")
