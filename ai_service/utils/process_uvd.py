import os
import cv2
import numpy as np
import glob
from pathlib import Path
import shutil
from tqdm import tqdm
import random

# Configuration
DATASET_ROOT = Path('dataset')
OUTPUT_ROOT = DATASET_ROOT / 'UVD' / 'extracted_frames'
IMG_SIZE = 224
NUM_FRAMES = 30

def extract_frames(video_path, output_dir, num_frames=30, resize=(224, 224)):
    """Extracts frames from a video with uniform sampling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return False

    # Uniform sampling indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            if len(frames) == num_frames:
                break
    
    cap.release()

    # Handle cases where video is shorter than expected or read failed
    if len(frames) < num_frames:
        # Pad with last frame
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((resize[1], resize[0], 3), dtype=np.uint8))

    # Save frames
    video_name = video_path.stem
    save_dir = output_dir / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(str(save_dir / f"frame_{idx:03d}.jpg"), frame)
    
    return True

def process_dataset(source_dir, split_ratio=0.8, is_pre_split=False):
    """
    Processes a dataset.
    is_pre_split: True if dataset already has train/val structure (like RWF-2000).
    """
    print(f"Processing {source_dir.name}...")
    
    if is_pre_split:
        for split in ['train', 'val']:
            for label in ['Fight', 'NonFight']:
                input_path = source_dir / split / label
                if not input_path.exists():
                    print(f"Warning: {input_path} does not exist, skipping.")
                    continue
                
                output_path = OUTPUT_ROOT / split / label
                output_path.mkdir(parents=True, exist_ok=True)
                
                videos = list(input_path.glob('*.*'))
                print(f"  {split}/{label}: {len(videos)} videos")
                
                for video in tqdm(videos, desc=f"{source_dir.name} {split}/{label}"):
                    extract_frames(video, output_path, NUM_FRAMES, (IMG_SIZE, IMG_SIZE))
    else:
        # For datasets without split (Hockey, RLVS), we split manually
        for label in ['Fight', 'NonFight']:
            input_path = source_dir / label
            if not input_path.exists():
                print(f"Warning: {input_path} does not exist, skipping.")
                continue

            videos = list(input_path.glob('*.*'))
            random.shuffle(videos)
            
            split_idx = int(len(videos) * split_ratio)
            train_videos = videos[:split_idx]
            val_videos = videos[split_idx:]
            
            print(f"  {label}: {len(videos)} videos -> Train: {len(train_videos)}, Val: {len(val_videos)}")
            
            # Process Train
            output_train = OUTPUT_ROOT / 'train' / label
            output_train.mkdir(parents=True, exist_ok=True)
            for video in tqdm(train_videos, desc=f"{source_dir.name} Train/{label}"):
                extract_frames(video, output_train, NUM_FRAMES, (IMG_SIZE, IMG_SIZE))
                
            # Process Val
            output_val = OUTPUT_ROOT / 'val' / label
            output_val.mkdir(parents=True, exist_ok=True)
            for video in tqdm(val_videos, desc=f"{source_dir.name} Val/{label}"):
                extract_frames(video, output_val, NUM_FRAMES, (IMG_SIZE, IMG_SIZE))

def main():
    random.seed(42)
    
    # Clean output directory
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Process RWF-2000 (Already split)
    process_dataset(DATASET_ROOT / 'RWF-2000', is_pre_split=True)

    # 2. Process HockeyFight (Not split)
    process_dataset(DATASET_ROOT / 'HockeyFight', is_pre_split=False)

    # 3. Process RLVS (Not split)
    process_dataset(DATASET_ROOT / 'RLVS', is_pre_split=False)

    print("\nProcessing Complete!")
    print(f"Data saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
