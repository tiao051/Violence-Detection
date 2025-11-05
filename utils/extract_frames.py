"""
Frame Extraction Utility

This module extracts a specified number of frames from a video file and saves them as JPEG images.
Frames are evenly distributed throughout the video to provide representative sampling.

Usage:
    python extract_frames.py --input video.mp4 --output frames_dir
    python extract_frames.py --input video.mp4 --output frames_dir --num-frames 60

Author: Violence Detection Project
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, num_frames=30):
    """
    Extract specified number of frames from a video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: 30)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Calculate the step to evenly distribute frames
    if total_frames > 0:
        step = max(1, total_frames // num_frames)
    else:
        print("Error: Video has no frames")
        cap.release()
        return False
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame at regular intervals
        if frame_count % step == 0 and saved_count < num_frames:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved: {frame_path}")
            saved_count += 1
        
        frame_count += 1
        
        if saved_count >= num_frames:
            break
    
    cap.release()
    print(f"\nSuccessfully extracted {saved_count} frames from {video_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--output", required=True, help="Directory to save extracted frames")
    parser.add_argument("--num-frames", type=int, default=30, help="Number of frames to extract (default: 30)")
    
    args = parser.parse_args()
    
    extract_frames(args.input, args.output, num_frames=args.num_frames)
