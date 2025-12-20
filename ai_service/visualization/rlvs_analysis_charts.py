"""
Generate RLVS Dataset Analysis Charts
Similar to RWF-2000 and Hockey Fight analysis visualizations
Creates:
1. Violence Statistics (4 subplots)
2. Fight vs NonFight Comparison (4 subplots)
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Paths
RLVS_FIGHT_PATH = Path("c:/Users/thang/Desktop/Violence-Detection/dataset/RLVS/Fight")
RLVS_NONFIGHT_PATH = Path("c:/Users/thang/Desktop/Violence-Detection/dataset/RLVS/NonFight")
OUTPUT_PATH = Path("c:/Users/thang/Desktop/Violence-Detection/ai_service/visualization")


def get_video_info(video_path: Path) -> dict:
    """Extract basic info from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


def compute_motion_intensity(video_path: Path, sample_frames: int = 30) -> list:
    """Compute motion intensity across video using frame differences."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return []
    
    # Sample evenly across video
    frame_indices = np.linspace(0, total_frames - 2, min(sample_frames, total_frames - 1), dtype=int)
    
    motion_intensities = []
    prev_frame = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion = np.mean(diff)
            motion_intensities.append(motion)
        
        prev_frame = gray
    
    cap.release()
    return motion_intensities


def compute_scene_complexity(video_path: Path) -> float:
    """Compute scene complexity using Canny edge detection on middle frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0)


def compute_color_diversity(video_path: Path) -> float:
    """Compute color diversity using histogram comparison."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return 0
    
    # Calculate color histogram and compute entropy/diversity
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros for log
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def analyze_violence_timing(video_path: Path) -> dict:
    """Analyze when violence/motion peaks occur in video."""
    motion = compute_motion_intensity(video_path, sample_frames=20)
    if len(motion) < 5:
        return None
    
    motion = np.array(motion)
    peak_idx = np.argmax(motion)
    peak_position = peak_idx / len(motion) * 100  # Percentage
    
    # Estimate onset (when motion first exceeds mean)
    mean_motion = np.mean(motion)
    onset_indices = np.where(motion > mean_motion)[0]
    onset_position = (onset_indices[0] / len(motion) * 100) if len(onset_indices) > 0 else 0
    
    # Estimate duration (percentage of video with high motion)
    high_motion_frames = np.sum(motion > mean_motion)
    duration_percent = high_motion_frames / len(motion) * 100
    
    return {
        'onset': onset_position,
        'peak': peak_position,
        'duration': duration_percent,
        'motion_over_time': motion.tolist()
    }


def create_violence_statistics_chart(fight_videos: list, output_path: Path):
    """Create 4-panel violence statistics chart."""
    print("Analyzing violence timing statistics...")
    
    random.seed(42)
    sampled = random.sample(fight_videos, min(30, len(fight_videos)))
    
    onsets = []
    durations = []
    peaks = []
    all_motion_curves = []
    
    for video in sampled:
        result = analyze_violence_timing(video)
        if result:
            onsets.append(result['onset'])
            durations.append(result['duration'])
            peaks.append(result['peak'])
            all_motion_curves.append(result['motion_over_time'])
    
    print(f"Analyzed {len(onsets)} videos")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RLVS Dataset - Violence Timing Analysis', fontsize=14, fontweight='bold')
    
    # 1. When Violence Starts (histogram)
    axes[0, 0].hist(onsets, bins=10, color='salmon', edgecolor='darkred', alpha=0.8)
    axes[0, 0].set_xlabel('Onset Time (% of video)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('When Violence Starts in Videos')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. How Long Violence Lasts (histogram)
    axes[0, 1].hist(durations, bins=10, color='salmon', edgecolor='darkred', alpha=0.8)
    axes[0, 1].set_xlabel('Violence Duration (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('How Long Violence Lasts')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. When Peak Motion Occurs (histogram)
    axes[1, 0].hist(peaks, bins=10, color='skyblue', edgecolor='darkblue', alpha=0.8)
    axes[1, 0].set_xlabel('Peak Motion Time (% of video)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('When Peak Motion Occurs')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Motion Intensity Over Time (line chart with average)
    # Normalize all curves to same length
    normalized_curves = []
    for curve in all_motion_curves:
        if len(curve) >= 5:
            x = np.linspace(0, 100, len(curve))
            x_new = np.linspace(0, 100, 10)
            normalized = np.interp(x_new, x, curve)
            normalized_curves.append(normalized)
    
    if normalized_curves:
        avg_curve = np.mean(normalized_curves, axis=0)
        std_curve = np.std(normalized_curves, axis=0)
        x_points = np.linspace(10, 100, 10)
        
        # Just plot average for Fight (no NonFight comparison in this chart)
        axes[1, 1].plot(x_points, avg_curve, 'r-o', label='Fight', linewidth=2, markersize=6)
        axes[1, 1].fill_between(x_points, avg_curve - std_curve, avg_curve + std_curve, 
                                color='red', alpha=0.2)
    
    axes[1, 1].set_xlabel('Video Progress (%)')
    axes[1, 1].set_ylabel('Average Motion Intensity')
    axes[1, 1].set_title('Motion Intensity Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / "rlvs_violence_statistics.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


def create_fight_vs_nonfight_chart(fight_videos: list, nonfight_videos: list, output_path: Path):
    """Create 4-panel Fight vs NonFight comparison chart."""
    print("Analyzing Fight vs NonFight characteristics...")
    
    random.seed(42)
    n_samples = 30
    fight_sample = random.sample(fight_videos, min(n_samples, len(fight_videos)))
    nonfight_sample = random.sample(nonfight_videos, min(n_samples, len(nonfight_videos)))
    
    # Collect metrics
    fight_motion = []
    nonfight_motion = []
    fight_temporal = []
    nonfight_temporal = []
    fight_complexity = []
    nonfight_complexity = []
    fight_color = []
    nonfight_color = []
    
    print("Processing Fight videos...")
    for video in fight_sample:
        motion = compute_motion_intensity(video, sample_frames=15)
        if motion:
            fight_motion.append(np.mean(motion))
            fight_temporal.append(np.var(motion))
        
        complexity = compute_scene_complexity(video)
        fight_complexity.append(complexity)
        
        color = compute_color_diversity(video)
        fight_color.append(color)
    
    print("Processing NonFight videos...")
    for video in nonfight_sample:
        motion = compute_motion_intensity(video, sample_frames=15)
        if motion:
            nonfight_motion.append(np.mean(motion))
            nonfight_temporal.append(np.var(motion))
        
        complexity = compute_scene_complexity(video)
        nonfight_complexity.append(complexity)
        
        color = compute_color_diversity(video)
        nonfight_color.append(color)
    
    print(f"Fight samples: {len(fight_motion)}, NonFight samples: {len(nonfight_motion)}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RLVS Dataset - Fight vs NonFight Comparison', fontsize=14, fontweight='bold')
    
    # 1. Motion Intensity Comparison (Violin plot)
    parts = axes[0, 0].violinplot([nonfight_motion, fight_motion], positions=[1, 2], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(['NonFight', 'Fight'])
    axes[0, 0].set_ylabel('Motion Intensity')
    axes[0, 0].set_title('Motion Intensity Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Temporal Consistency (Box plot)
    bp = axes[0, 1].boxplot([nonfight_temporal, fight_temporal], labels=['NonFight', 'Fight'], patch_artist=True)
    colors = ['lightgreen', 'salmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Temporal Variance')
    axes[0, 1].set_title('Temporal Consistency (lower = more consistent)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scene Complexity (Histogram)
    bins = np.linspace(min(min(fight_complexity), min(nonfight_complexity)), 
                       max(max(fight_complexity), max(nonfight_complexity)), 15)
    axes[1, 0].hist(nonfight_complexity, bins=bins, alpha=0.6, label='NonFight', color='green')
    axes[1, 0].hist(fight_complexity, bins=bins, alpha=0.6, label='Fight', color='salmon')
    axes[1, 0].set_xlabel('Edge Density')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Scene Complexity (Canny Edges)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Color Diversity (Histogram)
    bins = np.linspace(min(min(fight_color), min(nonfight_color)), 
                       max(max(fight_color), max(nonfight_color)), 15)
    axes[1, 1].hist(nonfight_color, bins=bins, alpha=0.6, label='NonFight', color='green')
    axes[1, 1].hist(fight_color, bins=bins, alpha=0.6, label='Fight', color='salmon')
    axes[1, 1].set_xlabel('Color Distribution Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Color Diversity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / "rlvs_fight_vs_nonfight.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("RLVS Dataset Analysis Charts")
    print("=" * 60)
    
    # Get video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    fight_videos = [f for f in RLVS_FIGHT_PATH.iterdir() if f.suffix.lower() in video_extensions]
    nonfight_videos = [f for f in RLVS_NONFIGHT_PATH.iterdir() if f.suffix.lower() in video_extensions]
    
    print(f"Found {len(fight_videos)} Fight videos and {len(nonfight_videos)} NonFight videos")
    
    # Create charts
    print("\n--- Creating Violence Statistics Chart ---")
    create_violence_statistics_chart(fight_videos, OUTPUT_PATH)
    
    print("\n--- Creating Fight vs NonFight Comparison Chart ---")
    create_fight_vs_nonfight_chart(fight_videos, nonfight_videos, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ All charts generated successfully!")
    print("=" * 60)
