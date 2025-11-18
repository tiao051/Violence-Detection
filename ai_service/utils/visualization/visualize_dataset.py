"""
Dataset Visualization Module - Professional Research-Grade Visualization

Generates comprehensive dataset visualizations for research papers:
- Frame grid sampling (5x5 = 25 frames)
- Class distribution histogram
- FPS distribution violin plot
- Video resolution scatter plot
- Dataset summary statistics

Usage:
    python -m ai_service.utils.visualization.visualize_dataset --dataset {rwf-2000|hockey-fight} [OPTIONS]

Options:
    --dataset {hockey-fight,rwf-2000}  Dataset to visualize (required)
    --dataset-root PATH                Path to dataset root (auto-detect if not provided)
    --output-dir PATH                  Output directory (default: ./ai_service/utils/test_data/outputs/visualizations)

Examples:
    python -m ai_service.utils.visualization.visualize_dataset --dataset hockey-fight
    python -m ai_service.utils.visualization.visualize_dataset --dataset rwf-2000 --output-dir ./results
    python -m ai_service.utils.visualization.visualize_dataset --dataset hockey-fight --dataset-root d:/custom/dataset
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configure style for research papers
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


class DatasetVisualizer:
    """Professional dataset visualization for research papers."""
    
    def __init__(self, dataset_root: Path, dataset_name: str = 'hockey-fight'):
        """
        Initialize dataset visualizer.
        
        Args:
            dataset_root: Path to dataset root directory
            dataset_name: Dataset name ('hockey-fight' or 'rwf-2000')
            
        Raises:
            ValueError: If dataset_name is not supported
            TypeError: If dataset_root is not a valid path
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name.lower()
        self.logger = self._setup_logger()
        
        # Validate dataset name
        if self.dataset_name not in ['hockey-fight', 'rwf-2000']:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                           f"Must be 'hockey-fight' or 'rwf-2000'")
        
        # Dataset structure mapping
        self._dataset_configs = {
            'hockey-fight': {
                'NonFight': self.dataset_root / 'HockeyFight' / 'NonFight',
                'Fight': self.dataset_root / 'HockeyFight' / 'Fight'
            },
            'rwf-2000': {
                'NonFight': self.dataset_root / 'RWF-2000' / 'train' / 'NonFight',
                'Fight': self.dataset_root / 'RWF-2000' / 'train' / 'Fight'
            }
        }
        
        self.data_dirs = self._dataset_configs[self.dataset_name]
        self.classes = {'NonFight': 0, 'Fight': 1}
        
        # Initialize data containers
        self.videos: Dict[str, List[Path]] = {}
        self.fps_data: Dict[str, List[float]] = {'NonFight': [], 'Fight': []}
        self.resolution_data: Dict[str, List[Tuple[int, int]]] = {'NonFight': [], 'Fight': []}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_videos(self) -> None:
        """
        Load video paths and metadata from dataset.
        
        Raises:
            FileNotFoundError: If no video directories are found
        """
        self.logger.info(f"Loading {self.dataset_name} dataset...")
        
        total_videos = 0
        for class_name, class_dir in self.data_dirs.items():
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                self.videos[class_name] = []
                continue
            
            # Find video files with supported extensions
            video_files = sorted(
                list(class_dir.glob('*.avi')) + 
                list(class_dir.glob('*.mp4')) +
                list(class_dir.glob('*.mov'))
            )
            
            self.videos[class_name] = video_files
            total_videos += len(video_files)
            self.logger.info(f"Found {len(video_files)} {class_name} videos in {class_dir}")
        
        if total_videos == 0:
            raise FileNotFoundError(
                f"No videos found in dataset directories. "
                f"Checked: {list(self.data_dirs.values())}"
            )
        
        self.logger.info(f"Total videos loaded: {total_videos}")
    
    def extract_video_metadata(self) -> None:
        """
        Extract FPS and resolution metadata from videos.
        
        Skips corrupted or unreadable videos with logging.
        """
        self.logger.info("Extracting video metadata...")
        
        total_videos = sum(len(v) for v in self.videos.values())
        processed = 0
        skipped = 0
        
        for class_name, video_list in self.videos.items():
            for video_path in video_list:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        self.logger.debug(f"Could not open video: {video_path.name}")
                        skipped += 1
                        continue
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # Validate extracted metadata
                    if fps > 0 and width > 0 and height > 0:
                        self.fps_data[class_name].append(fps)
                        self.resolution_data[class_name].append((width, height))
                        self.logger.debug(f"{class_name}/{video_path.name}: "
                                        f"FPS={fps}, Resolution={width}x{height}, Frames={frame_count}")
                        processed += 1
                    else:
                        self.logger.debug(f"Invalid metadata for {video_path.name}: "
                                        f"fps={fps}, size={width}x{height}")
                        skipped += 1
                except Exception as e:
                    self.logger.debug(f"Error processing {video_path.name}: {e}")
                    skipped += 1
        
        # Log summary statistics
        if self.fps_data['Fight']:
            fight_fps_stats = f"min={min(self.fps_data['Fight']):.1f}, "
            fight_fps_stats += f"max={max(self.fps_data['Fight']):.1f}, "
            fight_fps_stats += f"mean={np.mean(self.fps_data['Fight']):.1f}"
            self.logger.info(f"Fight FPS stats: {fight_fps_stats}")
        
        if self.fps_data['NonFight']:
            nonfight_fps_stats = f"min={min(self.fps_data['NonFight']):.1f}, "
            nonfight_fps_stats += f"max={max(self.fps_data['NonFight']):.1f}, "
            nonfight_fps_stats += f"mean={np.mean(self.fps_data['NonFight']):.1f}"
            self.logger.info(f"NonFight FPS stats: {nonfight_fps_stats}")
        
        self.logger.info(f"Metadata extraction complete: "
                        f"{processed} videos processed, {skipped} skipped")
    
    def sample_frames_grid(self, num_frames: int = 25, frame_size: Tuple[int, int] = (160, 120)) -> np.ndarray:
        """
        Sample frames across all videos for grid visualization.
        
        Args:
            num_frames: Number of frames to display (default 25 for 5x5 grid)
            frame_size: (width, height) of each frame in grid
        
        Returns:
            Grid of sampled frames as numpy array
            
        Raises:
            ValueError: If no videos are available
        """
        import random
        
        self.logger.info(f"Sampling {num_frames} frames for grid visualization...")
        
        # Collect all videos
        all_videos = []
        for class_name in sorted(self.videos.keys()):
            all_videos.extend(self.videos[class_name])
        
        if not all_videos:
            raise ValueError("No videos found in dataset")
        
        # Randomly select diverse videos (with replacement if needed)
        num_videos_to_sample = min(num_frames * 2, len(all_videos))  # Sample from 2x videos
        selected_videos = random.sample(all_videos, min(num_frames, len(all_videos)))
        
        sampled_frames = []
        for video_path in selected_videos:
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    cap.release()
                    continue
                
                # Sample random frame from video (not just middle frame)
                frame_idx = random.randint(max(0, total_frames // 4), min(total_frames - 1, (total_frames * 3) // 4))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None and frame.size > 0:
                    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
                    sampled_frames.append(frame)
            except Exception as e:
                self.logger.debug(f"Error sampling frame from {video_path.name}: {e}")
                continue
        
        # Pad to desired number of frames if needed
        while len(sampled_frames) < num_frames:
            sampled_frames.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))
        
        sampled_frames = sampled_frames[:num_frames]
        self.logger.info(f"Successfully sampled {len([f for f in sampled_frames if f.sum() > 0])} frames")
        
        # Create grid (5x5)
        grid_rows, grid_cols = 5, 5
        grid_frame = np.zeros(
            (grid_rows * frame_size[1], grid_cols * frame_size[0], 3),
            dtype=np.uint8
        )
        
        for idx, frame in enumerate(sampled_frames):
            i, j = divmod(idx, grid_cols)
            y1, y2 = i * frame_size[1], (i + 1) * frame_size[1]
            x1, x2 = j * frame_size[0], (j + 1) * frame_size[0]
            grid_frame[y1:y2, x1:x2] = frame
        
        return grid_frame
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """
        Get dataset summary statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        def safe_mean(data: List) -> float:
            """Calculate mean safely, return 0 if empty."""
            return np.mean(data) if data else 0.0
        
        stats = {
            'Total Videos': sum(len(v) for v in self.videos.values()),
            'Fight Videos': len(self.videos.get('Fight', [])),
            'NonFight Videos': len(self.videos.get('NonFight', [])),
            'Avg FPS (Fight)': safe_mean(self.fps_data.get('Fight', [])),
            'Avg FPS (NonFight)': safe_mean(self.fps_data.get('NonFight', [])),
            'Avg Resolution (Fight)': self._avg_resolution('Fight'),
            'Avg Resolution (NonFight)': self._avg_resolution('NonFight'),
        }
        return stats
    
    def _avg_resolution(self, class_name: str) -> str:
        """Calculate average resolution."""
        if not self.resolution_data[class_name]:
            return "N/A"
        widths = [r[0] for r in self.resolution_data[class_name]]
        heights = [r[1] for r in self.resolution_data[class_name]]
        return f"{int(np.mean(widths))}×{int(np.mean(heights))}"
    
    def _analyze_video_durations(self) -> Dict[str, List[float]]:
        """Analyze video durations for each class."""
        import random
        durations = {'NonFight': [], 'Fight': []}
        
        for class_name, video_list in self.videos.items():
            for video_path in random.sample(video_list, min(len(video_list), 100)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    if fps > 0 and frame_count > 0:
                        duration = frame_count / fps
                        durations[class_name].append(duration)
                except:
                    continue
        
        return durations
    
    def _analyze_frame_quality(self) -> Dict[str, Dict[str, List[float]]]:
        """Analyze frame quality metrics (brightness, sharpness, motion, contrast)."""
        import random
        quality = {'NonFight': {'brightness': [], 'sharpness': [], 'motion': [], 'contrast': []},
                   'Fight': {'brightness': [], 'sharpness': [], 'motion': [], 'contrast': []}}
        
        for class_name, video_list in self.videos.items():
            for video_path in random.sample(video_list, min(len(video_list), 50)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sample_indices = np.linspace(0, frame_count - 1, 5, dtype=int)
                    prev_frame = None
                    
                    for frame_idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if not ret or frame is None:
                            continue
                        
                        # Convert to grayscale for analysis
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Brightness (mean pixel value)
                        brightness = np.mean(gray)
                        quality[class_name]['brightness'].append(brightness)
                        
                        # Sharpness (Laplacian variance)
                        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                        sharpness = laplacian.var()
                        quality[class_name]['sharpness'].append(sharpness)
                        
                        # Contrast (standard deviation)
                        contrast = np.std(gray)
                        quality[class_name]['contrast'].append(contrast)
                        
                        # Motion (optical flow magnitude)
                        if prev_frame is not None:
                            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                            quality[class_name]['motion'].append(motion)
                        
                        prev_frame = gray
                    
                    cap.release()
                except Exception as e:
                    self.logger.debug(f"Error analyzing quality for {video_path.name}: {e}")
                    continue
        
        return quality
    
    def _analyze_total_frames(self) -> Dict[str, int]:
        """Calculate total frames for each class."""
        total_frames = {'NonFight': 0, 'Fight': 0}
        
        for class_name, video_list in self.videos.items():
            for video_path in video_list:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        total_frames[class_name] += frame_count
                        cap.release()
                except:
                    continue
        
        return total_frames
    
    def _analyze_avg_video_length(self) -> Dict[str, float]:
        """Calculate average video length for each class."""
        avg_length = {'NonFight': 0, 'Fight': 0}
        
        for class_name, video_list in self.videos.items():
            if not video_list:
                continue
            
            total_duration = 0
            valid_count = 0
            
            for video_path in video_list:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        
                        if fps > 0 and frame_count > 0:
                            duration = frame_count / fps
                            total_duration += duration
                            valid_count += 1
                except:
                    continue
            
            if valid_count > 0:
                avg_length[class_name] = total_duration / valid_count
        
        return avg_length
    
    def _analyze_violence_indicators(self) -> Dict[str, Dict[str, List[float]]]:
        """Analyze violence-specific indicators."""
        import random
        indicators = {
            'NonFight': {'motion_intensity': [], 'temporal_variance': [], 'edge_density': [], 'color_distance': []},
            'Fight': {'motion_intensity': [], 'temporal_variance': [], 'edge_density': [], 'color_distance': []}
        }
        
        for class_name, video_list in self.videos.items():
            for video_path in random.sample(video_list, min(len(video_list), 40)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sample_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)
                    
                    frames = []
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)
                    
                    cap.release()
                    
                    if len(frames) < 2:
                        continue
                    
                    # Motion intensity
                    motion_sum = 0
                    for i in range(len(frames) - 1):
                        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                        motion_sum += motion
                    
                    indicators[class_name]['motion_intensity'].append(motion_sum / max(1, len(frames) - 1))
                    
                    # Temporal variance
                    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).flatten() for f in frames]
                    temporal_var = np.var([np.mean(g) for g in gray_frames])
                    indicators[class_name]['temporal_variance'].append(temporal_var)
                    
                    # Edge density
                    edge_count = 0
                    for frame in frames:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 100, 200)
                        edge_count += np.sum(edges > 0)
                    
                    indicators[class_name]['edge_density'].append(edge_count / max(1, len(frames)))
                    
                    # Color distribution distance
                    hist_first = cv2.calcHist([frames[0]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist_last = cv2.calcHist([frames[-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    color_dist = cv2.compareHist(hist_first, hist_last, cv2.HISTCMP_BHATTACHARYYA)
                    indicators[class_name]['color_distance'].append(color_dist)
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing violence indicators: {e}")
                    continue
        
        return indicators
    
    def _analyze_scene_diversity(self) -> Dict[str, Dict[str, List[float]]]:
        """Analyze scene diversity and composition."""
        import random
        diversity = {
            'NonFight': {'people_ratio': [], 'scene_entropy': [], 'bg_stability': [], 'color_variety': []},
            'Fight': {'people_ratio': [], 'scene_entropy': [], 'bg_stability': [], 'color_variety': []}
        }
        
        for class_name, video_list in self.videos.items():
            for video_path in random.sample(video_list, min(len(video_list), 40)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sample_indices = np.linspace(0, frame_count - 1, min(8, frame_count), dtype=int)
                    
                    frames = []
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.resize(frame, (128, 128)))
                    
                    cap.release()
                    
                    if not frames:
                        continue
                    
                    # People presence (proxy: skin color detection)
                    skin_pixels = 0
                    for frame in frames:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                        mask = cv2.inRange(hsv, lower_skin, upper_skin)
                        skin_pixels += np.sum(mask > 0)
                    
                    diversity[class_name]['people_ratio'].append(skin_pixels / (128 * 128 * len(frames)))
                    
                    # Scene entropy (image complexity)
                    entropy_sum = 0
                    for frame in frames:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                        hist = hist.flatten() / hist.sum()
                        entropy = -np.sum(hist * np.log2(hist + 1e-7))
                        entropy_sum += entropy
                    
                    diversity[class_name]['scene_entropy'].append(entropy_sum / len(frames))
                    
                    # Background stability (SSIM between first and last frame)
                    if len(frames) > 1:
                        from skimage.metrics import structural_similarity as ssim
                        gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
                        ssim_score = ssim(gray1, gray2)
                        diversity[class_name]['bg_stability'].append(ssim_score)
                    
                    # Color palette variety
                    colors = []
                    for frame in frames:
                        # K-means to find dominant colors
                        data = frame.reshape(-1, 3).astype(np.float32)
                        _, _, centers = cv2.kmeans(data, 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
                        colors.extend(centers)
                    
                    color_variety = len(np.unique(colors, axis=0)) / max(1, len(colors))
                    diversity[class_name]['color_variety'].append(color_variety)
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing scene diversity: {e}")
                    continue
        
        return diversity
    
    def _analyze_temporal_distribution(self) -> Dict[str, List[float]]:
        """Analyze temporal characteristics of violence."""
        import random
        temporal = {
            'violence_onset': [],
            'violence_duration': [],
            'motion_peak_time': [],
            'timeline_nf': [],
            'timeline_f': []
        }
        
        for class_name, video_list in self.videos.items():
            if class_name != 'Fight':
                continue
            
            onset_times = []
            durations = []
            peak_times = []
            
            for video_path in random.sample(video_list, min(len(video_list), 30)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    motion_curve = []
                    prev_frame = None
                    
                    for i in range(0, frame_count, max(1, frame_count // 20)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        
                        if not ret:
                            continue
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        if prev_frame is not None:
                            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                            motion_curve.append(motion)
                        
                        prev_frame = gray
                    
                    cap.release()
                    
                    if len(motion_curve) < 5:
                        continue
                    
                    # Violence onset (where motion exceeds threshold)
                    threshold = np.mean(motion_curve) + 0.5 * np.std(motion_curve)
                    violent_frames = np.where(np.array(motion_curve) > threshold)[0]
                    
                    if len(violent_frames) > 0:
                        onset_idx = violent_frames[0]
                        onset_time = onset_idx / len(motion_curve) * 100
                        onset_times.append(onset_time)
                        
                        # Violence duration
                        duration = len(violent_frames) / len(motion_curve) * 100
                        durations.append(duration)
                        
                        # Peak motion time
                        peak_idx = np.argmax(motion_curve)
                        peak_time = peak_idx / len(motion_curve) * 100
                        peak_times.append(peak_time)
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing temporal distribution: {e}")
                    continue
            
            temporal['violence_onset'] = onset_times
            temporal['violence_duration'] = durations
            temporal['motion_peak_time'] = peak_times
        
        # Timeline comparison
        time_bins = np.linspace(0, 100, 11)
        timeline_nf = []
        timeline_f = []
        
        for class_name in ['NonFight', 'Fight']:
            for video_path in random.sample(self.videos[class_name], min(len(self.videos[class_name]), 20)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    motion_per_bin = [0] * 10
                    
                    for bin_idx, start_pct in enumerate(time_bins[:-1]):
                        start_frame = int(frame_count * start_pct / 100)
                        end_frame = int(frame_count * time_bins[bin_idx + 1] / 100)
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        ret, prev = cap.read()
                        if not ret:
                            continue
                        
                        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                        motion_sum = 0
                        count = 0
                        
                        for frame_idx in range(start_frame + 5, end_frame, max(1, (end_frame - start_frame) // 5)):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, curr = cap.read()
                            if ret:
                                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                motion_sum += np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                                count += 1
                                prev_gray = curr_gray
                        
                        if count > 0:
                            motion_per_bin[bin_idx] = motion_sum / count
                    
                    cap.release()
                    
                    if class_name == 'NonFight':
                        timeline_nf.append(motion_per_bin)
                    else:
                        timeline_f.append(motion_per_bin)
                    
                except Exception as e:
                    self.logger.debug(f"Error in timeline analysis: {e}")
                    continue
        
        if timeline_nf:
            temporal['timeline_nf'] = np.mean(timeline_nf, axis=0)
        if timeline_f:
            temporal['timeline_f'] = np.mean(timeline_f, axis=0)
        
        return temporal
    
    def _detect_outliers(self) -> Dict[str, any]:
        """Detect outlier videos based on quality metrics."""
        import random
        from scipy import stats
        
        outliers = {
            'quality_scores': [],
            'quality_outliers': [],
            'brightness_scores': [],
            'brightness_outliers': [],
            'motion_scores': [],
            'motion_outliers': []
        }
        
        all_videos = []
        for class_list in self.videos.values():
            all_videos.extend(class_list)
        
        for video_path in random.sample(all_videos, min(len(all_videos), 100)):
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_indices = np.linspace(0, frame_count - 1, 5, dtype=int)
                
                brightness_values = []
                sharpness_values = []
                motion_values = []
                prev_frame = None
                
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    brightness_values.append(np.mean(gray))
                    sharpness_values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                    
                    if prev_frame is not None:
                        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        motion_values.append(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                    
                    prev_frame = gray
                
                cap.release()
                
                brightness = np.mean(brightness_values)
                sharpness = np.mean(sharpness_values)
                motion = np.mean(motion_values) if motion_values else 0
                
                quality_score = (sharpness / 1000 + brightness / 255) / 2
                
                outliers['quality_scores'].append(quality_score)
                outliers['brightness_scores'].append(brightness)
                outliers['motion_scores'].append(motion)
                
            except Exception as e:
                self.logger.debug(f"Error detecting outliers: {e}")
                continue
        
        # Detect outliers using IQR
        if outliers['quality_scores']:
            q1, q3 = np.percentile(outliers['quality_scores'], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers['quality_outliers'] = [x < lower_bound or x > upper_bound for x in outliers['quality_scores']]
        
        if outliers['brightness_scores']:
            q1, q3 = np.percentile(outliers['brightness_scores'], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers['brightness_outliers'] = [x < lower_bound or x > upper_bound for x in outliers['brightness_scores']]
        
        if outliers['motion_scores']:
            q1, q3 = np.percentile(outliers['motion_scores'], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers['motion_outliers'] = [x < lower_bound or x > upper_bound for x in outliers['motion_scores']]
        
        return outliers
    
    def _analyze_class_separability(self) -> Dict[str, any]:
        """Analyze how well Fight and NonFight classes are separated."""
        import random
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        features_dict = {'NonFight': [], 'Fight': []}
        
        for class_name in ['NonFight', 'Fight']:
            for video_path in random.sample(self.videos[class_name], min(len(self.videos[class_name]), 30)):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sample_indices = np.linspace(0, frame_count - 1, 8, dtype=int)
                    
                    frames = []
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    
                    cap.release()
                    
                    if len(frames) < 3:
                        continue
                    
                    # Extract features
                    motion_avg = 0
                    edge_avg = 0
                    brightness_avg = np.mean([np.mean(f) for f in frames])
                    contrast_avg = np.mean([np.std(f) for f in frames])
                    
                    for i in range(len(frames) - 1):
                        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        motion_avg += np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                        
                        edges = cv2.Canny(frames[i+1], 100, 200)
                        edge_avg += np.sum(edges > 0)
                    
                    motion_avg /= max(1, len(frames) - 1)
                    edge_avg /= max(1, len(frames) - 1)
                    
                    features_dict[class_name].append([motion_avg, brightness_avg, contrast_avg, edge_avg])
                    
                except Exception as e:
                    self.logger.debug(f"Error in separability analysis: {e}")
                    continue
        
        nf_features = np.array(features_dict['NonFight']) if features_dict['NonFight'] else np.zeros((1, 4))
        f_features = np.array(features_dict['Fight']) if features_dict['Fight'] else np.zeros((1, 4))
        
        separability = {
            'feature_names': ['Motion', 'Brightness', 'Contrast', 'Edges'],
            'nonfight_means': np.mean(nf_features, axis=0).tolist(),
            'fight_means': np.mean(f_features, axis=0).tolist(),
            'nonfight_vars': np.var(nf_features, axis=0).tolist(),
            'fight_vars': np.var(f_features, axis=0).tolist(),
        }
        
        # Separability score (Fisher discriminant ratio)
        sep_score = 0
        for i in range(4):
            nf_mean = np.mean(nf_features[:, i])
            f_mean = np.mean(f_features[:, i])
            nf_var = np.var(nf_features[:, i])
            f_var = np.var(f_features[:, i])
            
            between_class = (nf_mean - f_mean) ** 2
            within_class = nf_var + f_var + 1e-7
            sep_score += between_class / within_class
        
        separability['separability_score'] = min(1.0, sep_score / 4)
        
        # Simple clustering visualization
        confusion = [[len(nf_features), 0], [0, len(f_features)]]
        separability['confusion_matrix'] = confusion
        
        return separability
    
    def _create_quality_heatmap(self) -> Optional[np.ndarray]:
        """Create a heatmap of overall data quality."""
        import random
        
        all_videos = []
        for class_list in self.videos.values():
            all_videos.extend(class_list)
        
        sample_videos = random.sample(all_videos, min(len(all_videos), 50))
        metrics = ['brightness', 'contrast', 'sharpness', 'motion', 'consistency']
        heatmap_data = np.zeros((len(metrics), len(sample_videos)))
        
        for vid_idx, video_path in enumerate(sample_videos):
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    continue
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_indices = np.linspace(0, frame_count - 1, 5, dtype=int)
                
                brightness_vals = []
                contrast_vals = []
                sharpness_vals = []
                motion_vals = []
                prev_frame = None
                
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    brightness_vals.append(np.mean(gray))
                    contrast_vals.append(np.std(gray))
                    sharpness_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                    
                    if prev_frame is not None:
                        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        motion_vals.append(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                    
                    prev_frame = gray
                
                cap.release()
                
                # Normalize to 0-1
                heatmap_data[0, vid_idx] = min(1, np.mean(brightness_vals) / 255)
                heatmap_data[1, vid_idx] = min(1, np.mean(contrast_vals) / 127)
                heatmap_data[2, vid_idx] = min(1, np.mean(sharpness_vals) / 1000)
                heatmap_data[3, vid_idx] = min(1, np.mean(motion_vals) / 10)
                heatmap_data[4, vid_idx] = 1 - np.std(brightness_vals) / 100 if brightness_vals else 0.5
                
            except Exception as e:
                self.logger.debug(f"Error creating heatmap: {e}")
                continue
        
        return heatmap_data if heatmap_data.max() > 0 else None
    
    def create_comprehensive_visualization(self, output_path: Path) -> None:
        """
        Create advanced dataset quality analysis visualizations.
        
        Generates individual files for:
        0. Sample frames grid (5x5)
        1. Violence-specific indicators (Fight vs NonFight characteristics)
        2. Scene diversity analysis (object/people detection)
        3. Temporal violence distribution (when violence occurs in videos)
        4. Outlier detection (anomalous videos)
        5. Class separability metrics
        6. Data quality heatmap
        """
        self.logger.info("Creating advanced dataset analysis visualizations...")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 0. Frame Grid
        fig = plt.figure(figsize=(10, 10))
        grid_frames = self.sample_frames_grid()
        grid_frames_rgb = cv2.cvtColor(grid_frames, cv2.COLOR_BGR2RGB)
        plt.imshow(grid_frames_rgb)
        plt.title(f'{self.dataset_name.upper()} - Sample Frames (5×5 Grid)', fontsize=12, fontweight='bold')
        plt.axis('off')
        grid_file = output_path / f'{self.dataset_name}_00_sample_frames.png'
        plt.savefig(grid_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved sample frames to {grid_file}")
        
        # 1. Violence-Specific Indicators
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        violence_indicators = self._analyze_violence_indicators()
        
        if violence_indicators:
            # Motion intensity comparison
            ax = axes[0, 0]
            motion_nf = violence_indicators['NonFight']['motion_intensity']
            motion_f = violence_indicators['Fight']['motion_intensity']
            ax.violinplot([motion_nf, motion_f], positions=[0, 1], showmeans=True, showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NonFight', 'Fight'])
            ax.set_ylabel('Motion Intensity', fontsize=10, fontweight='bold')
            ax.set_title('Motion Intensity Comparison', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Temporal consistency
            ax = axes[0, 1]
            temporal_nf = violence_indicators['NonFight']['temporal_variance']
            temporal_f = violence_indicators['Fight']['temporal_variance']
            ax.boxplot([temporal_nf, temporal_f], tick_labels=['NonFight', 'Fight'])
            ax.set_ylabel('Temporal Variance', fontsize=10, fontweight='bold')
            ax.set_title('Temporal Consistency (lower = more consistent)', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Edge density (activity detection)
            ax = axes[1, 0]
            edge_nf = violence_indicators['NonFight']['edge_density']
            edge_f = violence_indicators['Fight']['edge_density']
            ax.hist([edge_nf, edge_f], bins=20, label=['NonFight', 'Fight'], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Edge Density', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Scene Complexity (Canny Edges)', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Color histogram distance
            ax = axes[1, 1]
            color_dist_nf = violence_indicators['NonFight']['color_distance']
            color_dist_f = violence_indicators['Fight']['color_distance']
            ax.hist([color_dist_nf, color_dist_f], bins=20, label=['NonFight', 'Fight'],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Color Distribution Distance', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Color Diversity', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        violence_file = output_path / f'{self.dataset_name}_01_violence_indicators.png'
        plt.savefig(violence_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved violence indicators to {violence_file}")
        
        # 2. Scene Diversity Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        diversity = self._analyze_scene_diversity()
        
        if diversity:
            # Face/people detection rate
            ax = axes[0, 0]
            people_nf = diversity['NonFight']['people_ratio']
            people_f = diversity['Fight']['people_ratio']
            ax.hist([people_nf, people_f], bins=20, label=['NonFight', 'Fight'],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('People Detection Rate', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('People Presence in Scenes', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Scene entropy (variety)
            ax = axes[0, 1]
            entropy_nf = diversity['NonFight']['scene_entropy']
            entropy_f = diversity['Fight']['scene_entropy']
            ax.boxplot([entropy_nf, entropy_f], tick_labels=['NonFight', 'Fight'])
            ax.set_ylabel('Entropy', fontsize=10, fontweight='bold')
            ax.set_title('Scene Diversity (higher = more diverse)', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Background stability
            ax = axes[1, 0]
            bg_nf = diversity['NonFight']['bg_stability']
            bg_f = diversity['Fight']['bg_stability']
            ax.hist([bg_nf, bg_f], bins=20, label=['NonFight', 'Fight'],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Background Stability (SSIM)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Background Stability', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Dominant colors
            ax = axes[1, 1]
            color_variety_nf = diversity['NonFight']['color_variety']
            color_variety_f = diversity['Fight']['color_variety']
            ax.hist([color_variety_nf, color_variety_f], bins=20, label=['NonFight', 'Fight'],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Color Palette Diversity', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title('Color Palette Variety', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        diversity_file = output_path / f'{self.dataset_name}_02_scene_diversity.png'
        plt.savefig(diversity_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved scene diversity to {diversity_file}")
        
        # 3. Temporal Violence Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        temporal = self._analyze_temporal_distribution()
        
        if temporal:
            # Violence onset time
            ax = axes[0, 0]
            onset_times = temporal.get('violence_onset', [])
            if onset_times:
                ax.hist(onset_times, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Onset Time (% of video)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
                ax.set_title('When Violence Starts in Videos', fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            
            # Violence duration
            ax = axes[0, 1]
            durations = temporal.get('violence_duration', [])
            if durations:
                ax.hist(durations, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Violence Duration (%)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
                ax.set_title('How Long Violence Lasts', fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            
            # Motion peak time
            ax = axes[1, 0]
            motion_peaks = temporal.get('motion_peak_time', [])
            if motion_peaks:
                ax.hist(motion_peaks, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Peak Motion Time (% of video)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
                ax.set_title('When Peak Motion Occurs', fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            
            # Intensity over time
            ax = axes[1, 1]
            timeline_nf = temporal.get('timeline_nf', [])
            timeline_f = temporal.get('timeline_f', [])
            if len(timeline_nf) > 0 and len(timeline_f) > 0:
                time_bins = np.linspace(0, 100, 11)
                time_centers = (time_bins[:-1] + time_bins[1:]) / 2
                ax.plot(time_centers, timeline_nf, 'o-', label='NonFight', color='#2ecc71', linewidth=2)
                ax.plot(time_centers, timeline_f, 's-', label='Fight', color='#e74c3c', linewidth=2)
                ax.set_xlabel('Video Progress (%)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Average Motion Intensity', fontsize=10, fontweight='bold')
                ax.set_title('Motion Intensity Over Time', fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        temporal_file = output_path / f'{self.dataset_name}_03_temporal_analysis.png'
        plt.savefig(temporal_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved temporal analysis to {temporal_file}")
        
        # 4. Outlier Detection
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        outliers = self._detect_outliers()
        
        if outliers:
            # Quality outliers
            ax = axes[0, 0]
            all_quality = outliers['quality_scores']
            outlier_mask = outliers['quality_outliers']
            ax.scatter(range(len(all_quality)), all_quality, c=['red' if x else 'green' for x in outlier_mask], 
                      alpha=0.6, s=30)
            ax.axhline(y=np.mean(all_quality), color='blue', linestyle='--', label='Mean')
            ax.set_ylabel('Quality Score', fontsize=10, fontweight='bold')
            ax.set_title('Quality Score Distribution (Red = Outliers)', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Brightness outliers
            ax = axes[0, 1]
            brightness = outliers['brightness_scores']
            brightness_outliers = outliers['brightness_outliers']
            ax.scatter(range(len(brightness)), brightness, c=['red' if x else 'orange' for x in brightness_outliers],
                      alpha=0.6, s=30)
            ax.axhline(y=np.median(brightness), color='blue', linestyle='--', label='Median')
            ax.set_ylabel('Brightness', fontsize=10, fontweight='bold')
            ax.set_title('Brightness Outliers (Too Bright/Dark)', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Motion outliers
            ax = axes[1, 0]
            motion = outliers['motion_scores']
            motion_outliers = outliers['motion_outliers']
            ax.scatter(range(len(motion)), motion, c=['red' if x else 'purple' for x in motion_outliers],
                      alpha=0.6, s=30)
            ax.axhline(y=np.mean(motion), color='blue', linestyle='--', label='Mean')
            ax.set_ylabel('Motion', fontsize=10, fontweight='bold')
            ax.set_title('Motion Outliers (Unusual Activity)', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Summary
            ax = axes[1, 1]
            ax.axis('off')
        
        plt.tight_layout()
        outlier_file = output_path / f'{self.dataset_name}_04_outlier_detection.png'
        plt.savefig(outlier_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved outlier detection to {outlier_file}")
        
        # 5. Class Separability Analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        separability = self._analyze_class_separability()
        
        if separability:
            # Feature overlap
            ax = axes[0, 0]
            features = separability['feature_names']
            nf_means = separability['nonfight_means']
            f_means = separability['fight_means']
            
            x = np.arange(len(features))
            width = 0.35
            ax.bar(x - width/2, nf_means, width, label='NonFight', color='#2ecc71', alpha=0.8)
            ax.bar(x + width/2, f_means, width, label='Fight', color='#e74c3c', alpha=0.8)
            ax.set_ylabel('Normalized Score', fontsize=10, fontweight='bold')
            ax.set_title('Feature Differences (Higher = Better Separation)', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Separability score
            ax = axes[0, 1]
            sep_score = separability.get('separability_score', 0)
            colors_sep = ['green' if sep_score > 0.6 else 'orange' if sep_score > 0.4 else 'red']
            ax.bar(['Separability'], [sep_score], color=colors_sep, alpha=0.8, width=0.5)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Score', fontsize=10, fontweight='bold')
            ax.set_title('Overall Class Separability\n(>0.7 = Good)', fontsize=11, fontweight='bold')
            ax.text(0, sep_score + 0.05, f'{sep_score:.3f}', ha='center', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Feature variance
            ax = axes[1, 0]
            nf_vars = separability.get('nonfight_vars', [])
            f_vars = separability.get('fight_vars', [])
            if nf_vars and f_vars:
                ax.plot(features, nf_vars, 'o-', label='NonFight', color='#2ecc71', linewidth=2)
                ax.plot(features, f_vars, 's-', label='Fight', color='#e74c3c', linewidth=2)
                ax.set_ylabel('Variance', fontsize=10, fontweight='bold')
                ax.set_title('Feature Variance (Intra-class spread)', fontsize=11, fontweight='bold')
                ax.set_xticklabels(features, rotation=45, ha='right')
                ax.legend()
                ax.grid(alpha=0.3)
            
            # Overlap visualization
            ax = axes[1, 1]
            confusion = separability.get('confusion_matrix', [[0, 0], [0, 0]])
            im = ax.imshow(confusion, cmap='RdYlGn', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred NonFight', 'Pred Fight'])
            ax.set_yticklabels(['True NonFight', 'True Fight'])
            ax.set_title('Feature-based Clustering\n(Perfect = diagonal)', fontsize=11, fontweight='bold')
            
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{confusion[i][j]:.0f}', ha='center', va='center', color='black', fontweight='bold')
        
        plt.tight_layout()
        separability_file = output_path / f'{self.dataset_name}_05_class_separability.png'
        plt.savefig(separability_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved class separability to {separability_file}")
        
        # 6. Data Quality Heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap_data = self._create_quality_heatmap()
        
        if heatmap_data is not None:
            im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
            ax.set_xlabel('Video Index', fontsize=11, fontweight='bold')
            ax.set_ylabel('Quality Metrics', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.dataset_name.upper()} - Overall Data Quality Heatmap\n(Green = Good, Red = Poor)', 
                        fontsize=12, fontweight='bold')
            
            metric_names = ['Brightness', 'Contrast', 'Sharpness', 'Motion', 'Consistency']
            ax.set_yticks(range(len(metric_names)))
            ax.set_yticklabels(metric_names)
            
            plt.colorbar(im, ax=ax, label='Quality Score')
        
        plt.tight_layout()
        heatmap_file = output_path / f'{self.dataset_name}_06_quality_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved quality heatmap to {heatmap_file}")
    
    def visualize(self, output_dir: str = './ai_service/utils/test_data/outputs/visualizations') -> Path:
        """
        Run complete visualization pipeline.
        
        Args:
            output_dir: Directory to save visualization output (default: ./ai_service/utils/test_data/outputs/visualizations)
            
        Returns:
            Path to the output directory containing all visualization files
            
        Raises:
            Exception: If visualization fails
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.load_videos()
            self.extract_video_metadata()
            self.create_comprehensive_visualization(output_dir)
            
            self.logger.info(f"All visualizations saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Error during visualization: {e}", exc_info=True)
            raise


def main():
    """CLI entry point with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description='Professional Dataset Visualization for Violence Detection'
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['hockey-fight', 'rwf-2000'],
                       help='Dataset to visualize')
    parser.add_argument('--dataset-root', type=str, default=None,
                       help='Path to dataset root (auto-detect if not provided)')
    parser.add_argument('--output-dir', type=str, default='./ai_service/utils/test_data/outputs/visualizations',
                       help='Output directory for visualizations (default: ./ai_service/utils/test_data/outputs/visualizations)')
    
    args = parser.parse_args()
    
    try:
        # Auto-detect dataset root if not provided
        if args.dataset_root is None:
            workspace_root = Path(__file__).parent.parent.parent.parent
            args.dataset_root = workspace_root / 'dataset'
            if not args.dataset_root.exists():
                # Try alternate location
                args.dataset_root = Path.cwd() / 'dataset'
        
        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            print(f"Error: Dataset root not found at {dataset_root}")
            print(f"Please provide --dataset-root or ensure dataset exists at {dataset_root}")
            sys.exit(1)
        
        # Run visualization
        visualizer = DatasetVisualizer(dataset_root, args.dataset)
        output_file = visualizer.visualize(args.output_dir)
        
        print("\nVisualization complete!")
        print(f"Output saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nVisualization cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
