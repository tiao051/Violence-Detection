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
    
    def create_comprehensive_visualization(self, output_path: Path) -> None:
        """
        Create separate dataset visualizations for research papers.
        
        Generates individual files for:
        1. Frame grid (5x5)
        2. Class distribution histogram
        3. FPS distribution violin plot
        4. Resolution scatter plot
        """
        self.logger.info("Creating dataset visualizations...")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Frame Grid
        fig = plt.figure(figsize=(10, 10))
        grid_frames = self.sample_frames_grid()
        grid_frames_rgb = cv2.cvtColor(grid_frames, cv2.COLOR_BGR2RGB)
        plt.imshow(grid_frames_rgb)
        plt.title(f'{self.dataset_name.upper()} - Sample Frames (5×5 Grid)', fontsize=12, fontweight='bold')
        plt.axis('off')
        grid_file = output_path / f'{self.dataset_name}_01_frame_grid.png'
        plt.savefig(grid_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved frame grid to {grid_file}")
        
        # 2. Class Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        class_counts = [len(self.videos.get('NonFight', [])), len(self.videos.get('Fight', []))]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['NonFight', 'Fight'], class_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Videos', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.dataset_name.upper()} - Class Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        dist_file = output_path / f'{self.dataset_name}_02_class_distribution.png'
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved class distribution to {dist_file}")
        
        # 3. FPS Distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        fps_lists = [self.fps_data.get('NonFight', []), self.fps_data.get('Fight', [])]
        fps_lists = [l for l in fps_lists if len(l) > 0]
        
        if fps_lists:
            parts = ax.violinplot(fps_lists, positions=[0, 1], showmeans=True, showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NonFight', 'Fight'])
            ax.set_ylabel('FPS', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.dataset_name.upper()} - FPS Distribution', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        fps_file = output_path / f'{self.dataset_name}_03_fps_distribution.png'
        plt.savefig(fps_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved FPS distribution to {fps_file}")
        
        # 4. Resolution Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if self.resolution_data['NonFight']:
            non_fight_widths = [r[0] for r in self.resolution_data['NonFight']]
            non_fight_heights = [r[1] for r in self.resolution_data['NonFight']]
            ax.scatter(non_fight_widths, non_fight_heights, alpha=0.6, s=100, label='NonFight', color='#2ecc71', edgecolor='black', linewidth=0.5)
        
        if self.resolution_data['Fight']:
            fight_widths = [r[0] for r in self.resolution_data['Fight']]
            fight_heights = [r[1] for r in self.resolution_data['Fight']]
            ax.scatter(fight_widths, fight_heights, alpha=0.6, s=100, label='Fight', color='#e74c3c', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Height (pixels)', fontsize=11, fontweight='bold')
        ax.set_title(f'{self.dataset_name.upper()} - Video Resolution Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        res_file = output_path / f'{self.dataset_name}_04_resolution_scatter.png'
        plt.savefig(res_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved resolution scatter to {res_file}")
    
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
