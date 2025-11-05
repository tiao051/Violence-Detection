"""
PyTorch YOLOv8 Performance Benchmark Suite

Benchmarks end-to-end performance (preprocessing + inference) of PyTorch YOLO
across test video frames with frame-by-frame measurement.

Key Features:
- End-to-end benchmarking (includes model inference)
- Frame-by-frame latency measurement
- Structured performance metrics (latency, FPS, std deviation)
- GPU/CPU auto-detection

Usage:
    python ai_service/tests/detector/pytorch_optimization.py
    pytest ai_service/tests/detector/pytorch_optimization.py -v -s
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.config import DEFAULT_MODEL
from ai_service.detection import PyTorchDetector


class PyTorchOptimizer:
    """PyTorch performance benchmark suite."""

    WARMUP_RUNS = 5
    BENCHMARK_RUNS = 15

    def __init__(self):
        """Initialize PyTorch optimizer."""
        self.test_frames = []

    def setup(self) -> bool:
        """Setup and validate environment."""
        try:
            # Load test frames from all videos
            test_videos_dir = Path("utils/test_data/test_inputs")
            test_videos = sorted(test_videos_dir.glob("*.mp4"))

            if not test_videos:
                print(f"No test videos found in {test_videos_dir}")
                return False

            for video_file in test_videos:
                cap = cv2.VideoCapture(str(video_file))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.test_frames.append(frame)
                else:
                    print(f"Failed to read: {video_file.name}")
                    return False

            if not self.test_frames:
                print("No valid test frames loaded")
                return False

            return True

        except Exception as e:
            print(f"Setup error: {e}")
            return False

    def benchmark_pytorch_model(self) -> Optional[Dict]:
        """Benchmark PyTorch model across all test frames (end-to-end)."""
        try:
            # Initialize detector
            detector = PyTorchDetector(
                model_path=DEFAULT_MODEL,
                conf_threshold=0.25,
                iou_threshold=0.5,
                class_id=0,
            )

            # Get device info
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Benchmark across all frames
            all_times = []

            for frame in self.test_frames:
                # Warmup
                for _ in range(self.WARMUP_RUNS):
                    detector.detect(frame)

                # Benchmark
                for _ in range(self.BENCHMARK_RUNS):
                    start = time.time()
                    detector.detect(frame)
                    all_times.append((time.time() - start) * 1000)

            avg_time = np.mean(all_times)
            std_time = np.std(all_times)
            fps = 1000.0 / avg_time

            return {
                "model": "yolov8n.pt",
                "avg_time_ms": round(avg_time, 2),
                "std_time_ms": round(std_time, 2),
                "fps": round(fps, 2),
                "device": device.upper(),
                "frames_tested": len(self.test_frames),
            }

        except Exception as e:
            print(f"Benchmark error: {e}")
            return None

    def run_all_tasks(self) -> None:
        """Run benchmark pipeline."""
        if not self.setup():
            print("Setup failed")
            return

        # Benchmark model
        metrics = self.benchmark_pytorch_model()

        # Print results
        print()
        self._print_results(metrics)
        print()

    def _print_results(self, metrics: Optional[Dict]) -> None:
        """Print benchmark results."""
        if not metrics:
            print("No results to display\n")
            return

        print(f"{'Model':<20} {'Avg (ms)':<12} {'FPS':<10} {'Std (ms)':<10} {'Device':<10} {'Videos':<8}")
        print(
            f"{metrics['model']:<20} {metrics['avg_time_ms']:<12} "
            f"{metrics['fps']:<10} {metrics['std_time_ms']:<10} {metrics['device']:<10} {metrics['frames_tested']:<8}"
        )

def test_pytorch_optimization():
    """Pytest-compatible test for PyTorch optimization suite."""
    optimizer = PyTorchOptimizer()
    optimizer.run_all_tasks()


if __name__ == "__main__":
    optimizer = PyTorchOptimizer()
    optimizer.run_all_tasks()
