"""
PHASE 3: ONNX Model Export and Benchmark Suite

Task 3.1: Check ONNX dependencies
Task 3.2: Export ONNX models (320, 384, 512)
Task 3.4: Benchmark ONNX models (CPU/GPU auto-detect)
Task 3.5: Print comparison table

Usage:
    python -m pytest ai_service/tests/test_onnx_optimization.py -v -s
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.config import DEFAULT_MODEL
from ai_service.common.preprocessing.augmentation import YOLOPreprocessor


class ONNXOptimizer:
    """ONNX export and benchmark suite."""

    INPUT_SIZES = [320, 384, 512]
    WARMUP_RUNS = 5
    BENCHMARK_RUNS = 15

    def __init__(self):
        """Initialize ONNX optimizer."""
        self.models_dir = ROOT_DIR / "ai_service" / "models" / "weights"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.test_frames = []  # Multiple test frames from 4 videos
        
        # Initialize YOLO preprocessor (will be updated per size in benchmark)
        self.preprocessor = None

    def setup(self) -> bool:
        """Setup and validate environment."""
        try:
            # Check ONNX dependencies
            try:
                import onnx
                import onnxruntime as ort
                print(f"ONNX {onnx.__version__}")
                print(f"ONNXRUNTIME {ort.__version__}")
            except ImportError as e:
                print(f"Missing: {e}")
                print("  pip install onnx onnxruntime")
                return False

            # Load test frames from all 4 videos
            test_videos_dir = Path("utils/test_data")
            test_videos = sorted(test_videos_dir.glob("*.mp4"))
            
            if not test_videos:
                print(f"No test videos found in {test_videos_dir}")
                return False
            
            print(f"Loading frames from {len(test_videos)} videos:")
            for video_file in test_videos:
                cap = cv2.VideoCapture(str(video_file))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.test_frames.append(frame)
                    print(f"  {video_file.name} ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    print(f"  {video_file.name} - Failed to read")
            
            if not self.test_frames:
                print("No valid test frames loaded")
                return False
            
            return True

        except Exception as e:
            print(f"Setup error: {e}")
            return False

    def export_onnx_model(self, imgsz: int) -> Tuple[bool, Optional[Path]]:
        """Export YOLOv8n to ONNX format."""
        try:
            from ultralytics import YOLO

            print(f"  Exporting yolov8n_{imgsz}.onnx...")
            
            model = YOLO(str(DEFAULT_MODEL))
            output_path = self.models_dir / f"yolov8n_{imgsz}.onnx"
            
            # Export with GPU if available
            device = 0 if torch.cuda.is_available() else "cpu"
            model.export(format="onnx", imgsz=imgsz, half=False, device=device)

            # Handle default filename from Ultralytics
            default_onnx = self.models_dir / "yolov8n.onnx"
            if default_onnx.exists() and default_onnx != output_path:
                default_onnx.rename(output_path)

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 ** 2)
                print(f"    {output_path.name} ({size_mb:.2f}MB)")
                return True, output_path
            else:
                print(f"    Export failed")
                return False, None

        except Exception as e:
            print(f"    Error: {e}")
            return False, None

    def benchmark_onnx_model(self, model_path: Path, imgsz: int) -> Optional[Dict]:
        """Benchmark ONNX model across all 4 test frames (end-to-end with preprocessing)."""
        try:
            import onnxruntime as ort

            # Create session with GPU auto-detect
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(str(model_path), providers=providers)
            provider = sess.get_providers()[0] if sess.get_providers() else "Unknown"
            
            input_name = sess.get_inputs()[0].name
            output_names = [o.name for o in sess.get_outputs()]

            # Initialize preprocessor for this input size
            preprocessor = YOLOPreprocessor(input_size=(imgsz, imgsz))

            # Benchmark across all 4 frames (end-to-end: preprocessing + inference)
            all_times = []
            
            for frame in self.test_frames:
                # Preprocess: BGR frame → CHW blob ready for ONNX
                blob, _ = preprocessor(frame)
                
                # Warmup
                for _ in range(self.WARMUP_RUNS):
                    sess.run(output_names, {input_name: blob})

                # Benchmark (end-to-end: preprocess + infer)
                for _ in range(self.BENCHMARK_RUNS):
                    # Preprocess
                    blob, _ = preprocessor(frame)
                    
                    # Infer
                    start = time.time()
                    sess.run(output_names, {input_name: blob})
                    all_times.append((time.time() - start) * 1000)

            avg_time = np.mean(all_times)
            std_time = np.std(all_times)
            fps = 1000.0 / avg_time

            return {
                "model": model_path.name,
                "imgsz": imgsz,
                "avg_time_ms": round(avg_time, 2),
                "std_time_ms": round(std_time, 2),
                "fps": round(fps, 2),
                "provider": provider,
                "frames_tested": len(self.test_frames)
            }

        except Exception as e:
            print(f"  Benchmark error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_tasks(self) -> None:
        """Run export and benchmark pipeline."""
        if not self.setup():
            print("\nSetup failed")
            return

        print("\n" + "="*80)
        print("PHASE 3: ONNX EXPORT AND BENCHMARK".center(80))
        print("="*80)

        # Task 3.2: Export all models
        print("\n[TASK 3.2] Exporting ONNX Models (320, 384, 512)")
        print("-"*80)
        
        export_results = {}
        for size in self.INPUT_SIZES:
            success, path = self.export_onnx_model(size)
            if success:
                export_results[size] = path

        if len(export_results) < len(self.INPUT_SIZES):
            print(f"\nOnly {len(export_results)}/{len(self.INPUT_SIZES)} models exported")

        # Task 3.4: Benchmark models
        print("\n[TASK 3.4] Benchmarking ONNX Models (4 videos)")
        print("-"*80)
        
        benchmark_results = {}
        for size in self.INPUT_SIZES:
            if size in export_results:
                print(f"  Benchmark {size}x{size}...")
                metrics = self.benchmark_onnx_model(export_results[size], size)
                if metrics:
                    benchmark_results[size] = metrics

        # Task 3.5: Print results
        print("\n[TASK 3.5] Results")
        print("-"*80)
        self._print_results(benchmark_results)

        print("\n" + "="*80)
        print("PHASE 3 COMPLETE - Ready for Phase 4 (Runtime Optimization)".center(80))
        print("="*80 + "\n")

    def _print_results(self, results: Dict) -> None:
        """Print benchmark results table."""
        if not results:
            print("No results to display\n")
            return

        print(f"{'Model':<20} {'Size':<8} {'Avg (ms)':<12} {'FPS':<10} {'Std (ms)':<10} {'Provider':<15} {'Videos':<8}")
        print("-"*90)
        
        for size in sorted(results.keys()):
            m = results[size]
            print(
                f"{m['model']:<20} {m['imgsz']:<8} {m['avg_time_ms']:<12} "
                f"{m['fps']:<10} {m['std_time_ms']:<10} {m['provider']:<15} {m['frames_tested']:<8}"
            )
        
        print()



def test_onnx_optimization():
    """Pytest-compatible test for ONNX optimization suite."""
    optimizer = ONNXOptimizer()
    optimizer.run_all_tasks()


if __name__ == "__main__":
    optimizer = ONNXOptimizer()
    optimizer.run_all_tasks()
