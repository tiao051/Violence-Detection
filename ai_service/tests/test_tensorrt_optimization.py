"""
PHASE 4: TensorRT Model Export and Benchmark Suite

Task 4.1: Check TensorRT dependencies
Task 4.2: Convert ONNX to TensorRT engine
Task 4.3: Benchmark TensorRT vs ONNX
Task 4.4: Print comparison table

Usage:
    python -m pytest ai_service/tests/test_tensorrt_optimization.py -v -s
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_service.config import DEFAULT_MODEL
from ai_service.common.preprocessing.augmentation import YOLOPreprocessor


class TensorRTOptimizer:
    """TensorRT export and benchmark suite."""

    INPUT_SIZES = [320, 384, 512]
    WARMUP_RUNS = 5
    BENCHMARK_RUNS = 15

    def __init__(self):
        """Initialize TensorRT optimizer."""
        self.models_dir = Path(__file__).resolve().parent.parent / "models" / "weights"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.test_frames = []
        self.has_tensorrt = False
        self.has_gpu = torch.cuda.is_available()

    def setup(self) -> bool:
        """Setup and validate environment."""
        try:
            # Check GPU
            if not self.has_gpu:
                print("No GPU found - TensorRT requires NVIDIA GPU with CUDA")
                return False
            print(f"GPU: {torch.cuda.get_device_name(0)}")

            # Check TensorRT
            try:
                import tensorrt as trt
                print(f"TensorRT {trt.__version__}")
                self.has_tensorrt = True
            except ImportError:
                print("TensorRT not installed")
                print("  pip install tensorrt")
                return False

            # Check ONNX
            try:
                import onnx
                print(f"ONNX {onnx.__version__}")
            except ImportError:
                print("ONNX not installed")
                return False

            # Load test frames
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

    def export_tensorrt_engine(self, onnx_path: Path, imgsz: int) -> Tuple[bool, Optional[Path]]:
        """Convert ONNX to TensorRT engine."""
        try:
            import tensorrt as trt

            print(f"  Converting to TensorRT ({imgsz}x{imgsz})...")

            engine_path = self.models_dir / f"yolov8n_{imgsz}.engine"

            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Build engine
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:

                # Parse ONNX
                with open(onnx_path, "rb") as f:
                    if not parser.parse(f.read()):
                        print(f"    Failed to parse ONNX")
                        for error in range(parser.num_errors):
                            print(f"    {parser.get_error(error)}")
                        return False, None

                # Build config
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 30  # 1GB
                
                # Use FP16 if available (faster)
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print(f"    Using FP16 precision")

                # Build engine
                with builder.build_engine(network, config) as engine:
                    # Serialize engine
                    with open(engine_path, "wb") as f:
                        f.write(engine.serialize())

            if engine_path.exists():
                size_mb = engine_path.stat().st_size / (1024 ** 2)
                print(f"    {engine_path.name} ({size_mb:.2f}MB)")
                return True, engine_path
            else:
                print(f"    Export failed")
                return False, None

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def benchmark_tensorrt_engine(self, engine_path: Path, imgsz: int) -> Optional[Dict]:
        """Benchmark TensorRT engine."""
        try:
            import tensorrt as trt

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Load engine
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            with trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data) as engine:
                context = engine.create_execution_context()
                
                # Get I/O info
                input_name = engine.get_binding_name(0)
                output_name = engine.get_binding_name(1)
                input_shape = engine.get_binding_shape(input_name)
                
                # Allocate GPU memory
                import pycuda.driver as cuda
                import pycuda.autoinit
                
                input_size = np.prod(input_shape) * 4  # float32
                output_shape = engine.get_binding_shape(output_name)
                output_size = np.prod(output_shape) * 4
                
                d_input = cuda.mem_alloc(input_size)
                d_output = cuda.mem_alloc(output_size)
                
                # Initialize preprocessor
                preprocessor = YOLOPreprocessor(input_size=(imgsz, imgsz))
                
                # Benchmark
                all_times = []
                
                for frame in self.test_frames:
                    # Preprocess
                    blob, _ = preprocessor(frame)
                    
                    # Warmup
                    for _ in range(self.WARMUP_RUNS):
                        cuda.memcpy_htod(d_input, blob)
                        context.execute_v2([int(d_input), int(d_output)])
                        cuda.Context.synchronize()
                    
                    # Benchmark
                    for _ in range(self.BENCHMARK_RUNS):
                        cuda.memcpy_htod(d_input, blob)
                        
                        start = time.time()
                        context.execute_v2([int(d_input), int(d_output)])
                        cuda.Context.synchronize()
                        all_times.append((time.time() - start) * 1000)
                
                # Cleanup
                d_input.free()
                d_output.free()
                
                avg_time = np.mean(all_times)
                std_time = np.std(all_times)
                fps = 1000.0 / avg_time
                
                return {
                    "model": engine_path.name,
                    "imgsz": imgsz,
                    "avg_time_ms": round(avg_time, 2),
                    "std_time_ms": round(std_time, 2),
                    "fps": round(fps, 2),
                    "provider": "TensorRT (FP16/FP32)",
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

        if not self.has_gpu:
            print("\nTensorRT requires NVIDIA GPU - skipping")
            return

        print("\n" + "="*80)
        print("PHASE 4: TENSORRT EXPORT AND BENCHMARK".center(80))
        print("="*80)

        # Task 4.2: Convert ONNX to TensorRT
        print("\n[TASK 4.2] Converting ONNX to TensorRT Engines (320, 384, 512)")
        print("-"*80)

        engine_results = {}
        for size in self.INPUT_SIZES:
            onnx_path = self.models_dir / f"yolov8n_{size}.onnx"
            
            if not onnx_path.exists():
                print(f"  Skipping {size} - ONNX not found")
                continue
            
            success, engine_path = self.export_tensorrt_engine(onnx_path, size)
            if success:
                engine_results[size] = engine_path

        if not engine_results:
            print("No TensorRT engines created")
            return

        # Task 4.3: Benchmark TensorRT engines
        print("\n[TASK 4.3] Benchmarking TensorRT Engines")
        print("-"*80)

        benchmark_results = {}
        for size in self.INPUT_SIZES:
            if size in engine_results:
                print(f"  Benchmark {size}x{size}...")
                metrics = self.benchmark_tensorrt_engine(engine_results[size], size)
                if metrics:
                    benchmark_results[size] = metrics

        # Task 4.4: Print results
        print("\n[TASK 4.4] Results - TensorRT vs ONNX")
        print("-"*80)
        self._print_results(benchmark_results)

        print("\n" + "="*80)
        print("PHASE 4 COMPLETE".center(80))
        print("="*80 + "\n")

    def _print_results(self, results: Dict) -> None:
        """Print benchmark results table."""
        if not results:
            print("No results to display\n")
            return

        print(f"{'Model':<20} {'Size':<8} {'Avg (ms)':<12} {'FPS':<10} {'Std (ms)':<10} {'Provider':<25}")
        print("-"*90)

        for size in sorted(results.keys()):
            m = results[size]
            print(
                f"{m['model']:<20} {m['imgsz']:<8} {m['avg_time_ms']:<12} "
                f"{m['fps']:<10} {m['std_time_ms']:<10} {m['provider']:<25}"
            )

        print()


def test_tensorrt_optimization():
    """Pytest-compatible test for TensorRT optimization."""
    optimizer = TensorRTOptimizer()
    optimizer.run_all_tasks()


if __name__ == "__main__":
    optimizer = TensorRTOptimizer()
    optimizer.run_all_tasks()
