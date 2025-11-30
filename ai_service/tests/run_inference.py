"""CLI script to run end-to-end inference on a local test video.

Usage:
    python ai_service/tests/run_inference.py --input <video_file> [OPTIONS]

Arguments:
    --input <video_file>             Input video filename (required)
                                     Available: violence_1.mp4, violence_2.mp4, violence_3.mp4,
                                               violence_4.mp4, violence_5.mp4, non_violence_1.mp4

Options:
    --model <path>                   Path to model checkpoint (optional)
                                     Auto-detects best available if not provided
                                     Priority: best_model_hf_v3.pt → best_model_rwf_v3.pt → others
    
    --backbone <backbone>            STE backbone used during training (optional)
                                     Options: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
                                             efficientnet_b0, mnasnet
                                     Auto-detects from model name if not provided
    
    --confidence-threshold <value>   Confidence threshold for violence detection (default: 0.5)
                                     Range: 0.0-1.0

Defaults:
    - Input folder: `ai_service/utils/test_data/inputs/videos/`
    - Output folder: `ai_service/utils/test_data/outputs/detection/`
    - Model: auto-detected from checkpoints folder
    - Backbone: auto-detected from model filename
    - Confidence threshold: 0.5

Examples:
    # Auto-detect everything (recommended)
    python ai_service/tests/run_inference.py --input violence_2.mp4
    
    # Specify backbone explicitly
    python ai_service/tests/run_inference.py --input violence_2.mp4 --backbone mobilenet_v3_small
    
    # Lower confidence threshold for more sensitivity
    python ai_service/tests/run_inference.py --input violence_2.mp4 --confidence-threshold 0.3
    
    # Use custom model
    python ai_service/tests/run_inference.py --input violence_2.mp4 --model /path/to/custom_model.pt

Output:
    - Detected frames: ai_service/utils/test_data/outputs/detection/detection_window_*.jpg
    - Metadata: ai_service/utils/test_data/outputs/detection/manifest.json
    - Inference results with confidence scores and latency metrics

The script will:
    - Read the input video, resize frames to 224x224 (model expectation)
    - Feed frames into the ViolenceDetectionModel
    - When buffer reaches 30 frames, perform inference and save results
    - Repeat for entire video with sliding window
    - Generate comprehensive statistics (accuracy, latency, FPS, etc.)

If the video has fewer frames than the model buffer size, the script will
repeat the last frame to fill the buffer and perform one final inference.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_service.inference.inference_model import ViolenceDetectionModel, InferenceConfig

logger = logging.getLogger("run_inference")


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_buffer_frames(frames_rgb: List[np.ndarray], out_dir: Path, window_idx: int) -> List[Path]:
    paths = []
    for i, fr in enumerate(frames_rgb):
        # frames are RGB in model buffer; convert to BGR for cv2.imwrite
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        fname = out_dir / f"detection_window_{window_idx:04d}_frame_{i:03d}.jpg"
        cv2.imwrite(str(fname), bgr)
        paths.append(fname)
    return paths


def run_on_video(video_path: Path, model_path: Path, out_base: Path, backbone: str = 'mobilenet_v2', confidence_threshold: float = 0.5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    out_dir = out_base / "detection"
    ensure_dirs(out_dir)

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"VIDEO INFO")
    logger.info(f"{'='*70}")
    logger.info(f"  File: {video_path.name}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Duration: {total_frames/fps:.2f}s")
    
    # Initialize model with backbone and confidence threshold
    config = InferenceConfig(
        model_path=str(model_path),
        backbone=backbone,
        confidence_threshold=confidence_threshold
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL CONFIG")
    logger.info(f"{'='*70}")
    logger.info(f"  Model path: {model_path.name}")
    logger.info(f"  Backbone: {backbone}")
    logger.info(f"  Confidence threshold: {confidence_threshold}")
    logger.info(f"  Buffer size: {config.num_frames} frames")
    logger.info("")
    
    model = ViolenceDetectionModel(config)

    frame_count = 0
    window_idx = 0
    manifest = []
    latencies = []
    violence_count = 0
    total_detections = 0

    # Read frames and feed model (model.add_frame converts to RGB internally)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to model expected size (224x224)
        resized = cv2.resize(frame, (config.frame_size[0], config.frame_size[1]), interpolation=cv2.INTER_LINEAR)
        # feed into model
        model.add_frame(resized)
        frame_count += 1

        result = None
        if len(model.frame_buffer) >= model.MAX_BUFFER_SIZE:
            result = model.predict()

        if result is not None:
            total_detections += 1
            is_violence = result.get('violence', False)
            confidence = result.get('confidence', 0.0)
            latency = result.get('latency_ms', 0.0)
            
            if is_violence:
                violence_count += 1
            
            latencies.append(latency)
            
            # Save current buffer frames and metadata
            buffered_frames = list(model.frame_buffer)
            saved_paths = save_buffer_frames(buffered_frames, out_dir, window_idx)
            meta = {
                "window_idx": window_idx,
                "video": str(video_path.name),
                "frame_count_at_inference": frame_count,
                "result": result,
                "saved_frames": [str(p.name) for p in saved_paths],
            }
            manifest.append(meta)
            
            # Log detection
            detection_status = "🔴 VIOLENCE" if is_violence else "🟢 NO VIOLENCE"
            logger.info(f"[Window {window_idx:3d}] Frame {frame_count:4d} | {detection_status} | Confidence: {confidence:.4f} | Latency: {latency:.2f}ms")
            
            # advance window index
            window_idx += 1

    # If video ended and model buffer not full but has frames, pad with last frame and run one inference
    if len(model.frame_buffer) > 0 and len(model.frame_buffer) < model.MAX_BUFFER_SIZE:
        # repeat last frame until buffer full
        last_frame_rgb = model.frame_buffer[-1]
        frames_padded = model.MAX_BUFFER_SIZE - len(model.frame_buffer)
        while len(model.frame_buffer) < model.MAX_BUFFER_SIZE:
            # model.add_frame expects BGR input (it converts to RGB internally), so convert back
            bgr = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)
            model.add_frame(bgr)
        
        result = model.predict()
        if result is not None:
            total_detections += 1
            is_violence = result.get('violence', False)
            confidence = result.get('confidence', 0.0)
            latency = result.get('latency_ms', 0.0)
            
            if is_violence:
                violence_count += 1
            
            latencies.append(latency)
            
            buffered_frames = list(model.frame_buffer)
            saved_paths = save_buffer_frames(buffered_frames, out_dir, window_idx)
            meta = {
                "window_idx": window_idx,
                "video": str(video_path.name),
                "frame_count_at_inference": frame_count,
                "frames_padded": frames_padded,
                "result": result,
                "saved_frames": [str(p.name) for p in saved_paths],
            }
            manifest.append(meta)
            
            detection_status = "🔴 VIOLENCE" if is_violence else "🟢 NO VIOLENCE"
            logger.info(f"[Window {window_idx:3d}] Frame {frame_count:4d} | {detection_status} | Confidence: {confidence:.4f} | Latency: {latency:.2f}ms (padded {frames_padded} frames)")

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    cap.release()
    
    # Print statistics
    logger.info(f"\n{'='*70}")
    logger.info(f"INFERENCE RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Total detections: {total_detections}")
    logger.info(f"  Violence detected: {violence_count} ({100*violence_count/total_detections:.1f}%)" if total_detections > 0 else "  Violence detected: 0")
    logger.info(f"  No violence: {total_detections - violence_count} ({100*(total_detections-violence_count)/total_detections:.1f}%)" if total_detections > 0 else "  No violence: 0")
    
    if latencies:
        logger.info(f"\n{'='*70}")
        logger.info(f"PERFORMANCE METRICS")
        logger.info(f"{'='*70}")
        logger.info(f"  Avg latency: {sum(latencies)/len(latencies):.2f}ms")
        logger.info(f"  Min latency: {min(latencies):.2f}ms")
        logger.info(f"  Max latency: {max(latencies):.2f}ms")
        logger.info(f"  Inference FPS: {1000/np.mean(latencies):.2f} fps")
        logger.info(f"  Total inference time: {sum(latencies)/1000:.2f}s")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"OUTPUT")
    logger.info(f"{'='*70}")
    logger.info(f"  Frames saved: {len(manifest) * config.num_frames}")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"  Detection frames: {out_dir}")
    logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end inference on a test video")
    parser.add_argument("--input", required=True, help="Input video filename from ai_service/utils/test_data/inputs/videos/")
    parser.add_argument("--model", required=False, help="Path to model checkpoint (optional, auto-detect if not provided)")
    parser.add_argument("--backbone", required=False,
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'mnasnet'],
                        help="STE backbone used during training (optional, auto-detect from model name if not provided)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold for violence detection (default: 0.5)")

    args = parser.parse_args()

    inputs_dir = PROJECT_ROOT / "ai_service" / "utils" / "test_data" / "inputs" / "videos"
    checkpoints_dir = PROJECT_ROOT / "ai_service" / "training" / "two-stage" / "checkpoints"
    outputs_dir = PROJECT_ROOT / "ai_service" / "utils" / "test_data" / "outputs"

    video_path = inputs_dir / args.input
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Try to find best model (prioritize Hockey Fight + V3)
        candidates = [
            checkpoints_dir / "best_model.pt",
            checkpoints_dir / "best_model_hf_v3.pt",   # Hockey Fight + V3 (prioritized)
            checkpoints_dir / "best_model_rwf_v3.pt",  # RWF-2000 + V3
            checkpoints_dir / "best_model_hf.pt",      # Hockey Fight
            checkpoints_dir / "best_model_rwf.pt",     # RWF-2000
        ]
        
        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                logger.info(f"Auto-detected model: {candidate.name}")
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"No model checkpoint found in {checkpoints_dir}\n"
                f"Available models: best_model.pt, best_model_rwf.pt, best_model_rwf_v3.pt, best_model_hf.pt, best_model_hf_v3.pt\n"
                f"Use --model to specify a path"
            )
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Auto-detect backbone from model name if not provided
    if args.backbone:
        backbone = args.backbone
    else:
        model_name = model_path.stem.lower()
        if "v3" in model_name:
            backbone = "mobilenet_v3_small"
            logger.info(f"Auto-detected backbone from model name: {backbone}")
        else:
            backbone = "mobilenet_v2"
            logger.info(f"Auto-detected backbone from model name: {backbone}")

    out_base = outputs_dir
    ensure_dirs(out_base)

    run_on_video(video_path, model_path, out_base, backbone, args.confidence_threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
