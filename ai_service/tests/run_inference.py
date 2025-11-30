"""CLI script to run end-to-end inference on a local test video.

Usage:
    python ai_service/tests/run_inference.py --input violence_1.mp4

Defaults:
    - Input folder: `ai_service/utils/test_data/inputs/videos/`
    - Output folder: `ai_service/utils/test_data/outputs/detection/`

The script will:
    - Read the input video, resize frames to 224x224 (model expectation)
    - Feed frames into the ViolenceDetectionModel (no logic changed)
    - When the model returns a detection result (buffer full), save the
      buffered frames and a JSON metadata file containing confidence/violence/etc.

If the video has fewer frames than the model buffer size, the script will
repeat the last frame to fill the buffer and perform one inference.
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

    # Initialize model with backbone and confidence threshold
    config = InferenceConfig(
        model_path=str(model_path),
        backbone=backbone,
        confidence_threshold=confidence_threshold
    )
    model = ViolenceDetectionModel(config)

    frame_count = 0
    window_idx = 0
    manifest = []

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
            # advance window index
            window_idx += 1

    # If video ended and model buffer not full but has frames, pad with last frame and run one inference
    if len(model.frame_buffer) > 0 and len(model.frame_buffer) < model.MAX_BUFFER_SIZE:
        # repeat last frame until buffer full
        last_frame_rgb = model.frame_buffer[-1]
        while len(model.frame_buffer) < model.MAX_BUFFER_SIZE:
            # model.add_frame expects BGR input (it converts to RGB internally), so convert back
            bgr = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)
            model.add_frame(bgr)
        result = model.predict()
        if result is not None:
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

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    cap.release()
    logger.info(f"Completed inference. {len(manifest)} windows saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end inference on a test video")
    parser.add_argument("--input", required=True, help="Input video filename from ai_service/utils/test_data/inputs/videos/")
    parser.add_argument("--model", required=False, help="Path to model checkpoint (optional)")
    parser.add_argument("--backbone", default='mobilenet_v2', 
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'mnasnet'],
                        help="STE backbone used during training (default: mobilenet_v2). If checkpoint has backbone info, it will be auto-corrected.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold for violence detection (default: 0.5)")

    args = parser.parse_args()

    inputs_dir = PROJECT_ROOT / "ai_service" / "utils" / "test_data" / "inputs" / "videos"
    default_model = PROJECT_ROOT / "ai_service" / "training" / "two-stage" / "checkpoints" / "best_model.pt"
    outputs_dir = PROJECT_ROOT / "ai_service" / "utils" / "test_data" / "outputs"

    video_path = inputs_dir / args.input
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    model_path = Path(args.model) if args.model else default_model
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    out_base = outputs_dir
    ensure_dirs(out_base)

    run_on_video(video_path, model_path, out_base, args.backbone, args.confidence_threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
