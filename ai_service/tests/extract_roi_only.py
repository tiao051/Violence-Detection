"""
Extract and save only ROI from video - simplified visualization.

This script processes video frames and extracts the Region Of Interest (ROI) from each frame
using the SME (Spatial Motion Extractor) module. The extracted ROI is what the model actually
uses for violence classification.

Usage:
    python extract_roi_only.py --input video.mp4 --output roi_frames
    python extract_roi_only.py --input video.mp4 --output roi_frames --max-frames 150
    python extract_roi_only.py --input video.mp4 --output roi_frames --device cuda

Arguments:
    --input <video_file>         Input video filename (required)
                                 Available: violence_1.mp4, violence_2.mp4, etc.
    
    --output <folder>            Output directory for ROI frames (default: 'extracted_roi')
    
    --model <path>               Path to model checkpoint (optional)
                                 Auto-detects best available if not provided
    
    --backbone <backbone>        Backbone type (optional)
                                 Auto-detects from model name if not provided
    
    --max-frames <int>           Max frames to process (default: 0 = all)
    
    --device <device>            Device to use: 'cpu' or 'cuda' (default: 'cpu')

Output:
    Each frame saved as: roi_0000.png, roi_0001.png, etc.
    - Size: 224×224 pixels (model input size)
    - Overlay: Prediction text + confidence %
    - Red text: VIOLENCE prediction
    - Green text: NO VIOLENCE prediction

Example:
    # Auto-detect everything
    python extract_roi_only.py --input violence_2.mp4 --output roi_output
    
    # With max frames limit
    python extract_roi_only.py --input violence_2.mp4 --output roi_output --max-frames 150
    
    # Use CUDA device
    python extract_roi_only.py --input violence_2.mp4 --output roi_output --device cuda
"""

import sys
from pathlib import Path
import argparse
import logging
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.inference_model import ViolenceDetectionModel, InferenceConfig
from visualization.simple_roi_saver import SimpleROISaver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_best_model():
    """Auto-detect best model in checkpoints directory."""
    checkpoints_dir = Path(__file__).parent.parent / "training" / "two-stage" / "checkpoints"
    for model_name in ["best_model_hf_v3.pt", "best_model_rwf_v3.pt", "best_model_hf.pt", "best_model_rwf.pt"]:
        model_path = checkpoints_dir / model_name
        if model_path.exists():
            logger.info(f"Auto-detected model: {model_name}")
            return str(model_path)
    raise FileNotFoundError(f"No model in {checkpoints_dir}")


def detect_backbone(model_path: str) -> str:
    """Auto-detect backbone from model filename."""
    return "mobilenet_v3_small" if "v3" in Path(model_path).stem else "mobilenet_v2"


def main():
    """Main entry point: extract ROI from video and save as PNG frames."""
    parser = argparse.ArgumentParser(description="Extract ROI only from video")
    parser.add_argument('--input', required=True, help='Input video')
    parser.add_argument('--output', default='extracted_roi', help='Output folder')
    parser.add_argument('--model', help='Model path (auto-detect if not provided)')
    parser.add_argument('--backbone', help='Backbone (auto-detect if not provided)')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames (0=all)')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Auto-detect model if not provided
    if not args.model:
        args.model = find_best_model()
    if not args.backbone:
        args.backbone = detect_backbone(args.model)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Input: {args.input}")
    
    # Load video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Cannot open video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {total_frames} frames")
    
    # Initialize model
    config = InferenceConfig(
        model_path=args.model,
        backbone=args.backbone,
        device=args.device,
        confidence_threshold=0.5
    )
    model = ViolenceDetectionModel(config)
    model.set_roi_tracking(True)
    
    # Initialize ROI saver
    roi_saver = SimpleROISaver(args.output)
    
    frame_count = 0
    window_idx = 0
    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    
    logger.info(f"Processing (max {max_frames} frames)...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (224, 224))
        model.add_frame(frame)
        frame_count += 1
        
        result = model.predict()
        if result:
            roi = model.get_last_roi_extracted()
            pred_text = "VIOLENCE" if result['violence'] else "NO VIOLENCE"
            roi_saver.save_roi_frame(
                roi=roi,
                window_idx=window_idx,
                prediction=pred_text,
                confidence=result['confidence']
            )
            window_idx += 1
            
            if window_idx % 20 == 0:
                logger.info(f"Saved {window_idx} windows")
            
            model.frame_buffer = model.frame_buffer[10:]
            model.motion_buffer = model.motion_buffer[10:]
            model.roi_extracted_buffer = model.roi_extracted_buffer[10:]
    
    cap.release()
    logger.info(f"Done! Processed {frame_count} frames, saved {window_idx} ROI windows")
    logger.info(roi_saver.get_summary())


if __name__ == "__main__":
    main()
