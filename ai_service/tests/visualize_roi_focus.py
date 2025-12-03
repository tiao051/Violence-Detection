"""
Visualize which parts of ROI the model focuses on.

Shows activation heatmaps overlaid on ROI frames.

Usage:
    python visualize_roi_focus.py --input roi_folder --output output_folder
"""

import sys
from pathlib import Path
import argparse
import logging
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.inference_model import ViolenceDetectionModel, InferenceConfig
from visualization.gradcam_visualizer import compute_input_gradient, overlay_heatmap_with_bbox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Visualize model focus on ROI frames."""
    parser = argparse.ArgumentParser(description="Visualize what model focuses on in ROI")
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', default='roi_focus_output', help='Output folder')
    parser.add_argument('--model', help='Model path (auto-detect if not provided)')
    parser.add_argument('--backbone', help='Backbone (auto-detect if not provided)')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames (0=all)')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Auto-detect model if not provided
    from ai_service.tests.extract_roi_only import find_best_model, detect_backbone
    
    if not args.model:
        args.model = find_best_model()
    if not args.backbone:
        args.backbone = detect_backbone(args.model)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Backbone: {args.backbone}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    config = InferenceConfig(
        model_path=args.model,
        backbone=args.backbone,
        device=args.device,
        confidence_threshold=0.5
    )
    model = ViolenceDetectionModel(config)
    model.set_roi_tracking(True)
    
    # Load video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.input}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {total_frames} frames")
    
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
            
            # Get activation from ROI directly (simplified)
            # Just use ROI as input instead of motion frames
            if roi is None:
                continue
            
            # Use input gradient to see what model is sensitive to
            try:
                gradient = compute_input_gradient(model, roi, device=model.device)
                cam = gradient
            except Exception as e:
                logger.warning(f"Gradient computation failed: {e}, skipping")
                continue
            
            # Prepare ROI for visualization
            if roi.dtype != np.uint8:
                roi_viz = np.clip(roi * 255, 0, 255).astype(np.uint8)
            else:
                roi_viz = roi
            
            # Ensure 3 channels
            if len(roi_viz.shape) == 2:
                roi_viz = cv2.cvtColor(roi_viz, cv2.COLOR_GRAY2BGR)
            elif roi_viz.shape[2] == 1:
                roi_viz = cv2.cvtColor(roi_viz, cv2.COLOR_GRAY2BGR)
            
            # Overlay heatmap WITH bounding boxes (easier to understand)
            roi_with_cam = overlay_heatmap_with_bbox(roi_viz, cam, threshold=0.1, alpha=0.8)
            
            # Add prediction text
            pred_text = "VIOLENCE" if result['violence'] else "NO VIOLENCE"
            color = (0, 0, 255) if result['violence'] else (0, 255, 0)
            cv2.rectangle(roi_with_cam, (5, 5), (219, 40), (0, 0, 0), -1)
            cv2.putText(
                roi_with_cam,
                f"{pred_text}: {result['confidence']:.1%}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            # Save
            output_path = output_dir / f"focus_{window_idx:04d}.png"
            cv2.imwrite(str(output_path), roi_with_cam)
            
            window_idx += 1
            
            if window_idx % 10 == 0:
                logger.info(f"Visualized {window_idx} windows")
            
            # Slide buffer
            model.frame_buffer = model.frame_buffer[10:]
            model.motion_buffer = model.motion_buffer[10:]
            model.roi_extracted_buffer = model.roi_extracted_buffer[10:]
    
    cap.release()
    logger.info(f"Done! Visualized {window_idx} windows with activation heatmaps")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
