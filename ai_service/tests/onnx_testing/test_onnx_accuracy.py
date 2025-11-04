"""
ONNX YOLO Detection Analysis - Test All Videos with Visualization

Analyze frame-by-frame detections on all .mp4 files and visualize with bounding boxes.
- Processes all video files in utils/test_inputs/
- Draws detection boxes with confidence scores
- Saves annotated video output to utils/test_outputs/
- Prints summary statistics

Output:
    - Console: Summary statistics only
    - Videos: {original_name}_detections_onnx.mp4 (annotated with green boxes)

Usage:
    python ai_service/tests/onnx_testing/test_onnx_accuracy.py
"""
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from ai_service.inference.onnx_inference import ONNXYOLOInference


def draw_detections(frame: np.ndarray, detections: List, frame_num: int, fps: float) -> np.ndarray:
    """Draw detection boxes and info on frame."""
    annotated = frame.copy()
    
    # Draw background for text info
    cv2.rectangle(annotated, (0, 0), (500, 80), (0, 0, 0), -1)
    
    # Frame info
    time_ms = (frame_num / fps * 1000) if fps > 0 else 0
    time_s = time_ms / 1000
    cv2.putText(annotated, f"Frame: {frame_num} | Time: {time_s:.2f}s", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detections: {len(detections)} persons", 
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw bounding boxes
    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        conf = det['conf']
        
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Clamp coordinates to frame bounds
        h, w = annotated.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Draw box (green)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence label
        label = f"Person {i}: {conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for label
        cv2.rectangle(annotated, 
                     (x1, max(0, y1 - label_size[1] - 4)),
                     (x1 + label_size[0], y1),
                     (0, 255, 0), -1)
        
        # Label text
        cv2.putText(annotated, label, (x1, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated


def test_violence_detection_analysis():
    """Analyze ONNX YOLO detections on all videos with visualization."""
    
    test_inputs_dir = Path('utils/test_inputs')
    test_outputs_dir = Path('utils/test_outputs')
    
    if not test_inputs_dir.exists():
        print(f"Error: Test inputs directory not found: {test_inputs_dir}")
        return
    
    test_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted(test_inputs_dir.glob('*.mp4'))
    if not video_files:
        print(f"Error: No .mp4 files found in {test_inputs_dir}")
        return
    
    # Initialize ONNX inference engine
    models_dir = Path('ai_service/models/weights')
    onnx_path = models_dir / 'yolov8n_320.onnx'
    
    if not onnx_path.exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        return
    
    onnx_engine = ONNXYOLOInference(
        model_path=str(onnx_path),
        input_size=320,
        conf_threshold=0.25,
        iou_threshold=0.5,
    )
    
    print(f"\nProcessing {len(video_files)} video(s)\n")
    print(f"{'Video File':<45} {'Total Det':<12} {'With Det':<12} {'Avg/Frame':<12}")
    print("-" * 85)
    
    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        total_detections = 0
        frames_with_detections = 0
        total_frames = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = video_path.stem + '_detections_onnx.mp4'
        output_path = test_outputs_dir / output_filename
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        # Process frames and write video in one pass
        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            detections = onnx_engine.infer(frame, class_id=0)
            num_detections = len(detections)
            
            total_detections += num_detections
            if num_detections > 0:
                frames_with_detections += 1
            total_frames += 1
            
            # Draw and write frame directly
            annotated_frame = draw_detections(frame, detections, frame_num, fps)
            writer.write(annotated_frame)
            frame_num += 1
        
        cap.release()
        writer.release()
        
        avg_per_frame = total_detections / total_frames if total_frames > 0 else 0
        print(f"{video_path.name:<45} {total_detections:<12} {frames_with_detections:<12} {avg_per_frame:<12.2f}")
    
if __name__ == '__main__':
    test_violence_detection_analysis()
