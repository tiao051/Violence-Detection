"""
PyTorch YOLOv8 Detection Analysis - Test All Videos with Visualization

Analyze frame-by-frame detections on all .mp4 files using standard PyTorch YOLOv8.
- Processes all video files in utils/test_inputs/
- Draws detection boxes with confidence scores
- Saves annotated video output to utils/test_outputs/
- Prints summary statistics

Output:
    - Console: Summary statistics only
    - Videos: {original_name}_detections_pytorch.mp4 (annotated with cyan boxes)

Usage:
    python ai_service/tests/detector_testing/test_pytorch_accuracy.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from ultralytics import YOLO

from ai_service.config import DEFAULT_MODEL


def draw_detections(frame: np.ndarray, results, frame_num: int, fps: float) -> np.ndarray:
    """Draw detection boxes and info on frame."""
    annotated = frame.copy()
    
    # Draw background for text info
    cv2.rectangle(annotated, (0, 0), (500, 80), (0, 0, 0), -1)
    
    # Frame info
    time_ms = (frame_num / fps * 1000) if fps > 0 else 0
    time_s = time_ms / 1000
    cv2.putText(annotated, f"Frame: {frame_num} | Time: {time_s:.2f}s", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Count detections
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    cv2.putText(annotated, f"Detections: {num_detections} persons", 
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw bounding boxes
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes, 1):
            # Get box coordinates
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf)
            
            # Clamp coordinates to frame bounds
            h, w = annotated.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Draw box (cyan for PyTorch)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw confidence label
            label = f"Person {i}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for label
            cv2.rectangle(annotated, 
                         (x1, max(0, y1 - label_size[1] - 4)),
                         (x1 + label_size[0], y1),
                         (255, 255, 0), -1)
            
            # Label text
            cv2.putText(annotated, label, (x1, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated


def test_pytorch_detection_analysis():
    """Analyze PyTorch YOLOv8 detections on all videos with visualization."""
    
    test_inputs_dir = Path('utils/test_inputs')
    test_outputs_dir = Path('utils/test_outputs')
    
    # Create outputs directory if it doesn't exist
    test_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = sorted(test_inputs_dir.glob('*.mp4'))
    
    if not video_files:
        return
    
    # Initialize PyTorch YOLO model
    model = YOLO(str(DEFAULT_MODEL))
    
    print(f"{'Video File':<40} {'Status':<15} {'Total':<12} {'With Det':<12} {'Avg/Frame':<12}")
    
    # Process each video file
    for video_path in video_files:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video with detections
        output_filename = video_path.stem + '_detections_pytorch.mp4'
        output_path = test_outputs_dir / output_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        # Track detections
        frame_count = 0
        total_detections = 0
        frames_with_detections = 0
        
        # Process frames: detect → draw → write (streaming, no buffering)
        cap.seek(0)  # Reset to beginning
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Run PyTorch detection
            results = model.predict(
                source=frame,
                conf=0.25,
                iou=0.5,
                classes=0,  # Person only
                verbose=False
            )
            
            # Count detections
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            total_detections += num_detections
            if num_detections > 0:
                frames_with_detections += 1
            
            # Draw and write frame
            annotated_frame = draw_detections(frame, results, frame_count, fps)
            writer.write(annotated_frame)
            
            frame_count += 1
        
        writer.release()
        
        # Print summary row
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
        print(f"{video_path.name:<40} {'Done':<15} {total_detections:<12} {frames_with_detections:<12} {avg_detections_per_frame:<12.2f}")

if __name__ == '__main__':
    test_pytorch_detection_analysis()
