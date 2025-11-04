"""
ONNX YOLO Detection Analysis - Test All Videos with Visualization

Analyze frame-by-frame detections on all .mp4 files and visualize with bounding boxes.
- Processes all video files in utils/test_data/
- Draws detection boxes with confidence scores
- Saves annotated video output for each file
- Prints summary statistics

Output:
    - Console: Summary statistics only
    - Videos: {original_name}_detections_onnx.mp4 (annotated with green boxes)

Usage:
    python ai_service/tests/onnx_testing/test_onnx_accuracy.py
"""
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from ai_service.inference.onnx_inference import ONNXYOLOInference


def draw_detections(frame: np.ndarray, detections: List[Dict], frame_num: int, fps: float) -> np.ndarray:
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
    
    test_data_dir = Path('utils/test_data')
    
    if not test_data_dir.exists():
        print(f"❌ Test data directory not found: {test_data_dir}")
        return
    
    # Get all video files
    video_files = sorted(test_data_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"❌ No .mp4 files found in {test_data_dir}")
        return
    
    print(f"\n📹 Processing {len(video_files)} video file(s) with ONNX model...\n")
    
    # Initialize ONNX inference engine
    models_dir = Path('ai_service/models/weights')
    onnx_path = models_dir / 'yolov8n_320.onnx'
    
    if not onnx_path.exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return
    
    onnx_engine = ONNXYOLOInference(
        model_path=str(onnx_path),
        input_size=320,
        conf_threshold=0.25,
        iou_threshold=0.5,
    )
    
    print(f"{'Video File':<40} {'Status':<15} {'Total':<12} {'With Det':<12} {'Avg/Frame':<12}")
    print("-" * 95)
    
    # Process each video file
    total_summary = {}
    
    for video_path in video_files:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Track detections
        frame_count = 0
        all_frames = []
        all_detections_count = []
        
        # Process frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Run ONNX detection (with NMS applied)
            detections = onnx_engine.infer(frame, class_id=0)
            num_detections = len(detections)
            all_detections_count.append(num_detections)
            
            # Store frame for video output
            all_frames.append({
                'frame_num': frame_count,
                'frame_data': frame,
                'detections': detections
            })
            
            frame_count += 1
        
        cap.release()
        
        # Create output video with detections
        output_filename = video_path.stem + '_detections_onnx.mp4'
        output_path = video_path.parent / output_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for detection_frame in all_frames:
            annotated_frame = draw_detections(
                detection_frame['frame_data'],
                detection_frame['detections'],
                detection_frame['frame_num'],
                fps
            )
            writer.write(annotated_frame)
        
        writer.release()
        
        # Calculate statistics
        if all_detections_count:
            frames_with_detections = sum(1 for d in all_detections_count if d > 0)
            total_detections = sum(all_detections_count)
            avg_detections_per_frame = total_detections / len(all_detections_count)
            
            total_summary[video_path.name] = {
                'frames': len(all_detections_count),
                'total_detections': total_detections,
                'avg_per_frame': avg_detections_per_frame,
                'frames_with_detections': frames_with_detections,
            }
            
            # Print summary row
            status = "✅ Done"
            print(f"{video_path.name:<40} {status:<15} {total_detections:<12} {frames_with_detections:<12} {avg_detections_per_frame:<12.2f}")
    
    print("\n✅ All videos processed successfully!")
    print()


if __name__ == '__main__':
    test_violence_detection_analysis()
