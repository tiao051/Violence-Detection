"""
ONNX YOLO Detection Analysis - Test All Videos with Visualization

Analyze frame-by-frame detections on all .mp4 files and visualize with bounding boxes.
- Processes all video files in utils/test_data/
- Draws detection boxes with confidence scores
- Displays detection count per frame
- Saves annotated video output for each file
- Prints detailed frame-by-frame analysis

Output:
    - Console: Detailed per-file and per-frame analysis
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
    
    print(f"\n📹 Found {len(video_files)} video file(s) to process:")
    for vf in video_files:
        print(f"  - {vf.name}")
    
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
    
    print("\n" + "="*120)
    print("ONNX YOLO DETECTION ANALYSIS - ALL VIDEOS".center(120))
    print("="*120)
    
    print(f"\nModel Info:")
    print(f"  Model: {onnx_path.name}")
    print(f"  Provider: {onnx_engine.provider}")
    print(f"  Input Size: 320x320")
    print(f"  Conf Threshold: 0.25")
    print(f"  IOU Threshold: 0.5 (NMS)")
    
    # Process each video file
    total_summary = {}
    
    for video_path in video_files:
        print(f"\n{'='*120}")
        print(f"Processing: {video_path.name}".center(120))
        print(f"{'='*120}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        print(f"\nVideo Info:")
        print(f"  File: {video_path.name}")
        print(f"  Resolution: {w}x{h}")
        print(f"  Total Frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {duration_seconds:.2f}s")
        print()
        
        # Track detections by second
        detections_by_second: Dict[int, List] = {}
        frame_count = 0
        all_frames = []  # Store all frames for video output
        
        # First pass: collect detections
        print("Processing frames...")
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Calculate current second
            current_second = int((frame_count / fps)) if fps > 0 else 0
            
            # Run ONNX detection (with NMS applied)
            detections = onnx_engine.infer(frame, class_id=0)
            
            # Extract detections info
            num_detections = len(detections)
            confidences = [det['conf'] for det in detections]
            
            # Store detections for this second
            if current_second not in detections_by_second:
                detections_by_second[current_second] = {
                    'detections': [],
                }
            
            # Add this frame's detections
            detections_by_second[current_second]['detections'].append({
                'frame': frame_count,
                'count': num_detections,
                'confidences': confidences,
            })
            
            # Store frame for video output
            all_frames.append({
                'frame_num': frame_count,
                'frame_data': frame,
                'detections': detections
            })
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Processed {frame_count} frames")
        
        # Create output video with detections
        output_filename = video_path.stem + '_detections_onnx.mp4'
        output_path = video_path.parent / output_filename
        print(f"\n🎥 Creating annotated video...")
        
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
        print(f"✅ Video saved: {output_filename}")
        
        # Print detailed analysis by second
        print("\n" + "-"*120)
        print("DETAILED FRAME-BY-FRAME ANALYSIS (ONNX with NMS)".center(120))
        print("-" * 120 + "\n")
        
        for second in sorted(detections_by_second.keys()):
            second_data = detections_by_second[second]
            frames_in_second = second_data['detections']
            
            print(f"\n⏱️  SECOND {second}:")
            print("-" * 120)
            
            # Analyze all frames in this second
            detection_counts = [f['count'] for f in frames_in_second]
            min_detections = min(detection_counts)
            max_detections = max(detection_counts)
            avg_detections = sum(detection_counts) / len(detection_counts)
            
            print(f"  Frames in second: {len(frames_in_second)}")
            print(f"  Detection range: {min_detections}-{max_detections} persons")
            print(f"  Average: {avg_detections:.1f} persons per frame")
            print()
            
            # Show frame details
            for frame_data in frames_in_second:
                frame_num = frame_data['frame']
                count = frame_data['count']
                confidences = frame_data['confidences']
                
                # Format confidences
                if confidences:
                    conf_str = ", ".join([f"{c:.3f}" for c in sorted(confidences, reverse=True)])
                    conf_range = f"[{min(confidences):.3f}, {max(confidences):.3f}]"
                else:
                    conf_str = "No detections"
                    conf_range = "-"
                
                time_ms = (frame_num / fps * 1000) if fps > 0 else 0
                
                print(f"    Frame {frame_num:>4d} ({time_ms:>6.0f}ms) | {count:>2d} persons | Conf: {conf_range} | {conf_str}")
        
        # Summary statistics for this video
        print("\n" + "-"*120)
        print("SUMMARY STATISTICS".center(120))
        print("-" * 120 + "\n")
        
        all_detections_count = []
        for second in detections_by_second.values():
            for frame_data in second['detections']:
                all_detections_count.append(frame_data['count'])
        
        if all_detections_count:
            frames_with_detections = sum(1 for d in all_detections_count if d > 0)
            frames_without_detections = sum(1 for d in all_detections_count if d == 0)
            total_detections = sum(all_detections_count)
            avg_detections_per_frame = total_detections / len(all_detections_count)
            
            print(f"Total Frames Processed: {len(all_detections_count)}")
            print(f"Frames with Detections: {frames_with_detections}")
            print(f"Frames without Detections: {frames_without_detections}")
            print(f"Min Detections per Frame: {min(all_detections_count)}")
            print(f"Max Detections per Frame: {max(all_detections_count)}")
            print(f"Average Detections per Frame: {avg_detections_per_frame:.2f}")
            print(f"Total Detections (sum): {total_detections}")
            
            total_summary[video_path.name] = {
                'frames': len(all_detections_count),
                'total_detections': total_detections,
                'avg_per_frame': avg_detections_per_frame,
                'frames_with_detections': frames_with_detections,
            }
    
    # Print global summary
    print("\n" + "="*120)
    print("GLOBAL SUMMARY - ALL VIDEOS".center(120))
    print("="*120 + "\n")
    
    if total_summary:
        print(f"{'Video File':<40} {'Total Frames':<15} {'Total Detections':<20} {'Avg/Frame':<15} {'With Detections':<15}")
        print("-" * 120)
        
        for video_name, stats in total_summary.items():
            print(
                f"{video_name:<40} {stats['frames']:<15} {stats['total_detections']:<20} "
                f"{stats['avg_per_frame']:<15.2f} {stats['frames_with_detections']:<15}"
            )
        
        print()
    
    print("✅ All videos processed successfully!")
    print("\n" + "="*120 + "\n")


if __name__ == '__main__':
    test_violence_detection_analysis()
