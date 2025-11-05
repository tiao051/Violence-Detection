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
    python ai_service/tests/detector/pytorch_accuracy.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
from ai_service.config import DEFAULT_MODEL
from ai_service.detection import PyTorchDetector


def test_pytorch_detection_analysis():
    """Analyze PyTorch YOLOv8 detections on all videos with visualization."""
    
    test_inputs_dir = Path('utils/test_data/inputs/videos')
    test_outputs_dir = Path('utils/test_data/outputs/videos')
    
    # Create outputs directory if it doesn't exist
    test_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = sorted(test_inputs_dir.glob('*.mp4'))
    
    if not video_files:
        return
    
    # Initialize detector
    detector = PyTorchDetector(
        model_path=DEFAULT_MODEL,
        conf_threshold=0.25,
        iou_threshold=0.5,
        class_id=0  # Person only
    )
    
    print(f"{'Video File':<40} {'Status':<15} {'Total':<12} {'Frame w/Det':<12} {'Avg/Frame':<12}")
    
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Detect persons
            detections = detector.detect(frame)
            
            # Count detections
            num_detections = len(detections)
            total_detections += num_detections
            if num_detections > 0:
                frames_with_detections += 1
            
            # Draw and write frame
            annotated_frame = detector.draw_detections(frame, detections, frame_count, fps)
            writer.write(annotated_frame)
            
            frame_count += 1
        
        writer.release()
        cap.release()
        
        # Print summary row
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
        print(f"{video_path.name:<40} {'Done':<15} {total_detections:<12} {frames_with_detections:<12} {avg_detections_per_frame:<12.2f}")

if __name__ == '__main__':
    test_pytorch_detection_analysis()

