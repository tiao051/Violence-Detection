"""
Performance benchmark for RealtimeDetector on test videos.
Measures FPS, latency, and throughput on violence detection test dataset.

Usage:
    python -m pytest ai_service/tests/yolo_testing/test_detector_performance.py -v -s
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import cv2
import time

from ai_service.detection.realtime_detector import RealtimeDetector
from ai_service.config import DEFAULT_MODEL



TEST_VIDEOS_DIR = Path('utils/test_inputs')
TEST_VIDEOS = [
    TEST_VIDEOS_DIR / 'non_violence_1.mp4',
    TEST_VIDEOS_DIR / 'violence_1.mp4',
    TEST_VIDEOS_DIR / 'violence_2.mp4',
    TEST_VIDEOS_DIR / 'violence_3.mp4',
]


def test_performance_benchmark():
    """Benchmark detector on all test videos."""
    detector = RealtimeDetector(model_path=DEFAULT_MODEL)
    assert detector.model is not None, "Detector initialization failed"

    print(f"\nBenchmarking {DEFAULT_MODEL} on {len(TEST_VIDEOS)} test videos\n")

    for video_path in TEST_VIDEOS:
        if not video_path.exists():
            print(f"SKIP: {video_path} not found")
            continue

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Cannot open video '{video_path}'"

        total_frames = 0
        total_time = 0.0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Resize to 640x640 before inference to reduce CPU load
                frame_resized = cv2.resize(frame, (640, 640))
                
                start = time.time()
                detector.process_frame(frame_resized, video_path.stem, "")
                total_time += time.time() - start
                total_frames += 1

        finally:
            cap.release()

        # Assertions
        assert total_frames > 0, f"No frames processed from {video_path}"
        assert total_time > 0, f"No processing time for {video_path}"
        
        avg_latency = (total_time / total_frames) * 1000
        avg_fps = total_frames / total_time
        
        print(f"{video_path.name}:")
        print(f"  Frames:    {total_frames}")
        print(f"  Time:      {total_time:.2f}s")
        print(f"  Latency:   {avg_latency:.2f}ms/frame")
        print(f"  FPS:       {avg_fps:.2f}\n")
        
        # Basic sanity checks
        assert avg_fps > 0, f"FPS must be positive for {video_path}"
        assert avg_latency > 0, f"Latency must be positive for {video_path}"


if __name__ == '__main__':
    test_performance_benchmark()