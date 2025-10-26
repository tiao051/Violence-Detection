#!/usr/bin/env python3
"""
Test RTSP camera streams
Simple script to verify all cameras are working
"""

import cv2
import sys


def test_camera(rtsp_url: str, camera_name: str, duration: int = 5):
    """Test a single RTSP camera stream"""
    print(f"\nTesting {camera_name}...")
    print(f"URL: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Failed to open {camera_name}")
        return False

    print(f"Successfully connected to {camera_name}")

    frame_count = 0
    while frame_count < duration * 25:  # 25 fps * duration
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame")
            break
        
        frame_count += 1
        
        # Show frame
        cv2.imshow(camera_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"   Frames read: {frame_count}")
    return True


def main():
    cameras = [
        ('Camera 1 - Entrance', 'rtsp://localhost:8554/camera1'),
        ('Camera 2 - Parking', 'rtsp://localhost:8554/camera2'),
        ('Camera 3 - Hallway', 'rtsp://localhost:8554/camera3'),
        ('Camera 4 - Storage', 'rtsp://localhost:8554/camera4'),
    ]
    
    print("RTSP Camera Test")
    print("Testing 4 cameras (5 seconds each)")
    print("Press 'q' to skip to next camera\n")
    
    results = []
    for name, url in cameras:
        success = test_camera(url, name, duration=5)
        results.append((name, success))
    
    for name, success in results:
        status = "success" if success else "failed"
        print(f"{status} {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\n{successful}/4 cameras working")
    
    return 0 if successful == 4 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
