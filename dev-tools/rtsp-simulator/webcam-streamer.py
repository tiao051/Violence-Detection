#!/usr/bin/env python3
"""
Webcam to RTSP Streamer for Windows
Since Docker Desktop on Windows cannot access host webcam devices,
this script runs on the host and streams webcam to the RTSP server.

Usage:
    python webcam-streamer.py [--camera 0] [--fps 25] [--resolution 1280x720]
"""

import cv2
import subprocess
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Stream webcam to RTSP server")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate (default: 25)")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Resolution WxH (default: 1280x720)")
    parser.add_argument("--rtsp-url", type=str, default="rtsp://localhost:8554/camera3", 
                        help="RTSP destination URL (default: rtsp://localhost:8554/camera3)")
    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Use WIDTHxHEIGHT (e.g., 1280x720)")
        return 1

    # Open webcam
    print(f"Opening webcam {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # Use DirectShow on Windows
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        print("Available camera indices: Try 0, 1, 2...")
        return 1

    # Set resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Webcam opened successfully")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps}")
    print(f"  Streaming to: {args.rtsp_url}")
    print()

    # FFmpeg command to stream to RTSP
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{actual_width}x{actual_height}',
        '-framerate', str(args.fps),
        '-i', 'pipe:0',
        '-vf', f"drawtext=text='CAM3 - Webcam Live - %{{localtime}}':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=blue@0.7",
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '2000k',
        '-f', 'rtsp',
        args.rtsp_url
    ]

    print("Starting FFmpeg process...")
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and add it to PATH")
        print("Download from: https://ffmpeg.org/download.html")
        cap.release()
        return 1

    print("Streaming started! Press Ctrl+C to stop.")
    print()

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam")
                break

            # Write frame to FFmpeg stdin
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Error: FFmpeg process terminated unexpectedly")
                break

            frame_count += 1

            # Print stats every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frames sent: {frame_count} | FPS: {current_fps:.2f}", end='\r')

    except KeyboardInterrupt:
        print("\n\nStopping stream...")

    finally:
        # Cleanup
        cap.release()
        if ffmpeg_process.poll() is None:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=5)
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nStream stopped.")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
