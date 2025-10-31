"""
USB Camera Manager
Detects USB camera and runs FFmpeg to stream to RTSP server
Must run BEFORE docker-compose up

Handles graceful shutdown: CTRL+C to stop
"""

import subprocess
import sys
import time
import threading
import signal
import atexit

# Global variable to track if FFmpeg process is running
ffmpeg_process = None

def cleanup_ffmpeg():
    """Cleanup: kill FFmpeg process on exit"""
    global ffmpeg_process
    if ffmpeg_process:
        try:
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=3)
        except:
            ffmpeg_process.kill()

# Register cleanup on exit
atexit.register(cleanup_ffmpeg)

def signal_handler(sig, frame):
    """Handle CTRL+C gracefully"""
    print("\n\nShutting down...")
    cleanup_ffmpeg()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Global variable to track if FFmpeg process is running
ffmpeg_process = None

def check_usb_camera():
    """Check if USB camera 'Web Camera' exists"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        output = result.stderr + result.stdout
        return '"Web Camera" (video)' in output
    except Exception as e:
        print(f"Error checking USB camera: {e}")
        return False

def run_ffmpeg_usb_stream():
    """Run FFmpeg to capture USB camera and stream to RTSP server"""
    global ffmpeg_process
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'dshow',
        '-i', 'video=Web Camera',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-g', '30',
        '-b:v', '2000k',
        '-an',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        'rtsp://host.docker.internal:8554/usb-cam' 
    ]
    
    try:
        print("Starting FFmpeg for USB camera...")
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=sys.stdout,   # show logs in console
            stderr=sys.stderr,   # show logs in console
            text=True
        )
        print("FFmpeg started successfully")
        
        # Keep process running
        ffmpeg_process.wait()
        
    except Exception as e:
        print(f"Error running FFmpeg: {e}")

def main():
    global ffmpeg_process
    
    print("\n=== USB Camera Manager ===")
    
    # Check for USB camera
    has_usb = check_usb_camera()
    
    if has_usb:
        print("Starting FFmpeg stream to rtsp://127.0.0.1:8554/usb-cam")
        
        # Run FFmpeg in background thread
        ffmpeg_thread = threading.Thread(target=run_ffmpeg_usb_stream, daemon=True)
        ffmpeg_thread.start()
        
        # Give FFmpeg time to start
        time.sleep(2)
        
        print("\nYou can now run: docker-compose up -d")
        print("USB camera will be available at: rtsp://127.0.0.1:8554/usb-cam")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down FFmpeg...")
            if ffmpeg_process:
                ffmpeg_process.terminate()
                ffmpeg_process.wait()
            sys.exit(0)
    else:
        print("No USB camera detected")

if __name__ == "__main__":
    main()