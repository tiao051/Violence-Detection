"""
test_realtimedetector.py

Description:
    This script serves as a live integration test for the `RealtimeDetector` class.
    It captures video from a local webcam, passes each frame to the detector for
    inference, and visualizes the results in real-time. The primary purpose is to
    verify that the detector, model, and data flow are functioning correctly in a
    simulated live environment.

Usage:
    python tests/test_realtimedetector.py

Key Functions:
    - Initializes the `RealtimeDetector`.
    - Accesses and streams video from the default webcam (index 0).
    - For each frame, it calls the detector's `process_frame` method.
    - Prints the JSON-formatted detection output to the console.
    - Renders bounding boxes, tracking IDs, and confidence scores on the video feed.
    - Provides a visual confirmation of the model's tracking performance.
"""
import cv2
from datetime import datetime, timezone
import sys
from pathlib import Path
import importlib.util

# --- Dynamic Module Import ---
# This section ensures that the `RealtimeDetector` module can be imported correctly,
# even though its parent directory 'ai-service' contains a hyphen, which is not a
# valid character for standard Python package names.

# 1. Define the project root by navigating two levels up from this script's location.
#    (tests/ -> root)
project_root = Path(__file__).resolve().parent.parent
# 2. Add the project root to the system path to make top-level packages discoverable.
sys.path.append(str(project_root))

try:
    # 3. Construct the full, absolute path to the target module.
    module_path = project_root / "ai-service" / "detection" / "realtime_detector.py"
    
    # 4. Create a module specification from the file path. This tells Python how to load it.
    #    "realtime_detector" is the name we'll use to refer to the module in this script.
    spec = importlib.util.spec_from_file_location("realtime_detector", module_path)
    
    # 5. Create an empty module object based on the specification.
    rt_module = importlib.util.module_from_spec(spec)
    
    # 6. Execute the module's code to populate the module object.
    spec.loader.exec_module(rt_module)
    
    # 7. Extract the `RealtimeDetector` class from the now-loaded module.
    RealtimeDetector = rt_module.RealtimeDetector
    print("INFO: `RealtimeDetector` class imported successfully.")

except (ImportError, FileNotFoundError) as e:
    print(f"FATAL: Could not import 'RealtimeDetector'.")
    print(f"Please ensure the file 'ai-service/detection/realtime_detector.py' exists.")
    print(f"Detailed Error: {e}")
    sys.exit(1)
# --- End of Dynamic Import Logic ---


def draw_detections(frame, detections_data: dict):
    """
    A utility function to render detection results onto a video frame.

    It draws bounding boxes, labels with tracking IDs, and confidence scores
    for each person detected in the frame.

    Args:
        frame: The original video frame (NumPy array) to draw on.
        detections_data (dict): The standardized output dictionary from the detector,
                                containing a list of detections.

    Returns:
        The annotated frame with all visualizations.
    """
    # Return the original frame if there's no valid detection data.
    if not detections_data or 'detections' not in detections_data:
        return frame
        
    for det in detections_data['detections']:
        x1, y1, x2, y2 = det['bbox']
        person_id = det.get('person_id', 'N/A') # Use .get for safety
        conf = det['conf']
        
        # Define colors and fonts
        box_color = (0, 255, 0)  # Green for the box
        text_color = (0, 0, 0)   # Black for the text
        bg_color = (0, 255, 0)   # Green for the label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Draw the bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        
        # Create the label text.
        label = f"ID: {person_id} | {conf:.2f}"
        
        # Calculate text size to create a background rectangle.
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw the filled rectangle as a background for the text.
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), bg_color, -1)
        
        # Put the label text on the background.
        cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, text_color, thickness)
        
    return frame

def main_test():
    """
    The main function to run the real-time detection test.
    """
    print("\n--- Starting Real-time Detector Test ---")
    
    # --- Step 1: Initialize the Detector ---
    # Use a known pretrained model for this test.
    detector = RealtimeDetector(model_path='yolov8n.pt')
    if detector.model is None:
        print("FATAL: Detector initialization failed. Check model path and dependencies. Exiting.")
        return

    # --- Step 2: Open Webcam Feed ---
    # `0` is typically the default built-in webcam.
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Please check if it is connected and not in use.")
        return

    print("\nINFO: Webcam opened successfully. Streaming video...")
    print("Press 'q' on the video window to exit the test.")

    # --- Step 3: Main Processing Loop ---
    while True:
        # Read a frame from the webcam.
        success, frame = cap.read()
        if not success:
            print("ERROR: Lost connection to the camera or video stream ended.")
            break
            
        # --- Timestamp Generation ---
        # Generate a UTC timestamp in ISO 8601 format for each frame.
        now_utc = datetime.now(timezone.utc)
        timestamp_iso = now_utc.isoformat().replace('+00:00', 'Z')

        # --- Inference ---
        # Call the detector's processing method to get results.
        output_data = detector.process_frame(frame, camera_id="webcam_test_01", timestamp=timestamp_iso)
        
        # --- Output and Visualization ---
        # Print the structured output to the console only if detections are found.
        if output_data and output_data['detections']:
            print(output_data)

        # Draw the detection results on the frame.
        annotated_frame = draw_detections(frame, output_data)
        
        # Display the annotated frame in a window.
        cv2.imshow("Realtime Detector Test", annotated_frame)

        # Exit condition: break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' key pressed. Shutting down...")
            break

    # --- Step 4: Cleanup ---
    # Release the camera and destroy all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()
    print("Test finished. Resources have been released.")

if __name__ == '__main__':
    main_test()