"""
realtime_detector.py

Description:
    This module defines the `RealtimeDetector` class, a specialized component for
    real-time object detection and tracking, specifically optimized for identifying
    persons using a YOLOv8 model. It encapsulates the model loading, inference,
    and result formatting logic, providing a clean interface for processing video frames.

Key Features:
    - Loads a specified YOLOv8 model and configures it for the available hardware (GPU/CPU).
    - Performs inference on individual frames to detect and track objects.
    - Filters detections to focus exclusively on the 'person' class.
    - Formats detection results into a standardized JSON-like dictionary, including
      bounding boxes, confidence scores, and tracking IDs.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Default model weights path (keeps everything local, no automatic hub download)
DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent.parent / "models" / "weights" / "yolov8n.pt")

class RealtimeDetector:
    """
    A dedicated class for detecting and tracking objects, specifically persons,
    in real-time video streams using a YOLOv8 model with its built-in tracker.
    """
    def __init__(self, model_path: str | Path | None = None):
        """
        Initializes the RealtimeDetector instance.

        This constructor loads the YOLO model, determines the optimal computation device
        (prioritizing CUDA if available), and identifies the class ID for 'person'
        to streamline the tracking process.

        Args:
            model_path (str | Path | None): File path to YOLOv8 weights.
                                            Defaults to local models/weights/yolov8n.pt.
        """
        # --- Step 1: Device Configuration ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Initializing RealtimeDetector ---")
        print(f"INFO: Computation device selected: {self.device.upper()}")

        # --- Step 2: Model Loading and Initialization ---
        self.model = None
        weights_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at '{weights_path}'. Please provide a valid local path.")

        try:
            print(f"INFO: Loading model from '{weights_path}'...")
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            
            # --- Step 3: Identify Target Class ID ---
            # Automatically find the class ID for 'person' to filter detections.
            self.person_class_id = list(self.model.names.keys())[list(self.model.names.values()).index('person')]
            
            print(f"INFO: Model '{model_path}' loaded successfully on device '{self.device}'.")
            print(f"INFO: Tracking configured for class 'person' (ID: {self.person_class_id}).")
            print("------------------------------------")

        except Exception as e:
            print(f"FATAL: Failed to load YOLO model from '{model_path}'. Aborting. Error: {e}")
            # Ensure the model is None if loading fails.
            self.model = None

    def process_frame(self, frame: np.ndarray, camera_id: str, timestamp: str) -> dict:
        """
        Processes a single video frame to perform person detection and tracking.

        Args:
            frame (np.ndarray): The input video frame in BGR format (from OpenCV).
            camera_id (str): A unique identifier for the camera source.
            timestamp (str): The ISO 8601 formatted timestamp for when the frame was captured.

        Returns:
            dict: A dictionary containing the detection results, camera ID, and timestamp.
                  Returns a formatted dictionary with an empty 'detections' list if the
                  model is not loaded or no persons are detected.
        """
        # Immediately return a formatted empty result if the model isn't available.
        if self.model is None:
            return self._format_output(None, camera_id, timestamp)

        # Perform inference and tracking on the frame.
        # - `persist=True` maintains tracking state across frames.
        # - `classes=[self.person_class_id]` filters detections for persons only.
        # - `conf=0.5` sets the confidence threshold for detections.
        # - `verbose=False` suppresses detailed console output from the YOLO model.
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[self.person_class_id], 
            conf=0.5, 
            verbose=False
        )
        
        # Format the raw results into the standardized output structure.
        output_dict = self._format_output(results[0], camera_id, timestamp)
        return output_dict

    def _format_output(self, result, camera_id: str, timestamp: str) -> dict:
        """
        Packages the raw detection results into a standardized dictionary.

        This private method ensures a consistent output format, whether detections
        are present or not.

        Args:
            result: The raw result object from the YOLOv8 model's track method.
            camera_id (str): The identifier for the camera source.
            timestamp (str): The ISO 8601 timestamp of the frame.

        Returns:
            dict: A dictionary structured for downstream consumption, containing
                  metadata and a list of detection records.
        """
        detections_list = []
        
        # Check if the result object is valid and contains tracking information.
        if result is not None and result.boxes.id is not None:
            # Extract bounding boxes, confidence scores, and tracking IDs.
            # Move data to CPU and convert to NumPy arrays for easier handling.
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()

            # Iterate through each detected person and create a record.
            for i in range(len(track_ids)):
                detections_list.append({
                    "bbox": [int(coord) for coord in boxes[i]],  # [x1, y1, x2, y2]
                    "conf": float(confs[i]),
                    "class": "person",
                    "person_id": int(track_ids[i])
                })

        # Assemble the final dictionary.
        final_output = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "detections": detections_list
        }
        return final_output