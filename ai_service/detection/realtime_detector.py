"""
Real-time person detection and tracking using YOLOv8.

This module provides the RealtimeDetector class for detecting and tracking persons
in video frames using YOLOv8 with built-in tracking capabilities.
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Default model weights path (local, no automatic downloads)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "weights" / "yolov8n.pt"

class RealtimeDetector:
    """
    Real-time object detection and tracking for persons using YOLOv8.
    
    Features:
    - Automatic GPU/CPU device selection
    - Person-class filtering
    - Persistent tracking across frames
    - Standardized output format
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize the detector with YOLOv8 model.

        Args:
            model_path: Path to .pt weights file. Defaults to local models/weights/yolov8n.pt
            
        Raises:
            FileNotFoundError: If weights file does not exist
            Exception: If model loading fails
        """
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing RealtimeDetector ---")
        print(f"Device: {self.device.upper()}")

        # Model loading
        self.model = None
        self.person_class_id = None
        
        weights_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        try:
            print(f"Loading model from: {weights_path}")
            self.model = YOLO(str(weights_path))
            self.model.to(self.device)
            
            # Get person class ID
            self.person_class_id = list(self.model.names.keys())[
                list(self.model.names.values()).index("person")
            ]
            
            print(f"Model loaded successfully. Person class ID: {self.person_class_id}")

        except Exception as e:
            print(f"FATAL: Failed to load YOLO model: {e}")
            self.model = None
            raise

    def process_frame(self, frame: np.ndarray, camera_id: str, timestamp: str) -> dict:
        """
        Detect and track persons in a single frame.

        Args:
            frame: Input frame (BGR, from OpenCV)
            camera_id: Camera identifier
            timestamp: ISO 8601 timestamp

        Returns:
            Dictionary with keys: camera_id, timestamp, detections (list)
        """
        if self.model is None:
            return self._format_output(None, camera_id, timestamp)

        # Run inference with tracking
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.person_class_id],
            conf=0.5,
            verbose=False
        )

        return self._format_output(results[0], camera_id, timestamp)

    def _format_output(self, result, camera_id: str, timestamp: str) -> dict:
        """
        Format detection results into standardized dictionary.

        Args:
            result: YOLOv8 result object or None
            camera_id: Camera identifier
            timestamp: Frame timestamp

        Returns:
            Dict with camera_id, timestamp, and detections list
        """
        detections = []

        if result is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()

            for i in range(len(track_ids)):
                detections.append({
                    "bbox": [int(x) for x in boxes[i]],  # [x1, y1, x2, y2]
                    "conf": float(confs[i]),
                    "class": "person",
                    "person_id": int(track_ids[i])
                })

        return {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "detections": detections
        }