"""
PyTorch YOLOv8 detector for frame analysis and visualization.

This module provides the PyTorchDetector class for:
- Real-time object detection using YOLOv8 PyTorch
- Frame-by-frame analysis
- Bounding box visualization
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from ai_service.ai_utils.visualization import draw_detections


class PyTorchDetector:
    """PyTorch YOLOv8 detector for frame-level analysis."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        class_id: int = 0,
    ):
        """
        Initialize PyTorch YOLOv8 detector.

        Args:
            model_path: Path to YOLOv8 .pt weights file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            class_id: Class ID to filter (0 = person in COCO)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_id = class_id

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in a frame (end-to-end pipeline).

        Args:
            frame: Input frame (BGR, HWC format)

        Returns:
            List of detections with keys: bbox (x1,y1,x2,y2), conf, class_id
        """
        # Run inference
        results = self._run_inference(frame)

        # Parse and filter results
        detections = self._parse_results(results)

        return detections

    def _run_inference(self, frame: np.ndarray):
        """
        Run YOLO inference on frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            YOLO results object
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.class_id],
            verbose=False,
        )
        return results

    def _parse_results(self, results) -> List[Dict]:
        """
        Parse YOLO results into detections list.

        Args:
            results: YOLO results object

        Returns:
            List of detections with bbox, conf, class_id
        """
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for i in range(len(confs)):
                x1, y1, x2, y2 = boxes[i]
                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(confs[i]),
                        "class_id": self.class_id,
                    }
                )

        return detections

    def draw_detections(
        self, frame: np.ndarray, detections: List[Dict], frame_num: int, fps: float
    ) -> np.ndarray:
        """
        Draw detection boxes on frame (cyan for PyTorch).

        Args:
            frame: Input frame
            detections: List of detections from detect()
            frame_num: Frame number for display
            fps: Video FPS for display

        Returns:
            Annotated frame with boxes and labels
        """
        # Cyan color for PyTorch
        return draw_detections(frame, detections, frame_num, fps, box_color=(255, 255, 0))

