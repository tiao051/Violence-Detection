"""
ONNX YOLOv8 Inference with proper NMS support.

This module provides optimized ONNX inference for YOLOv8 models with:
- Preprocessing pipeline (YOLOPreprocessor)
- Post-processing with NMS
- Confidence filtering
- Fast CPU inference
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from ai_service.common.preprocessing.augmentation import YOLOPreprocessor


class ONNXYOLOInference:
    """ONNX YOLOv8 inference engine with NMS post-processing."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 320,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_det: int = 50,
        use_cuda: bool = False,
    ):
        """
        Initialize ONNX YOLOv8 inference.

        Args:
            model_path: Path to ONNX model file
            input_size: Input image size (square, e.g., 320, 384, 512)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            max_det: Maximum number of detections to keep
            use_cuda: Whether to use CUDA execution provider
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Setup execution providers
        providers = []
        if use_cuda:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Create ONNX session
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.provider = self.session.get_providers()[0] if self.session.get_providers() else "Unknown"

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Preprocessor
        self.preprocessor = YOLOPreprocessor(input_size=(input_size, input_size))

    def infer(self, frame: np.ndarray, class_id: int = 0) -> List[Dict]:
        """
        Run inference on a frame.

        Args:
            frame: Input frame (BGR, HWC format)
            class_id: Class ID to filter (0 = person in COCO)

        Returns:
            List of detections with keys: bbox (x1,y1,x2,y2), conf, class_id
        """
        # Preprocess
        blob, metadata = self.preprocessor(frame)

        # Run inference
        output = self.session.run(self.output_names, {self.input_name: blob})

        # Parse and filter detections
        detections = self._parse_output(output[0], metadata, class_id)

        # Apply NMS
        detections = self._apply_nms(detections)

        # Sort by confidence and limit
        detections = sorted(detections, key=lambda x: x["conf"], reverse=True)[: self.max_det]

        return detections

    def _parse_output(
        self, output: np.ndarray, metadata: Dict, class_id: int = 0
    ) -> List[Dict]:
        """
        Parse ONNX output to detections.

        ONNX output format: [batch, 84, 8400]
        - 84 channels: [x, y, w, h, conf, class_0, class_1, ..., class_79]
        - 8400 anchor positions (grid)

        Args:
            output: ONNX model output [1, 84, 8400]
            metadata: Preprocessing metadata
            class_id: Class ID to filter

        Returns:
            List of detections
        """
        predictions = output[0]  # Remove batch dimension [84, 8400]

        detections = []

        # Extract center coordinates and dimensions
        x_centers = predictions[0]  # [8400]
        y_centers = predictions[1]  # [8400]
        widths = predictions[2]      # [8400]
        heights = predictions[3]     # [8400]
        confidences = predictions[4] # [8400] - objectness confidence

        # Filter by confidence
        for i in range(len(x_centers)):
            conf = float(confidences[i])

            if conf < self.conf_threshold:
                continue

            # Convert from center format to corner format
            x_center = float(x_centers[i])
            y_center = float(y_centers[i])
            w = float(widths[i])
            h = float(heights[i])

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # Reverse preprocessing (undo letterbox offset and scaling)
            scale = metadata["scale"]
            x_offset, y_offset = metadata["offset"]

            x1 = (x1 - x_offset) / scale
            y1 = (y1 - y_offset) / scale
            x2 = (x2 - x_offset) / scale
            y2 = (y2 - y_offset) / scale

            # Clip to frame bounds
            orig_h, orig_w = metadata["orig_size"][1], metadata["orig_size"][0]
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": class_id,
                }
            )

        return detections

    def _apply_nms(self, detections: List[Dict], iou_threshold: Optional[float] = None) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            detections: List of detections
            iou_threshold: IOU threshold (uses self.iou_threshold if not provided)

        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []

        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x["conf"], reverse=True)

        keep = []
        while sorted_dets:
            # Take detection with highest confidence
            current = sorted_dets.pop(0)
            keep.append(current)

            # Remove detections with high IOU overlap
            remaining = []
            for det in sorted_dets:
                iou = self._calculate_iou(current["bbox"], det["bbox"])
                if iou < iou_threshold:
                    remaining.append(det)
            sorted_dets = remaining

        return keep

    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union between two boxes.

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IOU value [0, 1]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
