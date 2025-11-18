"""
Visualization utilities for detection results.

Provides common functions for drawing detection boxes on frames.
"""

from typing import Dict, List

import cv2
import numpy as np


def draw_detections(frame: np.ndarray, detections: List[Dict], frame_num: int, fps: float, box_color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw detection boxes on frame with frame info."""
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, 0), (500, 80), (0, 0, 0), -1)

    time_s = (frame_num / fps) if fps > 0 else 0
    cv2.putText(annotated, f"Frame: {frame_num} | Time: {time_s:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    cv2.putText(annotated, f"Detections: {len(detections)} persons", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        h, w = annotated.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        label = f"Person {i}: {det['conf']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated, (x1, max(0, y1 - label_size[1] - 4)), (x1 + label_size[0], y1), box_color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated
