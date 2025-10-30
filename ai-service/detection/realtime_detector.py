# File: VIOLENCE-DETECTION/ai-service/detection/realtime_detector.py

import cv2
import torch
import numpy as np
from ultralytics import YOLO

class RealtimeDetector:
    """
    Một class chuyên dụng để phát hiện và theo dõi đối tượng (cụ thể là người)
    sử dụng YOLOv8 và tracker tích hợp sẵn.
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Khởi tạo detector.
        - Tải model YOLO.
        - Xác định thiết bị (GPU/CPU).
        - Lấy ID của class 'person'.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.person_class_id = list(self.model.names.keys())[list(self.model.names.values()).index('person')]
            print(f"INFO: RealtimeDetector loaded model '{model_path}' on device '{self.device}'.")
            print(f"INFO: Tracking for class 'person' with ID: {self.person_class_id}")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model. Error: {e}")
            self.model = None

    def process_frame(self, frame: np.ndarray, camera_id: str, timestamp: str) -> dict:
        """
        Xử lý một frame video duy nhất.
        """
        if self.model is None:
            return self._format_output(None, camera_id, timestamp)

        results = self.model.track(
            frame, 
            persist=True, 
            classes=[self.person_class_id], 
            conf=0.5, 
            verbose=False
        )
        
        output_dict = self._format_output(results[0], camera_id, timestamp)
        return output_dict

    def _format_output(self, result, camera_id: str, timestamp: str) -> dict:
        """
        Đóng gói kết quả thành dictionary output chuẩn.
        """
        detections_list = []
        if result is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()

            for i in range(len(track_ids)):
                detections_list.append({
                    "bbox": [int(coord) for coord in boxes[i]], 
                    "conf": float(confs[i]),
                    "class": "person",
                    "person_id": int(track_ids[i])
                })

        final_output = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "detections": detections_list
        }
        return final_output