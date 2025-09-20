# File: Violence-Detection/ai-service/recognition/pose_module.py
import numpy as np
from ultralytics import YOLO

class PoseRecognitionModule:
    def __init__(self, model_path: str = "models/yolov8n-pose.pt"):
        """
        Khởi tạo và tải model Pose Estimation.
        Đường dẫn model là tương đối so với thư mục gốc của project.
        """
        try:
            self.model = YOLO(model_path)
            print("INFO: Pose Recognition model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load pose model. Error: {e}")
            self.model = None

    def process(self, preprocessed_data: dict) -> np.ndarray:
        """
        Hàm chính, thực hiện Inference và Post-processing.
        """
        if self.model is None:
            return np.zeros(51)

        input_tensor = preprocessed_data["image"]
        metadata = preprocessed_data["metadata"]

        # === BƯỚC INFERENCE ===
        raw_results = self.model.predict(input_tensor, verbose=False)

        # === BƯỚC POST-PROCESSING ===
        keypoints_raw = self._decode_results(raw_results)
        if keypoints_raw.sum() == 0: # Nếu không detect được gì
             return np.zeros(51)

        keypoints_rescaled = self._rescale_coordinates(keypoints_raw, metadata)
        
        keypoints_normalized = self._normalize_pose(keypoints_rescaled)
        
        final_feature_vector = self._filter_and_format(keypoints_normalized)

        return final_feature_vector
    
    def _decode_results(self, raw_results) -> np.ndarray:
        """Bóc tách keypoints thô từ output của model."""
        if raw_results[0].keypoints is not None and len(raw_results[0].keypoints.data) > 0:
            return raw_results[0].keypoints.data[0].cpu().numpy()
        return np.zeros((17, 3))

    def _rescale_coordinates(self, keypoints: np.ndarray, metadata: dict) -> np.ndarray:
        """Sử dụng metadata để phục hồi tọa độ về không gian ảnh crop."""
        scale = metadata["scale"]
        x_offset, y_offset = metadata["offset"]
        
        rescaled_keypoints = keypoints.copy()
        rescaled_keypoints[:, 0] -= x_offset  # Tọa độ x
        rescaled_keypoints[:, 1] -= y_offset  # Tọa độ y
        
        if scale > 1e-8:
            rescaled_keypoints[:, :2] /= scale
        
        return rescaled_keypoints

    def _normalize_pose(self, keypoints: np.ndarray) -> np.ndarray:
        """
        1. Chuẩn hóa vị trí (lấy hông làm gốc).
        2. Chuẩn hóa kích thước (lấy chiều dài thân làm đơn vị).
        """
        # 1. Locate anchor points
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # 2. Normalize Location (Translation)
        # Only execute when both hips are detected
        if np.all(left_hip[:2] > 0) and np.all(right_hip[:2] > 0):
            # Make the center of the hip to be the new origin (0,0)
            hip_center = (left_hip[:2] + right_hip[:2]) / 2
            
            # Make a copy to avoid modifying original keypoints
            normalized_keypoints = keypoints.copy()
            
            # Minus hip center to all keypoints
            normalized_keypoints[:, :2] -= hip_center 
        else:
            # If hips are not detected, skip normalization
            return keypoints
        
        # 3. Normalize Scale (Scaling)
        # Only execute when both shoulders are detected
        if np.all(left_shoulder[:2] > 0) and np.all(right_shoulder[:2] > 0):
            # Calculate shoulder's center position
            shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
            
            # Translate shoulder center to be relative to new origin
            translated_shoulder_center = shoulder_center - hip_center
            
            # Calculate torso length (distance from hip center to shoulder center) to be a "unit"
            torso_distance = np.linalg.norm(translated_shoulder_center)
            
            # Avoid division by zero
            if torso_distance > 1e-6:
                # Divide all keypoints by torso distance
                # This makes the torso length = 1 unit as above
                normalized_keypoints[:, :2] /= torso_distance
                
            return normalized_keypoints
        else:
            return normalized_keypoints
    
    def _filter_and_format(self, keypoints: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Loại bỏ các keypoints có confidence thấp và làm phẳng thành vector 1D.
        
        Args:
            keypoints (np.array): Mảng (17, 3) với keypoints đã được chuẩn hoá.
            confidence_threshold (float): Ngưỡng giữ lại một keypoint.
            
        Returns:
            np.array: Một vector 1D có shape (,) chứa thông tin về pose.
        """
        
        filtered_keypoints = keypoints.copy()
        
        # 1. Filter using Confidence
        # Find all rows (keypoints) with confidence below the threshold
        low_confidence_indices = filtered_keypoints[:, 2] < confidence_threshold
        
        # Assign (x, y) = 0 for all low_confidence_indices.
        filtered_keypoints[low_confidence_indices, :2] = 0
        
        # 2. Flatten
        # Turn a (17, 3) matrix into a horizontal vector (51,)
        # For example: [x1, y1, c1, x2, y2, c2, ..., x17, y17, c17]
        feature_vector = filtered_keypoints.flatten()
        
        return feature_vector