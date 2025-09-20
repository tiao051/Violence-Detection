# File: Violence-Detection/test_recognition_pipeline.py
import cv2
import numpy as np
import sys
import importlib.util
import os

# Thêm đường dẫn của project vào Python path để có thể import
sys.path.append('.')

# Hàm import động module từ đường dẫn file
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import pipeline của Người 1 từ đúng vị trí (ai-service)
base_preprocessor = import_from_path(
    "base_preprocessor",
    os.path.join("ai-service", "common", "preprocessing", "base_preprocessor.py")
)
augmentation = import_from_path(
    "augmentation",
    os.path.join("ai-service", "common", "preprocessing", "augmentation.py")
)

# Import module của BẠN từ đúng vị trí
pose_module = import_from_path(
    "pose_module",
    os.path.join("ai-service", "recognition", "pose_module.py")
)

Compose = base_preprocessor.Compose
Resize = augmentation.Resize
PoseRecognitionModule = pose_module.PoseRecognitionModule

print("--- Bắt đầu Kịch bản Test Tích hợp cho Module Recognition ---")

# 1. Giả lập công việc của Người 1: Crop, Mask, và Preprocess
print("[Giả lập Người 1] Bắt đầu...")
original_frame = cv2.imread('ai-service/test-data/test_image.jpg') # Giả sử ảnh test nằm trong test-data
if original_frame is None:
    print("LỖI: không tìm thấy file test-data/test_image.jpg. Hãy chắc chắn file tồn tại.")
    exit()

mock_bbox = [40, 30, 270, 470] # bbox của người bên trái

# Crop
x, y, w, h = mock_bbox
cropped_image = original_frame[y:y+h, x:x+w]

# (Bỏ qua bước mask để test cho nhanh)

resize_transform = Resize("yolo", input_size=(256, 192))

# Gọi transform với ảnh crop và lấy kết quả
preprocessed_data = resize_transform(cropped_image)
print("[Giả lập Người 1] Hoàn tất. Dữ liệu đã sẵn sàng cho Người 2.")

# 2. Khởi tạo và sử dụng Module của BẠN
print("\n[Bắt đầu Người 2]...")
# Đường dẫn model đúng vì script được chạy từ thư mục gốc Violence-Detection
pose_analyzer = PoseRecognitionModule(model_path="models/yolov8n-pose.pt")

# Lấy dữ liệu cần thiết từ dictionary của Người 1
image_for_model = preprocessed_data['image']
metadata_for_postprocessing = preprocessed_data['metadata']

# Tạo một dictionary input mới cho hàm process của bạn
# Trong thực tế, Người 3 sẽ làm việc này, nhưng ở đây ta giả lập
input_for_recognition = {
    "image": image_for_model,
    "metadata": metadata_for_postprocessing
}

keypoints_result = pose_analyzer.process(input_for_recognition)
print("[Kết thúc Người 2] Hoàn tất.")

# 3. In và kiểm tra kết quả
print("\n--- KẾT QUẢ CUỐI CÙNG TỪ MODULE RECOGNITION ---")
print(f"Shape của kết quả: {keypoints_result.shape}") 
print("Tọa độ và Confidence của 17 keypoints (đã rescale):")
print(keypoints_result)
print("\n--- Test hoàn tất! ---")