# File: tests/test_detector_performance.py
import cv2
import time
import sys
from pathlib import Path
import importlib.util

# --- LOGIC IMPORT TỪ FILE TEST TRƯỚC ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ... (copy toàn bộ block importlib từ file test_realtimedetector.py vào đây) ...
# Kết quả là bạn sẽ có class RealtimeDetector
try:
    module_path = project_root / "ai-service" / "detection" / "realtime_detector.py"
    spec = importlib.util.spec_from_file_location("realtime_detector", module_path)
    rt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt_module)
    RealtimeDetector = rt_module.RealtimeDetector
except ImportError as e:
    print(f"LỖI: Không thể import 'RealtimeDetector'. Lỗi: {e}")
    sys.exit(1)
# ----------------------------------------

# --- PHẦN CẤU HÌNH ---
MODEL_CHOICE = 'yolov8n.pt'
VIDEO_FOR_TEST = 'test-data/val_video_01.avi' # Chọn một video test dài khoảng 10-20s

def main_evaluation():
    print("--- Bắt đầu Đánh giá Hiệu năng (FPS) ---")
    
    # 1. Khởi tạo detector
    detector = RealtimeDetector(model_path=MODEL_CHOICE)
    if detector.model is None: return

    # 2. Mở video test
    cap = cv2.VideoCapture(VIDEO_FOR_TEST)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video test '{VIDEO_FOR_TEST}'")
        return

    total_frames = 0
    total_processing_time = 0

    print(f"Đang xử lý video '{VIDEO_FOR_TEST}'...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        total_frames += 1
        start_time = time.time()
        
        # Chỉ chạy inference, không cần timestamp
        _ = detector.process_frame(frame, "eval_cam", "timestamp")
        
        end_time = time.time()
        total_processing_time += (end_time - start_time)

    print("Xử lý hoàn tất.")

    # 3. Tính toán và In kết quả
    if total_frames > 0:
        avg_latency_ms = (total_processing_time / total_frames) * 1000
        avg_fps = total_frames / total_processing_time
        print("\n--- KẾT QUẢ HIỆU NĂNG ---")
        print(f"Model: {MODEL_CHOICE}")
        print(f"Tổng số frame đã xử lý: {total_frames}")
        print(f"Tổng thời gian xử lý: {total_processing_time:.2f} giây")
        print(f"Độ trễ trung bình: {avg_latency_ms:.2f} ms/frame")
        print(f"Tốc độ trung bình: {avg_fps:.2f} FPS")
        print("-------------------------")
    else:
        print("Không có frame nào được xử lý.")

    cap.release()

if __name__ == '__main__':
    main_evaluation()