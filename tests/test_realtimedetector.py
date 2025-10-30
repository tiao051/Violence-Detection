# File: tests/test_realtimedetector.py (NẰM TRONG THƯ MỤC tests/)

import cv2
from datetime import datetime, timezone
import sys
from pathlib import Path
import importlib.util  # <-- THÊM THƯ VIỆN NÀY

# --- SỬA LOGIC IMPORT (CHO TÊN CÓ DẤU GẠCH NGANG) ---
# Thêm thư mục gốc (lùi ra 2 cấp) vào Python path
project_root = Path(__file__).resolve().parent.parent 
sys.path.append(str(project_root))

try:
    # 1. Xác định đường dẫn đầy đủ đến file module
    module_path = project_root / "ai-service" / "detection" / "realtime_detector.py"
    
    # 2. Tạo 'spec' (thông số kỹ thuật) từ đường dẫn file
    spec = importlib.util.spec_from_file_location(
        "realtime_detector",  # Tên module (ảo) mà Python sẽ gọi
        module_path
    )
    
    # 3. Tạo module từ spec
    rt_module = importlib.util.module_from_spec(spec)
    
    # 4. "Thực thi" (tải) module
    spec.loader.exec_module(rt_module)
    
    # 5. Lấy class 'RealtimeDetector' từ module vừa tải
    RealtimeDetector = rt_module.RealtimeDetector

except ImportError as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể import 'RealtimeDetector'.")
    print(f"Đảm bảo file 'ai-services/detection/realtime_detector.py' tồn tại.")
    print(f"Lỗi chi tiết: {e}")
    sys.exit(1)
# --- KẾT THÚC SỬA LOGIC ---


# --- PHẦN BÊN DƯỚI GIỮ NGUYÊN ---

def draw_detections(frame, detections_data):
    """Hàm tiện ích để vẽ kết quả lên frame."""
    if not detections_data or 'detections' not in detections_data:
        return frame
        
    for det in detections_data['detections']:
        x1, y1, x2, y2 = det['bbox']
        person_id = det['person_id']
        conf = det['conf']
        
        # Vẽ box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tạo nhãn
        label = f"ID: {person_id} | {conf:.2f}"
        
        # Vẽ nền cho nhãn
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1 - 10), (0, 255, 0), -1)
        
        # Viết chữ
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Chữ đen
        
    return frame

def main_test():
    # Khởi tạo detector của bạn
    detector = RealtimeDetector(model_path='yolov8n.pt')
    if detector.model is None:
        print("Lỗi nghiêm trọng: Không thể tải model. Thoát.")
        return

    # Mở webcam
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.")
        return

    print("\nBắt đầu Test... Nhấn 'q' trên cửa sổ webcam để thoát.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Lỗi: Mất kết nối camera.")
            break
            
        # Lấy timestamp chuẩn ISO
        now_utc = datetime.now(timezone.utc)
        timestamp_iso = now_utc.isoformat().replace('+00:00', 'Z')

        # === GỌI MODULE CỦA BẠN ===
        output_data = detector.process_frame(frame, camera_id="webcam_01", timestamp=timestamp_iso)
        
        # In kết quả JSON ra terminal (chỉ in nếu có phát hiện)
        if output_data['detections']:
            print(output_data)

        # Vẽ kết quả lên frame
        annotated_frame = draw_detections(frame, output_data)
        
        # Hiển thị frame
        cv2.imshow("Realtime Detector Test (Tach file chuan)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_test()