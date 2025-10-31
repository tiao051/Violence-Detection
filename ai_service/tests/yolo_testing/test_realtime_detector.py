"""
Live integration test for RealtimeDetector with webcam.
Captures frames, runs inference, and visualizes detections in real-time.

Usage:
    python -m pytest ai_service/tests/test_realtime_detector.py
    # or
    python ai_service/tests/test_realtime_detector.py
"""
import cv2
from datetime import datetime, timezone

from ai_service.detection.realtime_detector import RealtimeDetector
from ai_service.config import DEFAULT_MODEL


def draw_detections(frame, detections_data: dict):
    """Draw bboxes and labels on frame."""
    if not detections_data or 'detections' not in detections_data:
        return frame
    
    for det in detections_data['detections']:
        x1, y1, x2, y2 = det['bbox']
        person_id = det.get('person_id', 'N/A')
        conf = det['conf']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID: {person_id} | {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame


def test_webcam_live():
    """Run real-time detection test with webcam."""
    detector = RealtimeDetector(model_path=DEFAULT_MODEL)
    if detector.model is None:
        print("ERROR: Detector init failed")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print("Running... Press 'q' to exit\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            output = detector.process_frame(frame, camera_id="webcam", timestamp=timestamp)
            
            if output and output.get('detections'):
                print(f"Detections: {len(output['detections'])}")

            annotated = draw_detections(frame, output)
            cv2.imshow("Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_webcam_live()