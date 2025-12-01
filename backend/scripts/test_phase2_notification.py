"""
Simple script to test Phase 2: Push Notification Trigger
Simulates a violence detection event to trigger notification flow
"""
import asyncio
import sys
import os
import logging
import numpy as np

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.firebase.setup import initialize_firebase
from src.infrastructure.storage.event_persistence import get_event_persistence_service
from src.infrastructure.memory import get_frame_buffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_notification_trigger():
    """Test the complete notification trigger flow"""
    
    print("\n" + "="*60)
    print("PHASE 2 TEST: Violence Detection Notification")
    print("="*60 + "\n")
    
    # 1. Initialize Firebase
    print("1. Initializing Firebase...")
    try:
        initialize_firebase()
        print("   ✅ Firebase initialized\n")
    except Exception as e:
        print(f"   ❌ Firebase init failed: {e}\n")
        return
    
    # 2. Create fake frames (simulating camera capture)
    print("2. Creating fake video frames...")
    frame_buffer = get_frame_buffer()
    
    # Generate 30 fake frames (5 seconds at 6 FPS)
    camera_id = "cam1"
    fake_frames = []
    for i in range(30):
        # Create a simple colored frame (blue background with frame number)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [100, 100, 200]  # Blue-ish color
        fake_frames.append(frame)
    
    # Add frames to buffer using put() method
    import time
    for i, frame in enumerate(fake_frames):
        frame_buffer.put(
            camera_id=camera_id,
            frame=frame,
            frame_id=f"test_frame_{i}",
            timestamp=time.time(),
            frame_seq=i
        )
    
    print(f"   ✅ Added {len(fake_frames)} frames to buffer\n")
    
    # 3. Trigger violence detection (fake)
    print("3. Simulating violence detection...")
    detection_result = {
        "camera_id": camera_id,
        "confidence": 0.87,
        "timestamp": "2025-12-01T12:48:00Z",
        "type": "violence"
    }
    print(f"   📹 Camera: {camera_id}")
    print(f"   ⚠️  Confidence: {detection_result['confidence']}\n")
    
    # 4. Save event (this will trigger notification)
    print("4. Saving event and triggering notification...")
    print("   (This will upload video, save to Firestore, and send notification)\n")
    
    persistence_service = get_event_persistence_service()
    event_id = await persistence_service.save_event(camera_id, detection_result)
    
    if event_id:
        print(f"   ✅ Event saved: {event_id}")
        print(f"   📱 Notification should be sent to your phone!\n")
    else:
        print("   ❌ Failed to save event\n")
        return
    
    # 5. Check results
    print("="*60)
    print("TEST COMPLETE!")
    print("="*60 + "\n")
    print("📱 CHECK YOUR PHONE FOR NOTIFICATION:")
    print("   Title: ⚠️ Phát Hiện Bạo Lực")
    print("   Body:  Camera Front Gate phát hiện...")
    print("\n💡 If notification appears → Phase 2 SUCCESS! 🎉")
    print("   If not, check backend logs for errors.\n")


if __name__ == "__main__":
    asyncio.run(test_notification_trigger())
