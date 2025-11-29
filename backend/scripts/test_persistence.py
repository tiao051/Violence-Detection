"""Test script for Event Persistence Service."""
import sys
import os
import asyncio
import numpy as np
import cv2

import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.infrastructure.firebase.setup import initialize_firebase
from src.infrastructure.storage.event_persistence import get_event_persistence_service
from src.infrastructure.memory import get_frame_buffer

async def main():
    print("1. Initializing Firebase...")
    initialize_firebase()
    
    print("2. Populating FrameBuffer with dummy frames...")
    fb = get_frame_buffer()
    
    # Create 20 dummy frames (red images)
    for i in range(20):
        # Create a 224x224 red image
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:] = (0, 0, 255) # BGR: Red
        
        # Add text to frame to make them distinct
        cv2.putText(frame, f"Frame {i}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        fb.put(
            camera_id="test_cam",
            frame=frame,
            frame_id=f"frame_{i}",
            timestamp=1000 + i,
            frame_seq=i
        )
        
    print("3. Testing save_event()...")
    service = get_event_persistence_service()
    
    detection = {
        "confidence": 0.95,
        "violence": True
    }
    
    event_id = await service.save_event("test_cam", detection)
    
    if event_id:
        print(f"SUCCESS: Event saved with ID: {event_id}")
        print("Check your Firebase Console for the new event and video.")
    else:
        print("FAILURE: Failed to save event.")

if __name__ == "__main__":
    asyncio.run(main())
