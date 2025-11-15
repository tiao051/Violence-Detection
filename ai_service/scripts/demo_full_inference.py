"""
Full Violence Detection Pipeline - End-to-End Demo

Demonstration of complete pipeline: Raw Frames → Violence Classification
Run: python ai_service/scripts/demo_full_inference.py
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_service.remonet.sme import SMEExtractor, SMEPreprocessor
from ai_service.remonet.ste import STEExtractor
from ai_service.remonet.gte import GTEExtractor


def simulate_video_stream(num_frames: int = 60, frame_size: tuple = (224, 224)):
    """Generate random video frames"""
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(50, 200, (frame_size[0], frame_size[1], 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def create_high_motion_frames(num_frames: int = 60):
    """Create high-motion frames (violence-like pattern)"""
    frames = []
    for i in range(num_frames):
        if i % 2 == 0:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = np.full((224, 224, 3), 100, dtype=np.uint8)
        
        y = (i * 5) % 180
        x = (i * 8 + (i % 2) * 50) % 180
        size = 40
        frame[y:y+size, x:x+size] = 255
        frames.append(frame)
    return frames


def create_fighting_frames(num_frames: int = 60):
    """Create simulated fighting frames"""
    frames = []
    for i in range(num_frames):
        frame = np.full((224, 224, 3), 80, dtype=np.uint8)
        
        p1_y = 50 + int(20 * np.sin(i * 0.3))
        p1_x = 40
        p2_y = 50 + int(20 * np.cos(i * 0.3))
        p2_x = 140
        
        frame[p1_y:p1_y+60, p1_x:p1_x+40] = 200
        frame[p2_y:p2_y+60, p2_x:p2_x+40] = 180
        
        interaction = frame[40:110, 80:144]
        noise = np.random.randint(50, 255, interaction.shape, dtype=np.uint8)
        frame[40:110, 80:144] = (interaction.astype(int) + noise.astype(int)) // 2
        
        frames.append(frame)
    return frames


def detect_violence(frames: list, device: str = 'cpu') -> dict:
    """
    Full pipeline: Frames → Violence Detection
    
    Returns: dict with violence prediction and performance metrics
    """
    if len(frames) < 30:
        raise ValueError(f"Need at least 30 frames, got {len(frames)}")
    
    sme = SMEExtractor(kernel_size=3, iteration=2)
    preprocessor = SMEPreprocessor(target_size=(224, 224))
    
    motion_frames = []
    for i in range(0, min(30, len(frames) - 1)):
        frame_t = preprocessor.preprocess(frames[i])
        frame_t1 = preprocessor.preprocess(frames[i + 1])
        roi, mask, diff, elapsed_ms = sme.process(frame_t, frame_t1)
        motion_frames.append(roi)
    
    motion_frames = np.array(motion_frames)
    
    ste = STEExtractor(device=device, training_mode=False)
    ste_output = ste.process(motion_frames, camera_id="test_camera", timestamp=1.0)
    
    print(f"STE Features: {ste_output.features.shape}")
    
    gte = GTEExtractor(device=device, training_mode=False)
    gte_output = gte.process(ste_output.features, camera_id="test_camera", timestamp=1.0)
    
    threshold = 0.5
    is_violent = gte_output.violence_prob > threshold
    
    if is_violent:
        print(f"VIOLENCE DETECTED: {gte_output.violence_prob*100:.2f}%")
    else:
        print(f"NO VIOLENCE: {gte_output.no_violence_prob*100:.2f}%")
    
    total_time = ste_output.latency_ms + gte_output.latency_ms
    print(f"Time: {total_time:.2f}ms ({1000/total_time:.1f} FPS)\n")
    
    return {
        'is_violent': is_violent,
        'violence_prob': gte_output.violence_prob,
        'no_violence_prob': gte_output.no_violence_prob,
        'ste_latency_ms': ste_output.latency_ms,
        'gte_latency_ms': gte_output.latency_ms,
        'total_latency_ms': total_time,
    }


def demo_1_random():
    print("DEMO 1: Random Frames")
    frames = simulate_video_stream(num_frames=60)
    return detect_violence(frames)


def demo_2_high_variation():
    print("DEMO 2: High Variation Frames")
    frames = []
    for i in range(60):
        base = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        variation = np.random.randint(0, 100, (224, 224, 3), dtype=np.uint8)
        frame = np.clip(base.astype(int) + variation, 0, 255).astype(np.uint8)
        frames.append(frame)
    return detect_violence(frames)


def demo_3_low_variation():
    print("DEMO 3: Low Variation Frames")
    frames = []
    for i in range(60):
        base = np.full((224, 224, 3), 128, dtype=np.uint8)
        noise = np.random.randint(-5, 5, (224, 224, 3), dtype=np.int16)
        frame = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return detect_violence(frames)


def demo_4_high_motion():
    print("DEMO 4: High Motion Frames")
    frames = create_high_motion_frames(num_frames=60)
    return detect_violence(frames)


def demo_5_fighting():
    print("DEMO 5: Simulated Fighting")
    frames = create_fighting_frames(num_frames=60)
    return detect_violence(frames)


if __name__ == "__main__":
    result_1 = demo_1_random()
    result_2 = demo_2_high_variation()
    result_3 = demo_3_low_variation()
    result_4 = demo_4_high_motion()
    result_5 = demo_5_fighting()
    
    print("\nSUMMARY")
    print(f"Demo 1 (Random):      {'VIOLENCE' if result_1['is_violent'] else 'NO VIOLENCE'} ({result_1['violence_prob']:.2%})")
    print(f"Demo 2 (High Var):    {'VIOLENCE' if result_2['is_violent'] else 'NO VIOLENCE'} ({result_2['violence_prob']:.2%})")
    print(f"Demo 3 (Low Var):     {'VIOLENCE' if result_3['is_violent'] else 'NO VIOLENCE'} ({result_3['violence_prob']:.2%})")
    print(f"Demo 4 (High Motion): {'VIOLENCE' if result_4['is_violent'] else 'NO VIOLENCE'} ({result_4['violence_prob']:.2%})")
    print(f"Demo 5 (Fighting):    {'VIOLENCE' if result_5['is_violent'] else 'NO VIOLENCE'} ({result_5['violence_prob']:.2%})")
    