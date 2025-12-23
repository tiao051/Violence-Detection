"""
Test script for Credibility Engine
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.credibility_engine import get_credibility_engine

def test_credibility_engine():
    print("=" * 60)
    print("Camera Credibility Engine - Test")
    print("=" * 60)
    
    # Get engine instance
    engine = get_credibility_engine()
    
    print(f"\nLoaded {len(engine.camera_scores)} cameras")
    print(f"Loaded {len(engine.camera_clusters)} clusters")
    print(f"Loaded {len(engine.false_alarm_patterns)} false alarm patterns")
    
    # Test scenarios
    print("\n" + "=" * 60)
    print("Test Scenarios:")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Scenario 1: Noisy camera (cam1) with low confidence",
            "camera_id": "cam1",
            "raw_confidence": 0.75,
            "context": {"hour": 23, "confidence": 0.75, "duration": 2.5}
        },
        {
            "name": "Scenario 2: Reliable camera (cam2) with high confidence", 
            "camera_id": "cam2",
            "raw_confidence": 0.85,
            "context": {"hour": 14, "confidence": 0.85, "duration": 28.0}
        },
        {
            "name": "Scenario 3: Noisy camera (cam1) - night + short duration (pattern match)",
            "camera_id": "cam1",
            "raw_confidence": 0.60,
            "context": {"hour": 2, "confidence": 0.45, "duration": 1.5}
        },
        {
            "name": "Scenario 4: Unknown camera (fallback)",
            "camera_id": "cam_unknown",
            "raw_confidence": 0.70,
            "context": None
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 60)
        
        result = engine.adjust_confidence(
            scenario["camera_id"],
            scenario["raw_confidence"],
            scenario.get("context")
        )
        
        print(f"  Camera: {scenario['camera_id']}")
        print(f"  Raw Confidence: {result['raw_confidence']}")
        print(f"  Camera Credibility: {result['camera_credibility']} ({result['camera_tier']})")
        print(f"  Adjusted Confidence: {result['adjusted_confidence']} (Δ {result['confidence_delta']:+.2f})")
        
        if result['pattern_matched']:
            print(f"  ⚠️ False Alarm Pattern Matched! (penalty: -{result['pattern_penalty']:.2f})")
        
        print(f"  → Action: {result['action']} (Priority: {result['priority']})")
        print(f"  → {result['camera_info']}")
    
    # Camera stats
    print("\n" + "=" * 60)
    print("Camera Statistics:")
    print("=" * 60)
    
    for cam_data in engine.get_all_cameras():
        print(f"\n📹 {cam_data['camera_name']} ({cam_data['camera_id']})")
        print(f"   Score: {cam_data['credibility_score']} | Tier: {cam_data['credibility_tier']} | Cluster: {cam_data['cluster']}")
        print(f"   TP Rate: {cam_data['metrics']['true_positive_rate']*100:.0f}% | FP Rate: {cam_data['metrics']['false_positive_rate']*100:.0f}%")
        print(f"   {cam_data['recommendation']}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_credibility_engine()
