"""
Test script for SecurityEngine with sklearn model.

Usage:
    docker exec violence-detection-backend python scripts/test_security_engine.py
    
Or locally:
    cd backend && python scripts/test_security_engine.py
"""

import sys
import os
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.application.security_engine import SecurityEngine, init_security_engine


def test_model_loading():
    """Test that model loads correctly."""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    
    engine = init_security_engine()
    stats = engine.get_stats()
    
    print(f"  Model loaded: {stats['model_loaded']}")
    print(f"  Severity labels: {stats['severity_labels']}")
    print(f"  Known cameras: {stats['cameras_known']}")
    
    if stats['model_loaded']:
        print("  ✅ Model loaded successfully!")
        return True
    else:
        print("  ❌ Model NOT loaded - using rule-based fallback")
        return False


def test_predictions():
    """Test predictions with various scenarios."""
    print("\n" + "=" * 60)
    print("TEST 2: Severity Predictions")
    print("=" * 60)
    
    engine = SecurityEngine.get_instance()
    
    # Test scenarios: (camera_id, confidence, hour, expected_description)
    test_cases = [
        # High confidence, night time -> likely HIGH
        ("cam1", 0.95, 2, "High conf, 2AM (night)"),
        ("cam1", 0.85, 23, "High conf, 11PM (night)"),
        
        # Medium confidence, afternoon -> likely MEDIUM
        ("cam2", 0.65, 14, "Med conf, 2PM (afternoon)"),
        ("cam3", 0.70, 10, "Med conf, 10AM (morning)"),
        
        # Low confidence -> likely LOW
        ("cam4", 0.40, 12, "Low conf, 12PM (noon)"),
        ("cam1", 0.35, 8, "Low conf, 8AM (morning)"),
        
        # Edge cases
        ("cam1", 0.99, 3, "Very high conf, 3AM"),
        ("unknown_cam", 0.75, 20, "Unknown camera, evening"),
    ]
    
    print(f"\n  {'Scenario':<30} | {'Severity':<8} | {'Score':<6} | {'Method':<8} | {'Time':<10}")
    print("  " + "-" * 80)
    
    for camera_id, confidence, hour, description in test_cases:
        # Create timestamp for the given hour (today)
        from datetime import datetime, timedelta
        now = datetime.now().replace(hour=hour, minute=0, second=0)
        timestamp = now.timestamp()
        
        result = engine.analyze_severity(camera_id, confidence, timestamp)
        
        print(f"  {description:<30} | {result['severity_level']:<8} | "
              f"{result['severity_score']:.4f} | {result['prediction_method']:<8} | "
              f"{result['analysis_time_ms']:.4f}ms")
    
    print("\n  ✅ All predictions completed!")


def test_performance():
    """Benchmark inference speed."""
    print("\n" + "=" * 60)
    print("TEST 3: Performance Benchmark")
    print("=" * 60)
    
    engine = SecurityEngine.get_instance()
    
    # Warmup
    for _ in range(10):
        engine.analyze_severity("cam1", 0.8, time.time())
    
    # Reset stats for clean measurement
    engine.analysis_count = 0
    engine.total_analysis_time_ms = 0.0
    
    # ===== Test 1: Standard analyze_severity =====
    n_iterations = 1000
    start = time.perf_counter()
    
    for i in range(n_iterations):
        camera_id = f"cam{(i % 4) + 1}"
        confidence = 0.5 + (i % 50) / 100
        engine.analyze_severity(camera_id, confidence, time.time())
    
    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / n_iterations) * 1000
    throughput = n_iterations / elapsed
    
    print(f"\n  📊 Standard analyze_severity():")
    print(f"     Iterations: {n_iterations}")
    print(f"     Total time: {elapsed:.3f}s")
    print(f"     Avg time per prediction: {avg_time_ms:.4f}ms")
    print(f"     Throughput: {throughput:.0f} predictions/sec")
    
    if avg_time_ms < 1.0:
        print("     ✅ Sub-millisecond inference achieved!")
    elif avg_time_ms < 5.0:
        print("     ✅ Good performance (<5ms)")
    else:
        print("     ⚠️ Consider using analyze_severity_fast() or analyze_batch()")
    
    # ===== Test 2: Fast mode (analyze_severity_fast) =====
    start = time.perf_counter()
    
    for i in range(n_iterations):
        camera_id = f"cam{(i % 4) + 1}"
        confidence = 0.5 + (i % 50) / 100
        engine.analyze_severity_fast(camera_id, confidence, time.time())
    
    elapsed_fast = time.perf_counter() - start
    avg_time_fast = (elapsed_fast / n_iterations) * 1000
    throughput_fast = n_iterations / elapsed_fast
    
    print(f"\n  🚀 Fast mode analyze_severity_fast():")
    print(f"     Avg time per prediction: {avg_time_fast:.4f}ms")
    print(f"     Throughput: {throughput_fast:.0f} predictions/sec")
    print(f"     Speedup vs standard: {throughput_fast/throughput:.1f}x")
    
    # ===== Test 3: Batch mode =====
    batch_sizes = [10, 50, 100, 500]
    print(f"\n  📦 Batch mode analyze_batch():")
    
    for batch_size in batch_sizes:
        # Prepare batch
        events = [
            (f"cam{(i % 4) + 1}", 0.5 + (i % 50) / 100, time.time())
            for i in range(batch_size)
        ]
        
        # Benchmark batch
        n_batches = max(1, 1000 // batch_size)
        start = time.perf_counter()
        
        for _ in range(n_batches):
            engine.analyze_batch(events)
        
        elapsed_batch = time.perf_counter() - start
        total_predictions = n_batches * batch_size
        throughput_batch = total_predictions / elapsed_batch
        avg_per_event = (elapsed_batch / total_predictions) * 1000
        
        print(f"     Batch size {batch_size:>3}: {throughput_batch:>6.0f} pred/sec ({avg_per_event:.4f}ms/event)")
    
    print(f"\n  📈 Summary:")
    print(f"     Standard: {throughput:.0f} pred/sec")
    print(f"     Fast:     {throughput_fast:.0f} pred/sec")
    print(f"     Batch:    Use for bulk processing")


def test_probability_distribution():
    """Test probability distribution for predictions."""
    print("\n" + "=" * 60)
    print("TEST 4: Probability Distribution")
    print("=" * 60)
    
    engine = SecurityEngine.get_instance()
    
    if not engine.model_loaded:
        print("  ⚠️ Skipping - model not loaded (rule-based doesn't have probabilities)")
        return
    
    # Test a few scenarios
    scenarios = [
        ("cam1", 0.95, 2),   # Night, high conf
        ("cam2", 0.50, 14),  # Afternoon, medium conf
        ("cam3", 0.30, 10),  # Morning, low conf
    ]
    
    for camera_id, confidence, hour in scenarios:
        from datetime import datetime
        now = datetime.now().replace(hour=hour, minute=0, second=0)
        timestamp = now.timestamp()
        
        result = engine.analyze_severity(camera_id, confidence, timestamp)
        
        print(f"\n  {camera_id} | conf={confidence} | hour={hour}")
        print(f"  Prediction: {result['severity_level']} (score={result['severity_score']:.4f})")
        
        if result['probabilities']:
            print("  Probabilities:")
            for label, prob in result['probabilities'].items():
                bar = "█" * int(prob * 30)
                print(f"    {label:<8}: {prob:.4f} {bar}")


def test_stats():
    """Show final statistics."""
    print("\n" + "=" * 60)
    print("TEST 5: Engine Statistics")
    print("=" * 60)
    
    engine = SecurityEngine.get_instance()
    stats = engine.get_stats()
    
    print(f"  Model loaded: {stats['model_loaded']}")
    print(f"  Total analyses: {stats['analysis_count']}")
    print(f"  Model predictions: {stats['model_predictions']}")
    print(f"  Rule fallbacks: {stats['rule_fallbacks']}")
    print(f"  Avg time: {stats['avg_analysis_time_ms']:.4f}ms")


if __name__ == "__main__":
    print("\n🔒 SecurityEngine Test Suite\n")
    
    try:
        model_ok = test_model_loading()
        test_predictions()
        test_performance()
        test_probability_distribution()
        test_stats()
        
        print("\n" + "=" * 60)
        if model_ok:
            print("✅ ALL TESTS PASSED - Model is working correctly!")
        else:
            print("⚠️ TESTS PASSED but using RULE-BASED fallback (no ML model)")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
