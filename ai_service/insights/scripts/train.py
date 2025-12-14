"""
Pre-train the analytics model.
Run this BEFORE docker compose up to ensure fast analytics loading.

Usage:
    python ai_service/insights/train_model.py
"""

import sys
import os

# Add ai_service to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insights import InsightsModel

def main():
    print("=" * 60)
    print("Pre-training Analytics Model")
    print("=" * 60)
    
    # Paths
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(data_dir, 'data')
    model_path = os.path.join(data_dir, 'trained_model.pkl')
    csv_path = os.path.join(data_dir, 'analytics_events.csv')
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        print("Run generate_analytics_dataset.py first!")
        sys.exit(1)
    
    # Train and save
    print(f"\nTraining with 20,000 mock events...")
    model = InsightsModel()
    model.fit_from_mock(n_events=20000, days=90)
    
    print(f"\nSaving to: {model_path}")
    model.save(model_path)
    
    # Verify
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n✓ Model saved successfully! ({size_mb:.1f} MB)")
        print("\nYou can now run: docker compose up -d")
    else:
        print("\n✗ Failed to save model!")
        sys.exit(1)

if __name__ == "__main__":
    main()
