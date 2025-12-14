"""
Generate Analytics Dataset

Creates a CSV file with mixed violence/nonviolence events for analytics.
The format matches event_persistence.py structure.

Usage:
    python generate_analytics_dataset.py
    
Output:
    analytics_events.csv - 20,000 events (30% violence, 70% nonviolence)
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from data.mock_generator import ViolenceEventGenerator


def main():
    print("=" * 60)
    print("Analytics Dataset Generator")
    print("=" * 60)
    
    # Create generator with seed for reproducibility
    generator = ViolenceEventGenerator(seed=42)
    
    # Generate 20,000 mixed events over 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    events = generator.generate_mixed(
        n_events=20000,
        violence_ratio=0.3,  # 30% violence, 70% nonviolence
        start_date=start_date,
        end_date=end_date,
    )
    
    # Convert to DataFrame and show stats
    df = generator.to_dataframe(events)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total events: {len(df)}")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    
    print(f"\n   Events by label:")
    for label, count in df.groupby("label").size().items():
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    print(f"\n   Events by camera:")
    for cam, count in df.groupby("cameraName").size().items():
        print(f"      {cam}: {count}")
    
    # Save to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "data", "analytics_events.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Show sample data
    print(f"\n📋 Sample data (first 5 rows):")
    print(df.head().to_string())
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
