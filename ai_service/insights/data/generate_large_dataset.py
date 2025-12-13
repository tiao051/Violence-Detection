"""
Generate large-scale mock violence event data for Analytics dashboard.
Creates ~100,000 events spanning 3 months.

Run: python ai_service/insights/data/generate_large_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta
from insights.data import ViolenceEventGenerator

def main():
    print("=" * 60)
    print("Generating Large-Scale Violence Event Dataset")
    print("=" * 60)
    
    # Config
    N_EVENTS = 100000
    DAYS = 90  # 3 months
    
    # Setup
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS)
    
    print(f"\nConfiguration:")
    print(f"  - Events: {N_EVENTS:,}")
    print(f"  - Date range: {start_date.date()} to {end_date.date()}")
    print(f"  - Duration: {DAYS} days")
    
    # Generate
    print("\nGenerating events... (this may take a minute)")
    generator = ViolenceEventGenerator(seed=42)
    events = generator.generate(
        n_events=N_EVENTS,
        start_date=start_date,
        end_date=end_date,
    )
    
    print(f"Generated {len(events):,} events")
    
    # Save to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, "violence_events_100k.csv")
    
    print(f"\nSaving to: {csv_path}")
    generator.save_to_csv(events, csv_path)
    
    # Show sample stats
    df = generator.to_dataframe(events)
    
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    print(f"\nEvents by Camera:")
    for cam, count in df.groupby("cameraName").size().items():
        pct = count / len(df) * 100
        print(f"  {cam}: {count:,} ({pct:.1f}%)")
    
    print(f"\nEvents by Day:")
    for day, count in df.groupby("day_name").size().items():
        print(f"  {day}: {count:,}")
    
    print(f"\nEvents by Time Period:")
    for period, count in df.groupby("time_period").size().items():
        print(f"  {period}: {count:,}")
    
    print(f"\nEvents by Severity:")
    for sev, count in df.groupby("severity").size().items():
        pct = count / len(df) * 100
        print(f"  {sev}: {count:,} ({pct:.1f}%)")
    
    print(f"\nConfidence Score:")
    print(f"  Min: {df['confidence'].min():.3f}")
    print(f"  Max: {df['confidence'].max():.3f}")
    print(f"  Mean: {df['confidence'].mean():.3f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
