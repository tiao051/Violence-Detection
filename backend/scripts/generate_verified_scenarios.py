"""
Synthetic Verified Data Generator for Camera Credibility System

Generates 200,000 rows of realistic violence detection data with verified labels.
Each camera has distinct behavior profile (Noisy/Reliable/Selective/Overcautious/Mixed).

Output: CSV file ready for Spark training with human-verified labels.
"""

import csv
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import argparse

# Camera Behavior Profiles
CAMERA_PROFILES = {
    "cam1": {
        "name": "Luy Ban Bich Street",
        "behavior": "Noisy",
        "false_positive_rate": 0.40,  # 40% false alarms
        "characteristics": {
            "avg_confidence": 0.55,    # Low-medium confidence
            "conf_std": 0.20,          # High variance (erratic)
            "avg_duration": 3.5,       # Short events
            "alerts_per_day": 35       # High frequency
        },
        "description": "High false alarm rate - needs recalibration"
    },
    "cam2": {
        "name": "Au Co Junction",
        "behavior": "Reliable",
        "false_positive_rate": 0.08,  # 8% false alarms
        "characteristics": {
            "avg_confidence": 0.88,    # High confidence
            "conf_std": 0.06,          # Low variance (consistent)
            "avg_duration": 25.0,      # Longer events
            "alerts_per_day": 12       # Moderate frequency
        },
        "description": "Highly reliable - low false alarms, consistent"
    },
    "cam3": {
        "name": "Tan Ky Tan Quy Street",
        "behavior": "Selective",
        "false_positive_rate": 0.12,  # 12% false alarms
        "characteristics": {
            "avg_confidence": 0.82,
            "conf_std": 0.08,
            "avg_duration": 18.0,
            "alerts_per_day": 8        # Low frequency
        },
        "description": "Low alert frequency but high accuracy when triggers"
    },
    "cam4": {
        "name": "Tan Phu Market",
        "behavior": "Overcautious",
        "false_positive_rate": 0.25,  # 25% false alarms
        "characteristics": {
            "avg_confidence": 0.68,
            "conf_std": 0.15,
            "avg_duration": 8.5,
            "alerts_per_day": 28       # High frequency
        },
        "description": "Overly sensitive - many false positives"
    },
    "cam5": {
        "name": "Dam Sen Park",
        "behavior": "Mixed",
        "false_positive_rate": 0.20,  # 20% false alarms
        "characteristics": {
            "avg_confidence": 0.72,
            "conf_std": 0.12,
            "avg_duration": 12.0,
            "alerts_per_day": 20
        },
        "description": "Moderate performance, room for improvement"
    }
}


def generate_event(camera_id: str, camera_profile: Dict, date_base: datetime) -> Dict[str, Any]:
    """
    Generate a single violence detection event with verified label.
    
    Args:
        camera_id: Camera identifier (e.g., "cam1")
        camera_profile: Camera behavior profile
        date_base: Base date for timestamp generation
        
    Returns:
        Event dictionary with all required fields including verification status
    """
    fp_rate = camera_profile["false_positive_rate"]
    char = camera_profile["characteristics"]
    
    # Decide if this event is a false positive
    is_false_positive = random.random() < fp_rate
    
    # Generate features based on verification status
    if is_false_positive:
        # FALSE POSITIVE characteristics:
        # - Lower confidence
        # - Shorter duration
        # - Rarely HIGH severity
        confidence = random.gauss(char["avg_confidence"] - 0.15, 0.12)
        duration = random.uniform(0.5, 5.0)  # Very short
        severity_level = random.choice(["LOW", "LOW", "MEDIUM"])  # Bias toward LOW
    else:
        # TRUE POSITIVE characteristics:
        # - Normal confidence distribution
        # - Normal duration distribution
        # - Realistic severity mix
        confidence = random.gauss(char["avg_confidence"], char["conf_std"])
        duration = random.gauss(char["avg_duration"], char["avg_duration"] * 0.3)
        severity_level = random.choices(
            ["LOW", "MEDIUM", "HIGH"],
            weights=[0.2, 0.5, 0.3]  # Realistic distribution
        )[0]
    
    # Clamp values to valid ranges
    confidence = max(0.3, min(0.99, confidence))
    duration = max(0.5, min(300, duration))
    
    # Generate realistic temporal pattern
    # Peak hours: 18-21 (evening), low hours: 00-05 (night)
    hour = random.choices(
        range(24),
        weights=[
            2, 1, 1, 1, 1, 2,    
            3, 5, 8, 10, 12, 15,
            15, 15, 12, 10, 8, 10, 
            15, 20, 25, 20, 10, 5  
        ]
    )[0]
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    # Create full timestamp
    dt = date_base.replace(hour=hour, minute=minute, second=second)
    timestamp_ms = int(dt.timestamp() * 1000)
    
    # Day info
    day_name = dt.strftime("%A")
    day_of_week = dt.weekday()
    is_weekend = day_of_week in [4, 5, 6]
    
    return {
        "timestamp": timestamp_ms,
        "camera_id": camera_id,
        "camera_name": camera_profile["name"],
        "confidence": round(confidence, 2),
        "label": "violence",
        "duration": round(duration, 1),
        "severity_level": severity_level,
        "user_id": "synthetic_user",
        "camera_description": f"{camera_profile['behavior']} Camera Simulation",
        "is_verified": True,  
        "verification_status": "false_positive" if is_false_positive else "true_positive",
        "verified_by": "data_generator",
        "verified_at": datetime.now().isoformat()
    }


def generate_dataset(total_rows: int = 200000, days: int = 180) -> List[Dict[str, Any]]:
    """
    Generate full dataset with specified number of rows.
    
    Args:
        total_rows: Total number of events to generate
        days: Number of days to spread events across
        
    Returns:
        List of event dictionaries
    """
    events = []
    
    # Calculate events per camera
    num_cameras = len(CAMERA_PROFILES)
    rows_per_camera = total_rows // num_cameras
    
    print(f"Generating {total_rows:,} events across {num_cameras} cameras over {days} days...")
    print(f"Target: ~{rows_per_camera:,} events per camera\n")
    
    # Base date: 180 days ago from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for camera_id, profile in CAMERA_PROFILES.items():
        print(f"Generating for {camera_id} ({profile['name']})...")
        print(f"  Behavior: {profile['behavior']}")
        print(f"  False Positive Rate: {profile['false_positive_rate']*100:.0f}%")
        print(f"  Avg Alerts/Day: {profile['characteristics']['alerts_per_day']}")
        
        camera_events = []
        
        # Generate events spread across days
        for day_offset in range(days):
            date = start_date + timedelta(days=day_offset)
            
            # Number of events for this day (Poisson-like distribution)
            avg_per_day = profile["characteristics"]["alerts_per_day"]
            daily_events = int(random.gauss(avg_per_day, avg_per_day * 0.3))
            daily_events = max(0, min(daily_events, avg_per_day * 2))  # Clamp
            
            for _ in range(daily_events):
                event = generate_event(camera_id, profile, date)
                camera_events.append(event)
        
        # Ensure we hit target count per camera
        while len(camera_events) < rows_per_camera:
            random_day = start_date + timedelta(days=random.randint(0, days-1))
            event = generate_event(camera_id, profile, random_day)
            camera_events.append(event)
        
        # Trim if exceeded
        camera_events = camera_events[:rows_per_camera]
        
        # Calculate actual false positive rate
        fp_count = sum(1 for e in camera_events if e["verification_status"] == "false_positive")
        actual_fp_rate = fp_count / len(camera_events)
        
        print(f"Generated {len(camera_events):,} events")
        print(f"Actual FP Rate: {actual_fp_rate*100:.1f}% (target: {profile['false_positive_rate']*100:.0f}%)\n")
        
        events.extend(camera_events)
    
    # Shuffle to mix cameras
    random.shuffle(events)
    
    print(f"✅ Total events generated: {len(events):,}")
    return events


def save_to_csv(events: List[Dict[str, Any]], output_file: str):
    """Save events to CSV file."""
    if not events:
        print("No events to save")
        return
    
    fieldnames = [
        "timestamp", "camera_id", "camera_name", "confidence", "label",
        "duration", "severity_level", "user_id", "camera_description",
        "is_verified", "verification_status", "verified_by", "verified_at"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Summary stats
    total = len(events)
    fp_count = sum(1 for e in events if e["verification_status"] == "false_positive")
    tp_count = total - fp_count
    
    print(f"\nDataset Summary:")
    print(f"   Total events: {total:,}")
    print(f"   True Positives: {tp_count:,} ({tp_count/total*100:.1f}%)")
    print(f"   False Positives: {fp_count:,} ({fp_count/total*100:.1f}%)")
    print(f"\nReady for upload to HDFS: hdfs dfs -put {output_file} /analytics/raw/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic verified violence detection data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tmp/verified_scenarios.csv",
        help="Output CSV file path (default: tmp/verified_scenarios.csv)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200000,
        help="Number of rows to generate (default: 200000)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days to spread data across (default: 180)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("Synthetic Verified Data Generator")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Rows: {args.rows:,}")
    print(f"  Days: {args.days}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print("=" * 60 + "\n")
    
    # Create output directory if needed
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate dataset
    events = generate_dataset(total_rows=args.rows, days=args.days)
    
    # Save to CSV
    save_to_csv(events, args.output)
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
