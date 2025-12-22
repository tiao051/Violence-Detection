"""
SCENARIO DIRECTOR: Synthetic Data Generator for Violence Detection Analytics.

This script generates high-volume, realistic historical data to train and verify
the Spark Analytics Engine and Random Forest models. It simulates specific
scenarios (Red/Green/Noise zones) with temporal patterns to ensure machine learning
algorithms can detect meaningful insights.

Scenarios:
1. Zone RED (High Risk): Critical cameras (e.g., Junctions). Frequent long-duration violence at night (Fri/Sat).
2. Zone GREEN (Low Risk): Safe areas (e.g., Parks). Short, low-confidence incidents during the day.
3. Zone NOISE (False Positives): Simulates sensor noise or glitches.

Usage:
    python scripts/generate_scenarios.py
"""

import csv
import random
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

OUTPUT_DIR = "tmp/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "violence_scenarios.csv")
DAYS_BACK = 30
TOTAL_EVENTS_TARGET = 100_000
EVENTS_PER_DAY = TOTAL_EVENTS_TARGET // DAYS_BACK + 100 

CAMERAS = {
    "cam1": {"name": "Luy Ban Bich Street", "zone": "RED"},
    "cam2": {"name": "Au Co Junction", "zone": "RED"},
    "cam3": {"name": "Tan Ky Tan Quy Street", "zone": "GREEN"},
    "cam4": {"name": "Tan Phu Market", "zone": "GREEN"},
    "cam5": {"name": "Dam Sen Park", "zone": "NOISE"},
}

class ScenarioDirector:
    """Orchestrates the generation of synthetic violence patterns."""

    def __init__(self, output_file: str = OUTPUT_FILE):
        self.output_file = output_file
        self.stats = {"total": 0, "zones": {"RED": 0, "GREEN": 0, "NOISE": 0}}
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def generate(self):
        """Main execution method."""
        start_time = time.time()
        print(f"🎬 Director Action! Generating ~{TOTAL_EVENTS_TARGET} events over {DAYS_BACK} days...")

        fieldnames = [
            "timestamp", "camera_id", "camera_name", "confidence", "label",
            "duration", "severity_level", "user_id", "camera_description", "is_verified"
        ]

        # Use efficient batch writing
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            batch: List[Dict[str, Any]] = []
            batch_size = 5000
            
            start_date = datetime.now() - timedelta(days=DAYS_BACK)

            for day_offset in range(DAYS_BACK + 1):
                current_date = start_date + timedelta(days=day_offset)
                day_name = current_date.strftime("%A")
                is_weekend = current_date.weekday() in [4, 5, 6]
                
                # Dynamic volume: Weekends get 20% more traffic
                daily_volume = EVENTS_PER_DAY + (int(EVENTS_PER_DAY * 0.2) if is_weekend else 0)
                
                for _ in range(daily_volume):
                    event = self._create_scene(current_date, day_name)
                    batch.append(event)
                    
                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        self.stats["total"] += len(batch)
                        self._print_progress(self.stats["total"])
                        batch = []

            # Write remaining
            if batch:
                writer.writerows(batch)
                self.stats["total"] += len(batch)

        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    def _create_scene(self, date_base: datetime, day_name: str) -> Dict[str, Any]:
        """Generates a single event based on zone logic."""
        cam_id = random.choice(list(CAMERAS.keys()))
        cam_info = CAMERAS[cam_id]
        zone = cam_info["zone"]
        self.stats["zones"][zone] += 1
        
        # Default Attributes
        label = "violence"
        severity = "LOW"
        
        # --- Logic Script (Matches User Requirements) ---
        if zone == "RED":
            # Scenario: Dangerous Weekend Nights
            # 70% chance of 'The Purge' pattern on Friday/Saturday nights
            is_purge_time = day_name in ["Friday", "Saturday"] and random.random() < 0.7
            
            if is_purge_time:
                # Night owl pattern: 20:00 - 02:00 (Cross-day handled below)
                hour = random.choice([20, 21, 22, 23, 0, 1, 2])
                duration = random.randint(60, 300) # Serious conflict (>1 min)
                confidence = random.uniform(0.85, 0.99)
                severity = "HIGH"
            else:
                # Normal chaos
                hour = random.randint(8, 22)
                duration = random.randint(10, 40)
                confidence = random.uniform(0.60, 0.85)
                severity = "MEDIUM"
                
        elif zone == "GREEN":
            # Scenario: Minor daytime scuffles
            hour = random.randint(7, 18) # Office hours
            duration = random.randint(5, 15) # Quick push/shove
            confidence = random.uniform(0.50, 0.75)
            severity = "LOW"
            
        else: # NOISE
            # Scenario: Sensor Glitches
            hour = random.randint(0, 23)
            duration = random.randint(0, 2) # Flash
            confidence = random.uniform(0.30, 0.60)
            label = "false_positive"
            severity = "IGNORE"

        # Timestamp Construction
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = date_base.replace(hour=hour if hour < 24 else 23, minute=minute, second=second)
        
        # Handle day overflow for late night hours (0, 1, 2)
        if hour < 3:
             dt += timedelta(days=1)
        
        timestamp_ms = int(dt.timestamp() * 1000)

        return {
            "timestamp": timestamp_ms,
            "camera_id": cam_id,
            "camera_name": cam_info["name"],
            "confidence": round(confidence, 2),
            "label": label,
            "duration": duration,
            "severity_level": severity,
            "user_id": "sim_user",
            "camera_description": f"Zone {zone} Simulation",
            "is_verified": False
        }

    def _print_progress(self, current: int):
        """Prints a simple inline progress update."""
        if current % 10000 == 0:
            print(f"   ...Generated {current} events")

    def _print_summary(self, elapsed: float):
        """Prints final cut report."""
        print(f"\nProduction Wrap.")
        print(f"Time Elapsed: {elapsed:.2f}s")
        print(f"Total Events: {self.stats['total']}")
        print(f"Distribution: {self.stats['zones']}")
        print(f"Saved to: {self.output_file}")


if __name__ == "__main__":
    director = ScenarioDirector()
    director.generate()
