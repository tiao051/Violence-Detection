import csv
import os
import random
from datetime import datetime, timedelta

"""
Script to generate synthetic analytics data (Rich Mock Data).
Generates ~200k rows of verified event scenarios to stress-test the Spark Analytics engine.

Usage (Docker - Recommended):
    docker exec violence-detection-inference python ai_service/utils/generate_rich_mock_data.py

Full Analytics Workflow:
    1. Generate Data:
       docker exec violence-detection-inference python ai_service/utils/generate_rich_mock_data.py
       
    2. (Optional) Push to HDFS (Simulating Production Data Lake):
       docker exec hdfs-namenode hdfs dfs -mkdir -p /analytics/raw
       docker exec violence-detection-inference cat /app/ai_service/tmp/verified_scenarios.csv | docker exec -i hdfs-namenode hdfs dfs -put -f - /analytics/raw/verified_events.csv
       
    3. Train Spark Model & Generate Insights:
       docker exec violence-detection-inference python ai_service/insights/spark/spark_insights_job.py

Output:
    /app/ai_service/tmp/verified_scenarios.csv
"""

# Constants
OUTPUT_DIR = "/app/ai_service/tmp"
OUTPUT_FILE = "verified_scenarios.csv"

# Camera Profiles (Scaled for ~200k total rows)
CAMERAS = {
    "cam1": {"type": "SAFE", "base_events": 1000, "tp_rate": 0.95, "conf_range": (0.8, 0.99)},
    "cam2": {"type": "NOISY", "base_events": 1600, "tp_rate": 0.20, "conf_range": (0.4, 0.7)}, # Lots of noise
    "cam3": {"type": "ANOMALY", "base_events": 1200, "tp_rate": 0.80, "conf_range": (0.7, 0.9)},
    "cam4": {"type": "NORMAL", "base_events": 1300, "tp_rate": 0.60, "conf_range": (0.6, 0.85)},
    "cam5": {"type": "HOTSPOT", "base_events": 1500, "tp_rate": 0.90, "conf_range": (0.85, 0.95)}
}

def generate_data():
    print(f"Generating rich mock data for Analytics...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    headers = ["timestamp", "camera_id", "confidence", "is_verified", "verification_status", "duration", "image_path"]
    
    now = datetime.now()
    rows = []
    
    # Generate 30 days of data
    for i in range(30):
        day_date = now - timedelta(days=29 - i)
        is_weekend = day_date.weekday() >= 5  # Sat, Sun
        is_friday = day_date.weekday() == 4
        
        print(f"Processing day {i+1}/30: {day_date.strftime('%Y-%m-%d')}")
        
        for cam_id, profile in CAMERAS.items():
            num_events = profile["base_events"] + random.randint(-1, 2)
            if num_events < 0: num_events = 0
            
            # --- SPECIAL PATTERNS ---
            
            # 1. Hotspot Pattern (Cam 5): Spikes on Friday/Saturday Nights
            if profile["type"] == "HOTSPOT" and (is_friday or is_weekend):
                num_events += random.randint(10, 15)
                
            # 2. Anomaly Pattern (Cam 3): Spike in last 2 days
            if profile["type"] == "ANOMALY" and i >= 28:
                num_events += random.randint(15, 20)
                
            # Generate events
            for _ in range(num_events):
                # Determine time of day
                if profile["type"] == "HOTSPOT" and (is_friday or is_weekend):
                    # Night spike: 18:00 - 02:00
                    hour = random.choice([18, 19, 20, 21, 22, 23, 0, 1, 2])
                elif profile["type"] == "NOISY":
                    # Random time
                    hour = random.randint(0, 23)
                else:
                    # Mostly day
                    hour = random.randint(7, 20)
                
                minute = random.randint(0, 59)
                event_time = day_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))
                
                # Check Verification
                is_tp = random.random() < profile["tp_rate"]
                status = "true_positive" if is_tp else "false_positive"
                
                # Confidence
                conf_min, conf_max = profile["conf_range"]
                conf = random.uniform(conf_min, conf_max)
                if not is_tp: 
                    conf -= 0.1 # Real mistakes usually have lower confidence
                    
                rows.append({
                    "timestamp": int(event_time.timestamp() * 1000),
                    "camera_id": cam_id,
                    "confidence": round(conf, 2),
                    "is_verified": True, # For training, we assume these are human labeled
                    "verification_status": status,
                    "duration": round(random.uniform(2.0, 10.0), 1),
                    "image_path": f"/images/{cam_id}_{event_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                })
                
    # Sort by time
    rows.sort(key=lambda x: x["timestamp"])
    
    # Save
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully generated {len(rows)} events to {filepath}")

if __name__ == "__main__":
    generate_data()
