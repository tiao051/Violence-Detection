"""
Mock Violence Event Generator

Generates realistic mock violence detection events for testing 
and developing analytics models when real data is limited.

The generator creates events with realistic patterns:
- More events during evening/night hours
- Higher frequency on weekends
- Certain cameras have higher incident rates
- Confidence scores follow realistic distribution
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import numpy as np

from .event_schema import ViolenceEvent


class ViolenceEventGenerator:
    """
    Generates mock violence detection events with realistic patterns.
    
    Features:
    - Configurable time range
    - Realistic time-of-day distribution (more events at night)
    - Realistic day-of-week distribution (more on weekends)
    - Camera hotspot simulation (some cameras have more events)
    - Configurable confidence score distribution
    
    Example:
        >>> generator = ViolenceEventGenerator(seed=42)
        >>> events = generator.generate(n_events=100)
        >>> df = generator.to_dataframe(events)
    """
    
    # Default camera configuration: (camera_id, camera_name, camera_description, weight)
    # Weight determines relative frequency of events (higher = more events)
    DEFAULT_CAMERAS = [
        ("cam1", "Ngã tư Lê Trọng Tấn", "Lê Trọng Tấn giao Tân Kỳ Tân Quý", 2.0),
        ("cam2", "Ngã tư Cộng Hòa", "Tân Kỳ Tân Quý giao Cộng Hòa", 1.5),
        ("cam3", "Ngã ba Âu Cơ", "Lũy Bán Bích giao Âu Cơ", 3.0),  # Hotspot
        ("cam4", "Ngã tư Hòa Bình", "Hòa Bình giao Lạc Long Quân", 1.0),
        ("cam5", "Ngã tư Tân Sơn Nhì", "Tân Sơn Nhì giao Tây Thạnh", 1.2),
    ]
    
    # Hour weights: probability distribution for each hour (0-23)
    # Higher values = more likely to have events at that hour
    DEFAULT_HOUR_WEIGHTS = {
        0: 1.5,   # Midnight
        1: 1.2,
        2: 1.0,
        3: 0.8,
        4: 0.6,
        5: 0.5,
        6: 0.4,   # Early morning - low
        7: 0.3,
        8: 0.4,
        9: 0.5,
        10: 0.6,
        11: 0.7,
        12: 0.8,  # Noon
        13: 0.9,
        14: 1.0,
        15: 1.2,
        16: 1.4,
        17: 1.6,  # After work/school
        18: 2.0,
        19: 2.5,  # Evening - peak
        20: 3.0,  # Peak hour
        21: 2.8,
        22: 2.5,
        23: 2.0,
    }
    
    # Day of week weights (0=Monday, 6=Sunday)
    DEFAULT_DAY_WEIGHTS = {
        0: 1.0,   # Monday
        1: 0.9,   # Tuesday
        2: 0.8,   # Wednesday - lowest
        3: 0.9,   # Thursday
        4: 1.5,   # Friday - higher
        5: 2.0,   # Saturday - peak
        6: 1.8,   # Sunday
    }
    
    def __init__(
        self,
        cameras: Optional[List[tuple]] = None,
        hour_weights: Optional[Dict[int, float]] = None,
        day_weights: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            cameras: List of (camera_id, camera_name, location, weight) tuples
            hour_weights: Dict mapping hour (0-23) to probability weight
            day_weights: Dict mapping day (0-6) to probability weight
            seed: Random seed for reproducibility
        """
        self.cameras = cameras or self.DEFAULT_CAMERAS
        self.hour_weights = hour_weights or self.DEFAULT_HOUR_WEIGHTS
        self.day_weights = day_weights or self.DEFAULT_DAY_WEIGHTS
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Normalize camera weights
        total_weight = sum(c[3] for c in self.cameras)
        self.camera_probs = [c[3] / total_weight for c in self.cameras]
    
    def _generate_timestamp(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> datetime:
        """
        Generate a random timestamp with realistic distribution.
        
        Uses weighted random selection based on hour and day patterns.
        """
        # Random day in range
        days_diff = (end_date - start_date).days
        random_day = random.randint(0, max(0, days_diff))
        base_date = start_date + timedelta(days=random_day)
        
        # Get day of week weight
        day_of_week = base_date.weekday()
        day_weight = self.day_weights.get(day_of_week, 1.0)
        
        # Weighted random hour selection
        hours = list(self.hour_weights.keys())
        hour_probs = list(self.hour_weights.values())
        
        # Apply day weight modifier
        hour_probs = [p * day_weight for p in hour_probs]
        
        # Normalize
        total = sum(hour_probs)
        hour_probs = [p / total for p in hour_probs]
        
        # Select hour
        hour = np.random.choice(hours, p=hour_probs)
        
        # Random minute and second
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
    
    def _generate_confidence(self) -> float:
        """
        Generate realistic confidence score.
        
        Uses beta distribution to create realistic scores
        that cluster around 0.7-0.9 with occasional low values.
        """
        # Beta distribution parameters for realistic confidence scores
        # alpha=5, beta=2 gives distribution skewed towards higher values
        score = np.random.beta(5, 2)
        
        # Scale to 0.5-1.0 range (we don't detect with very low confidence)
        score = 0.5 + score * 0.5
        
        return round(score, 3)
    
    def _select_camera(self) -> tuple:
        """Select a camera using weighted random selection."""
        idx = np.random.choice(len(self.cameras), p=self.camera_probs)
        return self.cameras[idx]
    
    def generate_event(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: str = "user_123",
        label: str = "violence",
    ) -> ViolenceEvent:
        """
        Generate a single mock event.
        
        Args:
            start_date: Start of time range
            end_date: End of time range
            user_id: User ID to assign to event
            label: Event label ("violence" or "nonviolence")
            
        Returns:
            ViolenceEvent instance
        """
        camera = self._select_camera()
        camera_id, camera_name, camera_description, _ = camera
        
        return ViolenceEvent(
            user_id=user_id,
            camera_id=camera_id,
            camera_name=camera_name,
            camera_description=camera_description,
            timestamp=self._generate_timestamp(start_date, end_date),
            confidence=self._generate_confidence(),
            video_url=f"https://storage.example.com/events/{uuid.uuid4()}.mp4",
            label=label,
        )
    
    def generate(
        self,
        n_events: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: str = "user_123",
        label: str = "violence",
    ) -> List[ViolenceEvent]:
        """
        Generate multiple mock events with the same label.
        
        Args:
            n_events: Number of events to generate
            start_date: Start of time range (default: 30 days ago)
            end_date: End of time range (default: now)
            user_id: User ID to assign to events
            label: Event label ("violence" or "nonviolence")
            
        Returns:
            List of ViolenceEvent instances, sorted by timestamp
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        events = [
            self.generate_event(start_date, end_date, user_id, label)
            for _ in range(n_events)
        ]
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def generate_mixed(
        self,
        n_events: int = 20000,
        violence_ratio: float = 0.3,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: str = "user_123",
    ) -> List[ViolenceEvent]:
        """
        Generate mixed events with both violence and nonviolence labels.
        
        Args:
            n_events: Total number of events to generate
            violence_ratio: Ratio of violence events (0.0-1.0), default 0.3 (30%)
            start_date: Start of time range (default: 90 days ago)
            end_date: End of time range (default: now)
            user_id: User ID to assign to events
            
        Returns:
            List of ViolenceEvent instances, sorted by timestamp
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        
        # Calculate counts
        n_violence = int(n_events * violence_ratio)
        n_nonviolence = n_events - n_violence
        
        print(f"Generating {n_violence} violence + {n_nonviolence} nonviolence events...")
        
        # Generate violence events
        violence_events = [
            self.generate_event(start_date, end_date, user_id, "violence")
            for _ in range(n_violence)
        ]
        
        # Generate nonviolence events
        nonviolence_events = [
            self.generate_event(start_date, end_date, user_id, "nonviolence")
            for _ in range(n_nonviolence)
        ]
        
        # Combine and sort by timestamp
        all_events = violence_events + nonviolence_events
        all_events.sort(key=lambda e: e.timestamp)
        
        print(f"Generated {len(all_events)} total events")
        return all_events
    
    def to_dataframe(self, events: List[ViolenceEvent]):
        """
        Convert events to pandas DataFrame.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            pandas DataFrame with all event attributes
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")
        
        data = [event.to_dict() for event in events]
        df = pd.DataFrame(data)
        
        # Convert timestamp to proper datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    
    def save_to_csv(self, events: List[ViolenceEvent], filepath: str) -> None:
        """
        Save events to CSV file.
        
        Args:
            events: List of ViolenceEvent instances
            filepath: Path to save CSV file
        """
        df = self.to_dataframe(events)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(events)} events to {filepath}")
    
    def save_to_json(self, events: List[ViolenceEvent], filepath: str) -> None:
        """
        Save events to JSON file.
        
        Args:
            events: List of ViolenceEvent instances
            filepath: Path to save JSON file
        """
        import json
        
        data = [event.to_dict() for event in events]
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Saved {len(events)} events to {filepath}")


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Violence Event Mock Data Generator")
    print("=" * 60)
    
    # Create generator with seed for reproducibility
    generator = ViolenceEventGenerator(seed=42)
    
    # Generate 200 events over the last 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    events = generator.generate(
        n_events=200,
        start_date=start_date,
        end_date=end_date,
    )
    
    print(f"\n✅ Generated {len(events)} mock violence events")
    print(f"   Time range: {start_date.date()} to {end_date.date()}")
    
    # Show sample events
    print("\n📋 Sample events:")
    for event in events[:5]:
        print(f"   {event}")
    
    # Convert to DataFrame and show stats
    try:
        df = generator.to_dataframe(events)
        
        print("\n📊 Quick Statistics:")
        print(f"   Events by camera:")
        for cam, count in df.groupby("camera_name").size().items():
            print(f"      {cam}: {count}")
        
        print(f"\n   Events by time period:")
        for period, count in df.groupby("time_period").size().items():
            print(f"      {period}: {count}")
        
        print(f"\n   Events by day:")
        for day, count in df.groupby("day_name").size().items():
            print(f"      {day}: {count}")
        
        print(f"\n   Average confidence: {df['confidence'].mean():.3f}")
        
        # Save to files
        generator.save_to_csv(events, "ai_service/insights/data/mock_events.csv")
        generator.save_to_json(events, "ai_service/insights/data/mock_events.json")
        
    except ImportError:
        print("\n⚠️  Install pandas to see detailed statistics: pip install pandas")
    
    print("\n✅ Mock data generation complete!")
