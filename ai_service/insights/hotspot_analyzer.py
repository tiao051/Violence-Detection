"""
Hotspot Analysis using Statistical Methods (No ML Training Required).

Analyzes camera locations to identify hotspots based on:
1. Violence event ratio
2. Average confidence scores
3. Z-score comparison with other cameras
4. Time-weighted recent events
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from collections import defaultdict
import csv
import os


@dataclass
class CameraStats:
    """Statistics for a single camera location."""
    camera_id: str
    camera_name: str
    camera_description: str
    total_events: int
    violence_events: int
    nonviolence_events: int
    avg_confidence: float
    violence_ratio: float
    z_score: float
    hotspot_score: float
    classification: str  # "hotspot" | "warning" | "safe"
    risk_level: str  # "HIGH" | "MEDIUM" | "LOW"


class HotspotAnalyzer:
    """
    Statistical analyzer for identifying violence hotspots.
    
    No training required - works directly on event data.
    """
    
    # Weights for scoring
    WEIGHT_VIOLENCE_RATIO = 0.40
    WEIGHT_CONFIDENCE = 0.30
    WEIGHT_ZSCORE = 0.30
    
    # Classification thresholds
    HOTSPOT_THRESHOLD = 0.6
    WARNING_THRESHOLD = 0.4
    
    # Time decay factor (events older than this get less weight)
    TIME_DECAY_DAYS = 30
    
    def __init__(self):
        self.camera_stats: Dict[str, CameraStats] = {}
        self.events: List[Dict] = []
        
    def load_from_csv(self, csv_path: str) -> "HotspotAnalyzer":
        """Load events from CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.events = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.events.append({
                    'camera_id': row.get('cameraId', ''),
                    'camera_name': row.get('cameraName', ''),
                    'camera_description': row.get('cameraDescription', ''),
                    'timestamp': row.get('timestamp', ''),
                    'confidence': float(row.get('confidence', 0)),
                    'label': row.get('label', 'nonviolence'),
                })
        
        return self
    
    def load_from_events(self, events: List[Dict]) -> "HotspotAnalyzer":
        """Load events from list of dictionaries."""
        self.events = events
        return self
    
    def analyze(self) -> Dict[str, CameraStats]:
        """
        Perform hotspot analysis on loaded events.
        
        Returns dict of camera_id -> CameraStats
        """
        if not self.events:
            return {}
        
        # Step 1: Aggregate events by camera
        camera_data = self._aggregate_by_camera()
        
        # Step 2: Calculate violence ratios
        for cam_id, data in camera_data.items():
            total = data['violence'] + data['nonviolence']
            data['violence_ratio'] = data['violence'] / total if total > 0 else 0
        
        # Step 3: Calculate Z-scores (how each camera compares to mean)
        ratios = [d['violence_ratio'] for d in camera_data.values()]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0
        std_ratio = self._std_dev(ratios, mean_ratio)
        
        for data in camera_data.values():
            if std_ratio > 0:
                data['z_score'] = (data['violence_ratio'] - mean_ratio) / std_ratio
            else:
                data['z_score'] = 0
        
        # Step 4: Calculate hotspot scores and classify
        self.camera_stats = {}
        for cam_id, data in camera_data.items():
            # Normalize z_score to 0-1 range (sigmoid-like)
            z_normalized = 1 / (1 + math.exp(-data['z_score']))
            
            # Calculate weighted score
            score = (
                self.WEIGHT_VIOLENCE_RATIO * data['violence_ratio'] +
                self.WEIGHT_CONFIDENCE * data['avg_confidence'] +
                self.WEIGHT_ZSCORE * z_normalized
            )
            
            # Classify based on score
            if score >= self.HOTSPOT_THRESHOLD:
                classification = "hotspot"
                risk_level = "HIGH"
            elif score >= self.WARNING_THRESHOLD:
                classification = "warning"
                risk_level = "MEDIUM"
            else:
                classification = "safe"
                risk_level = "LOW"
            
            self.camera_stats[cam_id] = CameraStats(
                camera_id=cam_id,
                camera_name=data['camera_name'],
                camera_description=data['camera_description'],
                total_events=data['violence'] + data['nonviolence'],
                violence_events=data['violence'],
                nonviolence_events=data['nonviolence'],
                avg_confidence=round(data['avg_confidence'], 3),
                violence_ratio=round(data['violence_ratio'], 3),
                z_score=round(data['z_score'], 3),
                hotspot_score=round(score, 3),
                classification=classification,
                risk_level=risk_level,
            )
        
        return self.camera_stats
    
    def _aggregate_by_camera(self) -> Dict[str, Dict]:
        """Aggregate events by camera ID."""
        camera_data = defaultdict(lambda: {
            'camera_name': '',
            'camera_description': '',
            'violence': 0,
            'nonviolence': 0,
            'confidences': [],
        })
        
        for event in self.events:
            cam_id = event['camera_id']
            camera_data[cam_id]['camera_name'] = event['camera_name']
            camera_data[cam_id]['camera_description'] = event.get('camera_description', '')
            
            if event['label'] == 'violence':
                camera_data[cam_id]['violence'] += 1
            else:
                camera_data[cam_id]['nonviolence'] += 1
            
            camera_data[cam_id]['confidences'].append(event['confidence'])
        
        # Calculate average confidence
        for data in camera_data.values():
            if data['confidences']:
                data['avg_confidence'] = sum(data['confidences']) / len(data['confidences'])
            else:
                data['avg_confidence'] = 0
        
        return dict(camera_data)
    
    def _std_dev(self, values: List[float], mean: float) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def get_hotspots(self) -> List[CameraStats]:
        """Get cameras classified as hotspots."""
        return [s for s in self.camera_stats.values() if s.classification == "hotspot"]
    
    def get_safe_zones(self) -> List[CameraStats]:
        """Get cameras classified as safe zones."""
        return [s for s in self.camera_stats.values() if s.classification == "safe"]
    
    def get_warnings(self) -> List[CameraStats]:
        """Get cameras classified as warnings."""
        return [s for s in self.camera_stats.values() if s.classification == "warning"]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of hotspot analysis."""
        if not self.camera_stats:
            return {"error": "No analysis performed yet"}
        
        return {
            "total_cameras": len(self.camera_stats),
            "hotspots": len(self.get_hotspots()),
            "warnings": len(self.get_warnings()),
            "safe_zones": len(self.get_safe_zones()),
            "total_events": sum(s.total_events for s in self.camera_stats.values()),
            "total_violence_events": sum(s.violence_events for s in self.camera_stats.values()),
            "cameras": [
                {
                    "camera_id": s.camera_id,
                    "camera_name": s.camera_name,
                    "camera_description": s.camera_description,
                    "total_events": s.total_events,
                    "violence_events": s.violence_events,
                    "violence_ratio": s.violence_ratio,
                    "avg_confidence": s.avg_confidence,
                    "z_score": s.z_score,
                    "hotspot_score": s.hotspot_score,
                    "classification": s.classification,
                    "risk_level": s.risk_level,
                }
                for s in sorted(self.camera_stats.values(), key=lambda x: x.hotspot_score, reverse=True)
            ]
        }


# Singleton instance for API use
_analyzer_instance: Optional[HotspotAnalyzer] = None


def get_hotspot_analyzer() -> HotspotAnalyzer:
    """Get or create the hotspot analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HotspotAnalyzer()
    return _analyzer_instance


def analyze_hotspots_from_csv(csv_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze hotspots from CSV file.
    
    Args:
        csv_path: Path to CSV file with event data
        
    Returns:
        Summary dict with hotspot analysis results
    """
    analyzer = HotspotAnalyzer()
    analyzer.load_from_csv(csv_path)
    analyzer.analyze()
    return analyzer.get_summary()


if __name__ == "__main__":
    # Test with sample data
    import os
    
    csv_path = os.path.join(os.path.dirname(__file__), "data", "analytics_events.csv")
    
    if os.path.exists(csv_path):
        result = analyze_hotspots_from_csv(csv_path)
        print(f"\n{'='*60}")
        print("HOTSPOT ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Total Cameras: {result['total_cameras']}")
        print(f"Hotspots: {result['hotspots']}")
        print(f"Warnings: {result['warnings']}")
        print(f"Safe Zones: {result['safe_zones']}")
        print(f"\nCamera Details:")
        for cam in result['cameras']:
            status_icon = "🔴" if cam['classification'] == 'hotspot' else "🟡" if cam['classification'] == 'warning' else "🟢"
            print(f"  {status_icon} {cam['camera_name']}: score={cam['hotspot_score']:.2f}, violence_ratio={cam['violence_ratio']:.1%}")
    else:
        print(f"CSV file not found: {csv_path}")
