"""
Location Pattern Analyzer

Analyzes spatial patterns in violence detection events to identify:
- Camera hotspots (which cameras detect most violence)
- Location risk ranking
- Confidence distribution by camera
- Camera performance metrics

This is a statistical analysis model.
"""

from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from insights.data.event_schema import ViolenceEvent


class LocationAnalyzer:
    """
    Analyzes location/camera patterns in violence events.
    
    Identifies hotspot cameras and areas that may need additional
    security attention or camera repositioning.
    
    Example:
        >>> from insights.data import ViolenceEventGenerator
        >>> generator = ViolenceEventGenerator(seed=42)
        >>> events = generator.generate(n_events=200)
        >>> 
        >>> analyzer = LocationAnalyzer()
        >>> analyzer.fit(events)
        >>> 
        >>> print(analyzer.get_hotspot_cameras(top_n=3))
        >>> print(analyzer.get_camera_stats())
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.events: List[ViolenceEvent] = []
        self.is_fitted: bool = False
        
        # Computed statistics
        self._camera_counts: Dict[str, int] = {}
        self._camera_names: Dict[str, str] = {}  # camera_id -> camera_name
        self._camera_confidences: Dict[str, List[float]] = defaultdict(list)
        self._camera_severities: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    def fit(self, events: List[ViolenceEvent]) -> "LocationAnalyzer":
        """
        Fit the analyzer with event data.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if not events:
            raise ValueError("Cannot fit with empty events list")
        
        self.events = events
        
        # Reset
        self._camera_counts = {}
        self._camera_names = {}
        self._camera_confidences = defaultdict(list)
        self._camera_severities = defaultdict(lambda: defaultdict(int))
        
        # Count events per camera
        camera_ids = [e.camera_id for e in events]
        self._camera_counts = dict(Counter(camera_ids))
        
        # Build camera name mapping and aggregate stats
        for event in events:
            cam_id = event.camera_id
            self._camera_names[cam_id] = event.camera_name
            self._camera_confidences[cam_id].append(event.confidence)
            self._camera_severities[cam_id][event.severity] += 1
        
        self.is_fitted = True
        return self
    
    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
    
    # ==================== Camera Ranking ====================
    
    def get_camera_counts(self) -> Dict[str, int]:
        """
        Get event count for each camera.
        
        Returns:
            Dict mapping camera_id to event count
        """
        self._check_fitted()
        return dict(sorted(self._camera_counts.items(), 
                          key=lambda x: x[1], reverse=True))
    
    def get_camera_percentages(self) -> Dict[str, float]:
        """
        Get percentage of events for each camera.
        
        Returns:
            Dict mapping camera_id to percentage (0-100)
        """
        self._check_fitted()
        total = sum(self._camera_counts.values())
        if total == 0:
            return {}
        return {cam: round(count / total * 100, 2) 
                for cam, count in sorted(self._camera_counts.items(),
                                         key=lambda x: x[1], reverse=True)}
    
    def get_hotspot_cameras(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get the cameras with most incidents (hotspots).
        
        Args:
            top_n: Number of top cameras to return
            
        Returns:
            List of camera dicts with id, name, count, percentage
        """
        self._check_fitted()
        
        total = sum(self._camera_counts.values())
        sorted_cameras = sorted(self._camera_counts.items(), 
                                key=lambda x: x[1], reverse=True)
        
        hotspots = []
        for cam_id, count in sorted_cameras[:top_n]:
            hotspots.append({
                "camera_id": cam_id,
                "camera_name": self._camera_names.get(cam_id, cam_id),
                "event_count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0,
            })
        
        return hotspots
    
    def get_safest_cameras(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get the cameras with fewest incidents.
        
        Args:
            top_n: Number of safest cameras to return
            
        Returns:
            List of camera dicts with id, name, count, percentage
        """
        self._check_fitted()
        
        total = sum(self._camera_counts.values())
        sorted_cameras = sorted(self._camera_counts.items(), 
                                key=lambda x: x[1])
        
        safest = []
        for cam_id, count in sorted_cameras[:top_n]:
            safest.append({
                "camera_id": cam_id,
                "camera_name": self._camera_names.get(cam_id, cam_id),
                "event_count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0,
            })
        
        return safest
    
    # ==================== Camera Statistics ====================
    
    def get_camera_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics for each camera.
        
        Returns:
            Dict mapping camera_id to stats dict
        """
        self._check_fitted()
        
        total_events = sum(self._camera_counts.values())
        stats = {}
        
        for cam_id in self._camera_counts:
            confidences = self._camera_confidences[cam_id]
            severities = dict(self._camera_severities[cam_id])
            
            stats[cam_id] = {
                "camera_name": self._camera_names.get(cam_id, cam_id),
                "event_count": self._camera_counts[cam_id],
                "percentage": round(self._camera_counts[cam_id] / total_events * 100, 2) if total_events > 0 else 0,
                "confidence": {
                    "mean": round(np.mean(confidences), 3),
                    "min": round(min(confidences), 3),
                    "max": round(max(confidences), 3),
                    "std": round(np.std(confidences), 3),
                },
                "severity_distribution": {
                    "Low": severities.get("Low", 0),
                    "Medium": severities.get("Medium", 0),
                    "High": severities.get("High", 0),
                },
            }
        
        return stats
    
    def get_risk_ranking(self) -> List[Dict[str, Any]]:
        """
        Get cameras ranked by risk level.
        
        Risk score is calculated based on:
        - Number of events (weight: 0.5)
        - Average confidence (weight: 0.3)
        - High severity percentage (weight: 0.2)
        
        Returns:
            List of cameras sorted by risk score (highest first)
        """
        self._check_fitted()
        
        stats = self.get_camera_stats()
        total_events = sum(self._camera_counts.values())
        
        rankings = []
        for cam_id, cam_stats in stats.items():
            # Normalize event count (0-1)
            event_score = cam_stats['event_count'] / max(self._camera_counts.values())
            
            # Confidence score (already 0-1)
            confidence_score = cam_stats['confidence']['mean']
            
            # High severity percentage
            total_cam_events = cam_stats['event_count']
            high_severity = cam_stats['severity_distribution']['High']
            severity_score = high_severity / total_cam_events if total_cam_events > 0 else 0
            
            # Weighted risk score
            risk_score = (
                0.5 * event_score + 
                0.3 * confidence_score + 
                0.2 * severity_score
            )
            
            rankings.append({
                "camera_id": cam_id,
                "camera_name": cam_stats['camera_name'],
                "risk_score": round(risk_score, 3),
                "event_count": cam_stats['event_count'],
                "avg_confidence": cam_stats['confidence']['mean'],
                "high_severity_pct": round(severity_score * 100, 1),
            })
        
        # Sort by risk score
        rankings.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return rankings
    
    # ==================== Confidence Analysis ====================
    
    def get_confidence_by_camera(self) -> Dict[str, Dict[str, float]]:
        """
        Get confidence score statistics for each camera.
        
        Returns:
            Dict mapping camera_id to confidence stats
        """
        self._check_fitted()
        
        result = {}
        for cam_id, confidences in self._camera_confidences.items():
            result[cam_id] = {
                "mean": round(np.mean(confidences), 3),
                "median": round(np.median(confidences), 3),
                "min": round(min(confidences), 3),
                "max": round(max(confidences), 3),
                "std": round(np.std(confidences), 3),
            }
        
        return result
    
    def get_overall_confidence_stats(self) -> Dict[str, float]:
        """
        Get overall confidence statistics across all events.
        
        Returns:
            Dict with mean, median, min, max, std
        """
        self._check_fitted()
        
        all_confidences = [e.confidence for e in self.events]
        
        return {
            "mean": round(np.mean(all_confidences), 3),
            "median": round(np.median(all_confidences), 3),
            "min": round(min(all_confidences), 3),
            "max": round(max(all_confidences), 3),
            "std": round(np.std(all_confidences), 3),
            "count": len(all_confidences),
        }
    
    # ==================== Summary ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive location pattern summary.
        
        Returns:
            Dict with all key insights
        """
        self._check_fitted()
        
        return {
            "total_events": len(self.events),
            "total_cameras": len(self._camera_counts),
            "hotspot_cameras": self.get_hotspot_cameras(3),
            "safest_cameras": self.get_safest_cameras(3),
            "risk_ranking": self.get_risk_ranking(),
            "camera_distribution": self.get_camera_percentages(),
            "overall_confidence": self.get_overall_confidence_stats(),
            "camera_stats": self.get_camera_stats(),
        }
    
    def print_report(self) -> None:
        """Print a human-readable analysis report."""
        self._check_fitted()
        
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("📍 LOCATION PATTERN ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n📊 Overview:")
        print(f"   Total Events: {summary['total_events']}")
        print(f"   Total Cameras: {summary['total_cameras']}")
        
        print(f"\n🔥 Hotspot Cameras (most incidents):")
        for cam in summary['hotspot_cameras']:
            print(f"   {cam['camera_name']}: {cam['event_count']} events ({cam['percentage']}%)")
        
        print(f"\n🛡️ Safest Cameras (fewest incidents):")
        for cam in summary['safest_cameras']:
            print(f"   {cam['camera_name']}: {cam['event_count']} events ({cam['percentage']}%)")
        
        print(f"\n⚠️ Risk Ranking:")
        for i, cam in enumerate(summary['risk_ranking'], 1):
            print(f"   {i}. {cam['camera_name']}: score={cam['risk_score']}, "
                  f"events={cam['event_count']}, high_severity={cam['high_severity_pct']}%")
        
        confidence = summary['overall_confidence']
        print(f"\n📈 Overall Confidence Stats:")
        print(f"   Mean: {confidence['mean']}")
        print(f"   Median: {confidence['median']}")
        print(f"   Range: {confidence['min']} - {confidence['max']}")
        print(f"   Std Dev: {confidence['std']}")
        
        print("\n" + "=" * 60)


# Quick test
if __name__ == "__main__":
    from insights.data import ViolenceEventGenerator
    
    print("Testing LocationAnalyzer...")
    
    # Generate mock data
    generator = ViolenceEventGenerator(seed=42)
    events = generator.generate(n_events=300)
    
    # Analyze
    analyzer = LocationAnalyzer()
    analyzer.fit(events)
    
    # Print report
    analyzer.print_report()
    
    # Get summary for API
    summary = analyzer.get_summary()
    print(f"\n✅ Summary keys: {list(summary.keys())}")
