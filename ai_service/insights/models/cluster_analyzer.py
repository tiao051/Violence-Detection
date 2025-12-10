"""
Cluster Pattern Analyzer

Uses K-means clustering to automatically discover patterns 
in violence detection events.

This is an actual ML model (unsupervised learning).
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from insights.data.event_schema import ViolenceEvent

# ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ClusterAnalyzer:
    """
    Uses K-means clustering to discover patterns in violence events.
    
    This is an ACTUAL ML MODEL that:
    1. Learns patterns from data (unsupervised)
    2. Groups similar events into clusters
    3. Describes each cluster as an insight
    
    Example:
        >>> analyzer = ClusterAnalyzer(n_clusters=3)
        >>> analyzer.fit(events)
        >>> 
        >>> # Get discovered patterns
        >>> patterns = analyzer.get_cluster_insights()
        >>> for p in patterns:
        ...     print(f"Pattern {p['cluster_id']}: {p['description']}")
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize cluster analyzer.
        
        Args:
            n_clusters: Number of clusters to find
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # ML components
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.camera_encoder = LabelEncoder()
        
        # Data
        self.events: List[ViolenceEvent] = []
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        
    def _prepare_features(self, events: List[ViolenceEvent]) -> np.ndarray:
        """
        Convert events to feature matrix for clustering.
        
        Features used:
        - hour (0-23)
        - day_of_week (0-6)
        - is_weekend (0/1)
        - time_period_encoded (0-3)
        - camera_encoded (0-n)
        - confidence (0-1)
        """
        # Encode cameras
        camera_ids = [e.camera_id for e in events]
        camera_encoded = self.camera_encoder.fit_transform(camera_ids)
        
        # Time period encoding
        period_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
        
        # Build feature matrix
        features = []
        for i, event in enumerate(events):
            features.append([
                event.hour,                           # 0-23
                event.day_of_week,                    # 0-6
                1 if event.is_weekend else 0,         # 0/1
                period_map[event.time_period],        # 0-3
                camera_encoded[i],                    # 0-n
                event.confidence,                     # 0-1
            ])
        
        return np.array(features)
    
    def fit(self, events: List[ViolenceEvent]) -> "ClusterAnalyzer":
        """
        Fit the clustering model on event data.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if len(events) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} events for {self.n_clusters} clusters")
        
        self.events = events
        
        # Prepare features
        self.features = self._prepare_features(events)
        
        # Scale features (important for K-means)
        features_scaled = self.scaler.fit_transform(self.features)
        
        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.labels = self.kmeans.fit_predict(features_scaled)
        
        self.is_fitted = True
        return self
    
    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster label for each event."""
        self._check_fitted()
        return self.labels
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get number of events in each cluster."""
        self._check_fitted()
        return dict(Counter(self.labels))
    
    def _analyze_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Analyze a single cluster to describe its pattern."""
        # Get events in this cluster
        cluster_mask = self.labels == cluster_id
        cluster_events = [e for e, m in zip(self.events, cluster_mask) if m]
        
        if not cluster_events:
            return {"cluster_id": cluster_id, "size": 0}
        
        # Analyze time patterns
        hours = [e.hour for e in cluster_events]
        avg_hour = np.mean(hours)
        
        days = [e.day_of_week for e in cluster_events]
        weekend_pct = sum(1 for e in cluster_events if e.is_weekend) / len(cluster_events)
        
        # Most common time period
        periods = Counter([e.time_period for e in cluster_events])
        top_period = periods.most_common(1)[0][0]
        
        # Most common camera
        cameras = Counter([e.camera_name for e in cluster_events])
        top_camera = cameras.most_common(1)[0][0]
        
        # Confidence stats
        confidences = [e.confidence for e in cluster_events]
        avg_confidence = np.mean(confidences)
        
        # Most common day
        day_names = Counter([e.day_name for e in cluster_events])
        top_day = day_names.most_common(1)[0][0]
        
        # Severity distribution
        severities = Counter([e.severity for e in cluster_events])
        high_severity_pct = severities.get("High", 0) / len(cluster_events)
        
        return {
            "cluster_id": cluster_id,
            "size": len(cluster_events),
            "percentage": round(len(cluster_events) / len(self.events) * 100, 1),
            "avg_hour": round(avg_hour, 1),
            "top_period": top_period,
            "top_camera": top_camera,
            "top_day": top_day,
            "weekend_pct": round(weekend_pct * 100, 1),
            "avg_confidence": round(avg_confidence, 3),
            "high_severity_pct": round(high_severity_pct * 100, 1),
        }
    
    def _generate_description(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable description for a cluster."""
        parts = []
        
        # Time description
        hour = analysis['avg_hour']
        if hour < 6:
            time_desc = "late night"
        elif hour < 12:
            time_desc = "morning"
        elif hour < 18:
            time_desc = "afternoon"
        else:
            time_desc = "evening/night"
        parts.append(f"{time_desc} hours")
        
        # Weekend
        if analysis['weekend_pct'] > 60:
            parts.append("mostly weekends")
        elif analysis['weekend_pct'] < 30:
            parts.append("mostly weekdays")
        
        # Location
        parts.append(f"near {analysis['top_camera']}")
        
        # Severity
        if analysis['high_severity_pct'] > 40:
            parts.append("HIGH severity")
        elif analysis['high_severity_pct'] < 20:
            parts.append("low severity")
        
        return ", ".join(parts)
    
    def get_cluster_insights(self) -> List[Dict[str, Any]]:
        """
        Get insights for all discovered clusters.
        
        Returns:
            List of cluster insight dicts with analysis and description
        """
        self._check_fitted()
        
        insights = []
        for cluster_id in range(self.n_clusters):
            analysis = self._analyze_cluster(cluster_id)
            analysis['description'] = self._generate_description(analysis)
            insights.append(analysis)
        
        # Sort by size (largest first)
        insights.sort(key=lambda x: x['size'], reverse=True)
        
        return insights
    
    def get_summary(self) -> Dict[str, Any]:
        """Get clustering summary for API."""
        self._check_fitted()
        
        insights = self.get_cluster_insights()
        
        return {
            "model": "K-means Clustering",
            "n_clusters": self.n_clusters,
            "total_events": len(self.events),
            "cluster_sizes": self.get_cluster_sizes(),
            "inertia": round(self.kmeans.inertia_, 2),  # Within-cluster sum of squares
            "patterns": insights,
        }
    
    def print_report(self) -> None:
        """Print human-readable clustering report."""
        self._check_fitted()
        
        print("\n" + "=" * 60)
        print("  CLUSTER ANALYSIS REPORT (K-means)")
        print("=" * 60)
        
        summary = self.get_summary()
        
        print(f"\nModel: {summary['model']}")
        print(f"Number of clusters: {summary['n_clusters']}")
        print(f"Total events: {summary['total_events']}")
        print(f"Inertia (lower = tighter clusters): {summary['inertia']}")
        
        print("\n" + "-" * 60)
        print("DISCOVERED PATTERNS:")
        print("-" * 60)
        
        for pattern in summary['patterns']:
            print(f"\n[Cluster {pattern['cluster_id']}] - {pattern['size']} events ({pattern['percentage']}%)")
            print(f"   Description: {pattern['description']}")
            print(f"   Avg Hour: {pattern['avg_hour']:.0f}:00")
            print(f"   Top Period: {pattern['top_period']}")
            print(f"   Top Camera: {pattern['top_camera']}")
            print(f"   Weekend %: {pattern['weekend_pct']}%")
            print(f"   Avg Confidence: {pattern['avg_confidence']}")
            print(f"   High Severity %: {pattern['high_severity_pct']}%")
        
        print("\n" + "=" * 60)


# Quick test
if __name__ == "__main__":
    from insights.data import ViolenceEventGenerator
    
    print("Testing ClusterAnalyzer (K-means)...")
    
    # Generate mock data
    generator = ViolenceEventGenerator(seed=42)
    events = generator.generate(n_events=300)
    
    # Cluster with 3 groups
    analyzer = ClusterAnalyzer(n_clusters=3)
    analyzer.fit(events)
    
    # Print report
    analyzer.print_report()
    
    print("\n[OK] ClusterAnalyzer test complete!")
    print("This is an ACTUAL ML MODEL using scikit-learn K-means!")
