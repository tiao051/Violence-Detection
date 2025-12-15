"""
K-means Clustering for Violence Event Pattern Discovery.

Groups similar events into K clusters to reveal temporal/spatial patterns.
Uses StandardScaler for feature normalization and scikit-learn's KMeans.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from insights.core.schema import ViolenceEvent

# ML imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ClusterAnalyzer:
    """
    Discovers patterns in violence events using K-means clustering.
    
    Example:
        analyzer = ClusterAnalyzer(n_clusters=3)
        analyzer.fit(events)
        patterns = analyzer.get_cluster_insights()
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
        """Convert events to feature matrix for clustering."""
        camera_ids = [e.camera_id for e in events]
        camera_encoded = self.camera_encoder.fit_transform(camera_ids)
        
        period_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
        
        features = []
        for i, event in enumerate(events):
            features.append([
                event.hour,
                event.day_of_week,
                1 if event.is_weekend else 0,
                period_map[event.time_period],
                camera_encoded[i],
                event.confidence,
            ])
        
        return np.array(features)
    
    def fit(self, events: List[ViolenceEvent]) -> "ClusterAnalyzer":
        """
        Fit clustering model on event data.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if len(events) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} events for {self.n_clusters} clusters")
        
        self.events = events
        self.features = self._prepare_features(events)
        features_scaled = self.scaler.fit_transform(self.features)
        
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
        cluster_mask = self.labels == cluster_id
        cluster_events = [e for e, m in zip(self.events, cluster_mask) if m]
        
        if not cluster_events:
            return {"cluster_id": cluster_id, "size": 0}
        
        hours = [e.hour for e in cluster_events]
        avg_hour = np.mean(hours)
        weekend_pct = sum(1 for e in cluster_events if e.is_weekend) / len(cluster_events)
        
        periods = Counter([e.time_period for e in cluster_events])
        top_period = periods.most_common(1)[0][0]
        
        cameras = Counter([e.camera_name for e in cluster_events])
        top_camera = cameras.most_common(1)[0][0]
        
        confidences = [e.confidence for e in cluster_events]
        avg_confidence = np.mean(confidences)
        
        day_names = Counter([e.day_name for e in cluster_events])
        top_day = day_names.most_common(1)[0][0]
        
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
        
        if analysis['weekend_pct'] > 60:
            parts.append("mostly weekends")
        elif analysis['weekend_pct'] < 30:
            parts.append("mostly weekdays")
        
        parts.append(f"near {analysis['top_camera']}")
        
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
    
    def predict_cluster(self, hour: int, day_of_week: int, camera: str, confidence: float = 0.8) -> Dict[str, Any]:
        """
        Predict which cluster a hypothetical event would belong to.
        
        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            camera: Camera name
            confidence: Confidence score (default 0.8)
            
        Returns:
            Dict with cluster_id, cluster_info, and risk_level
        """
        self._check_fitted()
        
        # Encode camera
        try:
            camera_encoded = self.camera_encoder.transform([camera])[0]
        except ValueError:
            # Unknown camera, use most common
            camera_encoded = 0
        
        # Build feature vector
        is_weekend = 1 if day_of_week >= 5 else 0
        
        if 6 <= hour < 12:
            period = 0  # Morning
        elif 12 <= hour < 18:
            period = 1  # Afternoon
        elif 18 <= hour < 22:
            period = 2  # Evening
        else:
            period = 3  # Night
        
        features = np.array([[hour, day_of_week, is_weekend, period, camera_encoded, confidence]])
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        cluster_id = self.kmeans.predict(features_scaled)[0]
        
        # Get cluster info
        cluster_info = self._analyze_cluster(cluster_id)
        cluster_info['description'] = self._generate_description(cluster_info)
        
        # Determine risk level
        high_sev_pct = cluster_info.get('high_severity_pct', 0)
        if high_sev_pct >= 70:
            risk_level = "HIGH"
        elif high_sev_pct >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "cluster_id": int(cluster_id),
            "cluster_info": cluster_info,
            "risk_level": risk_level,
            "high_severity_pct": high_sev_pct,
        }
    
    def forecast_next_hours(self, camera: str, current_hour: int, current_day: int, hours_ahead: int = 12) -> List[Dict[str, Any]]:
        """
        Forecast risk levels for the next N hours at a specific camera.
        
        Args:
            camera: Camera name
            current_hour: Current hour (0-23)
            current_day: Current day of week (0=Monday, 6=Sunday)
            hours_ahead: Number of hours to forecast (default 12)
            
        Returns:
            List of hourly forecasts with cluster and risk info
        """
        self._check_fitted()
        
        forecasts = []
        
        for h in range(hours_ahead):
            # Calculate future hour and day
            future_hour = (current_hour + h) % 24
            # Day changes when we pass midnight
            days_passed = (current_hour + h) // 24
            future_day = (current_day + days_passed) % 7
            
            # Predict for this hour
            prediction = self.predict_cluster(future_hour, future_day, camera)
            
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            forecasts.append({
                "hours_from_now": h,
                "hour": future_hour,
                "day": day_names[future_day],
                "is_weekend": future_day >= 5,
                "cluster_id": prediction["cluster_id"],
                "risk_level": prediction["risk_level"],
                "high_severity_pct": prediction["high_severity_pct"],
            })
        
        return forecasts
    
    def get_forecast_summary(self, camera: str, current_hour: int, current_day: int, hours_ahead: int = 12) -> Dict[str, Any]:
        """
        Get forecast summary with peak risk times.
        
        Returns formatted forecast with warnings and recommendations.
        """
        forecasts = self.forecast_next_hours(camera, current_hour, current_day, hours_ahead)
        
        # Find high risk windows
        high_risk_hours = [f for f in forecasts if f["risk_level"] == "HIGH"]
        peak_hour = max(forecasts, key=lambda x: x["high_severity_pct"]) if forecasts else None
        
        # Calculate risk by time window
        risk_by_window = {}
        for f in forecasts:
            window = f"hour_{f['hours_from_now']}"
            risk_by_window[window] = f["risk_level"]
        
        return {
            "camera": camera,
            "forecast_hours": hours_ahead,
            "current": {
                "hour": current_hour,
                "day_of_week": current_day,
            },
            "forecasts": forecasts,
            "summary": {
                "high_risk_count": len(high_risk_hours),
                "peak_hour": peak_hour,
                "next_high_risk": high_risk_hours[0] if high_risk_hours else None,
            },
            "warning": f"⚠️ {len(high_risk_hours)} high-risk hours in next {hours_ahead}h" if high_risk_hours else "✅ No high-risk periods detected",
        }
