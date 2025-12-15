"""
Unified Violence Insights Model.

Combines K-means Clustering, FP-Growth Association Rules, and Random Forest Prediction.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import joblib
import os

from .schema import ViolenceEvent
from ..data import ViolenceEventGenerator
from ..algorithms import ClusterAnalyzer, AssociationRuleAnalyzer, RiskPredictor


class InsightsModel:
    """
    Unified ML model combining K-means, FP-Growth, and Random Forest.
    Provides pattern discovery, association rules, and risk prediction.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        min_support: float = 0.05,
        min_confidence: float = 0.5,
        n_estimators: int = 100,
    ):
        """
        Initialize the unified model.
        
        Args:
            n_clusters: Number of clusters for K-means
            min_support: Minimum support for FP-Growth
            min_confidence: Minimum confidence for association rules
            n_estimators: Number of trees for Random Forest
        """
        self.cluster_model = ClusterAnalyzer(n_clusters=n_clusters)
        self.association_model = AssociationRuleAnalyzer(
            min_support=min_support, 
            min_confidence=min_confidence
        )
        self.prediction_model = RiskPredictor(n_estimators=n_estimators)
        
        self.events: List[ViolenceEvent] = []
        self.is_fitted: bool = False
        self.fit_time: Optional[datetime] = None
    
    def fit(self, events: List[ViolenceEvent]) -> "InsightsModel":
        """
        Train all models on the event data.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if len(events) < 50:
            raise ValueError("Need at least 50 events for training")
        
        self.events = events
        
        print("Training K-means Clustering...")
        self.cluster_model.fit(events)
        
        print("Training FP-Growth Association Rules...")
        self.association_model.fit(events)
        
        print("Training Random Forest Predictor...")
        self.prediction_model.fit(events)
        
        self.is_fitted = True
        self.fit_time = datetime.now()
        return self
    
    def fit_from_mock(self, n_events: int = 500, days: int = 60) -> "InsightsModel":
        """
        Train using mock data (for demo/testing).
        
        Args:
            n_events: Number of mock events to generate
            days: Number of days to span
            
        Returns:
            self for method chaining
        """
        generator = ViolenceEventGenerator(seed=42)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        events = generator.generate(n_events=n_events, start_date=start_date, end_date=end_date)
        return self.fit(events)
    
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get discovered patterns from K-means clustering."""
        self._check_fitted()
        return self.cluster_model.get_cluster_insights()
    
    def get_rules(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top association rules from FP-Growth."""
        self._check_fitted()
        return self.association_model.get_rules(top_n=top_n)
    
    def get_rules_for_severity(self, severity: str = "High", top_n: int = 5) -> List[Dict[str, Any]]:
        """Get rules leading to specific severity level."""
        self._check_fitted()
        return self.association_model.get_rules_for_target(f"severity_{severity}", top_n=top_n)
    
    def predict(self, hour: int, day: str, camera: str) -> Dict[str, Any]:
        """
        Predict risk level for given conditions.
        
        Args:
            hour: Hour (0-23)
            day: Day name (Monday, Tuesday, etc.)
            camera: Camera name
            
        Returns:
            Dict with risk_level, probabilities, change vs average
        """
        self._check_fitted()
        return self.prediction_model.predict(hour=hour, day=day, camera=camera)
    
    def get_high_risk_conditions(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get conditions with highest risk."""
        self._check_fitted()
        return self.prediction_model.get_high_risk_conditions(top_n=top_n)
    
    def get_forecast(self, camera: str, hours_ahead: int = 12) -> Dict[str, Any]:
        """
        Get hotspot forecast for a camera.
        
        Args:
            camera: Camera name
            hours_ahead: Number of hours to forecast (default 12)
            
        Returns:
            Forecast with hourly predictions and summary
        """
        self._check_fitted()
        now = datetime.now()
        return self.cluster_model.get_forecast_summary(
            camera=camera,
            current_hour=now.hour,
            current_day=now.weekday(),
            hours_ahead=hours_ahead
        )
    
    def get_full_report(self) -> Dict[str, Any]:
        """
        Get comprehensive report from all models.
        
        Returns:
            Dict containing insights from all 3 models
        """
        self._check_fitted()
        
        return {
            "metadata": {
                "total_events": len(self.events),
                "fit_time": self.fit_time.isoformat() if self.fit_time else None,
                "models": ["K-means", "FP-Growth", "Random Forest"],
            },
            "clustering": {
                "algorithm": "K-means",
                "n_clusters": self.cluster_model.n_clusters,
                "patterns": self.get_patterns(),
            },
            "association_rules": {
                "algorithm": "FP-Growth",
                "top_rules": self.get_rules(10),
                "high_severity_rules": self.get_rules_for_severity("High", 5),
            },
            "prediction": {
                "algorithm": "Random Forest",
                "accuracy": self.prediction_model.accuracy,
                "feature_importance": self.prediction_model.get_feature_importance(),
                "high_risk_conditions": self.get_high_risk_conditions(5),
            },
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary of key insights."""
        self._check_fitted()
        
        patterns = self.get_patterns()
        rules = self.get_rules(3)
        high_risk = self.get_high_risk_conditions(3)
        
        return {
            "total_events": len(self.events) if self.events else getattr(self, '_n_events', 0),
            "patterns_discovered": len(patterns),
            "rules_discovered": len(self.association_model.rules) if self.association_model.rules is not None else 0,
            "prediction_accuracy": round(self.prediction_model.accuracy, 3),
            "top_pattern": patterns[0] if patterns else None,
            "top_rule": rules[0] if rules else None,
            "highest_risk": high_risk[0] if high_risk else None,
        }
    
    
    def save(self, filepath: str) -> str:
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model file
            
        Returns:
            Absolute path to the saved file
        """
        self._check_fitted()
        
        save_data = {
            "version": "1.0",
            "fit_time": self.fit_time.isoformat() if self.fit_time else None,
            "n_events": len(self.events),
            "cluster_model": self.cluster_model,
            "association_model": self.association_model,
            "prediction_model": self.prediction_model,
            "params": {
                "n_clusters": self.cluster_model.n_clusters,
                "min_support": self.association_model.min_support,
                "min_confidence": self.association_model.min_confidence,
                "n_estimators": self.prediction_model.n_estimators,
            }
        }
        
        abs_path = os.path.abspath(filepath)
        joblib.dump(save_data, abs_path)
        print(f"Model saved to: {abs_path}")
        return abs_path
    
    @classmethod
    def load(cls, filepath: str) -> "InsightsModel":
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            InsightsModel instance with trained models
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        instance = cls()
        instance.cluster_model = save_data["cluster_model"]
        instance.association_model = save_data["association_model"]
        instance.prediction_model = save_data["prediction_model"]
        
        instance.is_fitted = True
        instance.fit_time = datetime.fromisoformat(save_data["fit_time"]) if save_data["fit_time"] else None
        instance.events = []
        instance._n_events = save_data.get("n_events", 0)
        
        print(f"Model loaded from: {os.path.abspath(filepath)}")
        print(f"  Trained on {save_data['n_events']} events")
        
        return instance


__all__ = ["InsightsModel"]
