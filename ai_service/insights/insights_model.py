"""
Violence Insights Model (Unified)

Combines all 3 ML models into a single interface:
- ClusterAnalyzer (K-means)
- AssociationRuleAnalyzer (FP-Growth)
- RiskPredictor (Random Forest)

Usage:
    from ai_service.insights import InsightsModel
    
    # Train and save
    model = InsightsModel()
    model.fit(events)
    model.save("insights_model.pkl")
    
    # Load and use
    model = InsightsModel.load("insights_model.pkl")
    report = model.get_full_report()
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import joblib
import os

from .data import ViolenceEventGenerator, ViolenceEvent
from .models import ClusterAnalyzer, AssociationRuleAnalyzer, RiskPredictor


class InsightsModel:
    """
    Unified ML model for violence event insights.
    
    Combines 3 ML algorithms:
    1. K-means Clustering - discovers patterns/groups
    2. FP-Growth - finds association rules
    3. Random Forest - predicts risk levels
    
    Example:
        >>> model = InsightsModel()
        >>> model.fit(events)
        >>> 
        >>> # Get all insights
        >>> report = model.get_full_report()
        >>> 
        >>> # Or use individual models
        >>> patterns = model.get_patterns()
        >>> rules = model.get_rules()
        >>> prediction = model.predict(hour=20, day="Saturday", camera="Parking Lot")
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
        Train all 3 models on the event data.
        
        Args:
            events: List of ViolenceEvent instances
            
        Returns:
            self for method chaining
        """
        if len(events) < 50:
            raise ValueError("Need at least 50 events for training")
        
        self.events = events
        
        # Train all models
        print("Training K-means Clustering...")
        self.cluster_model.fit(events)
        
        print("Training FP-Growth Association Rules...")
        self.association_model.fit(events)
        
        print("Training Random Forest Predictor...")
        self.prediction_model.fit(events)
        
        self.is_fitted = True
        self.fit_time = datetime.now()
        
        print(f"All models trained on {len(events)} events!")
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
        from datetime import timedelta
        
        generator = ViolenceEventGenerator(seed=42)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        events = generator.generate(n_events=n_events, start_date=start_date, end_date=end_date)
        return self.fit(events)
    
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
    
    # ==================== Clustering ====================
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get discovered patterns from K-means clustering."""
        self._check_fitted()
        return self.cluster_model.get_cluster_insights()
    
    # ==================== Association Rules ====================
    
    def get_rules(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top association rules from FP-Growth."""
        self._check_fitted()
        return self.association_model.get_rules(top_n=top_n)
    
    def get_rules_for_severity(self, severity: str = "High", top_n: int = 5) -> List[Dict[str, Any]]:
        """Get rules leading to specific severity level."""
        self._check_fitted()
        return self.association_model.get_rules_for_target(f"severity_{severity}", top_n=top_n)
    
    # ==================== Prediction ====================
    
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
    
    # ==================== Combined Reports ====================
    
    def get_full_report(self) -> Dict[str, Any]:
        """
        Get comprehensive report from all 3 models.
        
        Returns:
            Dict containing insights from all models
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
            "total_events": len(self.events),
            "patterns_discovered": len(patterns),
            "rules_discovered": len(self.association_model.rules) if self.association_model.rules is not None else 0,
            "prediction_accuracy": round(self.prediction_model.accuracy, 3),
            "top_pattern": patterns[0] if patterns else None,
            "top_rule": rules[0] if rules else None,
            "highest_risk": high_risk[0] if high_risk else None,
        }
    
    def print_report(self) -> None:
        """Print human-readable summary report."""
        self._check_fitted()
        
        print("\n" + "=" * 70)
        print("  VIOLENCE INSIGHTS MODEL - FULL REPORT")
        print("=" * 70)
        
        print(f"\nTotal events analyzed: {len(self.events)}")
        print(f"Models: K-means, FP-Growth, Random Forest")
        
        # Clustering
        print("\n" + "-" * 70)
        print("PATTERNS (K-means Clustering)")
        print("-" * 70)
        for i, p in enumerate(self.get_patterns(), 1):
            print(f"  Cluster {i}: {p['size']} events - {p['description']}")
        
        # Association Rules
        print("\n" + "-" * 70)
        print("ASSOCIATION RULES (FP-Growth)")
        print("-" * 70)
        for i, r in enumerate(self.get_rules(5), 1):
            print(f"  {i}. {r['rule_str']} (conf={r['confidence']:.0%})")
        
        # Prediction
        print("\n" + "-" * 70)
        print("RISK PREDICTION (Random Forest)")
        print("-" * 70)
        print(f"  Accuracy: {self.prediction_model.accuracy:.1%}")
        print("  High-risk conditions:")
        for c in self.get_high_risk_conditions(3):
            print(f"    - {c['day']} {c['hour']:02d}:00 at {c['camera']}: {c['high_prob']:.0%}")
        
        print("\n" + "=" * 70)
    
    # ==================== Save / Load ====================
    
    def save(self, filepath: str) -> str:
        """
        Save the trained model to a file.
        
        This saves all 3 trained ML models (K-means, FP-Growth, Random Forest)
        along with metadata, so you can reload them later without retraining.
        
        Args:
            filepath: Path to save the model (e.g., "insights_model.pkl")
            
        Returns:
            Absolute path to the saved file
            
        Example:
            >>> model = InsightsModel()
            >>> model.fit(events)
            >>> model.save("trained_insights_model.pkl")
            'C:/path/to/trained_insights_model.pkl'
        """
        self._check_fitted()
        
        # Create save data dictionary
        save_data = {
            "version": "1.0",
            "fit_time": self.fit_time.isoformat() if self.fit_time else None,
            "n_events": len(self.events),
            
            # Model instances (these contain trained sklearn/mlxtend objects)
            "cluster_model": self.cluster_model,
            "association_model": self.association_model,
            "prediction_model": self.prediction_model,
            
            # Model parameters (for reference)
            "params": {
                "n_clusters": self.cluster_model.n_clusters,
                "min_support": self.association_model.min_support,
                "min_confidence": self.association_model.min_confidence,
                "n_estimators": self.prediction_model.n_estimators,
            }
        }
        
        # Save using joblib (efficient for sklearn models)
        abs_path = os.path.abspath(filepath)
        joblib.dump(save_data, abs_path)
        
        print(f"Model saved to: {abs_path}")
        return abs_path
    
    @classmethod
    def load(cls, filepath: str) -> "InsightsModel":
        """
        Load a trained model from a file.
        
        This restores all 3 trained ML models so you can use them
        immediately without retraining.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            InsightsModel instance with trained models
            
        Example:
            >>> model = InsightsModel.load("trained_insights_model.pkl")
            >>> prediction = model.predict(hour=20, day="Saturday", camera="Parking Lot")
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load saved data
        save_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls()
        
        # Restore models
        instance.cluster_model = save_data["cluster_model"]
        instance.association_model = save_data["association_model"]
        instance.prediction_model = save_data["prediction_model"]
        
        # Restore metadata
        instance.is_fitted = True
        instance.fit_time = datetime.fromisoformat(save_data["fit_time"]) if save_data["fit_time"] else None
        instance.events = []  # Events are not saved (too large)
        
        print(f"Model loaded from: {os.path.abspath(filepath)}")
        print(f"  - Originally trained on {save_data['n_events']} events")
        print(f"  - Trained at: {save_data['fit_time']}")
        
        return instance


# For easy import
__all__ = ["InsightsModel"]
