"""
Random Forest for Violence Risk Level Prediction.

Predicts severity level (High/Medium/Low) given temporal/spatial conditions.
Returns risk level, probability, and comparison vs baseline distribution.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from insights.core.schema import ViolenceEvent

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class RiskPredictor:
    """
    Random Forest model for violence risk level prediction.
    
    Given conditions (hour, day, camera), predicts severity and probability.
    
    Example:
        predictor = RiskPredictor()
        predictor.fit(events)
        result = predictor.predict(hour=20, day="Saturday", camera="Parking Lot")
    """
    
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    PERIODS = ["Morning", "Afternoon", "Evening", "Night"]
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize the predictor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Model components
        self.model: Optional[RandomForestClassifier] = None
        self.camera_encoder = LabelEncoder()
        self.day_encoder = LabelEncoder()
        self.period_encoder = LabelEncoder()
        
        # Baseline stats
        self.baseline_probs: Dict[str, float] = {}
        self.severity_counts: Dict[str, int] = {}
        
        # Fitted state
        self.cameras: List[str] = []
        self.is_fitted: bool = False
        self.accuracy: float = 0.0
        
    def _prepare_features(self, events: List[ViolenceEvent]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels from events."""
        self.cameras = list(set(e.camera_name for e in events))
        
        self.camera_encoder.fit(self.cameras)
        self.day_encoder.fit(self.DAYS)
        self.period_encoder.fit(self.PERIODS)
        
        X = []
        y = []
        
        for event in events:
            features = [
                event.hour,
                event.day_of_week,
                1 if event.is_weekend else 0,
                self.period_encoder.transform([event.time_period])[0],
                self.camera_encoder.transform([event.camera_name])[0],
            ]
            X.append(features)
            y.append(event.severity)
        
        return np.array(X), np.array(y)
    
    def fit(self, events: List[ViolenceEvent], test_size: float = 0.2) -> "RiskPredictor":
        """
        Train the Random Forest model.
        
        Args:
            events: List of ViolenceEvent instances
            test_size: Fraction of data for testing
            
        Returns:
            self for method chaining
        """
        if len(events) < 50:
            raise ValueError("Need at least 50 events for training")
        
        severities = [e.severity for e in events]
        self.severity_counts = dict(Counter(severities))
        total = len(severities)
        self.baseline_probs = {
            sev: count / total for sev, count in self.severity_counts.items()
        }
        
        X, y = self._prepare_features(events)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        self.is_fitted = True
        return self
    
    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period."""
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Night"
    
    def predict(
        self,
        hour: int,
        day: str,
        camera: str,
    ) -> Dict[str, Any]:
        """
        Predict risk level for given conditions.
        
        Args:
            hour: Hour of day (0-23)
            day: Day name (Monday, Tuesday, etc.)
            camera: Camera name
            
        Returns:
            Dict with risk_level, probabilities, comparison vs baseline
        """
        self._check_fitted()
        
        if day not in self.DAYS:
            raise ValueError(f"Invalid day: {day}. Must be one of {self.DAYS}")
        if camera not in self.cameras:
            raise ValueError(f"Unknown camera: {camera}. Known: {self.cameras}")
        
        day_of_week = self.DAYS.index(day)
        is_weekend = 1 if day_of_week >= 5 else 0
        time_period = self._get_time_period(hour)
        
        features = np.array([[
            hour,
            day_of_week,
            is_weekend,
            self.period_encoder.transform([time_period])[0],
            self.camera_encoder.transform([camera])[0],
        ]])
        
        risk_level = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        prob_dict = {}
        for i, cls in enumerate(self.model.classes_):
            prob_dict[cls] = round(probabilities[i], 3)
        
        baseline_prob = self.baseline_probs.get(risk_level, 0)
        predicted_prob = prob_dict.get(risk_level, 0)
        
        if baseline_prob > 0:
            change_pct = ((predicted_prob - baseline_prob) / baseline_prob) * 100
        else:
            change_pct = 0
        
        if change_pct > 0:
            change_str = f"+{change_pct:.0f}% vs average"
        elif change_pct < 0:
            change_str = f"{change_pct:.0f}% vs average"
        else:
            change_str = "same as average"
        
        return {
            "hour": hour,
            "day": day,
            "camera": camera,
            "time_period": time_period,
            "risk_level": risk_level,
            "probabilities": prob_dict,
            "change_vs_avg": change_str,
            "change_pct": round(change_pct, 1),
            "insight": f"{day} {hour:02d}:00 at {camera} → {risk_level} risk ({change_str})",
        }
    
    def predict_batch(
        self,
        conditions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Predict risk for multiple conditions.
        
        Args:
            conditions: List of dicts with hour, day, camera
            
        Returns:
            List of prediction results
        """
        results = []
        for cond in conditions:
            try:
                result = self.predict(
                    hour=cond['hour'],
                    day=cond['day'],
                    camera=cond['camera'],
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), **cond})
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get importance of each feature.
        
        Returns:
            Dict mapping feature name to importance score
        """
        self._check_fitted()
        
        feature_names = ["hour", "day_of_week", "is_weekend", "time_period", "camera"]
        importances = self.model.feature_importances_
        
        return {name: round(imp, 4) for name, imp in zip(feature_names, importances)}
    
    def get_high_risk_conditions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Find conditions with highest High-severity probability.
        
        Args:
            top_n: Number of top conditions to return
            
        Returns:
            List of high-risk conditions
        """
        self._check_fitted()
        
        results = []
        
        # Check all combinations
        for hour in range(24):
            for day in self.DAYS:
                for camera in self.cameras:
                    pred = self.predict(hour, day, camera)
                    high_prob = pred['probabilities'].get('High', 0)
                    results.append({
                        "hour": hour,
                        "day": day,
                        "camera": camera,
                        "high_prob": high_prob,
                        "risk_level": pred['risk_level'],
                    })
        
        # Sort by high probability
        results.sort(key=lambda x: x['high_prob'], reverse=True)
        
        return results[:top_n]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary for API."""
        self._check_fitted()
        
        return {
            "model": "Random Forest Classifier",
            "parameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
            },
            "accuracy": round(self.accuracy, 3),
            "baseline_probabilities": self.baseline_probs,
            "feature_importance": self.get_feature_importance(),
            "known_cameras": self.cameras,
            "high_risk_conditions": self.get_high_risk_conditions(5),
        }
    
