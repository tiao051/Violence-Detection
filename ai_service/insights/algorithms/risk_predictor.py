"""
Random Forest for Violence Risk Level Prediction.

Predicts severity level given temporal/spatial conditions.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
import pandas as pd

from . import BaseAnalyzer
from ..time_utils import DAYS, PERIODS, categorize_hour
from insights.core.schema import ViolenceEvent

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RiskPredictor(BaseAnalyzer):
    """Random Forest model for violence risk prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model: Optional[RandomForestClassifier] = None
        self.camera_encoder = LabelEncoder()
        self.day_encoder = LabelEncoder()
        self.period_encoder = LabelEncoder()
        
        self.baseline_probs: Dict[str, float] = {}
        self.severity_counts: Dict[str, int] = {}
        self.cameras: List[str] = []
        self.accuracy: float = 0.0
        
    def _prepare_features(self, events: List[ViolenceEvent]) -> Tuple[np.ndarray, np.ndarray]:
        self.cameras = list(set(e.camera_name for e in events))
        
        self.camera_encoder.fit(self.cameras)
        self.day_encoder.fit(DAYS)
        self.period_encoder.fit(PERIODS)
        
        X = [
            [
                event.hour,
                event.day_of_week,
                1 if event.is_weekend else 0,
                self.period_encoder.transform([event.time_period])[0],
                self.camera_encoder.transform([event.camera_name])[0],
            ]
            for event in events
        ]
        y = [event.severity for event in events]
        
        return np.array(X), np.array(y)
    
    def fit(self, events: List[ViolenceEvent], test_size: float = 0.2) -> "RiskPredictor":
        if len(events) < 50:
            raise ValueError("Need at least 50 events for training")
        
        severities = [e.severity for e in events]
        self.severity_counts = dict(Counter(severities))
        total = len(severities)
        self.baseline_probs = {sev: count / total for sev, count in self.severity_counts.items()}
        
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
    
    def predict(self, hour: int, day: str, camera: str) -> Dict[str, Any]:
        self._check_fitted()
        
        if day not in DAYS:
            raise ValueError(f"Invalid day: {day}. Must be one of {DAYS}")
        if camera not in self.cameras:
            raise ValueError(f"Unknown camera: {camera}. Known: {self.cameras}")
        
        day_of_week = DAYS.index(day)
        is_weekend = 1 if day_of_week >= 5 else 0
        time_period = categorize_hour(hour).capitalize()
        
        features = np.array([[
            hour, day_of_week, is_weekend,
            self.period_encoder.transform([time_period])[0],
            self.camera_encoder.transform([camera])[0],
        ]])
        
        risk_level = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        prob_dict = {cls: round(probabilities[i], 3) for i, cls in enumerate(self.model.classes_)}
        
        baseline_prob = self.baseline_probs.get(risk_level, 0)
        predicted_prob = prob_dict.get(risk_level, 0)
        
        change_pct = ((predicted_prob - baseline_prob) / baseline_prob * 100) if baseline_prob > 0 else 0
        change_str = f"+{change_pct:.0f}% vs average" if change_pct > 0 else f"{change_pct:.0f}% vs average" if change_pct < 0 else "same as average"
        
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
    
    def predict_batch(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for cond in conditions:
            try:
                result = self.predict(hour=cond['hour'], day=cond['day'], camera=cond['camera'])
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), **cond})
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        self._check_fitted()
        feature_names = ["hour", "day_of_week", "is_weekend", "time_period", "camera"]
        importances = self.model.feature_importances_
        return {name: round(imp, 4) for name, imp in zip(feature_names, importances)}
    
    def get_high_risk_conditions(self, top_n: int = 10) -> List[Dict[str, Any]]:
        self._check_fitted()
        
        results = []
        for hour in range(24):
            for day in DAYS:
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
        
        results.sort(key=lambda x: x['high_prob'], reverse=True)
        return results[:top_n]
    
    def get_summary(self) -> Dict[str, Any]:
        self._check_fitted()
        
        return {
            "model": "Random Forest Classifier",
            "parameters": {"n_estimators": self.n_estimators, "max_depth": self.max_depth},
            "accuracy": round(self.accuracy, 3),
            "baseline_probabilities": self.baseline_probs,
            "feature_importance": self.get_feature_importance(),
            "known_cameras": self.cameras,
            "high_risk_conditions": self.get_high_risk_conditions(5),
        }
    
