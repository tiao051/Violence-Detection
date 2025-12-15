"""
Risk Prediction Model - Random Forest for Violence Risk Assessment

================================================================================
WHAT IS RANDOM FOREST?
================================================================================
Random Forest is a supervised machine learning algorithm that builds multiple
decision trees and combines their predictions (ensemble learning). It's called
"forest" because it creates many trees, and "random" because each tree is trained
on a random subset of data and features.

HOW IT WORKS (simplified):
1. Create N decision trees (we use 100 trees)
2. Each tree is trained on a random sample of the data (with replacement)
3. Each tree also uses a random subset of features at each split
4. For prediction, each tree "votes" and the majority vote wins
5. We also get probability estimates based on voting percentages

WHY RANDOM FOREST FOR RISK PREDICTION?
- Robust: Multiple trees reduce overfitting
- Handles non-linear relationships well
- Works with both categorical and numerical features
- Provides feature importance (which factors matter most)
- Returns probability estimates (not just class labels)

WHAT ARE WE PREDICTING?
- Target: Severity level (High / Medium / Low)
- Given: Hour, Day of week, Camera location
- Output: Predicted severity + probability + comparison vs average

FEATURES USED:
- Hour (0-23): Time of day
- Day of week (0-6): Which day (Monday=0, Sunday=6)
- Is weekend (0/1): Weekend or weekday
- Time period (encoded): Morning/Afternoon/Evening/Night
- Camera (encoded): Location of the camera

EXAMPLE OUTPUT:
  "Saturday 20:00 at Parking Lot → High risk (+27% vs average)"
  This means:
  - The model predicts HIGH severity
  - The probability is 27% HIGHER than the baseline average

KEY METRICS:
- Accuracy: How often the model predicts correctly (e.g., 65%)
- Feature Importance: Which features matter most for prediction
  - Higher importance = feature is more predictive

PARAMETERS:
- n_estimators: Number of trees (more trees = more stable but slower)
- max_depth: Maximum depth of each tree (deeper = more complex patterns)

TECHNICAL NOTES:
- Uses scikit-learn's RandomForestClassifier
- Uses stratified train/test split (preserves class distribution)
- LabelEncoder for categorical features (day, camera, period)
================================================================================
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
    Random Forest model to predict violence risk levels.
    
    Given conditions (hour, day, camera), predicts:
    - Risk level (High/Medium/Low)
    - Probability of each level
    - Comparison vs baseline (e.g., "+25% higher than average")
    
    Example:
        >>> predictor = RiskPredictor()
        >>> predictor.fit(events)
        >>> 
        >>> # Predict risk
        >>> result = predictor.predict(hour=20, day="Saturday", camera="Parking Lot")
        >>> print(f"Risk: {result['risk_level']} ({result['change_vs_avg']})")
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
        """
        Prepare feature matrix and labels from events.
        
        Features:
        - hour (0-23)
        - day_of_week_encoded (0-6)
        - is_weekend (0/1)
        - time_period_encoded (0-3)
        - camera_encoded (0-n)
        
        Labels:
        - severity (Low/Medium/High)
        """
        # Collect unique values
        self.cameras = list(set(e.camera_name for e in events))
        
        # Fit encoders
        self.camera_encoder.fit(self.cameras)
        self.day_encoder.fit(self.DAYS)
        self.period_encoder.fit(self.PERIODS)
        
        # Build feature matrix
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
        
        # Calculate baseline probabilities
        severities = [e.severity for e in events]
        self.severity_counts = dict(Counter(severities))
        total = len(severities)
        self.baseline_probs = {
            sev: count / total for sev, count in self.severity_counts.items()
        }
        
        # Prepare data
        X, y = self._prepare_features(events)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
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
        
        # Validate inputs
        if day not in self.DAYS:
            raise ValueError(f"Invalid day: {day}. Must be one of {self.DAYS}")
        if camera not in self.cameras:
            raise ValueError(f"Unknown camera: {camera}. Known: {self.cameras}")
        
        # Prepare features
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
        
        # Predict
        risk_level = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Create probability dict
        prob_dict = {}
        for i, cls in enumerate(self.model.classes_):
            prob_dict[cls] = round(probabilities[i], 3)
        
        # Calculate change vs baseline
        baseline_prob = self.baseline_probs.get(risk_level, 0)
        predicted_prob = prob_dict.get(risk_level, 0)
        
        if baseline_prob > 0:
            change_pct = ((predicted_prob - baseline_prob) / baseline_prob) * 100
        else:
            change_pct = 0
        
        # Format change string
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
    
    def print_report(self) -> None:
        """Print human-readable model report."""
        self._check_fitted()
        
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("  RISK PREDICTION MODEL REPORT (Random Forest)")
        print("=" * 70)
        
        print(f"\nModel: {summary['model']}")
        print(f"Accuracy: {summary['accuracy']:.1%}")
        print(f"Trees: {summary['parameters']['n_estimators']}")
        print(f"Max Depth: {summary['parameters']['max_depth']}")
        
        print("\n" + "-" * 70)
        print("BASELINE SEVERITY DISTRIBUTION:")
        print("-" * 70)
        for sev, prob in summary['baseline_probabilities'].items():
            print(f"   {sev}: {prob:.1%}")
        
        print("\n" + "-" * 70)
        print("FEATURE IMPORTANCE:")
        print("-" * 70)
        for feat, imp in sorted(summary['feature_importance'].items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 50)
            print(f"   {feat:<15} {bar} {imp:.2%}")
        
        print("\n" + "-" * 70)
        print("TOP HIGH-RISK CONDITIONS:")
        print("-" * 70)
        for i, cond in enumerate(summary['high_risk_conditions'], 1):
            print(f"   {i}. {cond['day']} {cond['hour']:02d}:00 at {cond['camera']}")
            print(f"      High probability: {cond['high_prob']:.1%}")
        
        print("\n" + "=" * 70)


# Quick test
if __name__ == "__main__":
    from insights.data import ViolenceEventGenerator
    
    print("Testing RiskPredictor (Random Forest)...")
    
    # Generate mock data
    generator = ViolenceEventGenerator(seed=42)
    events = generator.generate(n_events=500)
    
    # Train model
    predictor = RiskPredictor(n_estimators=100, max_depth=10)
    predictor.fit(events)
    
    # Print report
    predictor.print_report()
    
    # Example predictions
    print("\nExample Predictions:")
    print("-" * 50)
    
    test_cases = [
        {"hour": 20, "day": "Saturday", "camera": "Parking Lot"},
        {"hour": 8, "day": "Monday", "camera": "Front Gate"},
        {"hour": 23, "day": "Friday", "camera": "Back Yard"},
    ]
    
    for case in test_cases:
        result = predictor.predict(**case)
        print(f"   {result['insight']}")
    
    print("\n[OK] RiskPredictor test complete!")
    print("This uses Random Forest from scikit-learn!")
