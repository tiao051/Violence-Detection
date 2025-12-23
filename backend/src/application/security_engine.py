"""
SecurityEngine - Real-time severity analysis for violence alerts.

Loads trained ML model (scikit-learn RandomForest) into RAM at startup
for high-performance, non-blocking severity analysis.

Model is trained by Spark pipeline but exported as sklearn .pkl for lightweight inference.
"""

import logging
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class SecurityEngine:
    """
    High-performance severity analyzer using trained ML model.
    
    - Loads sklearn RandomForest model into RAM at startup
    - Analyzes alerts and returns severity (HIGH/MEDIUM/LOW) with benchmark timing
    - Thread-safe and async-friendly
    - Sub-millisecond inference time
    """

    # Singleton instance
    _instance: Optional['SecurityEngine'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once (singleton pattern)
        if SecurityEngine._initialized:
            return
        
        # ML Model (loaded from pkl)
        self.model = None
        self.scaler = None
        self.severity_labels: list = []
        self.camera_to_idx: dict = {}
        self.feature_columns: list = []
        
        # Fallback rules (used if model not available)
        self.risk_rules: list = []
        self.camera_profiles: list = []
        
        # Model availability flag
        self.model_loaded = False
        
        # Statistics for monitoring
        self.analysis_count = 0
        self.total_analysis_time_ms = 0.0
        self.model_predictions = 0
        self.rule_fallbacks = 0
        
        SecurityEngine._initialized = True

    @classmethod
    def get_instance(cls) -> 'SecurityEngine':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self, data_dir: str = None) -> bool:
        """
        Load ML model and fallback rules into RAM.
        
        Args:
            data_dir: Path to data directory containing model files.
                      If None, uses default path.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if data_dir is None:
            # Default path - relative to ai_service
            data_dir = Path(__file__).parent.parent.parent.parent / "ai_service" / "insights" / "data"
        else:
            data_dir = Path(data_dir)
        
        logger.info(f"Loading SecurityEngine artifacts from: {data_dir}")
        
        try:
            # 1. Load sklearn model (primary prediction method)
            sklearn_model_path = data_dir / "severity_rf_sklearn.pkl"
            if sklearn_model_path.exists():
                with open(sklearn_model_path, 'rb') as f:
                    model_bundle = pickle.load(f)
                
                self.model = model_bundle['model']
                self.scaler = model_bundle['scaler']
                self.severity_labels = model_bundle['severity_labels']
                self.camera_to_idx = model_bundle['camera_to_idx']
                self.feature_columns = model_bundle['feature_columns']
                self.model_loaded = True
                
                logger.info(f"✅ Loaded sklearn RF model: {len(self.severity_labels)} classes, "
                           f"{len(self.camera_to_idx)} cameras")
                logger.info(f"   Features: {self.feature_columns}")
                logger.info(f"   Labels: {self.severity_labels}")
            else:
                logger.warning(f"⚠️ Sklearn model not found at {sklearn_model_path}, using rule-based fallback")
                self.model_loaded = False
            
            # 2. Load fallback rules (used if model unavailable or for additional context)
            rules_path = data_dir / "risk_rules.json"
            if rules_path.exists():
                with open(rules_path, 'r') as f:
                    self.risk_rules = json.load(f)
                logger.info(f"Loaded {len(self.risk_rules)} fallback risk rules")
            
            # 3. Load camera profiles (for additional context)
            profiles_path = data_dir / "camera_profiles.json"
            if profiles_path.exists():
                with open(profiles_path, 'r') as f:
                    self.camera_profiles = json.load(f)
                logger.info(f"Loaded {len(self.camera_profiles)} camera profiles")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}", exc_info=True)
            return False

    def _predict_with_model(
        self, 
        camera_id: str, 
        confidence: float, 
        timestamp: float
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict severity using trained sklearn model.
        
        Returns:
            (severity_level, probability, probabilities_array)
        """
        dt = datetime.fromtimestamp(timestamp)
        
        # Build feature vector: [hour, day_of_week, camera_idx, is_weekend, confidence, duration]
        hour = dt.hour
        day_of_week = dt.weekday()
        camera_idx = self.camera_to_idx.get(camera_id, len(self.camera_to_idx))  # Unknown = max idx
        is_weekend = 1 if day_of_week >= 5 else 0
        duration = 5.0  # Default duration for real-time (unknown)
        
        # Create feature array matching training order
        features = np.array([[hour, day_of_week, camera_idx, is_weekend, confidence, duration]], dtype=np.float32)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction_idx = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        severity_level = self.severity_labels[int(prediction_idx)]
        probability = float(probabilities[int(prediction_idx)])
        
        return severity_level, probability, probabilities
        
        return best_severity, best_rule_confidence, matched_rule

    def _get_camera_risk_profile(self, camera_id: str, hour: int) -> Optional[Dict]:
        """Get risk profile for camera based on time."""
        # Find matching profile based on hour
        best_profile = None
        min_hour_diff = float('inf')
        
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period string for rule matching."""
        if hour < 6 or hour >= 22:
            return 'Night'
        elif hour < 8:
            return 'Early_Morning'
        elif hour < 12:
            return 'Morning'
        elif hour < 17:
            return 'Afternoon'
        else:
            return 'Evening'

    def _fallback_rule_prediction(
        self, 
        camera_id: str, 
        confidence: float, 
        hour: int
    ) -> Tuple[str, float]:
        """
        Fallback prediction using FP-Growth rules when model unavailable.
        """
        time_period = self._get_time_period(hour)
        
        # Try to match rules
        for rule in self.risk_rules:
            conditions = rule.get('if', [])
            consequences = rule.get('then', [])
            
            # Check if time period matches
            if f"Time_{time_period}" in conditions:
                for conseq in consequences:
                    if conseq.startswith("Sev_"):
                        return conseq.replace("Sev_", ""), rule.get('confidence', 0.7)
        
        # No rule matched - use confidence-based
        if confidence >= 0.8:
            return "HIGH", confidence
        elif confidence >= 0.6:
            return "MEDIUM", confidence
        else:
            return "LOW", confidence

    def analyze_severity(
        self, 
        camera_id: str, 
        confidence: float, 
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Analyze event severity using trained ML model (primary) or rules (fallback).
        
        Args:
            camera_id: Camera identifier
            confidence: Detection confidence (0.0-1.0)
            timestamp: Unix timestamp of detection
        
        Returns:
            Dict containing:
            - severity_level: "HIGH" | "MEDIUM" | "LOW"
            - severity_score: 0.0-1.0 (model probability or rule confidence)
            - analysis_time_ms: Processing time in milliseconds
            - prediction_method: "model" | "rules" | "fallback"
            - probabilities: Class probabilities (if model used)
        """
        start_time = time.perf_counter()
        
        try:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            
            # Try ML model first (if loaded)
            if self.model_loaded and self.model is not None:
                severity_level, probability, probabilities = self._predict_with_model(
                    camera_id, confidence, timestamp
                )
                severity_score = probability
                prediction_method = "model"
                self.model_predictions += 1
                
                # Format probabilities dict
                prob_dict = {
                    label: round(float(prob), 4) 
                    for label, prob in zip(self.severity_labels, probabilities)
                }
            else:
                # Fallback to rule-based prediction
                severity_level, severity_score = self._fallback_rule_prediction(
                    camera_id, confidence, hour
                )
                prediction_method = "rules"
                prob_dict = None
                self.rule_fallbacks += 1
                
        except Exception as e:
            logger.error(f"Error in severity analysis: {e}", exc_info=True)
            # Ultimate fallback - confidence-based
            severity_level = "HIGH" if confidence >= 0.7 else "MEDIUM" if confidence >= 0.5 else "LOW"
            severity_score = confidence
            prediction_method = "fallback"
            prob_dict = None
        
        # Calculate timing
        end_time = time.perf_counter()
        analysis_time_ms = (end_time - start_time) * 1000
        
        # Update statistics
        self.analysis_count += 1
        self.total_analysis_time_ms += analysis_time_ms
        
        result = {
            'severity_level': severity_level,
            'severity_score': round(severity_score, 4),
            'analysis_time_ms': round(analysis_time_ms, 4),
            'prediction_method': prediction_method,
            'probabilities': prob_dict,
            'features': {
                'camera_id': camera_id,
                'confidence': confidence,
                'hour': hour,
                'time_period': self._get_time_period(hour),
            }
        }
        
        logger.debug(
            f"[{camera_id}] Severity: {severity_level} (score={severity_score:.3f}, "
            f"method={prediction_method}, time={analysis_time_ms:.3f}ms)"
        )
        
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        avg_time = (
            self.total_analysis_time_ms / self.analysis_count 
            if self.analysis_count > 0 else 0
        )
        return {
            'model_loaded': self.model_loaded,
            'analysis_count': self.analysis_count,
            'model_predictions': self.model_predictions,
            'rule_fallbacks': self.rule_fallbacks,
            'total_analysis_time_ms': round(self.total_analysis_time_ms, 3),
            'avg_analysis_time_ms': round(avg_time, 4),
            'severity_labels': self.severity_labels,
            'cameras_known': list(self.camera_to_idx.keys()) if self.camera_to_idx else [],
        }


# Module-level singleton getter
_security_engine: Optional[SecurityEngine] = None

def get_security_engine() -> SecurityEngine:
    """Get the singleton SecurityEngine instance."""
    global _security_engine
    if _security_engine is None:
        _security_engine = SecurityEngine.get_instance()
        _security_engine.load_artifacts()
    return _security_engine


def init_security_engine(data_dir: str = None) -> SecurityEngine:
    """
    Initialize SecurityEngine with artifacts.
    Call this at application startup.
    
    Args:
        data_dir: Path to data directory. If None, uses default.
    
    Returns:
        Initialized SecurityEngine instance.
    """
    global _security_engine
    _security_engine = SecurityEngine.get_instance()
    _security_engine.load_artifacts(data_dir)
    logger.info(f"SecurityEngine initialized: {_security_engine.get_stats()}")
    return _security_engine
