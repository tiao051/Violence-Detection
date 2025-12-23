"""
Camera Credibility Inference Engine

Lightweight runtime inference that loads trained artifacts (JSON) 
and adjusts alert confidence based on camera credibility.

NO model prediction - just JSON lookup and confidence multiplication.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Artifact paths
ARTIFACTS_DIR = Path(__file__).parent.parent.parent.parent / "ai_service" / "insights" / "data"


class CredibilityEngine:
    """
    High-performance camera credibility engine.
    
    Loads pre-computed credibility scores from JSON artifacts.
    Runtime: Simple lookup + multiplication (< 1ms).
    """
    
    # Singleton
    _instance: Optional['CredibilityEngine'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if CredibilityEngine._initialized:
            return
        
        # Loaded artifacts
        self.camera_credibility: Dict[str, Dict] = {}
        self.camera_clusters: List[Dict] = []
        self.false_alarm_patterns: List[Dict] = []
        
        # Quick lookup maps
        self.camera_scores: Dict[str, float] = {}
        self.camera_tiers: Dict[str, str] = {}
        
        # Load artifacts
        self._load_artifacts()
        
        CredibilityEngine._initialized = True
    
    def _load_artifacts(self):
        """Load pre-trained artifacts from JSON files."""
        try:
            # 1. Camera Credibility (main source)
            credibility_file = ARTIFACTS_DIR / "camera_credibility.json"
            if credibility_file.exists():
                with open(credibility_file, 'r', encoding='utf-8') as f:
                    credibility_list = json.load(f)
                
                # Build lookup maps
                for cam_data in credibility_list:
                    cam_id = cam_data["camera_id"]
                    self.camera_credibility[cam_id] = cam_data
                    self.camera_scores[cam_id] = cam_data["credibility_score"]
                    self.camera_tiers[cam_id] = cam_data["credibility_tier"]
                
                logger.info(f"Loaded credibility scores for {len(self.camera_scores)} cameras")
            else:
                logger.warning(f"Credibility file not found: {credibility_file}")
            
            # 2. Camera Clusters (for context)
            clusters_file = ARTIFACTS_DIR / "camera_clusters.json"
            if clusters_file.exists():
                with open(clusters_file, 'r', encoding='utf-8') as f:
                    self.camera_clusters = json.load(f)
                logger.info(f"Loaded {len(self.camera_clusters)} camera clusters")
            
            # 3. False Alarm Patterns (for pattern matching)
            patterns_file = ARTIFACTS_DIR / "false_alarm_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self.false_alarm_patterns = json.load(f)
                logger.info(f"Loaded {len(self.false_alarm_patterns)} false alarm patterns")
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}", exc_info=True)
    
    def adjust_confidence(
        self, 
        camera_id: str, 
        raw_confidence: float,
        alert_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adjust alert confidence based on camera credibility.
        
        Args:
            camera_id: Camera identifier
            raw_confidence: Original confidence from violence detection model
            alert_context: Optional context (hour, duration, etc.) for pattern matching
        
        Returns:
            Dictionary with adjusted confidence and metadata
        """
        # Get camera credibility score (default to 0.5 if unknown)
        camera_score = self.camera_scores.get(camera_id, 0.5)
        camera_tier = self.camera_tiers.get(camera_id, "MEDIUM")
        
        # Base adjustment: multiply by camera credibility
        adjusted_confidence = raw_confidence * camera_score
        
        # Optional: Check for false alarm patterns
        pattern_matched = False
        pattern_penalty = 0.0
        
        if alert_context and self.false_alarm_patterns:
            pattern_match = self._check_false_alarm_patterns(camera_id, alert_context)
            if pattern_match:
                pattern_matched = True
                pattern_penalty = 0.15  # Additional 15% reduction
                adjusted_confidence = max(0.0, adjusted_confidence - pattern_penalty)
        
        # Clamp to [0, 1]
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # Determine action recommendation
        if adjusted_confidence < 0.3:
            action = "likely_false_alarm"
            priority = "low"
        elif adjusted_confidence < 0.5:
            action = "needs_verification"
            priority = "medium"
        else:
            action = "credible_alert"
            priority = "high"
        
        return {
            "raw_confidence": round(raw_confidence, 2),
            "camera_credibility": round(camera_score, 2),
            "adjusted_confidence": round(adjusted_confidence, 2),
            "confidence_delta": round(adjusted_confidence - raw_confidence, 2),
            "camera_tier": camera_tier,
            "pattern_matched": pattern_matched,
            "pattern_penalty": round(pattern_penalty, 2) if pattern_matched else 0.0,
            "action": action,
            "priority": priority,
            "camera_info": self.camera_credibility.get(camera_id, {}).get("recommendation", "Unknown camera")
        }
    
    def _check_false_alarm_patterns(self, camera_id: str, context: Dict[str, Any]) -> bool:
        """
        Check if alert matches any known false alarm patterns.
        
        Args:
            camera_id: Camera ID
            context: Alert context (hour, confidence, duration, etc.)
        
        Returns:
            True if matches a false alarm pattern
        """
        # Extract context features
        hour = context.get("hour", 12)
        confidence = context.get("confidence", 0.5)
        duration = context.get("duration", 5.0)
        
        # Categorize
        if hour < 6 or hour >= 22:
            time_period = "Hour_night"
        elif hour < 12:
            time_period = "Hour_morning"
        elif hour < 17:
            time_period = "Hour_afternoon"
        else:
            time_period = "Hour_evening"
        
        conf_level = "Conf_low" if confidence < 0.5 else "Conf_medium" if confidence < 0.75 else "Conf_high"
        dur_level = "Dur_short" if duration < 3 else "Dur_medium" if duration < 15 else "Dur_long"
        
        alert_signature = {f"Cam_{camera_id}", time_period, conf_level, dur_level}
        
        # Check against patterns
        for pattern in self.false_alarm_patterns:
            pattern_items = set(pattern["pattern"])
            
            # If alert contains all pattern items, it's a match
            if pattern_items.issubset(alert_signature):
                logger.debug(f"Alert matched false alarm pattern: {pattern['pattern']}")
                return True
        
        return False
    
    def get_camera_stats(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get full statistics for a camera."""
        return self.camera_credibility.get(camera_id)
    
    def get_all_cameras(self) -> List[Dict[str, Any]]:
        """Get credibility info for all cameras."""
        return list(self.camera_credibility.values())
    
    def reload_artifacts(self):
        """Reload artifacts from disk (useful after retraining)."""
        logger.info("Reloading credibility artifacts...")
        self.camera_credibility.clear()
        self.camera_scores.clear()
        self.camera_tiers.clear()
        self.camera_clusters.clear()
        self.false_alarm_patterns.clear()
        self._load_artifacts()


# Singleton accessor
_engine_instance: Optional[CredibilityEngine] = None

def get_credibility_engine() -> CredibilityEngine:
    """Get singleton instance of CredibilityEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CredibilityEngine()
    return _engine_instance
