"""
Hotspot Analysis using Statistical Methods (No ML Training Required).
Optimized for performance using Pandas and Caching.

Analyzes camera locations to identify hotspots based on:
1. Violence event ratio
2. Average confidence scores
3. Z-score comparison with other cameras
4. Time-weighted recent events
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math
import os
import time
import pandas as pd
import numpy as np


@dataclass
class CameraStats:
    """Statistics for a single camera location."""
    camera_id: str
    camera_name: str
    camera_description: str
    total_events: int
    violence_events: int
    nonviolence_events: int
    avg_confidence: float
    violence_ratio: float
    z_score: float
    hotspot_score: float
    classification: str  # "hotspot" | "warning" | "safe"
    risk_level: str  # "HIGH" | "MEDIUM" | "LOW"


class HotspotAnalyzer:
    """
    Statistical analyzer for identifying violence hotspots.
    Optimized with Pandas for high performance.
    """
    
    # Weights for scoring
    WEIGHT_VIOLENCE_RATIO = 0.40
    WEIGHT_CONFIDENCE = 0.30
    WEIGHT_ZSCORE = 0.30
    
    HOTSPOT_THRESHOLD = 0.6
    WARNING_THRESHOLD = 0.4
    
    def __init__(self):
        self.camera_stats: Dict[str, CameraStats] = {}
        self.df: Optional[pd.DataFrame] = None
        self._last_file_mtime = 0
        self._last_file_path = ""
        
    def load_from_csv(self, csv_path: str) -> "HotspotAnalyzer":
        """Load events from CSV file with caching check."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Check if file has changed
        mtime = os.path.getmtime(csv_path)
        if self.df is not None and csv_path == self._last_file_path and mtime == self._last_file_mtime:
            # Cache hit - file unchanged
            return self
            
        try:
            # Fast load with pandas
            # types map for faster parsing
            dtype = {
                'cameraId': 'string',
                'cameraName': 'string',
                'cameraDescription': 'string',
                'confidence': 'float32',
                'label': 'category'
            }
            
            self.df = pd.read_csv(csv_path, dtype=dtype, usecols=dtype.keys())
            
            # Normalize column names for internal use
            self.df = self.df.rename(columns={
                'cameraId': 'camera_id',
                'cameraName': 'camera_name',
                'cameraDescription': 'camera_description'
            })
            
            self._last_file_path = csv_path
            self._last_file_mtime = mtime
            
        except Exception as e:
            # Fallback for older CSV formats or empty files
            print(f"Pandas load error: {e}. File might be empty.")
            self.df = pd.DataFrame(columns=['camera_id', 'camera_name', 'camera_description', 'confidence', 'label'])
        
        return self
    
    def analyze(self) -> Dict[str, CameraStats]:
        """
        Perform hotspot analysis using vectorized operations.
        Returns dict of camera_id -> CameraStats
        """
        if self.df is None or self.df.empty:
            return {}
        
        # Step 1: Aggregation by camera (Vectorized)
        # Group by camera_id and calculate metrics in one pass
        grouped = self.df.groupby('camera_id')
        
        # Calculate counts
        total_counts = grouped.size()
        violence_counts = self.df[self.df['label'] == 'violence'].groupby('camera_id').size()
        
        # Reindex violence counts to ensure all cameras are present (fill 0)
        violence_counts = violence_counts.reindex(total_counts.index, fill_value=0)
        nonviolence_counts = total_counts - violence_counts
        
        # Calculate means
        avg_confidence = grouped['confidence'].mean()
        
        # Get metadata (take first value)
        meta = grouped[['camera_name', 'camera_description']].first()
        
        # Step 2: Vectorized Calculations
        violence_ratios = violence_counts / total_counts
        
        # Step 3: Z-scores
        mean_ratio = violence_ratios.mean()
        std_ratio = violence_ratios.std()
        
        if std_ratio > 0:
            z_scores = (violence_ratios - mean_ratio) / std_ratio
        else:
            z_scores = pd.Series(0, index=violence_ratios.index)
            
        # Sigmoid normalization for Z-score
        z_normalized = 1 / (1 + np.exp(-z_scores))
        
        # Step 4: Weighted Score
        hotspot_scores = (
            self.WEIGHT_VIOLENCE_RATIO * violence_ratios +
            self.WEIGHT_CONFIDENCE * avg_confidence +
            self.WEIGHT_ZSCORE * z_normalized
        )
        
        # Step 5: Build Result Objects
        self.camera_stats = {}
        
        for cam_id in total_counts.index:
            score = hotspot_scores[cam_id]
            
            if score >= self.HOTSPOT_THRESHOLD:
                classification = "hotspot"
                risk = "HIGH"
            elif score >= self.WARNING_THRESHOLD:
                classification = "warning"
                risk = "MEDIUM"
            else:
                classification = "safe"
                risk = "LOW"
                
            self.camera_stats[cam_id] = CameraStats(
                camera_id=cam_id,
                camera_name=str(meta.loc[cam_id, 'camera_name']),
                camera_description=str(meta.loc[cam_id, 'camera_description']),
                total_events=int(total_counts[cam_id]),
                violence_events=int(violence_counts[cam_id]),
                nonviolence_events=int(nonviolence_counts[cam_id]),
                avg_confidence=round(float(avg_confidence[cam_id]), 3),
                violence_ratio=round(float(violence_ratios[cam_id]), 3),
                z_score=round(float(z_scores[cam_id]), 3),
                hotspot_score=round(float(score), 3),
                classification=classification,
                risk_level=risk
            )
            
        return self.camera_stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of hotspot analysis."""
        if not self.camera_stats:
            return {"error": "No analysis performed yet"}
        
        # Calculate totals
        total_ev = sum(s.total_events for s in self.camera_stats.values())
        total_viol = sum(s.violence_events for s in self.camera_stats.values())
        
        # Group stats
        hotspots = [s for s in self.camera_stats.values() if s.classification == "hotspot"]
        warnings = [s for s in self.camera_stats.values() if s.classification == "warning"]
        safe = [s for s in self.camera_stats.values() if s.classification == "safe"]
        
        return {
            "total_cameras": len(self.camera_stats),
            "hotspots": len(hotspots),
            "warnings": len(warnings),
            "safe_zones": len(safe),
            "total_events": total_ev,
            "total_violence_events": total_viol,
            "cameras": [
                {
                    "camera_id": s.camera_id,
                    "camera_name": s.camera_name,
                    "camera_description": s.camera_description,
                    "total_events": s.total_events,
                    "violence_events": s.violence_events,
                    "violence_ratio": s.violence_ratio,
                    "avg_confidence": s.avg_confidence,
                    "z_score": s.z_score,
                    "hotspot_score": s.hotspot_score,
                    "classification": s.classification,
                    "risk_level": s.risk_level,
                }
                for s in sorted(self.camera_stats.values(), key=lambda x: x.hotspot_score, reverse=True)
            ]
        }


# Singleton instance for API use
_analyzer_instance: Optional[HotspotAnalyzer] = None


def get_hotspot_analyzer() -> HotspotAnalyzer:
    """Get or create the hotspot analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HotspotAnalyzer()
    return _analyzer_instance


def analyze_hotspots_from_csv(csv_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze hotspots from CSV file.
    Uses singleton to benefit from caching.
    """
    analyzer = get_hotspot_analyzer()
    analyzer.load_from_csv(csv_path)
    analyzer.analyze()
    return analyzer.get_summary()
