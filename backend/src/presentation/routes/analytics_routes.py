"""
Analytics API Routes (Hotspots Only)

Endpoints for visual analytics, mainly Hotspot Map.
Refactored for Clean Code & DRY principles.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import os
import sys
import logging
import threading
import json
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

AI_SERVICE_PATHS = [
    '/app/ai_service',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service'),
]

for path in AI_SERVICE_PATHS:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from insights.hotspot_analyzer import HotspotAnalyzer
except ImportError as e:
    logger.warning(f"Failed to import HotspotAnalyzer: {e}")
    HotspotAnalyzer = None

# Camera Metadata
CAMERA_COORDINATES = {
    "cam1": {"lat": 10.7912, "lng": 106.6294, "name": "Luy Ban Bich Street"},
    "cam2": {"lat": 10.8024, "lng": 106.6401, "name": "Au Co Junction"},
    "cam3": {"lat": 10.7935, "lng": 106.6512, "name": "Tan Ky Tan Quy Street"},
    "cam4": {"lat": 10.8103, "lng": 106.6287, "name": "Tan Phu Market"},
    "cam5": {"lat": 10.7856, "lng": 106.6523, "name": "Dam Sen Park"},
}

CAMERA_NAME_MAP = {
    "Luy Ban Bich Street": "Luy Ban Bich Street",
    "Au Co Junction": "Au Co Junction",
    "Tan Ky Tan Quy Street": "Tan Ky Tan Quy Street",
    "Tan Phu Market": "Tan Phu Market",
    "Dam Sen Park": "Dam Sen Park",
}

DATA_DIRS = [
    '/app/ai_service/insights/data',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service', 'insights', 'data'),
]

# Cache state
_hotspot_cache: Optional[Dict[str, Any]] = None
_is_computing: bool = False

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

def _get_data_dir() -> Optional[str]:
    """Resolve the data directory path."""
    for path in DATA_DIRS:
        if os.path.exists(path):
            return path
    return None

def _get_file_path(filename: str) -> Optional[str]:
    """Resolve full path for a data file."""
    data_dir = _get_data_dir()
    if not data_dir:
        return None
    
    file_path = os.path.join(data_dir, filename)
    return file_path if os.path.exists(file_path) else None

def _load_json_data(filename: str, default: Any = None) -> Any:
    """Safely load JSON data."""
    if default is None:
        default = []
    
    file_path = _get_file_path(filename)
    if not file_path:
        return default

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return default

def _process_single_event(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Process a single event row into a structured dictionary."""
    camera_id = str(row.get('cameraId', ''))
    coords = CAMERA_COORDINATES.get(camera_id)
    
    if not coords:
        return None

    timestamp_str = str(row.get('timestamp', ''))
    try:
        confidence = float(row.get('thumbnailConfidence', 0))
    except (ValueError, TypeError):
        confidence = 0.0

    label = str(row.get('label', ''))

    # Parse time
    hour = 12
    try:
        if timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour = dt.hour
    except ValueError:
        pass

    # Logic: Cluster Definition (Simplified)
    # Cluster 1 (High Risk/Critical): 00:00-04:00 & High Conf
    # Cluster 0 (Evening): 15:00-20:00
    # Cluster 2 (Day/Normal): Others
    if 0 <= hour < 4 and confidence > 0.8:
        cluster = 1
    elif 15 <= hour < 20:
        cluster = 0
    else:
        cluster = 2

    return {
        'camera_id': camera_id,
        'lat': coords['lat'],
        'lng': coords['lng'],
        'hour': hour,
        'is_night': (hour >= 22 or hour < 6),
        'confidence': confidence,
        'is_violence': (label == 'violence'),
        'cluster': cluster
    }

def _compute_hotspots() -> None:
    """Compute hotspot analysis in background."""
    global _hotspot_cache, _is_computing
    
    try:
        if HotspotAnalyzer is None:
            logger.error("HotspotAnalyzer not available")
            return

        logger.info("Computing hotspots...")
        
        # Initialize and load data
        analyzer = HotspotAnalyzer()
        csv_path = _get_file_path('analytics_events.csv')
        
        if csv_path:
            analyzer.load_from_csv(csv_path)
            analyzer.analyze()
            summary = analyzer.get_summary()
        else:
            logger.warning("Hotspot CSV not found, using dummy data")
            summary = {"cameras": []}
        
        # Enrich camera data
        filtered_cameras = []
        for cam in summary.get("cameras", []):
            cam_id = cam.get("camera_id", "")
            coords = CAMERA_COORDINATES.get(cam_id)
            
            if coords:
                vn_name = cam.get("camera_name", "")
                filtered_cameras.append({
                    **cam,
                    "lat": coords["lat"],
                    "lng": coords["lng"],
                    "camera_name_en": CAMERA_NAME_MAP.get(vn_name, vn_name)
                })
        
        # Result construction
        _hotspot_cache = {
            "success": True,
            "algorithm": "Weighted Hotspot Scoring",
            "description": "Statistical analysis using violence ratio and confidence scores",
            "weights": {
                "violence_ratio": 0.4,
                "avg_confidence": 0.3,
                "z_score": 0.3,
            },
            "thresholds": {"hotspot": 0.6, "warning": 0.4},
            "cameras": filtered_cameras,
            "total_analyzed": len(filtered_cameras)
        }
        logger.info(f"Hotspots computed for {len(filtered_cameras)} cameras")
        
    except Exception as e:
        logger.error(f"Error computing hotspots: {e}", exc_info=True)
    finally:
        _is_computing = False

def _start_hotspot_computation() -> None:
    """Trigger background computation."""
    global _is_computing
    if not _is_computing:
        _is_computing = True
        threading.Thread(target=_compute_hotspots, daemon=True).start()

@router.get("/hotspots")
async def get_hotspot_analysis() -> Dict[str, Any]:
    """Get hotspot analysis for all cameras (Cached/Async)."""
    try:
        # Lazy loading
        if _hotspot_cache is None:
            if not _is_computing:
                _start_hotspot_computation()
            
            # Return accepted status while computing
            raise HTTPException(
                status_code=202, 
                detail="Hotspots computing. Please retry in a few seconds."
            )
            
        return _hotspot_cache
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_hotspot_analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# SMART CAMERA PLACEMENT - Voronoi + Weighted Scoring Algorithm
# ==============================================================================

@router.get("/camera-placement")
async def get_camera_placement_data() -> Dict[str, Any]:
    """
    Get data for smart camera placement suggestions.
    
    Returns:
    - events: List of violence events with coordinates and timestamps
    - cameras: List of existing cameras with coordinates
    - risk_clusters: K-means cluster profiles (from camera_profiles.json)
    - risk_rules: FP-Growth association rules (from risk_rules.json)
    
    Frontend uses this data to compute optimal camera placement using:
    - Voronoi diagram for gap analysis
    - Weighted scoring based on:
      - Event density (30%)
      - Distance to existing cameras (25%)
      - Proximity to schools (20%)
      - Critical risk cluster membership (15%)
      - Night activity ratio (10%)
    """
    try:
        import json
        from datetime import datetime
        
        # Find data directory
        data_dir_options = [
            '/app/ai_service/insights/data',
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service', 'insights', 'data'),
        ]
        
        data_dir = None
        for path in data_dir_options:
            if os.path.exists(path):
                data_dir = path
                break
        
        if data_dir is None:
            raise HTTPException(status_code=404, detail="Data directory not found")
        
        # Load events from CSV
        csv_path = os.path.join(data_dir, 'analytics_events.csv')
        events = []
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                camera_id = row.get('cameraId', '')
                timestamp_str = row.get('timestamp', '')
                confidence = row.get('confidence', 0)
                label = row.get('label', '')
                
                # Get camera coordinates
                coords = CAMERA_COORDINATES.get(camera_id, {})
                if not coords:
                    continue
                
                # Parse timestamp to get hour
                hour = 12  # default
                try:
                    if isinstance(timestamp_str, str) and timestamp_str:
                        # Try parsing ISO format
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        hour = dt.hour
                except:
                    pass
                
                # Determine if night (22:00 - 06:00)
                is_night = hour >= 22 or hour < 6
                
                # Determine cluster based on hour and confidence (simplified)
                # Based on camera_profiles.json:
                # Cluster 1 (Critical): ~1h, high confidence
                # Cluster 0 (Moderate): ~17h
                # Cluster 2 (Moderate): ~10h
                if hour >= 0 and hour < 4 and confidence > 0.8:
                    cluster = 1  # Critical
                elif hour >= 15 and hour < 20:
                    cluster = 0  # Moderate (evening)
                else:
                    cluster = 2  # Moderate (day)
                
                events.append({
                    'camera_id': camera_id,
                    'lat': coords['lat'],
                    'lng': coords['lng'],
                    'hour': hour,
                    'is_night': is_night,
                    'confidence': float(confidence) if confidence else 0,
                    'is_violence': label == 'violence',
                    'cluster': cluster
                })
        
        # Load risk clusters (camera_profiles.json)
        risk_clusters = []
        clusters_path = os.path.join(data_dir, 'camera_profiles.json')
        if os.path.exists(clusters_path):
            with open(clusters_path, 'r') as f:
                risk_clusters = json.load(f)
        
        # Load risk rules (risk_rules.json)
        risk_rules = []
        rules_path = os.path.join(data_dir, 'risk_rules.json')
        if os.path.exists(rules_path):
            with open(rules_path, 'r') as f:
                risk_rules = json.load(f)
        
        # Prepare camera data
        cameras = [
            {
                'camera_id': cam_id,
                'lat': coords['lat'],
                'lng': coords['lng'],
                'name': coords['name']
            }
            for cam_id, coords in CAMERA_COORDINATES.items()
        ]
        
        # Compute statistics
        violence_events = [e for e in events if e['is_violence']]
        night_events = [e for e in violence_events if e['is_night']]
        critical_events = [e for e in violence_events if e['cluster'] == 1]
        
        return {
            'success': True,
            'events': events,
            'violence_events_count': len(violence_events),
            'cameras': cameras,
            'risk_clusters': risk_clusters,
            'risk_rules': risk_rules,
            'statistics': {
                'total_events': len(events),
                'violence_events': len(violence_events),
                'night_violence_ratio': len(night_events) / len(violence_events) if violence_events else 0,
                'critical_cluster_ratio': len(critical_events) / len(violence_events) if violence_events else 0,
            },
            'weights': {
                'event_density': 0.30,
                'gap_distance': 0.25,
                'school_proximity': 0.20,
                'critical_cluster': 0.15,
                'night_activity': 0.10
            },
            'algorithm': 'Voronoi + Weighted Scoring'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_camera_placement_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

