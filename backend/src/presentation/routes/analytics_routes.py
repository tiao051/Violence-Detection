"""
Analytics API Routes (Hotspots Only)

Endpoints for visual analytics, mainly Hotspot Map.
Removed legacy severity prediction and rule mining.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import os
import sys
import logging
import threading

logger = logging.getLogger(__name__)

# Add ai_service to path for HotspotAnalyzer
ai_service_paths = [
    '/app/ai_service',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service'),
]
for path in ai_service_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from insights.hotspot_analyzer import HotspotAnalyzer
except ImportError as e:
    logger.warning(f"Failed to import HotspotAnalyzer: {e}")
    HotspotAnalyzer = None

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Cache state for hotspots
_hotspot_cache: Optional[Dict[str, Any]] = None
_is_computing: bool = False

# Camera Metadata (Coordinates)
CAMERA_COORDINATES = {
    "cam1": {"lat": 10.7912, "lng": 106.6294, "name": "Luy Ban Bich Street"},
    "cam2": {"lat": 10.8024, "lng": 106.6401, "name": "Au Co Junction"},
    "cam3": {"lat": 10.7935, "lng": 106.6512, "name": "Tan Ky Tan Quy Street"},
    "cam4": {"lat": 10.8103, "lng": 106.6287, "name": "Tan Phu Market"},
    "cam5": {"lat": 10.7856, "lng": 106.6523, "name": "Dam Sen Park"},
}

# English names map
CAMERA_NAME_MAP = {
    "Luy Ban Bich Street": "Luy Ban Bich Street",
    "Au Co Junction": "Au Co Junction",
    "Tan Ky Tan Quy Street": "Tan Ky Tan Quy Street",
    "Tan Phu Market": "Tan Phu Market",
    "Dam Sen Park": "Dam Sen Park",
}

def _compute_hotspots() -> None:
    """Compute hotspot analysis in background."""
    global _hotspot_cache, _is_computing
    try:
        if HotspotAnalyzer is None:
            logger.error("HotspotAnalyzer not available")
            return

        logger.info("Computing hotspots...")
        
        # Find CSV data file
        data_dir_options = [
            '/app/ai_service/insights/data',
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service', 'insights', 'data'),
        ]
        
        csv_path = None
        for path in data_dir_options:
            potential_path = os.path.join(path, 'analytics_events.csv')
            if os.path.exists(potential_path):
                csv_path = potential_path
                break
        
        # If no CSV, try to use default data or empty
        analyzer = HotspotAnalyzer()
        
        if csv_path:
            analyzer.load_from_csv(csv_path)
            analyzer.analyze()
            summary = analyzer.get_summary()
        else:
            logger.warning("Hotspot CSV not found, using dummy data structure")
            summary = {"cameras": []}
        
        # Enrich with coordinates
        filtered_cameras = []
        for cam in summary.get("cameras", []):
            cam_id = cam.get("camera_id", "")
            
            # Add coordinates if known
            if cam_id in CAMERA_COORDINATES:
                cam["lat"] = CAMERA_COORDINATES[cam_id]["lat"]
                cam["lng"] = CAMERA_COORDINATES[cam_id]["lng"]
                
                # Add display name
                vn_name = cam.get("camera_name", "")
                cam["camera_name_en"] = CAMERA_NAME_MAP.get(vn_name, vn_name)
                
                filtered_cameras.append(cam)
        
        # Update cache
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

def start_hotspot_computation() -> None:
    """Start hotspot computation in background."""
    global _is_computing
    if not _is_computing:
        _is_computing = True
        thread = threading.Thread(target=_compute_hotspots, daemon=True)
        thread.start()

@router.get("/hotspots")
async def get_hotspot_analysis() -> Dict[str, Any]:
    """
    Get hotspot analysis for all cameras.
    Used by Map Dashboard to show heatmaps.
    """
    try:
        # Trigger computation if cache empty
        if _hotspot_cache is None and not _is_computing:
            start_hotspot_computation()
            
            # If blocking wait is acceptable for first load:
            # _compute_hotspots() 
            # return _hotspot_cache
            
            # Else return computing status
            raise HTTPException(
                status_code=202, 
                detail="Hotspots computing. Please retry in a few seconds."
            )
        
        if _hotspot_cache is None:
             raise HTTPException(status_code=202, detail="Hotspots computing...")
             
        return _hotspot_cache
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_hotspot_analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))