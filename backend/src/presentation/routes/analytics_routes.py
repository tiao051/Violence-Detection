"""Analytics API Routes"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import sys
import logging

logger = logging.getLogger(__name__)

ai_service_paths = [
    '/app/ai_service',
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service'),
]
for path in ai_service_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from insights import InsightsModel, ViolenceEvent
    from insights.hotspot_analyzer import HotspotAnalyzer, analyze_hotspots_from_csv
except ImportError as e:
    logger.error(f"Failed to import ai_service modules: {e}", exc_info=True)
    raise

import pandas as pd
import threading
import time

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_model: Optional[InsightsModel] = None
_model_loaded: bool = False

# Per-component cache state
_cache_state: Dict[str, Optional[Any]] = {
    "summary": None,
    "patterns": None,
    "rules": None,
    "high_risk": None,
    "full_report": None,
    "hotspots": None,  # NEW: hotspot cache
}
_computing_tasks: Dict[str, bool] = {
    "summary": False,
    "patterns": False,
    "rules": False,
    "high_risk": False,
    "full_report": False,
    "hotspots": False,  # NEW: hotspot computing flag
}
_cache_lock = threading.Lock()

def init_insights_model() -> None:
    """
    Initialize InsightsModel at startup.
    Loads pre-trained model from pkl file.
    Called from backend main.py during lifespan.startup().
    """
    global _model, _model_loaded
    
    logger.info("Loading InsightsModel...")
    
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
        raise FileNotFoundError("Could not find ai_service/insights/data directory")
    
    model_path = os.path.join(data_dir, 'trained_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pre-trained model not found at {model_path}. "
            "Please ensure trained_model.pkl exists in ai_service/insights/data/"
        )
    
    logger.info(f"Loading model from {model_path}")
    _model = InsightsModel.load(model_path)
    _model_loaded = True

    # Pre-compute analytics so they are ready when requested
    start_analytics_computation()

def get_model() -> InsightsModel:
    """
    Get the loaded InsightsModel.
    Assumes init_insights_model() has been called during startup.
    """
    global _model, _model_loaded
    
    if not _model_loaded or _model is None:
        raise RuntimeError("InsightsModel not initialized. Call init_insights_model() first.")
    
    return _model


def _compute_summary() -> None:
    """Compute summary in background."""
    try:
        logger.info("Computing summary...")
        model = get_model()
        _cache_state["summary"] = model.get_summary()
        logger.info("Summary computed")
    except Exception as e:
        logger.error(f"Error computing summary: {e}", exc_info=True)
    finally:
        _computing_tasks["summary"] = False


def _compute_patterns() -> None:
    """Compute patterns in background."""
    try:
        logger.info("Computing patterns...")
        model = get_model()
        _cache_state["patterns"] = model.get_patterns()
        logger.info("Patterns computed")
    except Exception as e:
        logger.error(f"Error computing patterns: {e}", exc_info=True)
    finally:
        _computing_tasks["patterns"] = False


def _compute_rules() -> None:
    """Compute rules in background."""
    try:
        logger.info("Computing rules...")
        model = get_model()
        _cache_state["rules"] = model.get_rules(top_n=20)
        logger.info("Rules computed")
    except Exception as e:
        logger.error(f"Error computing rules: {e}", exc_info=True)
    finally:
        _computing_tasks["rules"] = False


def _compute_high_risk() -> None:
    """Compute high risk conditions in background."""
    try:
        logger.info("Computing high risk conditions...")
        model = get_model()
        _cache_state["high_risk"] = model.get_high_risk_conditions(top_n=10)
        logger.info("High risk conditions computed")
    except Exception as e:
        logger.error(f"Error computing high risk: {e}", exc_info=True)
    finally:
        _computing_tasks["high_risk"] = False


def _compute_full_report() -> None:
    """Compute full report in background."""
    try:
        logger.info("Computing full report...")
        model = get_model()
        _cache_state["full_report"] = model.get_full_report()
        logger.info("Full report computed")
    except Exception as e:
        logger.error(f"Error computing full report: {e}", exc_info=True)
    finally:
        _computing_tasks["full_report"] = False


def _compute_hotspots() -> None:
    """Compute hotspot analysis in background."""
    try:
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
        
        if csv_path is None:
            logger.warning("Hotspot CSV not found, skipping hotspot computation")
            return
        
        # Perform analysis
        analyzer = HotspotAnalyzer()
        analyzer.load_from_csv(csv_path)
        analyzer.analyze()
        summary = analyzer.get_summary()
        
        # Enrich with coordinates and English names
        for cam in summary.get("cameras", []):
            cam_id = cam.get("camera_id", "")
            if cam_id in CAMERA_COORDINATES:
                cam["lat"] = CAMERA_COORDINATES[cam_id]["lat"]
                cam["lng"] = CAMERA_COORDINATES[cam_id]["lng"]
            
            vn_name = cam.get("camera_name", "")
            cam["camera_name_en"] = CAMERA_NAME_MAP.get(vn_name, vn_name)
        
        _cache_state["hotspots"] = {
            "success": True,
            "algorithm": "Weighted Hotspot Scoring",
            "description": "Statistical analysis using violence ratio, confidence scores, and Z-score comparison",
            "weights": {
                "violence_ratio": 0.4,
                "avg_confidence": 0.3,
                "z_score": 0.3,
            },
            "thresholds": {
                "hotspot": 0.6,
                "warning": 0.4,
            },
            **summary,
        }
        logger.info("Hotspots computed")
    except Exception as e:
        logger.error(f"Error computing hotspots: {e}", exc_info=True)
    finally:
        _computing_tasks["hotspots"] = False


def start_analytics_computation() -> None:
    """Start all analytics computations in parallel (non-blocking)."""
    global _computing_tasks
    
    tasks = [
        ("summary", _compute_summary),
        ("patterns", _compute_patterns),
        ("rules", _compute_rules),
        ("high_risk", _compute_high_risk),
        ("full_report", _compute_full_report),
        ("hotspots", _compute_hotspots),  # NEW: include hotspots
    ]
    
    for task_name, task_func in tasks:
        if not _computing_tasks[task_name] and _cache_state[task_name] is None:
            _computing_tasks[task_name] = True
            thread = threading.Thread(target=task_func, daemon=True)
            thread.start()
            logger.info(f"Started background task: {task_name}")
    
    logger.info("All analytics tasks started in parallel")


def wait_for_analytics_completion(timeout_seconds: int = 60) -> bool:
    """
    Block until all analytics computations are complete.
    
    Called during startup to ensure all data is ready before serving requests.
    This only blocks the startup thread, not the main event loop or inference pipeline.
    
    Args:
        timeout_seconds: Maximum time to wait (default 60s)
        
    Returns:
        True if all completed, False if timeout reached
    """
    import time as time_module
    
    start = time_module.time()
    check_interval = 0.5  # Check every 500ms
    
    required_keys = ["summary", "patterns", "rules", "high_risk", "full_report", "hotspots"]
    
    while time_module.time() - start < timeout_seconds:
        # Check if all are done
        all_done = True
        pending = []
        
        for key in required_keys:
            if _cache_state[key] is None:
                all_done = False
                status = "computing" if _computing_tasks[key] else "pending"
                pending.append(f"{key}({status})")
        
        if all_done:
            elapsed = time_module.time() - start
            logger.info(f"All analytics ready in {elapsed:.1f}s")
            return True
        
        # Log progress periodically (every 5 seconds)
        elapsed = time_module.time() - start
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            logger.info(f"Waiting for analytics: {', '.join(pending)} ({elapsed:.0f}s elapsed)")
        
        time_module.sleep(check_interval)
    
    # Timeout reached
    logger.warning(f"Analytics timeout after {timeout_seconds}s. Some data may not be ready.")
    return False


# Request/Response models
class PredictRequest(BaseModel):
    hour: int
    day: str
    camera: str


class PredictResponse(BaseModel):
    hour: int
    day: str
    camera: str
    time_period: str
    risk_level: str
    probabilities: Dict[str, float]
    change_vs_avg: str
    change_pct: float
    insight: str


# Endpoints
@router.post("/compute")
async def trigger_computation() -> Dict[str, str]:
    """Trigger all analytics computations in parallel."""
    try:
        start_analytics_computation()
        return {"status": "computation started", "message": "Check individual endpoints for results"}
    except Exception as e:
        logger.error(f"Error triggering computation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_computation_status() -> Dict[str, str]:
    """Get status of ongoing computations."""
    return {
        "summary": "done" if _cache_state["summary"] is not None else ("computing" if _computing_tasks["summary"] else "pending"),
        "patterns": "done" if _cache_state["patterns"] is not None else ("computing" if _computing_tasks["patterns"] else "pending"),
        "rules": "done" if _cache_state["rules"] is not None else ("computing" if _computing_tasks["rules"] else "pending"),
        "high_risk": "done" if _cache_state["high_risk"] is not None else ("computing" if _computing_tasks["high_risk"] else "pending"),
        "full_report": "done" if _cache_state["full_report"] is not None else ("computing" if _computing_tasks["full_report"] else "pending"),
        "hotspots": "done" if _cache_state["hotspots"] is not None else ("computing" if _computing_tasks["hotspots"] else "pending"),
    }


@router.get("/summary")
async def get_summary() -> Dict[str, Any]:
    """Get quick summary of insights (returns 202 if still computing)."""
    try:
        # Trigger computation if not started
        start_analytics_computation()
        
        if _cache_state["summary"] is None:
            raise HTTPException(
                status_code=202,
                detail="Summary still computing. Check /api/analytics/status for progress."
            )
        
        return _cache_state["summary"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_patterns() -> List[Dict[str, Any]]:
    """Get K-means cluster patterns (returns 202 if still computing)."""
    try:
        start_analytics_computation()
        
        if _cache_state["patterns"] is None:
            raise HTTPException(
                status_code=202,
                detail="Patterns still computing. Check /api/analytics/status for progress."
            )
        
        return _cache_state["patterns"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_rules(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get FP-Growth association rules (returns 202 if still computing)."""
    try:
        start_analytics_computation()
        
        if _cache_state["rules"] is None:
            raise HTTPException(
                status_code=202,
                detail="Rules still computing. Check /api/analytics/status for progress."
            )
        
        return _cache_state["rules"][:top_n]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_rules: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions")
async def get_high_risk_conditions(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get high risk conditions from Random Forest (returns 202 if still computing)."""
    try:
        start_analytics_computation()
        
        if _cache_state["high_risk"] is None:
            raise HTTPException(
                status_code=202,
                detail="High risk analysis still computing. Check /api/analytics/status for progress."
            )
        
        return _cache_state["high_risk"][:top_n]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_high_risk_conditions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict_risk(request: PredictRequest) -> Dict[str, Any]:
    """Predict risk level for specific conditions."""
    try:
        model = get_model()
        return model.predict(
            hour=request.hour,
            day=request.day,
            camera=request.camera
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/full-report")
async def get_full_report() -> Dict[str, Any]:
    """Get comprehensive report from all 3 models (returns 202 if still computing)."""
    try:
        start_analytics_computation()
        
        if _cache_state["full_report"] is None:
            raise HTTPException(
                status_code=202,
                detail="Full report still computing. Check /api/analytics/status for progress."
            )
        
        return _cache_state["full_report"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_full_report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{camera}")
async def get_forecast(camera: str, hours: int = 12) -> Dict[str, Any]:
    """Get hotspot forecast for a specific camera."""
    try:
        model = get_model()
        hours = min(hours, 24)
        return model.get_forecast(camera=camera, hours_ahead=hours)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# HOTSPOT ANALYSIS - Statistical Method (No ML Training Required)
# ==============================================================================

# Camera coordinates (real locations in Tan Phu/Tan Binh, HCMC)
CAMERA_COORDINATES = {
    "cam1": {"lat": 10.766761, "lng": 106.641934, "name": "Ngã tư Hòa Bình"},
    "cam2": {"lat": 10.804096, "lng": 106.637517, "name": "Ngã tư Cộng Hòa"},
    "cam3": {"lat": 10.801625, "lng": 106.636815, "name": "Ngã ba Âu Cơ"},
    "cam4": {"lat": 10.803639, "lng": 106.632466, "name": "Ngã ba Lê Trọng Tấn"},
    "cam5": {"lat": 10.794952, "lng": 106.629491, "name": "Ngã tư Tân Sơn Nhì"},
}

# English names for display (kept for backwards compatibility)
CAMERA_NAME_MAP = {
    "Ngã tư Hòa Bình": "Hoa Binh Junction",
    "Ngã tư Cộng Hòa": "Cong Hoa Junction",
    "Ngã ba Âu Cơ": "Au Co Junction",
    "Ngã ba Lê Trọng Tấn": "Le Trong Tan Junction",
    "Ngã tư Tân Sơn Nhì": "Tan Son Nhi Junction",
}


@router.get("/hotspots")
async def get_hotspot_analysis() -> Dict[str, Any]:
    """
    Get hotspot analysis for all cameras using statistical methods.
    
    This endpoint uses a weighted scoring algorithm:
    - Violence ratio (40%): Percentage of violence events
    - Average confidence (30%): Mean confidence of violence detections
    - Z-score (30%): How camera compares to others (statistical outlier detection)
    
    Classification thresholds:
    - Hotspot: score >= 0.6
    - Warning: score >= 0.4
    - Safe: score < 0.4
    
    Now uses cached data computed at startup for instant response.
    """
    try:
        # Trigger computation if not started
        start_analytics_computation()
        
        # Return cached data if available
        if _cache_state["hotspots"] is not None:
            return _cache_state["hotspots"]
        
        # If still computing, return 202
        if _computing_tasks["hotspots"]:
            raise HTTPException(
                status_code=202,
                detail="Hotspot analysis still computing. Check /api/analytics/status for progress."
            )
        
        # Fallback: compute synchronously if cache miss and not computing
        # This should rarely happen after startup
        _compute_hotspots()
        
        if _cache_state["hotspots"] is not None:
            return _cache_state["hotspots"]
        
        raise HTTPException(
            status_code=404,
            detail="Hotspot data not available. CSV file may be missing."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_hotspot_analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# SPARK INSIGHTS - Trend Analysis & Anomaly Detection
# ==============================================================================

# Import Spark insights (optional - may not be available)
try:
    from insights.spark.spark_insights_job import get_spark_insights
    SPARK_INSIGHTS_AVAILABLE = True
except ImportError:
    SPARK_INSIGHTS_AVAILABLE = False
    logger.warning("SparkInsightsJob not available - insights endpoints disabled")


@router.get("/insights")
async def get_full_insights(refresh: bool = False) -> Dict[str, Any]:
    """
    Get comprehensive insights from Spark processing.
    
    Includes:
    - Weekly trends (this week vs last week per camera)
    - Monthly trends (this month vs last month per camera)
    - Anomaly detection (cameras with unusual activity)
    - Patrol schedule recommendations
    - Camera statistics
    
    Results are cached for 24 hours unless refresh=true.
    """
    if not SPARK_INSIGHTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Spark insights not available. Module not installed."
        )
    
    try:
        insights = get_spark_insights(force_refresh=refresh)
        return insights
    except Exception as e:
        logger.error(f"Error getting insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_trends() -> Dict[str, Any]:
    """
    Get trend analysis comparing current vs previous periods.
    
    Returns:
    - Weekly: This week vs last week
    - Monthly: This month vs last month
    """
    if not SPARK_INSIGHTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Insights not available")
    
    try:
        insights = get_spark_insights(force_refresh=False)
        
        if not insights.get("success"):
            raise HTTPException(status_code=500, detail=insights.get("error", "Unknown error"))
        
        return {
            "weekly": insights.get("weekly_trends", []),
            "monthly": insights.get("monthly_trends", []),
            "computed_at": insights.get("computed_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_anomalies() -> Dict[str, Any]:
    """
    Get cameras with unusual activity detected via Z-score analysis.
    
    Returns list of cameras flagged as anomalies with severity levels:
    - critical: Z-score >= 2.5
    - warning: Z-score >= 1.5
    - normal: Z-score < 1.5
    """
    if not SPARK_INSIGHTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Insights not available")
    
    try:
        insights = get_spark_insights(force_refresh=False)
        
        if not insights.get("success"):
            raise HTTPException(status_code=500, detail=insights.get("error", "Unknown error"))
        
        anomalies = insights.get("anomalies", [])
        
        return {
            "anomalies": anomalies,
            "critical_count": len([a for a in anomalies if a.get("severity") == "critical"]),
            "warning_count": len([a for a in anomalies if a.get("severity") == "warning"]),
            "computed_at": insights.get("computed_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patrol-schedule")
async def get_patrol_schedule() -> Dict[str, Any]:
    """
    Get recommended patrol schedule based on historical patterns.
    
    Returns time slots prioritized by expected incident rate.
    """
    if not SPARK_INSIGHTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Insights not available")
    
    try:
        insights = get_spark_insights(force_refresh=False)
        
        if not insights.get("success"):
            raise HTTPException(status_code=500, detail=insights.get("error", "Unknown error"))
        
        schedule = insights.get("patrol_schedule", [])
        
        return {
            "schedule": schedule,
            "high_priority_count": len([s for s in schedule if s.get("priority") == "high"]),
            "computed_at": insights.get("computed_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patrol schedule: {e}", exc_info=True)
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
                confidence = row.get('thumbnailConfidence', 0)
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

