"""
Camera Credibility API Routes

Endpoints for camera credibility system:
- Report false alarms
- Get camera intelligence/stats
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging

from ...application.credibility_engine import get_credibility_engine

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response models
class FalseAlarmReport(BaseModel):
    alert_id: str
    camera_id: str
    reason: Optional[str] = None


class ConfidenceAdjustmentRequest(BaseModel):
    camera_id: str
    raw_confidence: float
    context: Optional[Dict[str, Any]] = None


# ==================== ENDPOINTS ====================

@router.post("/adjust-confidence")
async def adjust_alert_confidence(request: ConfidenceAdjustmentRequest) -> Dict[str, Any]:
    """
    Adjust alert confidence based on camera credibility.
    
    This is the main runtime endpoint used by the detection system.
    """
    try:
        engine = get_credibility_engine()
        
        result = engine.adjust_confidence(
            camera_id=request.camera_id,
            raw_confidence=request.raw_confidence,
            alert_context=request.context
        )
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Failed to adjust confidence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


from ...infrastructure.storage.event_persistence import get_event_persistence_service

# ... (imports)

@router.post("/mark-false-alarm/{alert_id}")
async def mark_false_alarm(alert_id: str, report: FalseAlarmReport) -> Dict[str, Any]:
    """
    Mark an alert as a false alarm.
    
    This updates the alert's verification status in the database.
    It is idempotent: calling it multiple times for the same alert is safe.
    """
    try:
        event_service = get_event_persistence_service()
        
        # Use the persistence service to atomic/safe update
        result = await event_service.mark_false_alarm(
            event_id=alert_id,
            reason=report.reason
        )
        
        if not result["success"]:
            # If error is "Event not found", return 404
            if result.get("error") == "Event not found":
                 raise HTTPException(status_code=404, detail="Alert not found")
            raise HTTPException(status_code=500, detail=result.get("error"))
            
        return {
            "success": True,
            "message": result.get("message", "Alert marked as false alarm"),
            "alert_id": alert_id,
            "camera_id": report.camera_id,
            "verification_status": "false_positive",
            "already_reported": result.get("already_reported", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark false alarm: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera-intelligence")
async def get_camera_intelligence() -> Dict[str, Any]:
    """
    Get camera credibility intelligence for all cameras.
    
    Returns credibility scores, clusters, patterns, and recommendations.
    """
    try:
        engine = get_credibility_engine()
        
        cameras = engine.get_all_cameras()
        
        # Sort by credibility score (lowest first - need attention)
        cameras_sorted = sorted(cameras, key=lambda x: x["credibility_score"])
        
        # Categorize cameras by tier
        high_tier = [c for c in cameras if c["credibility_tier"] == "HIGH"]
        medium_tier = [c for c in cameras if c["credibility_tier"] == "MEDIUM"]
        low_tier = [c for c in cameras if c["credibility_tier"] == "LOW"]
        
        return {
            "success": True,
            "total_cameras": len(cameras),
            "summary": {
                "high_credibility": len(high_tier),
                "medium_credibility": len(medium_tier),
                "low_credibility": len(low_tier)
            },
            "cameras": cameras_sorted,
            "clusters": engine.camera_clusters,
            "false_alarm_patterns": engine.false_alarm_patterns,
            "recommendations": {
                "needs_attention": [
                    {
                        "camera_id": c["camera_id"],
                        "camera_name": c["camera_name"],
                        "credibility_score": c["credibility_score"],
                        "fp_rate": c["metrics"]["false_positive_rate"],
                        "recommendation": c["recommendation"]
                    }
                    for c in low_tier
                ],
                "most_reliable": [
                    {
                        "camera_id": c["camera_id"],
                        "camera_name": c["camera_name"],
                        "credibility_score": c["credibility_score"],
                        "tp_rate": c["metrics"]["true_positive_rate"]
                    }
                    for c in sorted(high_tier, key=lambda x: x["credibility_score"], reverse=True)[:3]
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get camera intelligence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera-intelligence/{camera_id}")
async def get_camera_stats(camera_id: str) -> Dict[str, Any]:
    """
    Get detailed statistics for a specific camera.
    """
    try:
        engine = get_credibility_engine()
        
        camera_data = engine.get_camera_stats(camera_id)
        
        if not camera_data:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not found in credibility database"
            )
        
        return {
            "success": True,
            **camera_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get camera stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload-artifacts")
async def reload_credibility_artifacts() -> Dict[str, Any]:
    """
    Reload credibility artifacts from disk.
    
    Call this after Spark retraining completes to update runtime artifacts.
    """
    try:
        engine = get_credibility_engine()
        engine.reload_artifacts()
        
        return {
            "success": True,
            "message": "Credibility artifacts reloaded",
            "cameras_loaded": len(engine.camera_scores),
            "clusters_loaded": len(engine.camera_clusters),
            "patterns_loaded": len(engine.false_alarm_patterns)
        }
        
    except Exception as e:
        logger.error(f"Failed to reload artifacts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
