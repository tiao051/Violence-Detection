"""Threat detection API routes."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from ...infrastructure.redis import get_redis_stream_producer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/threats", tags=["threats"])


@router.get("/cameras/{camera_id}/latest")
async def get_latest_threat(camera_id: str) -> Dict[str, Any]:
    """
    Get latest threat detection for a specific camera.
    
    Args:
        camera_id: Camera ID (e.g., "cam1")
    
    Returns:
        Latest detection result with violence status and confidence
    """
    try:
        redis_producer = await get_redis_stream_producer()
        threat = await redis_producer.get_latest_threat(camera_id)
        
        if threat is None:
            return {
                "camera_id": camera_id,
                "violence": False,
                "confidence": 0.0,
                "timestamp": None,
            }
        
        return threat
    except Exception as e:
        logger.error(f"Error getting threat for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get threat data")


@router.get("/alerts")
async def get_threat_alerts(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent threat alerts across all cameras.
    
    Args:
        limit: Maximum number of alerts to return
    
    Returns:
        List of threat alerts with violence detections
    """
    try:
        redis_producer = await get_redis_stream_producer()
        alerts = []
        
        # Collect threats from all 4 cameras
        for camera_id in ["cam1", "cam2", "cam3", "cam4"]:
            try:
                threat = await redis_producer.get_latest_threat(camera_id)
                if threat and threat.get("violence"):
                    alerts.append(threat)
            except Exception:
                continue
        
        # Sort by timestamp descending and limit
        alerts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return alerts[:limit]
    except Exception as e:
        logger.error(f"Error getting threat alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get threat alerts")


@router.get("/cameras")
async def get_all_camera_threats() -> Dict[str, Dict[str, Any]]:
    """
    Get latest threat detection for all cameras.
    
    Returns:
        Dictionary mapping camera_id to latest detection result
    """
    try:
        redis_producer = await get_redis_stream_producer()
        threats = await redis_producer.get_all_threats()
        
        if not threats:
            return {
                "cam1": {"camera_id": "cam1", "violence": False, "confidence": 0.0},
                "cam2": {"camera_id": "cam2", "violence": False, "confidence": 0.0},
                "cam3": {"camera_id": "cam3", "violence": False, "confidence": 0.0},
                "cam4": {"camera_id": "cam4", "violence": False, "confidence": 0.0},
            }
        
        return threats
    except Exception as e:
        logger.error(f"Error getting camera threats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get camera threats")
