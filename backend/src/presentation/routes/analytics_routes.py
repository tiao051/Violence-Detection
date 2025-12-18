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
}
_computing_tasks: Dict[str, bool] = {
    "summary": False,
    "patterns": False,
    "rules": False,
    "high_risk": False,
    "full_report": False,
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


def start_analytics_computation() -> None:
    """Start all analytics computations in parallel (non-blocking)."""
    global _computing_tasks
    
    tasks = [
        ("summary", _compute_summary),
        ("patterns", _compute_patterns),
        ("rules", _compute_rules),
        ("high_risk", _compute_high_risk),
        ("full_report", _compute_full_report),
    ]
    
    for task_name, task_func in tasks:
        if not _computing_tasks[task_name] and _cache_state[task_name] is None:
            _computing_tasks[task_name] = True
            thread = threading.Thread(target=task_func, daemon=True)
            thread.start()
            logger.info(f"Started background task: {task_name}")
    
    logger.info("All analytics tasks started in parallel")


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

