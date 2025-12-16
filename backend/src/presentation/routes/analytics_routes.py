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

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_model: Optional[InsightsModel] = None
_model_loaded: bool = False

_cache: Dict[str, Any] = {}
_cache_ready: bool = False
_cache_lock = threading.Lock()

def get_model() -> InsightsModel:
    """Get or load the insights model."""
    global _model, _model_loaded
    
    if _model_loaded and _model is not None:
        return _model
    
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
    csv_path = os.path.join(data_dir, 'analytics_events.csv')
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        _model = InsightsModel.load(model_path)
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        print(f"Training model from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        events = []
        for _, row in df.iterrows():
            event = ViolenceEvent.from_dict(row.to_dict())
            events.append(event)
        
        _model = InsightsModel()
        _model.fit(events)
        _model.save(model_path)
        print(f"Model saved to {model_path}")
    
    _model_loaded = True
    return _model


def get_cached_results() -> Dict[str, Any]:
    """Get or compute cached results for all analytics endpoints."""
    global _cache, _cache_ready
    
    if _cache_ready:
        return _cache
    
    with _cache_lock:
        if _cache_ready:
            return _cache
        
        print("Pre-computing analytics results...")
        
        model = get_model()
        
        _cache = {
            "summary": model.get_summary(),
            "patterns": model.get_patterns(),
            "rules": model.get_rules(top_n=20),
            "high_risk": model.get_high_risk_conditions(top_n=10),
            "full_report": model.get_full_report(),
        }
        
        _cache_ready = True
    
    return _cache


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
@router.get("/summary")
async def get_summary() -> Dict[str, Any]:
    """Get quick summary of insights."""
    try:
        cache = get_cached_results()
        return cache["summary"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_patterns() -> List[Dict[str, Any]]:
    """Get K-means cluster patterns."""
    try:
        cache = get_cached_results()
        return cache["patterns"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_rules(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get FP-Growth association rules."""
    try:
        cache = get_cached_results()
        # Return cached rules, sliced to top_n
        return cache["rules"][:top_n]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions")
async def get_high_risk_conditions(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get high risk conditions from Random Forest."""
    try:
        cache = get_cached_results()
        return cache["high_risk"][:top_n]
    except Exception as e:
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
    """Get comprehensive report from all 3 models."""
    try:
        cache = get_cached_results()
        return cache["full_report"]
    except Exception as e:
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

