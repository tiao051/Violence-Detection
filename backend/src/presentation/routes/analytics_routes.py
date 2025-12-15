"""
Analytics API Routes

Provides endpoints to serve ML insights from InsightsModel:
- GET /api/analytics/summary - Quick stats
- GET /api/analytics/patterns - K-means clusters
- GET /api/analytics/rules - FP-Growth association rules  
- GET /api/analytics/predictions - High risk conditions
- POST /api/analytics/predict - Predict risk for specific conditions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import sys

# Add ai_service to path (works for both local and Docker)
# In Docker: /app/ai_service (mounted via docker-compose volume)
# Local: ../../ai_service relative to backend folder
ai_service_paths = [
    '/app/ai_service',  # Docker path
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service'),  # Local path
]
for path in ai_service_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from insights import InsightsModel, ViolenceEvent
import pandas as pd
import threading

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Global model instance (loaded once)
_model: Optional[InsightsModel] = None
_model_loaded: bool = False

# Cache for computed results (avoid recalculating on every request)
_cache: Dict[str, Any] = {}
_cache_ready: bool = False
_cache_lock = threading.Lock()  # Prevent race conditions

def get_model() -> InsightsModel:
    """Get or load the insights model."""
    global _model, _model_loaded
    
    if _model_loaded and _model is not None:
        return _model
    
    # Find insight data directory (works for both Docker and local)
    data_dir_options = [
        '/app/ai_service/insights/data',  # Docker path
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service', 'insights', 'data'),  # Local
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
        print(f"Loading pre-trained model from {model_path}")
        _model = InsightsModel.load(model_path)
    elif os.path.exists(csv_path):
        # Train from CSV if available
        print(f"Training model from {csv_path} (this may take 1-2 minutes)...")
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame rows to ViolenceEvent objects
        events = []
        for _, row in df.iterrows():
            event = ViolenceEvent.from_dict(row.to_dict())
            events.append(event)
        
        _model = InsightsModel()
        _model.fit(events)
        
        # Save for next time
        _model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        # No pkl, no csv - generate CSV first then train
        print("No training data found. Generating analytics_events.csv (20,000 events)...")
        
        from insights.data import ViolenceEventGenerator
        from datetime import datetime, timedelta
        
        generator = ViolenceEventGenerator(seed=42)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        events = generator.generate_mixed(
            n_events=20000,
            violence_ratio=0.3,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Save CSV
        df = generator.to_dataframe(events)
        df.to_csv(csv_path, index=False)
        print(f"Generated CSV saved to {csv_path}")
        
        # Train model
        _model = InsightsModel()
        _model.fit(events)
        
        # Save model
        _model.save(model_path)
        print(f"Model saved to {model_path}")
    
    _model_loaded = True
    return _model


def get_cached_results() -> Dict[str, Any]:
    """Get or compute cached results for all analytics endpoints."""
    global _cache, _cache_ready
    
    # Fast path - return if already cached
    if _cache_ready:
        return _cache
    
    # Use lock to prevent multiple threads from computing cache simultaneously
    with _cache_lock:
        # Double-check after acquiring lock
        if _cache_ready:
            return _cache
        
        print("Pre-computing analytics results (one-time)...")
        import time
        start = time.time()
        
        model = get_model()
        
        # Compute all results once
        _cache = {
            "summary": model.get_summary(),
            "patterns": model.get_patterns(),
            "rules": model.get_rules(top_n=20),
            "high_risk": model.get_high_risk_conditions(top_n=10),
            "full_report": model.get_full_report(),
        }
        
        _cache_ready = True
        print(f"Analytics cache ready in {time.time() - start:.1f}s")
    
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
    """
    Get hotspot forecast for a specific camera.
    
    Uses K-means clusters to predict risk levels for the next N hours.
    
    Args:
        camera: Camera name (URL encoded)
        hours: Number of hours to forecast (default 12, max 24)
        
    Returns:
        Forecast with hourly predictions, peak times, and warnings
    """
    try:
        model = get_model()
        hours = min(hours, 24)  # Cap at 24 hours
        return model.get_forecast(camera=camera, hours_ahead=hours)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

