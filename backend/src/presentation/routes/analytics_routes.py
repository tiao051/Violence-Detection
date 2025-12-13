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

from insights import InsightsModel
from insights.data import ViolenceEvent
import pandas as pd

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Global model instance (loaded once)
_model: Optional[InsightsModel] = None
_model_loaded: bool = False

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
    csv_path = os.path.join(data_dir, 'violence_events_100k.csv')
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        _model = InsightsModel.load(model_path)
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No training data found at {csv_path}")
        
        print(f"Training model from {csv_path} (this may take 1-2 minutes)...")
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame rows to ViolenceEvent objects
        events = []
        for _, row in df.iterrows():
            event = ViolenceEvent.from_dict(row.to_dict(), event_id=row.get('event_id', ''))
            events.append(event)
        
        _model = InsightsModel()
        _model.fit(events)
        
        # Save for next time
        _model.save(model_path)
        print(f"Model saved to {model_path}")
    
    _model_loaded = True
    return _model


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
        model = get_model()
        return model.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_patterns() -> List[Dict[str, Any]]:
    """Get K-means cluster patterns."""
    try:
        model = get_model()
        return model.get_patterns()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_rules(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get FP-Growth association rules."""
    try:
        model = get_model()
        return model.get_rules(top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions")
async def get_high_risk_conditions(top_n: int = 10) -> List[Dict[str, Any]]:
    """Get high risk conditions from Random Forest."""
    try:
        model = get_model()
        return model.get_high_risk_conditions(top_n=top_n)
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
        model = get_model()
        return model.get_full_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
