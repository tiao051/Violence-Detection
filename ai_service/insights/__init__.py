"""
Violence Insights Module

Provides ML models to extract insights from violence detection events.

Main Interface:
    InsightsModel - Unified model combining K-means, FP-Growth, Random Forest

Usage:
    from ai_service.insights import InsightsModel
    
    model = InsightsModel()
    model.fit(events)  # or model.fit_from_mock(500)
    report = model.get_full_report()
"""

__version__ = "0.2.0"

# Main unified model
from .insights_model import InsightsModel

# Data utilities
from .data import ViolenceEventGenerator, ViolenceEvent

# Individual ML models
from .models import (
    ClusterAnalyzer,
    AssociationRuleAnalyzer,
    RiskPredictor,
)

__all__ = [
    # Main interface
    "InsightsModel",
    # Data
    "ViolenceEventGenerator",
    "ViolenceEvent",
    # Models
    "ClusterAnalyzer",
    "AssociationRuleAnalyzer",
    "RiskPredictor",
]

