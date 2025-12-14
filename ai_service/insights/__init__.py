"""
Violence Insights Module

ML-powered analytics for violence detection patterns.

Quick start:
    from insights import InsightsModel, ViolenceEvent
    
    model = InsightsModel()
    model.fit(events)
    patterns = model.get_patterns()
"""

from .core import InsightsModel, ViolenceEvent

__all__ = ["InsightsModel", "ViolenceEvent"]
