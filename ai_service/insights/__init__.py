"""
Violence Insights Module

This module provides analytics and ML models to extract insights 
from violence detection events data.

Components:
- data: Data loading and mock data generation
- models: Analytics and ML models for pattern detection
- engine: Main InsightsEngine interface

Usage:
    from ai_service.insights import InsightsEngine
    
    engine = InsightsEngine()
    engine.load_mock_data(n_events=200)
    report = engine.get_full_report()
"""

__version__ = "0.1.0"

from .engine import InsightsEngine
from .data import ViolenceEventGenerator, ViolenceEvent
from .models import TimePatternAnalyzer, LocationAnalyzer

__all__ = [
    "InsightsEngine",
    "ViolenceEventGenerator",
    "ViolenceEvent",
    "TimePatternAnalyzer",
    "LocationAnalyzer",
]
