"""Presentation routes package."""

from src.presentation.routes.threat_routes import router as threat_router
from src.presentation.routes.auth_routes import router as auth_router

__all__ = ['threat_router', 'auth_router']
