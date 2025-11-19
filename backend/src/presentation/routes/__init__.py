"""Presentation routes package."""

from src.presentation.routes.auth_routes import router as auth_router
from src.presentation.routes.websocket_routes import router as websocket_router


__all__ = ['auth_router', 'websocket_router']