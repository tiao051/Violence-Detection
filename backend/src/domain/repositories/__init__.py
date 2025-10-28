"""Repository Interfaces - Abstract data access contracts"""
from .camera_repository import ICameraRepository
from .stream_repository import IStreamRepository

__all__ = ["ICameraRepository", "IStreamRepository"]
