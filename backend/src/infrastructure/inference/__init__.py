"""Inference infrastructure module."""

from src.infrastructure.inference.inference_service import (
    InferenceService,
    get_inference_service,
)

__all__ = [
    'InferenceService',
    'get_inference_service',
]
