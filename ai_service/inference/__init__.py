"""Inference module for violence detection."""

from .inference_model import (
    ViolenceDetectionModel,
    InferenceConfig,
    get_violence_detection_model,
)

__all__ = [
    'ViolenceDetectionModel',
    'InferenceConfig',
    'get_violence_detection_model',
]
