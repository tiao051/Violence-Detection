"""
Violence Detection AI Service

A unified YOLO-based approach for violence detection in videos and images.
Previously split into detection, recognition, and classification phases,
now streamlined into a single fine-tuned YOLO pipeline.
"""

from .models import YOLOModel
from .config import SystemConfig, default_config
from .preprocessing import create_yolo_pipeline
from .utils import VideoProcessor, run_inference, batch_inference

__version__ = "2.0.0"
__all__ = [
    'YOLOModel',
    'SystemConfig',
    'default_config',
    'create_yolo_pipeline',
    'VideoProcessor',
    'run_inference',
    'batch_inference'
]
