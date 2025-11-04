"""
Violence Detection AI Service

A unified YOLO-based approach for violence detection in videos and images.
Streamlined pipeline using YOLOv8 via RealtimeDetector.
"""

from .detection import PyTorchDetector, ONNXYOLOInference

__version__ = "2.0.0"
__all__ = [
    'PyTorchDetector',
    'ONNXYOLOInference',
]