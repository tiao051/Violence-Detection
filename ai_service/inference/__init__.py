"""
AI Service Inference Module

Provides optimized inference engines for YOLOv8 models:
- ONNXYOLOInference: CPU-optimized ONNX inference with NMS
"""

from ai_service.inference.onnx_inference import ONNXYOLOInference

__all__ = ["ONNXYOLOInference"]
