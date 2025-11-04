"""Detection module."""
from .pytorch_detector import PyTorchDetector
from .onnx_inference import ONNXYOLOInference

__all__ = ['PyTorchDetector', 'ONNXYOLOInference']
