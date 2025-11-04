"""
Model configuration for violence detection.
"""
from pathlib import Path

# Path to local model weights directory
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models" / "weights"

# Default YOLO model for real-time detection (PyTorch)
DEFAULT_MODEL = WEIGHTS_DIR / 'yolov8n.pt'

# PyTorch model variants (for training/inference)
PYTORCH_MODELS = {
    'nano': WEIGHTS_DIR / 'yolov8n.pt'
}

# ONNX model variants (for optimized inference)
ONNX_MODELS = {
    'nano_320': WEIGHTS_DIR / 'yolov8n_320.onnx'
}

# Legacy alias for backward compatibility
MODELS = PYTORCH_MODELS

__all__ = ['DEFAULT_MODEL', 'PYTORCH_MODELS', 'ONNX_MODELS', 'MODELS']
