"""
Model configuration for violence detection.
"""
from pathlib import Path
from enum import Enum

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

# ==================== ReMotNet Configuration ====================

class STEBackbone(str, Enum):
    """Supported CNN backbones for STE (Short Temporal Extractor) feature extraction."""
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    EFFICIENTNET_B0 = "efficientnet_b0"
    MNASNET = "mnasnet"


# Default STE backbone (MobileNetV2 for balance of speed and accuracy)
DEFAULT_STE_BACKBONE = STEBackbone.MOBILENET_V2

# STE backbone configuration
STE_BACKBONE_CONFIG = {
    STEBackbone.MOBILENET_V2: {
        'out_channels': 1280,
        'spatial_size': 7,
        'params_millions': 3.5,
        'speed_ms': 8.2,  # Approximate inference time per composite on GPU
        'description': 'Balanced backbone - default choice'
    },
    STEBackbone.MOBILENET_V3_SMALL: {
        'out_channels': 576,
        'spatial_size': 7,
        'params_millions': 2.5,
        'speed_ms': 6.1,  # Fastest
        'description': 'Lightweight, fastest inference'
    },
    STEBackbone.MOBILENET_V3_LARGE: {
        'out_channels': 960,
        'spatial_size': 7,
        'params_millions': 5.4,
        'speed_ms': 9.8,
        'description': 'Larger capacity, better accuracy'
    },
    STEBackbone.EFFICIENTNET_B0: {
        'out_channels': 1280,
        'spatial_size': 7,
        'params_millions': 5.3,
        'speed_ms': 10.5,  # Slowest
        'description': 'Highest accuracy, slower inference'
    },
    STEBackbone.MNASNET: {
        'out_channels': 1280,
        'spatial_size': 7,
        'params_millions': 4.3,
        'speed_ms': 7.8,
        'description': 'Mobile NAS optimized'
    },
}

__all__ = [
    'DEFAULT_MODEL', 'PYTORCH_MODELS', 'ONNX_MODELS', 'MODELS',
    'STEBackbone', 'DEFAULT_STE_BACKBONE', 'STE_BACKBONE_CONFIG'
]
