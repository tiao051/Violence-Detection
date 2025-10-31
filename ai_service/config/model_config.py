"""
Model configuration for violence detection.
"""
from pathlib import Path

# Path to local model weights directory
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models" / "weights"

# Default YOLO model for real-time detection (local path)
DEFAULT_MODEL = WEIGHTS_DIR / 'yolov8n.pt'

# Model variants available (local paths)
MODELS = {
    'nano': WEIGHTS_DIR / 'yolov8n.pt',      # Fastest, lowest accuracy
    'small': WEIGHTS_DIR / 'yolov8s.pt',     # Balanced
    'medium': WEIGHTS_DIR / 'yolov8m.pt',    # Better accuracy
    'large': WEIGHTS_DIR / 'yolov8l.pt',     # High accuracy, slower
}

__all__ = ['DEFAULT_MODEL', 'MODELS']
