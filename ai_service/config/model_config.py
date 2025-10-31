"""
Model configuration for violence detection.
"""

# Default YOLO model for real-time detection
DEFAULT_MODEL = 'yolov8n.pt'

# Model variants available
MODELS = {
    'nano': 'yolov8n.pt',      # Fastest, lowest accuracy
    'small': 'yolov8s.pt',     # Balanced
    'medium': 'yolov8m.pt',    # Better accuracy
    'large': 'yolov8l.pt',     # High accuracy, slower
}

__all__ = ['DEFAULT_MODEL', 'MODELS']
