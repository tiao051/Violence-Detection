from typing import Dict, Any, List


class BasePreprocessor:
    """Base class for preprocessing operations"""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing operation"""
        pass


class Compose:
    """Chain multiple preprocessing operations"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, data):
        """Apply all transformations"""
        # Convert input to dict format if needed
        if not isinstance(data, dict):
            data = {"image": data}
        
        # Apply each transform
        for transform in self.transforms:
            data = transform(data)
        
        return data


# Simple factory functions
def create_yolo_pipeline(input_size=None):
    """Create YOLO preprocessing pipeline"""
    from .augmentation import DecodeImage, Resize, Normalize, ToCHWImage
    
    return Compose([
        DecodeImage(img_mode="RGB"),
        Resize("yolo", input_size=input_size), 
        Normalize("yolo"),
        ToCHWImage()
    ])


def create_ssd_pipeline(input_size=None):
    """Create SSD preprocessing pipeline"""
    from .augmentation import DecodeImage, Resize, Normalize
    
    return Compose([
        DecodeImage(img_mode="RGB"),
        Resize("ssd", input_size=input_size),
        Normalize("ssd")
    ])