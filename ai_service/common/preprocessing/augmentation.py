import cv2
import numpy as np
import time
from typing import Union, Dict, Any, Tuple

class Resize:
    """Resize image with letterbox (YOLO) or stretch (SSD) method."""
    
    def __init__(self, model_type: str = "yolo", input_size: Tuple[int, int] = None):
        """
        Args:
            model_type: "yolo" (letterbox) or "ssd" (stretch)
            input_size: Target (height, width). Defaults based on model_type.
        """
        self.model_type = model_type.lower()
        
        if "yolo" in self.model_type:
            self.input_size = input_size or (640, 640)
            self.pad_value = 114
        elif "ssd" in self.model_type:
            self.input_size = input_size or (300, 300)
            self.pad_value = 0
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def __call__(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Resize frame. Input assumed to be BGR from OpenCV.
        
        Args:
            frame: BGR image (HWC) from OpenCV
            
        Returns:
            Dict with resized image (RGB HWC uint8) and metadata
        """
        # Convert BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        if "yolo" in self.model_type:
            resized, metadata = self._letterbox_resize(image)
        else:
            resized, metadata = self._stretch_resize(image)
        
        return {"image": resized, "metadata": metadata}
    
    def _letterbox_resize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Letterbox resize (keeps aspect ratio, adds padding)."""
        target_h, target_w = self.input_size
        src_h, src_w = image.shape[:2]
        
        # Calculate scale to fit image in target size
        scale = min(target_w / src_w, target_h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded canvas
        padded = np.full((target_h, target_w, 3), self.pad_value, dtype=np.uint8)
        
        # Center image on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        metadata = {
            "method": "letterbox",
            "scale": scale,
            "offset": (x_offset, y_offset),
            "orig_size": (src_w, src_h),
            "new_size": (new_w, new_h)
        }
        
        return padded, metadata
    
    def _stretch_resize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Stretch resize (changes aspect ratio)."""
        target_h, target_w = self.input_size
        src_h, src_w = image.shape[:2]
        
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        metadata = {
            "method": "stretch",
            "scale_x": target_w / src_w,
            "scale_y": target_h / src_h,
            "orig_size": (src_w, src_h)
        }
        
        return resized, metadata

class Normalize:
    """Normalize image for model inference."""
    
    def __init__(self, model_type: str = "yolo"):
        """
        Args:
            model_type: "yolo" (div by 255) or "ssd" (ImageNet normalization)
        """
        self.model_type = model_type.lower()
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize image to float32 [0, 1] or with ImageNet stats.
        Input assumed to be uint8 RGB HWC.
        
        Args:
            data: Dict with "image" (uint8 RGB HWC)
            
        Returns:
            Dict with normalized image (float32 RGB HWC)
        """
        image = data["image"]
        
        # Convert to float32 and scale to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        if "ssd" in self.model_type:
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
        
        data["image"] = normalized
        return data

class ToCHW:
    """Convert HWC to CHW format."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data: Dict with "image" (HWC format)
            
        Returns:
            Dict with "image" (CHW format)
        """
        image = data["image"]
        
        if len(image.shape) == 3:
            data["image"] = image.transpose((2, 0, 1))
        elif len(image.shape) == 2:
            # Grayscale: add channel dimension
            data["image"] = image[np.newaxis, :, :]
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
        
        return data

class YOLOPreprocessor:
    """End-to-end YOLO preprocessing pipeline."""
    
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            input_size: Target (height, width)
        """
        self.resize = Resize(model_type="yolo", input_size=input_size)
        self.normalize = Normalize(model_type="yolo")
        self.to_chw = ToCHW()
    
    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Full preprocessing pipeline: BGR → RGB → letterbox → normalize → CHW.
        
        Args:
            frame: BGR image (HWC uint8) from OpenCV
            
        Returns:
            (CHW float32 array ready for ONNX, metadata dict)
        """
        # Resize (BGR → RGB)
        result = self.resize(frame)
        image = result["image"]
        metadata = result["metadata"]
        
        # Normalize
        result = self.normalize({"image": image})
        image = result["image"]
        
        # To CHW
        result = self.to_chw({"image": image})
        image = result["image"]
        
        # Add batch dimension (1, C, H, W)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        return image, metadata

class SMEPreprocessor:
    """Spatial Motion Extractor preprocessing pipeline."""

    def __init__(self, kernel_size = 3, iteration = 12):
        self.kernel_size = np.ones((kernel_size, kernel_size), np.uint8)
        self.iteration = iteration

    def process(self, frame_t, frame_t1):
        start = time.perf_counter()

        # Calculate absolute difference between two frames
        diff = np.sqrt(np.sum((frame_t1.astype(np.float32) - frame_t.astype(np.float32)) ** 2, axis=2))
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Dilate to enhance motion areas
        mask = cv2.dilate(diff, self.kernel_size, iterations=self.iteration)
        
        # Threshold mask to binary using Otsu's method for proper ROI extraction
        _, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Multiply mask with current frame to highlight motion
        roi = cv2.bitwise_and(frame_t1, frame_t1, mask=mask_binary)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return roi, mask_binary, diff, elapsed_ms