import cv2
import numpy as np
from typing import Union, Dict, Any

class DecodeImage:
    """Decode image from bytes to numpy array"""
    
    def __init__(self, img_mode="RGB", channel_first=False):
        """
        Args:
            img_mode (str): "RGB", "BGR", or "GRAY"
            channel_first (bool): If True, return CHW format, else HWC
        """
        self.img_mode = img_mode
        self.channel_first = channel_first
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data (dict): {"image": bytes, ...}
        Returns:
            dict: {"image": np.ndarray, ...}
        """
        img = data["image"]
        assert isinstance(img, bytes) and len(img) > 0, "Invalid input 'img' in DecodeImage"
        
        # Convert bytes to numpy array
        img_array = np.frombuffer(img, dtype="uint8")
        
        # Set decode flag
        if self.img_mode == "GRAY":
            decode_flag = cv2.IMREAD_GRAYSCALE
        else:
            decode_flag = cv2.IMREAD_COLOR
        
        # Decode image
        img = cv2.imdecode(img_array, decode_flag)
        if img is None:
            raise ValueError("Could not decode image bytes")
        
        # Convert color format
        if self.img_mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == "RGB":
            assert img.shape[2] == 3, f"Invalid shape of image: {img.shape}"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # For "BGR", keep as is
        
        # Convert to channel first if needed
        if self.channel_first:
            img = img.transpose((2, 0, 1))  # HWC -> CHW
        
        data["image"] = img
        return data


class Resize:
    """Simple preprocessor for YOLO and MobileNet-SSD models"""
    
    # Constants
    YOLO_INPUT_SIZE = (640, 640)
    SSD_INPUT_SIZE = (300, 300)
    YOLO_PAD_VALUE = 114
    SSD_PAD_VALUE = 0
    
    def __init__(self, model_type="yolo", input_size=None, pad_value=None):
        """
        Args:
            model_type (str): "yolo" or "ssd"
            input_size (tuple): Custom input size, overrides default
            pad_value (int): Custom padding value, overrides default
        """
        self.model_type = model_type.lower()
        
        # Set default configurations
        if "yolo" in self.model_type:
            default_size = self.YOLO_INPUT_SIZE
            default_pad = self.YOLO_PAD_VALUE
            self.resize_method = "letterbox"
        elif "ssd" in self.model_type:
            default_size = self.SSD_INPUT_SIZE
            default_pad = self.SSD_PAD_VALUE
            self.resize_method = "stretch"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Allow custom overrides
        self.input_size = input_size if input_size is not None else default_size
        self.pad_value = pad_value if pad_value is not None else default_pad
        
        # Initialize decoder
        self.decoder = DecodeImage(img_mode="RGB", channel_first=False)
    
    def __call__(self, input_data: Union[np.ndarray, bytes, str]):
        """Preprocess image for model
        Args:
            input_data: numpy array (HWC RGB), bytes, or file path
        """
        # Convert input to numpy array
        if isinstance(input_data, bytes):
            data = {"image": input_data}
            data = self.decoder(data)
            image = data["image"]  # RGB format
        elif isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                raise ValueError(f"Could not load image from {input_data}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        elif isinstance(input_data, np.ndarray):
            # Assume input is already RGB
            image = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Resize image
        if self.resize_method == "letterbox":
            processed_image, metadata = self.letterbox_resize(image)
        else:  # stretch
            processed_image, metadata = self.stretch_resize(image)
        
        return {
            "image": processed_image,
            "metadata": metadata
        }
    
    def letterbox_resize(self, image):
        """Letterbox resize for YOLO (keeps aspect ratio)"""
        target_height, target_width = self.input_size
        source_height, source_width = image.shape[:2]
        
        # Calculate scale to fit image in target size
        scale_factor = min(target_width / source_width, target_height / source_height)
        new_width, new_height = int(source_width * scale_factor), int(source_height * scale_factor)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Create padded canvas
        if len(image.shape) == 3:
            padded_image = np.full((target_height, target_width, 3), self.pad_value, dtype=np.uint8)
        else:
            padded_image = np.full((target_height, target_width), self.pad_value, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        if len(image.shape) == 3:
            padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        else:
            padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        metadata = {
            "method": "letterbox",
            "scale": scale_factor,
            "offset": (x_offset, y_offset),
            "original_size": (source_width, source_height)
        }
        
        return padded_image, metadata
    
    def stretch_resize(self, image):
        """Stretch resize for SSD (changes aspect ratio)"""
        target_height, target_width = self.input_size
        source_height, source_width = image.shape[:2]
        
        # Direct resize to target size
        resized_image = cv2.resize(image, (target_width, target_height))
        
        metadata = {
            "method": "stretch",
            "scale_x": target_width / source_width,
            "scale_y": target_height / source_height,
            "original_size": (source_width, source_height)
        }
        
        return resized_image, metadata
