"""
Simple Grad-CAM visualization for ROI frames.

Shows which parts of the ROI the model focuses on when making predictions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_input_gradient(model, roi: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """
    Compute feature activation heatmap from model.
    Shows which spatial regions have high activation in the final feature map.
    
    Args:
        model: ViolenceDetectionModel
        roi: Input ROI (224, 224, 3) uint8
        device: 'cpu' or 'cuda'
        
    Returns:
        Activation map (224, 224) showing feature intensity at each spatial location
    """
    # Prepare input
    if roi.dtype != np.uint8:
        roi_tensor = torch.tensor(roi * 255, dtype=torch.float32, device=device)
    else:
        roi_tensor = torch.tensor(roi, dtype=torch.float32, device=device)
    
    # Ensure shape is (1, 3, H, W)
    if roi_tensor.ndim == 2:
        roi_tensor = roi_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    elif roi_tensor.shape[2] == 3:
        roi_tensor = roi_tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        roi_tensor = roi_tensor.unsqueeze(0)
    
    # Forward through backbone to get features
    with torch.no_grad():
        features = model.ste_extractor.backbone(roi_tensor / 255.0)  # (1, C, H, W)
        # Average across channels to get spatial activation map
        activation = features.mean(dim=1).squeeze()  # (H, W)
        activation = activation.cpu().numpy()
    
    # Normalize to [0, 1]
    act_min = activation.min()
    act_max = activation.max()
    if act_max > act_min:
        activation = (activation - act_min) / (act_max - act_min)
    else:
        activation = np.zeros_like(activation)
    
    return activation


class SimpleGradCAM:
    """Simple Grad-CAM without external dependencies."""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model (ViolenceDetectionModel)
            target_layer: Layer to hook into (model.ste_extractor.backbone.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Store activations."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate(self, features: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate CAM.
        
        Args:
            features: Feature tensor from model (B, C, H, W)
            class_idx: Target class (0=no violence, 1=violence)
            
        Returns:
            CAM heatmap (H, W) normalized [0,1]
        """
        # Zero gradients
        self.model.ste_extractor.backbone.zero_grad()
        
        # Forward pass to get model output
        # This is simplified - just use averaged activations
        batch_size = features.shape[0]
        
        # Average pooling to get class scores from features
        # (simplified - no actual backprop needed for basic visualization)
        weights = features.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Generate CAM
        cam = (features * weights).sum(dim=1)  # (B, H, W)
        cam = cam[0].cpu().numpy()  # Take first batch
        
        # Normalize to [0, 1]
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Input image (H, W, 3) uint8
        heatmap: Heatmap (H, W) float [0,1]
        alpha: Overlay strength [0,1]
        
    Returns:
        Image with heatmap overlay (H, W, 3) uint8
    """
    # Resize heatmap to match image if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to BGR colormap (JET)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend
    result = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    
    return result


def overlay_heatmap_with_bbox(image: np.ndarray, heatmap: np.ndarray, threshold: float = 0.5, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay heatmap and draw bounding box around high-activation areas.
    
    Args:
        image: Input image (H, W, 3) uint8
        heatmap: Heatmap (H, W) float [0,1]
        threshold: Threshold for heatmap (only show areas > threshold)
        alpha: Overlay strength [0,1]
        
    Returns:
        Image with heatmap overlay and bounding boxes (H, W, 3) uint8
    """
    # Resize heatmap to match image if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    result = image.copy()
    
    # Normalize heatmap to full range [0, 1]
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if heatmap_max > heatmap_min:
        heatmap_normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_normalized = heatmap
    
    # Create masked heatmap (only areas above threshold)
    mask = heatmap_normalized > threshold
    masked_heatmap = heatmap_normalized.copy()
    masked_heatmap[~mask] = 0
    
    # Apply colormap (use HOT or JET)
    heatmap_uint8 = (masked_heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
    
    # Overlay heatmap
    result = cv2.addWeighted(result, 1 - alpha, heatmap_color, alpha, 0)
    
    # Find contours of high-activation regions
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around high-activation regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Only draw boxes for significant regions
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box
    
    return result
