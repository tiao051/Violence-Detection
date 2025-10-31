"""
Inference utilities for violence detection
"""

import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path


def run_inference(model, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Run inference on a single image
    
    Args:
        model: YOLO model instance
        image (np.ndarray): Input image (RGB)
        conf_threshold (float): Confidence threshold
        
    Returns:
        List[Dict]: Detections with bbox, confidence, and class
    """
    # TODO: Implement single image inference
    predictions = model.predict(image)
    
    # Filter by confidence
    filtered_predictions = [
        pred for pred in predictions 
        if pred['confidence'] >= conf_threshold
    ]
    
    return filtered_predictions


def batch_inference(model, images: List[np.ndarray], conf_threshold: float = 0.5) -> List[List[Dict[str, Any]]]:
    """
    Run inference on a batch of images
    
    Args:
        model: YOLO model instance
        images (List[np.ndarray]): List of input images (RGB)
        conf_threshold (float): Confidence threshold
        
    Returns:
        List[List[Dict]]: List of detections for each image
    """
    # TODO: Implement batch inference if supported
    # For now, run sequential inference
    results = []
    for image in images:
        predictions = run_inference(model, image, conf_threshold)
        results.append(predictions)
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Save inference results to file
    
    Args:
        results (List[Dict]): Inference results
        output_path (Union[str, Path]): Path to save results
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
