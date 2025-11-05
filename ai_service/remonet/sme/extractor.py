"""
SME (Spatial Motion Extractor) - Extract spatial motion features from frames.

This module extracts motion features by computing the difference between
consecutive frames and enhancing motion regions through dilation and thresholding.
"""

import cv2
import numpy as np
import time


class SMEPreprocessor:
    """Spatial Motion Extractor - Extract motion features from consecutive frames."""

    def __init__(self, kernel_size=3, iteration=8, target_size=(224, 224)):
        """
        Initialize SME processor.
        
        Args:
            kernel_size: Size of dilation kernel (default: 3)
            iteration: Number of dilation iterations (default: 8)
            target_size: Target frame size for resizing (default: (224, 224))
        """
        self.kernel_size = np.ones((kernel_size, kernel_size), np.uint8)
        self.iteration = iteration
        self.target_size = target_size  # (width, height)

    def process(self, frame_t, frame_t1):
        """
        Extract motion features from two consecutive frames.
        
        Args:
            frame_t: First frame (BGR, uint8)
            frame_t1: Second frame (BGR, uint8)
            
        Returns:
            roi: Motion region extracted from resized frame_t1 (uint8, 224x224)
            mask_binary: Binary motion mask (uint8, 224x224)
            diff: Grayscale motion difference (uint8, 224x224)
            elapsed_ms: Processing time in milliseconds
        """
        start = time.perf_counter()

        # Resize frames to target size (224x224) for consistent processing
        frame_t = cv2.resize(frame_t, self.target_size)
        frame_t1 = cv2.resize(frame_t1, self.target_size)

        # Calculate absolute difference between two frames
        diff = np.sqrt(np.sum((frame_t1.astype(np.float32) - frame_t.astype(np.float32)) ** 2, axis=2))
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Dilate to enhance motion areas (minimal dilation)
        mask = cv2.dilate(diff, self.kernel_size, iterations=self.iteration)
        
        # Threshold mask to binary - moderate threshold (50) for balanced motion detection
        _, mask_binary = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

        # Multiply mask with current frame to highlight motion
        roi = cv2.bitwise_and(frame_t1, frame_t1, mask=mask_binary)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return roi, mask_binary, diff, elapsed_ms
