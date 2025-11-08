"""
SME (Spatial Motion Extractor) - Extract spatial motion features from frames.

This module extracts motion features by computing the difference between
consecutive frames and enhancing motion regions through dilation and thresholding.
"""

import cv2
import numpy as np
import time


class SMEPreprocessor:
    """Preprocessor for SME - Handles frame resizing and color space conversion."""

    def __init__(self, target_size=(224, 224)):
        """
        Initialize SME preprocessor.
        
        Args:
            target_size: Target frame size (width, height), default: (224, 224)
        """
        self.target_size = target_size  # (width, height)

    def preprocess(self, frames):
        """
        Preprocess frames for SME module.
        
        Args:
            frames: Input frames, can be:
                - Sequence of frames (N, H, W, C) where C can be 1, 3, or 4
                - Single frame (H, W, C) where C can be 1, 3, or 4
                - Grayscale frame (H, W)
            
        Returns:
            processed_frames: Frames in shape (N, 224, 224, 3) or (224, 224, 3)
                            uint8, RGB color space, value range [0, 255]
        """
        # Handle single frame vs sequence
        if len(frames.shape) == 2:
            # Grayscale (H, W) - treat as single frame
            return self._process_single_frame(frames)
        elif len(frames.shape) == 3:
            # Single frame (H, W, C)
            return self._process_single_frame(frames)
        elif len(frames.shape) == 4:
            # Sequence of frames (N, H, W, C)
            return np.array([self._process_single_frame(frame) for frame in frames], dtype=np.uint8)
        else:
            raise ValueError(f"Expected 2D, 3D or 4D array, got shape {frames.shape}")

    def _process_single_frame(self, frame):
        """
        Process a single frame: resize and convert to RGB.
        
        Args:
            frame: Single frame (H, W, C) where C can be 1, 3, or 4
            
        Returns:
            processed_frame: Frame in shape (224, 224, 3), uint8, RGB
        """
        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() > 1:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)

        # Get number of channels
        if len(frame.shape) == 2:
            # Grayscale (H, W) -> convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            # Single channel (H, W, 1) -> convert to RGB
            frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            # Assume BGR and convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.shape[2] == 4:
            # BGRA -> convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")

        # Resize to target size
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        return frame.astype(np.uint8)


class SMEExtractor:
    """Spatial Motion Extractor - Extract motion features from consecutive frames."""

    def __init__(self, kernel_size=3, iteration=8):
        """
        Initialize SME processor.
        
        Args:
            kernel_size: Size of dilation kernel (default: 3)
            iteration: Number of dilation iterations (default: 8)
        """
        self.kernel_size = np.ones((kernel_size, kernel_size), np.uint8)
        self.iteration = iteration

    def process(self, frame_t, frame_t1):
        """
        Extract motion features from two consecutive frames.
        
        Args:
            frame_t: First frame (RGB, uint8, pre-resized to 224x224)
            frame_t1: Second frame (RGB, uint8, pre-resized to 224x224)
            
        Returns:
            roi: Motion region extracted from frame_t1 (uint8, 224x224)
            mask_binary: Binary motion mask (uint8, 224x224)
            diff: Grayscale motion difference (uint8, 224x224)
            elapsed_ms: Processing time in milliseconds
        """
        start = time.perf_counter()

        # Calculate Euclidean distance between two frames
        diff = np.sqrt(np.sum((frame_t1.astype(np.float32) - frame_t.astype(np.float32)) ** 2, axis=2))
        # Keep raw values, just clip to [0, 255]
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        # Dilate motion mask directly
        mask = cv2.dilate(diff, self.kernel_size, iterations=self.iteration)
        
        # Threshold mask to binary - moderate threshold (50) for balanced motion detection
        _, mask_binary = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

        # Multiply mask with current frame to highlight motion
        roi = cv2.bitwise_and(frame_t1, frame_t1, mask=mask_binary)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return roi, mask_binary, diff, elapsed_ms
