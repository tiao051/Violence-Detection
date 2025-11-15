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
        # Get number of channels and convert to RGB
        if len(frame.shape) == 2:
            # Grayscale (H, W) -> convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            # Single channel (H, W, 1) -> convert to RGB
            frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            # Assume BGR and convert to RGB using numpy (faster than cv2.cvtColor)
            # BGR to RGB: swap channels 0 and 2
            frame = frame[:, :, ::-1]
        elif frame.shape[2] == 4:
            # BGRA -> RGB using numpy (drop alpha, swap BGR to RGB)
            frame = frame[:, :, 2::-1]
        else:
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")

        # Ensure uint8 dtype
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Always resize to target size (required for input handling)
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        return frame


class SMEExtractor:
    """Spatial Motion Extractor - Extract motion features from consecutive frames."""

    def __init__(self, kernel_size=3, iteration=8, threshold=50, use_squared_distance=False):
        """
        Initialize SME processor.
        
        Args:
            kernel_size: Size of dilation kernel (default: 3)
            iteration: Number of dilation iterations (default: 8)
            threshold: Binary threshold for motion mask (default: 50)
                      Higher values = stricter motion detection
                      Lower values = more sensitive to small changes
            use_squared_distance: If True, use squared Euclidean distance (faster)
                                 If False, use Euclidean distance (default, more intuitive)
        """
        self.kernel_size = np.ones((kernel_size, kernel_size), np.uint8)
        self.iteration = iteration
        self.threshold = threshold
        self.use_squared_distance = use_squared_distance

    def _validate_frames(self, frame_t, frame_t1):
        """
        Validate input frames.
        
        Args:
            frame_t: First frame to validate
            frame_t1: Second frame to validate
            
        Raises:
            ValueError: If frames don't meet requirements
        """
        if frame_t.shape != (224, 224, 3):
            raise ValueError(f"frame_t must be shape (224, 224, 3), got {frame_t.shape}")
        if frame_t.dtype != np.uint8:
            raise ValueError(f"frame_t must be uint8, got {frame_t.dtype}")

    def process(self, frame_t, frame_t1):
        """
        Extract motion features from two consecutive frames.
        
        Args:
            frame_t: First frame (RGB, uint8, 224x224)
            frame_t1: Second frame (RGB, uint8, 224x224)
            
        Returns:
            roi: Motion region extracted from frame_t1 (float32 [0, 1], 224x224, 3)
                 Only pixels with motion (mask > threshold) are retained, normalized to [0, 1]
            mask_binary: Binary motion mask (uint8, 224x224)
                        255 where motion detected, 0 elsewhere
            diff: Raw motion difference map (uint8, 224x224)
                 Euclidean distance between frame_t and frame_t1
            elapsed_ms: Processing time in milliseconds
        """
        self._validate_frames(frame_t, frame_t1)
        
        start = time.perf_counter()

        # Calculate motion magnitude between frames using uint32 for intermediate calculations
        # Using Euclidean distance per pixel across RGB channels
        # Formula: sqrt((R1-R2)² + (G1-G2)² + (B1-B2)²)
        
        # Compute differences using int32 to avoid overflow (uint8: -255 to 255)
        frame_diff = frame_t1.astype(np.int32) - frame_t.astype(np.int32)
        
        if self.use_squared_distance:
            # Squared distance: faster, preserves ordering
            # Sum squared differences across RGB channels
            diff = np.sum(frame_diff ** 2, axis=2, dtype=np.int32)
            diff = np.sqrt(diff.astype(np.float32))
        else:
            # Standard Euclidean distance: more intuitive
            # sum(x^2 + y^2 + z^2) then sqrt
            diff = np.sqrt(np.sum(frame_diff ** 2, axis=2, dtype=np.int32).astype(np.float32))
        
        # Clip to uint8 range [0, 255] while preserving raw distance values
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        # Enhance motion regions using morphological dilation
        # This expands motion areas to fill small gaps
        mask_dilated = cv2.dilate(diff, self.kernel_size, iterations=self.iteration)
        
        # Threshold to create binary motion mask
        # Pixels above threshold are marked as motion (255), below as static (0)
        _, mask_binary = cv2.threshold(mask_dilated, self.threshold, 255, cv2.THRESH_BINARY)

        # Extract motion regions from original frame using dot product
        # Normalize frame to [0, 1], then multiply by normalized mask
        # (frame / 255.0) * (mask / 255.0) → float32 [0, 1]
        roi = frame_t1.astype(np.float32) / 255.0 * mask_binary.astype(np.float32)[:, :, np.newaxis] / 255.0

        elapsed_ms = (time.perf_counter() - start) * 1000

        return roi, mask_binary, diff, elapsed_ms

    def process_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        Process a batch of frames to extract motion features.
        
        Takes N frames and produces N motion frames (with last frame duplicated).
        Processes consecutive frame pairs: (frame[i], frame[i+1]) → motion[i]
        Then duplicates the last motion frame to match input count.
        
        Args:
            frames: Batch of frames (N, 224, 224, 3), RGB, uint8
                   Example: 30 frames for STE processing
        
        Returns:
            motion_frames: Motion features (N, 224, 224, 3), float32 [0, 1]
                          N = same as input frame count
        """
        if len(frames) < 2:
            raise ValueError(f"Expected at least 2 frames, got {len(frames)}")
        
        motion_frames = []
        
        # Process consecutive frame pairs
        for i in range(len(frames) - 1):
            frame_t = frames[i]
            frame_t1 = frames[i + 1]
            roi, _, _, _ = self.process(frame_t, frame_t1)
            motion_frames.append(roi)
        
        # Convert to numpy array
        motion_array = np.array(motion_frames, dtype=np.float32)  # (N-1, 224, 224, 3)
        
        # Duplicate last motion frame to match input count
        # 30 input frames → 29 motion frames → duplicate last → 30 motion frames
        if len(motion_array) < len(frames):
            last_frame = motion_array[-1:].copy()  # Shape: (1, 224, 224, 3)
            motion_array = np.vstack([motion_array, last_frame])
        
        return motion_array
