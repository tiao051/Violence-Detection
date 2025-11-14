"""
Short Temporal Extractor (STE) Module

This module captures short-term motion patterns from consecutive video frames,
transforming visual changes into compact feature embeddings that represent
short-term temporal dynamics.

Supports multiple CNN backbones for feature extraction:
- MobileNetV2 (default)
- MobileNetV3 Small/Large
- EfficientNet B0
- MNasNet
"""

from .extractor import STEExtractor, STEOutput, BackboneType, BACKBONE_CONFIG

__all__ = ['STEExtractor', 'STEOutput', 'BackboneType', 'BACKBONE_CONFIG']
