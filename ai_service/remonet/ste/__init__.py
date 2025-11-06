"""
Short Temporal Extractor (STE) Module

This module captures short-term motion patterns from consecutive video frames,
transforming visual changes into compact feature embeddings that represent
short-term temporal dynamics.
"""

from .extractor import STEExtractor, STEOutput, create_ste_extractor

__all__ = ['STEExtractor', 'STEOutput', 'create_ste_extractor']
