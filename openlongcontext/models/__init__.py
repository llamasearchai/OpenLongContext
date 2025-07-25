"""
OpenLongContext Models Module

This module contains implementations of various long-context transformer models
including Longformer, BigBird, Hyena, and other efficient attention mechanisms.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .longformer import LongformerForQuestionAnswering

__all__ = [
    "LongformerForQuestionAnswering",
]