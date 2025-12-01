"""
Models package

Contains model implementations for receipt processing.

Available models:
- LayoutLMv3Model: Layout-aware transformer for token classification (Microsoft)
- DonutModel: OCR-free document understanding transformer (NAVER, MIT license)
- IDEFICS2Model: Multimodal vision-language model (HuggingFace, Apache 2.0)
"""

from .base import BaseModel
from .layoutlmv3 import LayoutLMv3Model
from .donut import DonutModel
from .idefics2 import IDEFICS2Model

__all__ = [
    "BaseModel",
    "LayoutLMv3Model",
    "DonutModel",
    "IDEFICS2Model",
]


def get_model(model_type: str, **kwargs):
    """
    Factory function to get a model instance by type.
    
    Args:
        model_type: One of 'layoutlmv3', 'donut', 'idefics2'
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    model_type_lower = model_type.lower()
    
    if model_type_lower == "layoutlmv3":
        return LayoutLMv3Model(**kwargs)
    elif model_type_lower == "donut":
        return DonutModel(**kwargs)
    elif model_type_lower == "idefics2":
        return IDEFICS2Model(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: layoutlmv3, donut, idefics2"
        )
