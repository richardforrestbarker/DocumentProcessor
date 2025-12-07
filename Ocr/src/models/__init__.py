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


def get_model(model_name_or_path: str, **kwargs):
    """
    Factory function to get a model instance based on the model name/path.
    Chooses implementation by inspecting the model identifier.
    """
    name = (model_name_or_path or "").lower()
    # Simple heuristics
    if "layoutlmv3" in name or name.startswith("microsoft/layoutlmv3"):
        return LayoutLMv3Model(model_name_or_path=model_name_or_path, **kwargs)
    if "idefics" in name or name.startswith("huggingfacem4/idefics2"):
        return IDEFICS2Model(model_name_or_path=model_name_or_path, **kwargs)
    # Default to Donut
    return DonutModel(model_name_or_path=model_name_or_path, **kwargs)
