"""
Models package

Contains model implementations for document processing.

Available models:
- LayoutLMv3Model: Layout-aware transformer for token classification (Microsoft)
- DonutModel: OCR-free document understanding transformer (NAVER, MIT license)
- IDEFICS2Model: Multimodal vision-language model (HuggingFace, Apache 2.0)
- Phi3VisionModel: Lightweight vision-language model (Microsoft, MIT license)
- InternVLModel: Powerful vision-language model (OpenGVLab, MIT license)
- Qwen2VLModel: Vision-language model (Alibaba, Apache 2.0)
"""

from .base import BaseModel
from .layoutlmv3 import LayoutLMv3Model
from .donut import DonutModel
from .idefics2 import IDEFICS2Model
from .phi3_vision import Phi3VisionModel
from .internvl import InternVLModel
from .qwen2_vl import Qwen2VLModel

__all__ = [
    "BaseModel",
    "LayoutLMv3Model",
    "DonutModel",
    "IDEFICS2Model",
    "Phi3VisionModel",
    "InternVLModel",
    "Qwen2VLModel",
]


def get_model(model_name_or_path: str, **kwargs):
    """
    Factory function to get a model instance based on the model name/path.
    Chooses implementation by inspecting the model identifier.
    """
    name = (model_name_or_path or "").lower()
    
    # LayoutLMv3
    if "layoutlmv3" in name or name.startswith("microsoft/layoutlmv3"):
        return LayoutLMv3Model(model_name_or_path=model_name_or_path, **kwargs)
    
    # IDEFICS2
    if "idefics" in name or name.startswith("huggingfacem4/idefics2"):
        return IDEFICS2Model(model_name_or_path=model_name_or_path, **kwargs)
    
    # Phi-3-Vision
    if "phi-3-vision" in name or "phi3" in name:
        return Phi3VisionModel(model_name_or_path=model_name_or_path, **kwargs)
    
    # InternVL
    if "internvl" in name or name.startswith("opengvlab/internvl"):
        return InternVLModel(model_name_or_path=model_name_or_path, **kwargs)
    
    # Qwen2-VL
    if "qwen2-vl" in name or "qwen2vl" in name or name.startswith("qwen/qwen2-vl"):
        return Qwen2VLModel(model_name_or_path=model_name_or_path, **kwargs)
    
    # Default to Donut
    return DonutModel(model_name_or_path=model_name_or_path, **kwargs)
