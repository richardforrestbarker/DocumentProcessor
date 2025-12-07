"""
Models package

Contains model implementations for document processing.

All models have commercial-friendly open source licenses (MIT or Apache 2.0).

Available models:
- DonutModel: OCR-free document understanding transformer (NAVER, MIT license)
- IDEFICS2Model: Multimodal vision-language model (HuggingFace, Apache 2.0)
- Phi3VisionModel: Lightweight vision-language model (Microsoft, MIT license)
- InternVLModel: Powerful vision-language model (OpenGVLab, MIT license)
- Qwen2VLModel: Vision-language model (Alibaba, Apache 2.0)
"""

from .base import BaseModel
from .donut import DonutModel
from .idefics2 import IDEFICS2Model
from .phi3_vision import Phi3VisionModel
from .internvl import InternVLModel
from .qwen2_vl import Qwen2VLModel

__all__ = [
    "BaseModel",
    "DonutModel",
    "IDEFICS2Model",
    "Phi3VisionModel",
    "InternVLModel",
    "Qwen2VLModel",
]


# Internal mapping of model identifiers to Hugging Face config `model_type`.
# This keeps the `model_type` hidden from the rest of the system.
_SUPPORTED_HF_MODEL_TYPES = {
    # InternVL family
    "opengvlab/internvl": "internvl",
    "opengvlab/internvl2": "internvl",
    # Alias patterns handled via lookup function below
    
    # IDEFICS2 family (AutoModelForVision2Seq resolves without explicit type)
    # Kept for completeness in case explicit config is needed later
    "huggingfacem4/idefics2": "idefics2",
    
    # Qwen2-VL (not typically needed, but mapped for symmetry)
    "qwen/qwen2-vl": "qwen2_vl",
    
    # Donut (uses explicit classes, not AutoModel)
    "naver-clova-ix/donut": "donut",
    
    # Phi-3-Vision (explicit classes)
    "microsoft/phi-3-vision": "phi3",
}


def get_hf_model_type(model_name_or_path: str) -> str | None:
    """
    Resolve Hugging Face `model_type` from a supported model identifier.
    Returns None if not required.
    """
    name = (model_name_or_path or "").lower()
    # Direct prefix matches
    for prefix, mtype in _SUPPORTED_HF_MODEL_TYPES.items():
        if name.startswith(prefix):
            return mtype
    # Heuristic fallbacks
    if "internvl" in name:
        return "internvl"
    if "idefics2" in name:
        return "idefics2"
    if "qwen2-vl" in name:
        return "qwen2_vl"
    if "donut" in name:
        return "donut"
    if "phi-3-vision" in name or "phi3" in name:
        return "phi3"
    return None


def get_model(model_name_or_path: str, **kwargs):
    """
    Factory function to get a model instance based on the model name/path.
    Chooses implementation by inspecting the model identifier.
    
    All models have commercial-friendly licenses (MIT or Apache 2.0).
    """
    name = (model_name_or_path or "").lower()
    
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
