"""
Bardcoded.Ocr Service

Python package for receipt OCR and structured data extraction.
"""

__version__ = "0.1.0"
__author__ = "Bardcode Team"
__description__ = "Document OCR with PaddleOCR and vision-language models"

from .receipt_processor import ReceiptProcessor
from .models import DonutModel, IDEFICS2Model, Phi3VisionModel, InternVLModel, Qwen2VLModel

__all__ = [
    "ReceiptProcessor",
    "DonutModel",
    "IDEFICS2Model",
    "Phi3VisionModel",
    "InternVLModel",
    "Qwen2VLModel",
]
