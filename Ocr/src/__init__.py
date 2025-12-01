"""
Bardcoded.Ocr Service

Python package for receipt OCR and structured data extraction.
"""

__version__ = "0.1.0"
__author__ = "Bardcode Team"
__description__ = "Receipt OCR with PaddleOCR and LayoutLMv3"

from .receipt_processor import ReceiptProcessor
from .models import LayoutLMv3Model

__all__ = [
    "ReceiptProcessor",
    "LayoutLMv3Model",
]
