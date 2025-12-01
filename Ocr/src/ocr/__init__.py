"""
OCR engines module.

Provides text detection and recognition using PaddleOCR and Tesseract.
"""

from .ocr_engine import OcrEngine, PaddleOcrEngine, TesseractOcrEngine, create_ocr_engine

__all__ = [
    'OcrEngine',
    'PaddleOcrEngine',
    'TesseractOcrEngine',
    'create_ocr_engine',
]
