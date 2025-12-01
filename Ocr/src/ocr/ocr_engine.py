"""
OCR engines for text detection and recognition.

Supports PaddleOCR and Tesseract for receipt text extraction.
"""

import logging
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OcrEngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def detect_and_recognize(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detect text regions and recognize text.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            List of detected words with boxes and text
        """
        pass


class PaddleOcrEngine(OcrEngine):
    """
    PaddleOCR engine for text detection and recognition.
    
    Uses PP-StructureV3 for high-accuracy OCR on receipts.
    """
    
    def __init__(
        self,
        lang: str = "en",
        detection_mode: str = "word",
        use_gpu: bool = True
    ):
        """
        Initialize PaddleOCR engine.
        
        Args:
            lang: Language code ('en', 'ch', etc.)
            detection_mode: 'word' or 'line' level detection
            use_gpu: Whether to use GPU acceleration
        """
        self.lang = lang
        self.detection_mode = detection_mode
        self.use_gpu = use_gpu
        self._ocr = None
        
        logger.info(f"Initialized PaddleOCR engine (lang={lang}, gpu={use_gpu})")
    
    def _load_ocr(self):
        """Lazy load PaddleOCR model."""
        if self._ocr is not None:
            return
        
        try:
            from paddleocr import PaddleOCR
            
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang
            )
            logger.info("PaddleOCR model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle"
            )
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            raise
    
    def detect_and_recognize(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detect and recognize text using PaddleOCR.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of words with format:
            [
                {
                    'text': 'TOTAL',
                    'box': [x0, y0, x1, y1],
                    'confidence': 0.98
                },
                ...
            ]
        """
        self._load_ocr()
        
        logger.info("Running PaddleOCR detection and recognition")
        
        try:
            # Run OCR
            result = self._ocr.ocr(image)
            
            if result is None or len(result) == 0:
                logger.warning("PaddleOCR returned no results")
                return []
            
            words = []
            
            # PaddleOCR returns nested list structure
            for page_result in result:
                if page_result is None:
                    continue
                    
                for line in page_result:
                    if line is None or len(line) < 2:
                        continue
                    
                    box_points = line[0]  # 4 corner points
                    text_info = line[1]   # (text, confidence)
                    
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                    else:
                        continue
                    
                    # Skip empty text
                    if not text.strip():
                        continue
                    
                    # Convert 4-point box to [x0, y0, x1, y1]
                    x_coords = [p[0] for p in box_points]
                    y_coords = [p[1] for p in box_points]
                    box = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    ]
                    
                    words.append({
                        'text': text,
                        'box': box,
                        'confidence': confidence
                    })
            
            logger.info(f"PaddleOCR detected {len(words)} text regions")
            return words
            
        except Exception as e:
            logger.error(f"PaddleOCR detection failed: {e}")
            raise


class TesseractOcrEngine(OcrEngine):
    """
    Tesseract OCR engine fallback.
    
    Used when PaddleOCR is not available or as a secondary option.
    """
    
    def __init__(self, lang: str = "eng", config: str = "--psm 6", use_gpu: bool = True):
        """
        Initialize Tesseract engine.
        
        Args:
            lang: Language code
            config: Tesseract configuration string
            use_gpu: Whether to use GPU acceleration
        """
        self.lang = lang
        self.config = config
        self.use_gpu = use_gpu
        logger.info(f"Initialized Tesseract engine (lang={lang}, use_gpu={use_gpu})")
        
        # Verify Tesseract is available
        self._verify_installation()
    
    def _verify_installation(self):
        """Verify Tesseract is installed."""
        try:
            import pytesseract
            # Try to get version to verify installation
            pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract"
            )
        except Exception as e:
            logger.warning(
                f"Tesseract may not be installed properly: {e}. "
                "Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract"
            )
    
    def detect_and_recognize(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detect and recognize text using Tesseract.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            List of words with boxes and text
        """
        logger.info("Running Tesseract OCR")
        
        try:
            import pytesseract
            from PIL import Image
            import numpy as np
            
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Get word-level data
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            words = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                conf = data['conf'][i]
                
                # Tesseract returns -1 for invalid entries
                if conf < 0:
                    continue
                
                box = [
                    int(data['left'][i]),
                    int(data['top'][i]),
                    int(data['left'][i] + data['width'][i]),
                    int(data['top'][i] + data['height'][i])
                ]
                
                words.append({
                    'text': text,
                    'box': box,
                    'confidence': conf / 100.0  # Tesseract returns 0-100
                })
            
            logger.info(f"Tesseract detected {len(words)} words")
            return words
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise


def create_ocr_engine(engine_type: str = "paddle", **kwargs) -> OcrEngine:
    """
    Factory function to create OCR engine.
    
    Args:
        engine_type: Type of engine ('paddle' or 'tesseract')
        **kwargs: Additional arguments for engine initialization
        
    Returns:
        OcrEngine instance
    """
    engine_type = engine_type.lower()
    
    if engine_type == "paddle":
        try:
            return PaddleOcrEngine(**kwargs)
        except ImportError:
            logger.warning("PaddleOCR not available, falling back to Tesseract")
            return TesseractOcrEngine()
    elif engine_type == "tesseract":
        return TesseractOcrEngine(**kwargs)
    else:
        raise ValueError(f"Unknown OCR engine type: {engine_type}")
