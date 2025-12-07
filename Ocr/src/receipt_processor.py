"""
Receipt Processor

Main orchestration class for receipt OCR and field extraction pipeline.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ReceiptProcessor:
    """
    Main processor for receipt OCR and structured extraction.
    
    Pipeline stages:
    1. Image preprocessing (denoise, deskew, normalize)
    2. Text detection and OCR
    3. Tokenization and box mapping
    4. Model inference (LayoutLMv3)
    5. Postprocessing and field extraction
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        ocr_engine: str = "paddle",
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize receipt processor.
        
        Args:
            model_name: HuggingFace model name or local path
            ocr_engine: OCR engine to use ('paddle' or 'tesseract')
            device: Device for inference ('auto', 'cuda', or 'cpu')
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.ocr_engine_type = ocr_engine
        self.device = self._resolve_device(device)
        self.config = config or {}
        
        logger.info(f"Initializing ReceiptProcessor with model={model_name}, device={self.device}")
        
        # Lazy load heavy dependencies
        self._model = None
        self._ocr = None
        self._preprocessor = None
        self._field_extractor = None
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("CUDA available, using GPU")
                    return "cuda"
                else:
                    logger.info("CUDA not available, using CPU")
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch not installed, defaulting to CPU")
                return "cpu"
        return device
    
    def _get_preprocessor(self):
        """Get or create image preprocessor."""
        if self._preprocessor is None:
            from .preprocessing.image_preprocessor import ImagePreprocessor
            self._preprocessor = ImagePreprocessor(
                denoise=self.config.get('denoise', True),
                deskew=self.config.get('deskew', True),
                enhance_contrast=self.config.get('enhance_contrast', True)
            )
        return self._preprocessor
    
    def _get_ocr(self):
        """Get or create OCR engine."""
        if self._ocr is None:
            from .ocr.ocr_engine import create_ocr_engine
            if self.ocr_engine_type == "paddle":
                self._ocr = create_ocr_engine(
                    self.ocr_engine_type,
                    lang="en",
                    detection_mode=self.config.get('detection_mode', 'word')
                )
            else:
                self._ocr = create_ocr_engine(
                    self.ocr_engine_type,
                    lang="eng"
                )
        return self._ocr
    
    def _get_model(self):
        """Get or create LayoutLMv3 model."""
        if self._model is None:
            from .models.layoutlmv3 import LayoutLMv3Model
            self._model = LayoutLMv3Model(
                model_name_or_path=self.model_name,
                device=self.device
            )
            self._model.load()
        return self._model
    
    def _get_field_extractor(self):
        """Get or create field extractor."""
        if self._field_extractor is None:
            from .postprocessing.field_extractor import FieldExtractor
            self._field_extractor = FieldExtractor(
                min_confidence=self.config.get('min_confidence', 0.5)
            )
        return self._field_extractor
    
    def process_receipt(
        self,
        image_paths: List[str],
        job_id: Optional[str] = None,
        skip_model: bool = False
    ) -> Dict[str, Any]:
        """
        Process receipt images and extract structured data.
        
        Args:
            image_paths: List of image file paths
            job_id: Optional job identifier
            skip_model: Skip LayoutLM model inference (use only heuristics)
            
        Returns:
            Dictionary with extracted receipt data
        """
        logger.info(f"Processing {len(image_paths)} receipt page(s)")
        
        result = {
            "job_id": job_id or f"job-{hash(tuple(image_paths)) % 100000:05d}",
            "status": "done",
            "pages": [],
            # Document classification
            "document_type": None,
            # Common fields
            "vendor_name": None,
            "merchant_address": None,
            "date": None,
            "total_amount": None,
            "subtotal": None,
            "tax_amount": None,
            "currency": None,
            "line_items": [],
            # Invoice-specific fields
            "invoice_number": None,
            "due_date": None,
            "payment_terms": None,
            "customer_name": None,
            "customer_address": None,
            "po_number": None,
            # Bill-specific fields
            "account_number": None,
            "billing_period": None,
            "previous_balance": None,
            "current_charges": None,
            "amount_due": None,
            # Receipt-specific fields
            "payment_method": None,
            "cashier_name": None,
            "register_number": None,
            # General fields
            "discount": None,
            "shipping": None,
            "notes": None
        }
        
        all_words = []
        
        try:
            for page_num, image_path in enumerate(image_paths):
                logger.info(f"Processing page {page_num + 1}: {image_path}")
                
                # Load and preprocess image
                preprocessed = self.preprocess_image(image_path)
                
                # Get original dimensions for box normalization
                from PIL import Image
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                
                # Run OCR
                words = self.run_ocr(preprocessed)
                logger.info(f"OCR detected {len(words)} words")
                
                # Normalize boxes
                normalized_words = self.normalize_boxes(words, img_width, img_height)
                
                # Build raw OCR text
                raw_text = ' '.join(w['text'] for w in words)
                
                # Add page result
                result["pages"].append({
                    "page_number": page_num + 1,
                    "raw_ocr_text": raw_text,
                    "words": [
                        {
                            "text": w['text'],
                            "box": {
                                "x0": w['box'][0],
                                "y0": w['box'][1],
                                "x1": w['box'][2],
                                "y1": w['box'][3]
                            },
                            "confidence": w['confidence']
                        }
                        for w in normalized_words
                    ]
                })
                
                all_words.extend(normalized_words)
            
            # Run model inference and extract fields
            if all_words:
                # Load first image for model's visual features
                first_image = self._load_image(image_paths[0])
                
                model_predictions = None
                if not skip_model:
                    try:
                        model_predictions = self.run_model_inference(all_words, first_image)
                    except Exception as e:
                        logger.warning(f"Model inference failed: {e}")
                
                # Postprocess results
                fields = self.postprocess_results(model_predictions, all_words)
                
                # Update result with extracted fields
                all_field_names = [
                    "document_type", "vendor_name", "date", "total_amount", "subtotal", 
                    "tax_amount", "currency", "merchant_address",
                    "invoice_number", "due_date", "payment_terms", "customer_name", 
                    "customer_address", "po_number",
                    "account_number", "billing_period", "previous_balance", 
                    "current_charges", "amount_due",
                    "payment_method", "cashier_name", "register_number",
                    "discount", "shipping", "notes"
                ]
                for field_name in all_field_names:
                    if fields.get(field_name):
                        result[field_name] = fields[field_name]
                
                result["line_items"] = fields.get("line_items", [])
        
        except Exception as e:
            logger.error(f"Error processing receipt: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image as numpy array."""
        from PIL import Image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        preprocessor = self._get_preprocessor()
        return preprocessor.preprocess(image_path)
    
    def run_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run OCR on preprocessed image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of detected words with bounding boxes and confidences
        """
        ocr = self._get_ocr()
        return ocr.detect_and_recognize(image)
    
    def normalize_boxes(
        self,
        words: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        scale: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Normalize bounding boxes to 0-1000 scale.
        
        Args:
            words: List of words with boxes
            image_width: Original image width
            image_height: Original image height
            scale: Target scale (default 1000)
            
        Returns:
            Words with normalized boxes
        """
        normalized = []
        for word in words:
            box = word['box']
            normalized_box = [
                int(box[0] * scale / image_width),
                int(box[1] * scale / image_height),
                int(box[2] * scale / image_width),
                int(box[3] * scale / image_height)
            ]
            # Clamp to valid range
            normalized_box = [max(0, min(scale, x)) for x in normalized_box]
            
            normalized.append({
                'text': word['text'],
                'box': normalized_box,
                'confidence': word['confidence']
            })
        
        return normalized
    
    def tokenize_and_map_boxes(
        self,
        words: List[Dict[str, Any]]
    ) -> tuple:
        """
        Tokenize words and map tokens to bounding boxes.
        
        Each token inherits its parent word's bounding box.
        
        Args:
            words: List of words with text and bounding boxes
            
        Returns:
            Tuple of (token_ids, token_boxes, word_indices)
        """
        model = self._get_model()
        
        token_ids = []
        token_boxes = []
        word_indices = []  # Maps each token to its source word
        
        for word_idx, word in enumerate(words):
            # Tokenize word
            word_tokens = model.tokenize(word['text'])
            
            # Assign parent word's box to each token
            for token_id in word_tokens:
                token_ids.append(token_id)
                token_boxes.append(word['box'])
                word_indices.append(word_idx)
        
        return token_ids, token_boxes, word_indices
    
    def run_model_inference(
        self,
        words: List[Dict[str, Any]],
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run LayoutLMv3 model inference.
        
        Args:
            words: List of words with boxes
            image: Original image for visual features
            
        Returns:
            Model predictions with entity labels and confidences
        """
        model = self._get_model()
        
        # Extract texts and boxes
        texts = [w['text'] for w in words]
        boxes = [w['box'] for w in words]
        
        # Run prediction
        return model.predict_from_words(
            words=texts,
            boxes=boxes,
            image=image
        )
    
    def postprocess_results(
        self,
        predictions: Optional[Dict[str, Any]],
        words: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Postprocess model predictions into structured receipt data.
        
        Args:
            predictions: Raw model predictions (may be None)
            words: OCR words for heuristic extraction
            
        Returns:
            Structured receipt data with extracted fields
        """
        field_extractor = self._get_field_extractor()
        
        # If we have model predictions with entities, use them
        if predictions and predictions.get("entities"):
            entities = predictions["entities"]
            
            result = {
                # Document classification
                "document_type": entities.get("document_type"),
                # Common fields
                "vendor_name": entities.get("vendor_name"),
                "date": entities.get("date"),
                "total_amount": entities.get("total_amount"),
                "subtotal": entities.get("subtotal"),
                "tax_amount": entities.get("tax_amount"),
                "currency": entities.get("currency"),
                "merchant_address": entities.get("merchant_address"),
                "line_items": entities.get("line_items", []),
                # Invoice-specific
                "invoice_number": entities.get("invoice_number"),
                "due_date": entities.get("due_date"),
                "payment_terms": entities.get("payment_terms"),
                "customer_name": entities.get("customer_name"),
                "customer_address": entities.get("customer_address"),
                "po_number": entities.get("po_number"),
                # Bill-specific
                "account_number": entities.get("account_number"),
                "billing_period": entities.get("billing_period"),
                "previous_balance": entities.get("previous_balance"),
                "current_charges": entities.get("current_charges"),
                "amount_due": entities.get("amount_due"),
                # Receipt-specific
                "payment_method": entities.get("payment_method"),
                "cashier_name": entities.get("cashier_name"),
                "register_number": entities.get("register_number"),
                # General
                "discount": entities.get("discount"),
                "shipping": entities.get("shipping"),
                "notes": entities.get("notes")
            }
            
            # Fill in any missing critical fields with heuristics
            if result["document_type"] is None:
                result["document_type"] = field_extractor.classify_document_type(words)
            if result["vendor_name"] is None:
                result["vendor_name"] = field_extractor.extract_vendor_name(words)
            if result["total_amount"] is None:
                result["total_amount"] = field_extractor.extract_total(words)
            
            return result
        
        # Fall back to heuristic extraction
        result = {
            # Document classification
            "document_type": field_extractor.classify_document_type(words),
            # Common fields
            "vendor_name": field_extractor.extract_vendor_name(words),
            "date": self._extract_date_heuristic(words),
            "total_amount": field_extractor.extract_total(words),
            "subtotal": self._extract_subtotal_heuristic(words),
            "tax_amount": self._extract_tax_heuristic(words),
            "currency": self._detect_currency(words),
            "merchant_address": field_extractor.extract_address(words),
            "line_items": field_extractor.extract_line_items(words),
            # Invoice-specific
            "invoice_number": field_extractor.extract_invoice_number(words),
            "due_date": field_extractor.extract_due_date(words),
            "payment_terms": field_extractor.extract_payment_terms(words),
            "customer_name": field_extractor.extract_customer_name(words),
            "customer_address": field_extractor.extract_customer_address(words),
            "po_number": field_extractor.extract_po_number(words),
            # Bill-specific
            "account_number": field_extractor.extract_account_number(words),
            "billing_period": field_extractor.extract_billing_period(words),
            "previous_balance": field_extractor.extract_previous_balance(words),
            "current_charges": field_extractor.extract_current_charges(words),
            "amount_due": field_extractor.extract_amount_due(words),
            # Receipt-specific
            "payment_method": field_extractor.extract_payment_method(words),
            "cashier_name": field_extractor.extract_cashier_name(words),
            "register_number": field_extractor.extract_register_number(words),
            # General
            "discount": field_extractor.extract_discount(words),
            "shipping": field_extractor.extract_shipping(words),
            "notes": None
        }
        
        return result
    
    def _extract_date_heuristic(self, words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract date using regex patterns."""
        import re
        
        full_text = ' '.join(w['text'] for w in words)
        
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                for w in words:
                    if date_str in w['text'] or w['text'] in date_str:
                        return {
                            "value": date_str,
                            "confidence": w['confidence'],
                            "box": {
                                "x0": w['box'][0],
                                "y0": w['box'][1],
                                "x1": w['box'][2],
                                "y1": w['box'][3]
                            }
                        }
        
        return None
    
    def _extract_subtotal_heuristic(self, words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract subtotal amount."""
        import re
        
        amount_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        keywords = ['subtotal', 'sub total', 'sub-total']
        
        for i, w in enumerate(words):
            if any(kw in w['text'].lower() for kw in keywords):
                for j in range(max(0, i-2), min(len(words), i+5)):
                    match = re.search(amount_pattern, words[j]['text'])
                    if match:
                        return {
                            "value": match.group(1).replace(',', ''),
                            "confidence": words[j]['confidence'],
                            "box": {
                                "x0": words[j]['box'][0],
                                "y0": words[j]['box'][1],
                                "x1": words[j]['box'][2],
                                "y1": words[j]['box'][3]
                            }
                        }
        
        return None
    
    def _extract_tax_heuristic(self, words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract tax amount."""
        import re
        
        amount_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        keywords = ['tax', 'vat', 'gst', 'hst']
        
        for i, w in enumerate(words):
            if any(kw in w['text'].lower() for kw in keywords):
                for j in range(max(0, i-2), min(len(words), i+5)):
                    match = re.search(amount_pattern, words[j]['text'])
                    if match:
                        return {
                            "value": match.group(1).replace(',', ''),
                            "confidence": words[j]['confidence'],
                            "box": {
                                "x0": words[j]['box'][0],
                                "y0": words[j]['box'][1],
                                "x1": words[j]['box'][2],
                                "y1": words[j]['box'][3]
                            }
                        }
        
        return None
    
    def _detect_currency(self, words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect currency from text."""
        full_text = ' '.join(w['text'] for w in words)
        
        if '$' in full_text or 'USD' in full_text:
            return {"value": "USD", "confidence": 0.9, "box": None}
        elif '€' in full_text or 'EUR' in full_text:
            return {"value": "EUR", "confidence": 0.9, "box": None}
        elif '£' in full_text or 'GBP' in full_text:
            return {"value": "GBP", "confidence": 0.9, "box": None}
        
        return None

# --- Standalone wrappers for CLI/test imports ---

def get_device(device: str) -> str:
    return ReceiptProcessor()._resolve_device(device)

def load_image(image_path: str):
    return ReceiptProcessor()._load_image(image_path)

def preprocess_image(image, denoise=False, deskew=False):
    import numpy as np
    import tempfile
    import os
    from PIL import Image
    if isinstance(image, str):
        return ReceiptProcessor(config={'denoise': denoise, 'deskew': deskew}).preprocess_image(image)
    elif isinstance(image, np.ndarray):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(image).save(tmp.name)
            result = ReceiptProcessor(config={'denoise': denoise, 'deskew': deskew}).preprocess_image(tmp.name)
        os.unlink(tmp.name)
        return result
    else:
        raise ValueError("Unsupported image type for preprocess_image")

def normalize_boxes(words, image_width, image_height, scale=1000):
    return ReceiptProcessor().normalize_boxes(words, image_width, image_height, scale)

def extract_fields_heuristic(words):
    proc = ReceiptProcessor()
    return proc.postprocess_results(None, words)

def run_ocr(image, ocr_engine="paddle", device="auto"):
    """Run OCR using specified engine and device."""
    proc = ReceiptProcessor(ocr_engine=ocr_engine, device=device)
    return proc.run_ocr(image)


def run_layoutlm_inference(words, image, model_name="microsoft/layoutlmv3-base", device="auto"):
    """Run LayoutLMv3 model inference."""
    proc = ReceiptProcessor(model_name=model_name, device=device)
    return proc.run_model_inference(words, image)


def process_receipt(
    image_paths,
    job_id=None,
    output_path=None,
    skip_model=False,
    ocr_engine="paddle",
    device="auto",
    model_name="microsoft/layoutlmv3-base",
    denoise=False,
    deskew=False,
    verbose=False
):
    """Process receipt images with all pipeline options."""
    proc = ReceiptProcessor(
        model_name=model_name,
        ocr_engine=ocr_engine,
        device=device,
        config={"denoise": denoise, "deskew": deskew}
    )
    result = proc.process_receipt(image_paths, job_id=job_id, skip_model=skip_model)
    if output_path:
        import json, os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result
