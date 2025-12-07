"""
Donut Model

Implementation of Donut (Document Understanding Transformer) for receipt field extraction.

Donut is an OCR-free document understanding model that directly generates structured
text output from document images without requiring external OCR preprocessing.

License: MIT (open source)
Model source: https://huggingface.co/naver-clova-ix/donut-base
"""

import logging
from typing import Dict, Any, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Default confidence scores for generated outputs (generation models don't provide real confidence)
DEFAULT_CONFIDENCE = 0.8
FALLBACK_CONFIDENCE = 0.6

# Task prompt for CORD-v2 fine-tuned model
CORD_TASK_PROMPT = "<s_cord-v2>"


class DonutModel(BaseModel):
    """
    Donut model for document understanding and field extraction.
    
    Donut (Document understanding transformer) is an OCR-free model that
    directly understands document images and generates structured output.
    
    Key features:
    - No external OCR required - end-to-end visual document understanding
    - Pre-trained on large document datasets
    - Fine-tuned variants available for specific document types
    - MIT license (open source)
    
    Recommended models:
    - naver-clova-ix/donut-base: Base model for fine-tuning
    - naver-clova-ix/donut-base-finetuned-cord-v2: Receipt understanding
    - naver-clova-ix/donut-base-finetuned-docvqa: Document QA
    """
    
    def __init__(
        self,
        model_name_or_path: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        device: str = "cpu",
        max_length: int = 768
    ):
        """
        Initialize Donut model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path.
                               Default is the CORD-v2 fine-tuned model for receipts.
            device: Device to run model on ('cpu' or 'cuda')
            max_length: Maximum output sequence length
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        
        self.model = None
        self.processor = None
        
        logger.info(f"Initialized DonutModel with {model_name_or_path}")
    
    def load(self):
        """Load model and processor from HuggingFace."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info("Loading Donut model components...")
        
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            
            # Load processor
            logger.info(f"Loading processor from {self.model_name_or_path}")
            self.processor = DonutProcessor.from_pretrained(self.model_name_or_path)
            
            # Load model
            logger.info(f"Loading model from {self.model_name_or_path}")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name_or_path)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Donut model loaded successfully on device: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install torch transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load Donut model: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using Donut tokenizer.
        
        Note: Donut is typically used for generation, not token classification.
        This method is provided for API compatibility.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.processor is None:
            self.load()
        
        encoding = self.processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None
        )
        return encoding["input_ids"]
    
    def predict(
        self,
        token_ids: List[int],
        token_boxes: List[List[int]],
        image: Any
    ) -> Dict[str, Any]:
        """
        Run Donut prediction on document image.
        
        Note: Donut doesn't use token_ids or boxes - it processes the image directly.
        These parameters are accepted for API compatibility but ignored.
        
        Args:
            token_ids: Ignored (kept for API compatibility)
            token_boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with parsed output from Donut
        """
        if self.model is None:
            self.load()
        
        import torch
        from PIL import Image as PILImage
        import numpy as np
        import re
        import json
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        logger.info("Running Donut inference...")
        
        # Prepare decoder input - task prompt for the model
        decoder_input_ids = self.processor.tokenizer(
            CORD_TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Process image
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,  # Use greedy decoding for speed
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        
        # Decode output
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove task token
        
        logger.info(f"Donut raw output: {sequence[:200]}...")
        
        # Parse the JSON-like output from CORD-v2 model
        entities = self._parse_cord_output(sequence)
        
        return {
            "raw_output": sequence,
            "entities": entities,
            "predictions": [],  # Not applicable for Donut
            "confidences": []   # Not available for generation models
        }
    
    def _parse_cord_output(self, sequence: str) -> Dict[str, Any]:
        """
        Parse the CORD-v2 format output from Donut.
        
        The CORD-v2 model outputs a structured JSON-like format with menu items,
        subtotal, tax, and total information.
        
        Args:
            sequence: Raw output string from Donut
            
        Returns:
            Dictionary with extracted entities
        """
        import json
        import re
        
        entities = {
            "document_type": {"value": "receipt", "confidence": 0.85, "box": None},  # CORD-v2 is receipt-specific
            "vendor_name": None,
            "date": None,
            "total_amount": None,
            "subtotal": None,
            "tax_amount": None,
            "line_items": []
        }
        
        try:
            # Try to parse as JSON first
            # CORD-v2 output is typically XML-like, convert to dict
            parsed = self.processor.token2json(sequence)
            
            if isinstance(parsed, dict):
                # Extract fields from CORD format
                if "menu" in parsed:
                    for item in parsed.get("menu", []):
                        line_item = {
                            "description": item.get("nm", ""),
                            "quantity": self._parse_number(item.get("cnt", "1")),
                            "unit_price": self._parse_amount(item.get("unitprice", "")),
                            "line_total": self._parse_amount(item.get("price", "")),
                            "confidence": 0.8,
                            "box": None
                        }
                        if line_item["description"]:
                            entities["line_items"].append(line_item)
                
                # Extract totals
                if "sub_total" in parsed:
                    subtotal = parsed.get("sub_total", {})
                    if isinstance(subtotal, dict):
                        entities["subtotal"] = {
                            "value": self._parse_amount(subtotal.get("subtotal_price", "")),
                            "confidence": 0.8,
                            "box": None
                        }
                    elif isinstance(subtotal, str):
                        entities["subtotal"] = {
                            "value": self._parse_amount(subtotal),
                            "confidence": 0.8,
                            "box": None
                        }
                
                if "total" in parsed:
                    total = parsed.get("total", {})
                    if isinstance(total, dict):
                        # Look for total_price, cashprice, or total_etc
                        total_val = total.get("total_price", "") or total.get("cashprice", "") or total.get("total_etc", "")
                        entities["total_amount"] = {
                            "value": self._parse_amount(total_val),
                            "confidence": 0.8,
                            "box": None
                        }
                    elif isinstance(total, str):
                        entities["total_amount"] = {
                            "value": self._parse_amount(total),
                            "confidence": 0.8,
                            "box": None
                        }
                
                # Tax is sometimes in sub_total
                if "sub_total" in parsed:
                    subtotal_data = parsed.get("sub_total", {})
                    if isinstance(subtotal_data, dict) and "tax_price" in subtotal_data:
                        entities["tax_amount"] = {
                            "value": self._parse_amount(subtotal_data.get("tax_price", "")),
                            "confidence": 0.8,
                            "box": None
                        }
                
        except Exception as e:
            logger.warning(f"Failed to parse CORD output: {e}")
            # Fallback: try regex-based extraction
            entities = self._fallback_parse(sequence, entities)
        
        return entities
    
    def _parse_amount(self, value: str) -> Optional[str]:
        """Parse a monetary amount string."""
        if not value:
            return None
        import re
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.]', '', str(value))
        return cleaned if cleaned else None
    
    def _parse_number(self, value: str) -> int:
        """Parse a number string."""
        if not value:
            return 1
        import re
        cleaned = re.sub(r'[^\d]', '', str(value))
        return int(cleaned) if cleaned else 1
    
    def _fallback_parse(self, sequence: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback regex-based parsing if JSON parsing fails."""
        import re
        
        # Look for total pattern
        total_match = re.search(r'total[:\s]*\$?(\d+\.?\d*)', sequence, re.IGNORECASE)
        if total_match:
            entities["total_amount"] = {
                "value": total_match.group(1),
                "confidence": 0.6,
                "box": None
            }
        
        # Look for tax pattern
        tax_match = re.search(r'tax[:\s]*\$?(\d+\.?\d*)', sequence, re.IGNORECASE)
        if tax_match:
            entities["tax_amount"] = {
                "value": tax_match.group(1),
                "confidence": 0.6,
                "box": None
            }
        
        return entities
    
    def predict_from_words(
        self,
        words: List[str],
        boxes: List[List[int]],
        image: Any
    ) -> Dict[str, Any]:
        """
        Run prediction from an image.
        
        Note: Donut doesn't use OCR words - it processes the image directly.
        This is provided for API compatibility with other models.
        
        Args:
            words: Ignored (kept for API compatibility)
            boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with predictions and extracted entities
        """
        # Donut ignores words/boxes and processes image directly
        return self.predict([], [], image)
