"""
InternVL Model

Implementation of InternVL for document understanding and field extraction.

InternVL is a powerful open-source vision-language model with strong multimodal
capabilities, suitable for complex document processing tasks.

License: MIT
Model source: https://huggingface.co/OpenGVLab/InternVL2-8B
"""

import logging
from typing import Dict, Any, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Default confidence scores
DEFAULT_CONFIDENCE = 0.8
FALLBACK_CONFIDENCE = 0.5

# Prompt for document extraction
DOCUMENT_EXTRACTION_PROMPT = """Analyze this financial document image and extract information in JSON format.

Identify the document type (receipt, invoice, bill, or financial_document) and extract:
- document_type: Type of document
- vendor_name: Business/company name
- date: Document date
- total_amount: Total amount
- subtotal: Subtotal before tax (if visible)
- tax_amount: Tax amount (if visible)
- line_items: Array of items with description, quantity, unit_price, line_total

For invoices, also extract:
- invoice_number, due_date, payment_terms, customer_name, po_number

For bills, also extract:
- account_number, billing_period, amount_due

For receipts, also extract:
- payment_method

Return only valid JSON."""


class InternVLModel(BaseModel):
    """
    InternVL model for multimodal document understanding.
    
    InternVL is a powerful vision-language model from OpenGVLab with
    strong performance on vision-language tasks including document understanding.
    
    Key features:
    - High accuracy on vision-language tasks
    - Strong OCR and document understanding capabilities
    - Open source (MIT license)
    - Multiple size variants available
    
    Recommended models:
    - OpenGVLab/InternVL2-8B: 8B parameter model (balanced)
    - OpenGVLab/InternVL2-4B: 4B parameter model (efficient)
    - OpenGVLab/InternVL2-2B: 2B parameter model (lightweight)
    """
    
    def __init__(
        self,
        model_name_or_path: str = "OpenGVLab/InternVL2-8B",
        device: str = "cpu",
        max_new_tokens: int = 512
    ):
        """
        Initialize InternVL model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to run model on ('cpu' or 'cuda')
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized InternVLModel with {model_name_or_path}")
    
    def load(self):
        """Load model and tokenizer from HuggingFace."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info("Loading InternVL model components...")
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Load model
            logger.info(f"Loading model from {self.model_name_or_path}")
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"InternVL model loaded successfully on device: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install torch transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load InternVL model: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using InternVL tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            self.load()
        
        encoding = self.tokenizer(
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
        Run InternVL prediction on document image.
        
        Args:
            token_ids: Ignored (kept for API compatibility)
            token_boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with parsed output from InternVL
        """
        if self.model is None:
            self.load()
        
        import torch
        from PIL import Image as PILImage
        import numpy as np
        import json
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        logger.info("Running InternVL inference...")
        
        # Generate with InternVL's chat API
        generation_config = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': False,
        }
        
        try:
            # InternVL uses a chat-based interface
            response = self.model.chat(
                self.tokenizer,
                pil_image,
                DOCUMENT_EXTRACTION_PROMPT,
                generation_config
            )
            
            logger.info(f"InternVL raw output: {response[:200]}...")
            
            # Parse the JSON output
            entities = self._parse_json_output(response)
            
            return {
                "raw_output": response,
                "entities": entities,
                "predictions": [],
                "confidences": []
            }
        except AttributeError as e:
            # Model doesn't support chat interface
            logger.error(f"InternVL model does not support chat interface: {e}")
            raise
        except RuntimeError as e:
            # CUDA or memory errors
            logger.error(f"Runtime error during InternVL inference: {e}")
            raise
        except Exception as e:
            # Log unexpected errors but provide fallback
            logger.warning(f"Unexpected error during InternVL inference: {e}")
            return {
                "raw_output": "",
                "entities": self._get_empty_entities(),
                "predictions": [],
                "confidences": []
            }
    
    def _get_empty_entities(self) -> Dict[str, Any]:
        """Get empty entities structure."""
        return {
            "document_type": None,
            "vendor_name": None,
            "date": None,
            "total_amount": None,
            "subtotal": None,
            "tax_amount": None,
            "line_items": [],
            "invoice_number": None,
            "due_date": None,
            "payment_terms": None,
            "customer_name": None,
            "po_number": None,
            "account_number": None,
            "billing_period": None,
            "amount_due": None,
            "payment_method": None
        }
    
    def _parse_json_output(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON output from InternVL.
        
        Args:
            response: Raw model output
            
        Returns:
            Dictionary with extracted entities
        """
        import json
        import re
        
        entities = self._get_empty_entities()
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Extract all possible fields
                field_mappings = {
                    "document_type": ("document_type", lambda x: str(x)),
                    "vendor_name": ("vendor_name", lambda x: str(x)),
                    "date": ("date", lambda x: str(x)),
                    "total_amount": ("total_amount", self._clean_amount),
                    "subtotal": ("subtotal", self._clean_amount),
                    "tax_amount": ("tax_amount", self._clean_amount),
                    "invoice_number": ("invoice_number", lambda x: str(x)),
                    "due_date": ("due_date", lambda x: str(x)),
                    "payment_terms": ("payment_terms", lambda x: str(x)),
                    "customer_name": ("customer_name", lambda x: str(x)),
                    "po_number": ("po_number", lambda x: str(x)),
                    "account_number": ("account_number", lambda x: str(x)),
                    "billing_period": ("billing_period", lambda x: str(x)),
                    "amount_due": ("amount_due", self._clean_amount),
                    "payment_method": ("payment_method", lambda x: str(x))
                }
                
                for field_key, (entity_key, transform) in field_mappings.items():
                    if field_key in parsed and parsed[field_key]:
                        entities[entity_key] = {
                            "value": transform(parsed[field_key]),
                            "confidence": 0.8,
                            "box": None
                        }
                
                if "line_items" in parsed and isinstance(parsed["line_items"], list):
                    for item in parsed["line_items"]:
                        if isinstance(item, dict):
                            line_item = {
                                "description": item.get("description", ""),
                                "quantity": self._parse_int(item.get("quantity", 1)),
                                "unit_price": self._clean_amount(item.get("unit_price", "")),
                                "line_total": self._clean_amount(item.get("line_total", "")),
                                "confidence": 0.8,
                                "box": None
                            }
                            if line_item["description"]:
                                entities["line_items"].append(line_item)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON output: {e}")
            entities = self._fallback_parse(response, entities)
        except Exception as e:
            logger.warning(f"Error parsing output: {e}")
        
        return entities
    
    def _clean_amount(self, value: Any) -> Optional[str]:
        """Clean and normalize monetary amount."""
        if value is None:
            return None
        import re
        # Extract valid decimal number (allows only one decimal point)
        match = re.search(r'(\d+\.?\d*)', str(value))
        return match.group(1) if match else None
    
    def _parse_int(self, value: Any) -> int:
        """Parse integer value."""
        if value is None:
            return 1
        try:
            return int(value)
        except (ValueError, TypeError):
            import re
            digits = re.sub(r'[^\d]', '', str(value))
            return int(digits) if digits else 1
    
    def _fallback_parse(self, response: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback regex-based parsing."""
        import re
        
        # Document type
        doc_types = ['invoice', 'bill', 'receipt', 'financial']
        for dtype in doc_types:
            if dtype in response.lower():
                entities["document_type"] = {
                    "value": dtype if dtype != 'financial' else 'financial_document',
                    "confidence": 0.5,
                    "box": None
                }
                break
        
        # Vendor name
        vendor_match = re.match(r'^([A-Z][A-Za-z\s&]+)', response)
        if vendor_match:
            entities["vendor_name"] = {
                "value": vendor_match.group(1).strip(),
                "confidence": 0.5,
                "box": None
            }
        
        # Date
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})', response)
        if date_match:
            entities["date"] = {
                "value": date_match.group(1),
                "confidence": 0.6,
                "box": None
            }
        
        # Total amount
        total_match = re.search(r'total[:\s]*\$?(\d+\.?\d*)', response, re.IGNORECASE)
        if total_match:
            entities["total_amount"] = {
                "value": total_match.group(1),
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
        
        Args:
            words: Ignored (kept for API compatibility)
            boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with predictions and extracted entities
        """
        return self.predict([], [], image)
