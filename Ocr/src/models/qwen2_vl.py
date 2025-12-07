"""
Qwen2-VL Model

Implementation of Qwen2-VL for document understanding and field extraction.

Qwen2-VL is Alibaba's vision-language model with strong performance on
document understanding and multimodal tasks.

License: Apache 2.0
Model source: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
"""

import logging
from typing import Dict, Any, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Default confidence scores
DEFAULT_CONFIDENCE = 0.8

# Prompt for document extraction
DOCUMENT_EXTRACTION_PROMPT = """Analyze this financial document and extract information in JSON format.

Identify the document type and extract all relevant fields:
- document_type: One of "receipt", "invoice", "bill", or "financial_document"
- vendor_name: Business name
- date: Document date
- total_amount: Total amount
- subtotal, tax_amount: If visible
- line_items: Array with description, quantity, unit_price, line_total

For invoices: invoice_number, due_date, payment_terms, customer_name, po_number
For bills: account_number, billing_period, amount_due
For receipts: payment_method

Return only valid JSON, no additional text."""


class Qwen2VLModel(BaseModel):
    """
    Qwen2-VL model for multimodal document understanding.
    
    Qwen2-VL is Alibaba's powerful vision-language model optimized for
    various vision-language tasks including document understanding.
    
    Key features:
    - Strong vision-language understanding
    - Efficient architecture
    - Apache 2.0 license (open source)
    - Multiple size variants
    
    Recommended models:
    - Qwen/Qwen2-VL-7B-Instruct: 7B parameter model
    - Qwen/Qwen2-VL-2B-Instruct: 2B parameter model (lightweight)
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 512
    ):
        """
        Initialize Qwen2-VL model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to run model on ('cpu' or 'cuda')
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.processor = None
        
        logger.info(f"Initialized Qwen2VLModel with {model_name_or_path}")
    
    def load(self):
        """Load model and processor from HuggingFace."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info("Loading Qwen2-VL model components...")
        
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # Load processor
            logger.info(f"Loading processor from {self.model_name_or_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Load model
            logger.info(f"Loading model from {self.model_name_or_path}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Qwen2-VL model loaded successfully on device: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install torch transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using Qwen2-VL tokenizer.
        
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
        Run Qwen2-VL prediction on document image.
        
        Args:
            token_ids: Ignored (kept for API compatibility)
            token_boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with parsed output from Qwen2-VL
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
        
        logger.info("Running Qwen2-VL inference...")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": DOCUMENT_EXTRACTION_PROMPT}
                ]
            }
        ]
        
        # Process with Qwen2-VL processor
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in generated_text:
            generated_text = generated_text.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in generated_text:
            generated_text = generated_text.split("<|im_end|>")[0]
        
        generated_text = generated_text.strip()
        
        logger.info(f"Qwen2-VL raw output: {generated_text[:200]}...")
        
        # Parse the JSON output
        entities = self._parse_json_output(generated_text)
        
        return {
            "raw_output": generated_text,
            "entities": entities,
            "predictions": [],
            "confidences": []
        }
    
    def _parse_json_output(self, response: str) -> Dict[str, Any]:
        """Parse JSON output from Qwen2-VL."""
        import json
        import re
        
        entities = {
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
