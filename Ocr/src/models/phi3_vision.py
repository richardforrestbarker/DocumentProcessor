"""
Phi-3-Vision Model

Implementation of Microsoft Phi-3-Vision for document understanding and field extraction.

Phi-3-Vision is a lightweight multimodal model with strong vision-language capabilities,
suitable for document processing tasks including receipts, invoices, and bills.

License: MIT
Model source: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
"""

import logging
from typing import Dict, Any, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Default confidence scores for generated outputs
DEFAULT_CONFIDENCE = 0.8
FALLBACK_CONFIDENCE = 0.5

# Prompt delimiter for extracting generated response
PROMPT_DELIMITER = "Now analyze this document:"

# Prompt template for document extraction
DOCUMENT_EXTRACTION_PROMPT = """<|user|>
You are analyzing a financial document image. First identify the document type (receipt, invoice, bill, or other financial document), then extract relevant information in JSON format.

For all documents, extract:
- document_type: One of "receipt", "invoice", "bill", or "financial_document"
- vendor_name: The business/company name
- date: The transaction/document date
- total_amount: The total amount
- subtotal: The subtotal before tax (if visible)
- tax_amount: The tax amount (if visible)
- line_items: List of items with description, quantity, unit_price, and line_total

For invoices, also extract:
- invoice_number: The invoice number
- due_date: Payment due date
- payment_terms: Payment terms (e.g., "Net 30")
- customer_name: Customer or "Bill To" name
- po_number: Purchase order number (if visible)

For bills, also extract:
- account_number: Account number
- billing_period: Billing period dates
- amount_due: Amount due

For receipts, also extract:
- payment_method: Payment method used

Return only valid JSON, no additional text.

Now analyze this document:<|end|>
<|assistant|>
"""


class Phi3VisionModel(BaseModel):
    """
    Phi-3-Vision model for multimodal document understanding.
    
    Phi-3-Vision is Microsoft's lightweight vision-language model optimized
    for efficiency while maintaining strong performance on vision tasks.
    
    Key features:
    - Lightweight and efficient (~7B parameters)
    - 128k context window
    - Strong vision-language understanding
    - MIT license (open source)
    
    Recommended models:
    - microsoft/Phi-3-vision-128k-instruct: Main model with 128k context
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/Phi-3-vision-128k-instruct",
        device: str = "cpu",
        max_new_tokens: int = 512
    ):
        """
        Initialize Phi-3-Vision model.
        
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
        
        logger.info(f"Initialized Phi3VisionModel with {model_name_or_path}")
    
    def load(self):
        """Load model and processor from HuggingFace."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info("Loading Phi-3-Vision model components...")
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            # Load processor
            logger.info(f"Loading processor from {self.model_name_or_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Load model
            logger.info(f"Loading model from {self.model_name_or_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Phi-3-Vision model loaded successfully on device: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install torch transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load Phi-3-Vision model: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using Phi-3-Vision tokenizer.
        
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
        Run Phi-3-Vision prediction on document image.
        
        Args:
            token_ids: Ignored (kept for API compatibility)
            token_boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with parsed output from Phi-3-Vision
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
        
        logger.info("Running Phi-3-Vision inference...")
        
        # Prepare messages for the model
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{DOCUMENT_EXTRACTION_PROMPT}"}
        ]
        
        # Process inputs
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            prompt,
            [pil_image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Phi-3-Vision raw output: {generated_text[:200]}...")
        
        # Parse the JSON output
        entities = self._parse_json_output(generated_text)
        
        return {
            "raw_output": generated_text,
            "entities": entities,
            "predictions": [],
            "confidences": []
        }
    
    def _parse_json_output(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON output from Phi-3-Vision.
        
        Args:
            response: Raw model output
            
        Returns:
            Dictionary with extracted entities
        """
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
        
        # Vendor name (first capitalized phrase)
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
