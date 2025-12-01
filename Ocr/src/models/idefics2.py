"""
IDEFICS2 Model

Implementation of IDEFICS2 (Image-aware Decoder Enhanced à la Flamingo with Interleaved Cross-attentionS)
for receipt field extraction.

IDEFICS2 is an open-source multimodal model that can understand images and text,
making it suitable for document understanding tasks like receipt parsing.

License: Apache 2.0 (open source)
Model source: https://huggingface.co/HuggingFaceM4/idefics2-8b
"""

import logging
from typing import Dict, Any, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Default confidence scores for generated outputs (generation models don't provide real confidence)
DEFAULT_CONFIDENCE = 0.8
FALLBACK_CONFIDENCE = 0.5

# Prompt delimiter for extracting generated response
PROMPT_DELIMITER = "Now analyze this receipt:"

# Prompt template for receipt extraction
RECEIPT_EXTRACTION_PROMPT = """You are analyzing a receipt image. Extract the following information in JSON format:
- vendor_name: The store or business name
- date: The date of the transaction
- total_amount: The total amount paid
- subtotal: The subtotal before tax (if visible)
- tax_amount: The tax amount (if visible)
- line_items: List of items with description, quantity, unit_price, and line_total

Return only valid JSON, no additional text.

Example output:
{
  "vendor_name": "Store Name",
  "date": "2024-01-15",
  "total_amount": "45.99",
  "subtotal": "42.00",
  "tax_amount": "3.99",
  "line_items": [
    {"description": "Product 1", "quantity": 2, "unit_price": "10.50", "line_total": "21.00"}
  ]
}

Now analyze this receipt:"""


class IDEFICS2Model(BaseModel):
    """
    IDEFICS2 model for multimodal document understanding.
    
    IDEFICS2 is a large multimodal model that combines vision and language
    understanding. It can process images along with text prompts to generate
    structured outputs.
    
    Key features:
    - Multimodal understanding (image + text)
    - Instruction-following capabilities
    - Apache 2.0 license (open source)
    - 8B parameter version available for good performance/resource balance
    
    Recommended models:
    - HuggingFaceM4/idefics2-8b: Full 8B parameter model
    - HuggingFaceM4/idefics2-8b-AWQ: 4-bit quantized for lower memory
    """
    
    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceM4/idefics2-8b",
        device: str = "cpu",
        max_new_tokens: int = 512,
        load_in_4bit: bool = False
    ):
        """
        Initialize IDEFICS2 model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to run model on ('cpu' or 'cuda')
            max_new_tokens: Maximum tokens to generate
            load_in_4bit: Whether to use 4-bit quantization (saves memory, requires GPU)
            load_in_4bit: Whether to use 4-bit quantization (saves memory)
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.processor = None
        
        logger.info(f"Initialized IDEFICS2Model with {model_name_or_path}")
    
    def load(self):
        """Load model and processor from HuggingFace."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info("Loading IDEFICS2 model components...")
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
            
            # Load processor
            logger.info(f"Loading processor from {self.model_name_or_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            )
            
            # Configure quantization if requested
            quantization_config = None
            if self.load_in_4bit and self.device != "cpu":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    logger.info("Using 4-bit quantization")
                except Exception as e:
                    logger.warning(f"4-bit quantization not available: {e}. Loading in full precision.")
            
            # Load model
            logger.info(f"Loading model from {self.model_name_or_path}")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not quantization_config:
                self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"IDEFICS2 model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install torch transformers accelerate bitsandbytes"
            )
        except Exception as e:
            logger.error(f"Failed to load IDEFICS2 model: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using IDEFICS2 tokenizer.
        
        Note: IDEFICS2 is a generative model, this is for API compatibility.
        
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
        Run IDEFICS2 prediction on document image.
        
        Uses a carefully crafted prompt to extract receipt information.
        
        Args:
            token_ids: Ignored (kept for API compatibility)
            token_boxes: Ignored (kept for API compatibility)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with parsed output from IDEFICS2
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
        
        logger.info("Running IDEFICS2 inference...")
        
        # Create chat-style messages for IDEFICS2
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": RECEIPT_EXTRACTION_PROMPT}
                ]
            }
        ]
        
        # Apply chat template and process
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            images=[pil_image],
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                num_beams=1,
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract just the generated portion (after the prompt)
        response = generated_text.split(PROMPT_DELIMITER)[-1].strip()
        
        logger.info(f"IDEFICS2 raw output: {response[:200]}...")
        
        # Parse the JSON output
        entities = self._parse_json_output(response)
        
        return {
            "raw_output": response,
            "entities": entities,
            "predictions": [],  # Not applicable for generation
            "confidences": []   # Not available for generation models
        }
    
    def _parse_json_output(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON output from IDEFICS2.
        
        Args:
            response: Raw model output
            
        Returns:
            Dictionary with extracted entities
        """
        import json
        import re
        
        entities = {
            "vendor_name": None,
            "date": None,
            "total_amount": None,
            "subtotal": None,
            "tax_amount": None,
            "line_items": []
        }
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Extract fields
                if "vendor_name" in parsed and parsed["vendor_name"]:
                    entities["vendor_name"] = {
                        "value": str(parsed["vendor_name"]),
                        "confidence": 0.8,
                        "box": None
                    }
                
                if "date" in parsed and parsed["date"]:
                    entities["date"] = {
                        "value": str(parsed["date"]),
                        "confidence": 0.8,
                        "box": None
                    }
                
                if "total_amount" in parsed and parsed["total_amount"]:
                    entities["total_amount"] = {
                        "value": self._clean_amount(parsed["total_amount"]),
                        "confidence": 0.8,
                        "box": None
                    }
                
                if "subtotal" in parsed and parsed["subtotal"]:
                    entities["subtotal"] = {
                        "value": self._clean_amount(parsed["subtotal"]),
                        "confidence": 0.8,
                        "box": None
                    }
                
                if "tax_amount" in parsed and parsed["tax_amount"]:
                    entities["tax_amount"] = {
                        "value": self._clean_amount(parsed["tax_amount"]),
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
            # Fallback to regex parsing
            entities = self._fallback_parse(response, entities)
        except Exception as e:
            logger.warning(f"Error parsing output: {e}")
        
        return entities
    
    def _clean_amount(self, value: Any) -> Optional[str]:
        """Clean and normalize monetary amount."""
        if value is None:
            return None
        import re
        cleaned = re.sub(r'[^\d.]', '', str(value))
        return cleaned if cleaned else None
    
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
        
        # Look for vendor name at the start (often in caps)
        vendor_match = re.match(r'^([A-Z][A-Za-z\s&]+)', response)
        if vendor_match:
            entities["vendor_name"] = {
                "value": vendor_match.group(1).strip(),
                "confidence": 0.5,
                "box": None
            }
        
        # Look for date pattern
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})', response)
        if date_match:
            entities["date"] = {
                "value": date_match.group(1),
                "confidence": 0.6,
                "box": None
            }
        
        # Look for total pattern
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
        
        Note: IDEFICS2 processes images directly with prompts.
        OCR words can be included in the prompt for additional context.
        
        Args:
            words: OCR-detected words (can be used as context)
            boxes: Bounding boxes (ignored)
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with predictions and extracted entities
        """
        # IDEFICS2 can use OCR text as additional context
        # For now, just use the image directly
        return self.predict([], [], image)
