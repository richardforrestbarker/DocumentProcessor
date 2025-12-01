"""
LayoutLMv3 Model

Implementation of LayoutLMv3 for receipt field extraction.
"""

import logging
from typing import Dict, Any, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Label mapping for receipt field extraction
# These labels correspond to the expected receipt fields
RECEIPT_LABELS = {
    0: "O",               # Outside any entity
    1: "B-VENDOR",        # Beginning of vendor name
    2: "I-VENDOR",        # Inside vendor name
    3: "B-DATE",          # Beginning of date
    4: "I-DATE",          # Inside date
    5: "B-TOTAL",         # Beginning of total amount
    6: "I-TOTAL",         # Inside total amount
    7: "B-SUBTOTAL",      # Beginning of subtotal
    8: "I-SUBTOTAL",      # Inside subtotal
    9: "B-TAX",           # Beginning of tax amount
    10: "I-TAX",          # Inside tax amount
    11: "B-ITEM",         # Beginning of line item
    12: "I-ITEM",         # Inside line item
}


class LayoutLMv3Model(BaseModel):
    """
    LayoutLMv3 model for document understanding and field extraction.
    
    This model combines text, layout, and visual information to extract
    structured fields from receipt images.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/layoutlmv3-base",
        device: str = "cpu",
        num_labels: int = 13  # Number of field types to extract
    ):
        """
        Initialize LayoutLMv3 model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to run model on ('cpu' or 'cuda')
            num_labels: Number of entity labels for classification
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.num_labels = num_labels
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.feature_extractor = None
        
        # Label mappings
        self.id2label = RECEIPT_LABELS
        self.label2id = {v: k for k, v in RECEIPT_LABELS.items()}
        
        logger.info(f"Initialized LayoutLMv3Model with {model_name_or_path}")
    
    def load(self):
        """Load model, tokenizer, and processor from HuggingFace."""
        if self.model is not None:
            return  # Already loaded
        
        logger.info("Loading LayoutLMv3 model components...")
        
        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForTokenClassification,
                AutoTokenizer
            )
            from PIL import Image
            
            # Load processor (includes feature extractor and tokenizer)
            logger.info(f"Loading processor from {self.model_name_or_path}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                apply_ocr=False  # We provide our own OCR
            )
            
            # Load tokenizer separately for tokenize() method
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            
            # Load model for token classification
            logger.info(f"Loading model from {self.model_name_or_path}")
            
            # Check if this is a fine-tuned model with our labels or base model
            try:
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name_or_path,
                    num_labels=self.num_labels,
                    id2label=self.id2label,
                    label2id=self.label2id,
                    ignore_mismatched_sizes=True  # Allow loading base model with different head
                )
            except Exception as e:
                logger.warning(f"Loading with custom labels failed: {e}")
                logger.info("Loading base model without custom labels")
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name_or_path,
                    ignore_mismatched_sizes=True
                )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}. "
                "Install with: pip install torch transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using LayoutLMv3 tokenizer.
        
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
        Run LayoutLMv3 prediction on receipt.
        
        Args:
            token_ids: List of token IDs from tokenizer
            token_boxes: List of normalized boxes [x0, y0, x1, y1] in 0-1000 scale
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with predictions including entity labels and confidences
        """
        if self.model is None:
            self.load()
        
        import torch
        from PIL import Image as PILImage
        import numpy as np
        
        logger.info(f"Running prediction on {len(token_ids)} tokens")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Validate inputs
        if not token_ids:
            logger.warning("Empty token_ids provided to predict()")
            return {"predictions": [], "confidences": [], "entities": {}}
        
        # Prepare inputs using processor
        # LayoutLMv3 expects: input_ids, attention_mask, bbox, pixel_values
        
        # Ensure token_ids is a list of lists for batch processing
        if isinstance(token_ids[0], int):
            token_ids = [token_ids]
            token_boxes = [token_boxes]
        
        # Pad sequences to same length
        max_len = min(512, max(len(t) for t in token_ids) if token_ids else 1)
        
        padded_ids = []
        padded_boxes = []
        attention_masks = []
        
        for ids, boxes in zip(token_ids, token_boxes):
            # Truncate if necessary
            ids = ids[:max_len]
            boxes = boxes[:max_len]
            
            # Calculate padding
            pad_len = max_len - len(ids)
            
            # Pad token ids
            padded_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            
            # Pad boxes with [0, 0, 0, 0]
            padded_boxes.append(boxes + [[0, 0, 0, 0]] * pad_len)
            
            # Create attention mask
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
        bbox = torch.tensor(padded_boxes, dtype=torch.long).to(self.device)
        
        # Process image for visual features
        # Resize to 224x224 as expected by LayoutLMv3
        pil_image_resized = pil_image.resize((224, 224))
        pixel_values = torch.tensor(
            np.array(pil_image_resized).transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values
            )
        
        # Extract predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        
        # Get confidence scores
        probs = torch.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values.squeeze().tolist()
        
        # Handle single prediction case
        if not isinstance(predictions, list):
            predictions = [predictions]
            confidences = [confidences]
        
        logger.info(f"LayoutLMv3 inference completed with {len(predictions)} predictions")
        
        return {
            "predictions": predictions,
            "confidences": confidences,
            "label_names": self.id2label
        }
    
    def predict_from_words(
        self,
        words: List[str],
        boxes: List[List[int]],
        image: Any
    ) -> Dict[str, Any]:
        """
        Run prediction directly from OCR words and boxes.
        
        This method handles tokenization and box mapping automatically.
        
        Args:
            words: List of words from OCR
            boxes: List of bounding boxes for each word
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with predictions and extracted entities
        """
        if self.model is None:
            self.load()
        
        import torch
        from PIL import Image as PILImage
        import numpy as np
        
        logger.info(f"Processing {len(words)} words")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if not words:
            return {"predictions": [], "confidences": [], "entities": {}}
        
        # Use processor to handle tokenization and box mapping
        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Move to device
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Extract predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        
        # Get confidence scores
        probs = torch.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values.squeeze().tolist()
        
        # Handle single prediction case
        if not isinstance(predictions, list):
            predictions = [predictions]
            confidences = [confidences]
        
        # Extract entities from predictions
        entities = self.extract_entities(words, predictions, confidences, boxes)
        
        return {
            "predictions": predictions,
            "confidences": confidences,
            "entities": entities
        }
    
    def extract_entities(
        self,
        tokens: List[str],
        predictions: List[int],
        confidences: List[float],
        boxes: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Extract structured entities from model predictions.
        
        Args:
            tokens: List of tokens/words
            predictions: List of predicted label IDs
            confidences: List of confidence scores
            boxes: List of bounding boxes
            
        Returns:
            Dictionary mapping field names to extracted values
        """
        entities = {
            "vendor_name": None,
            "date": None,
            "total_amount": None,
            "subtotal": None,
            "tax_amount": None,
            "line_items": []
        }
        
        # Group consecutive tokens with same entity type
        current_entity = None
        current_tokens = []
        current_boxes = []
        current_confidences = []
        
        # Limit to number of tokens we have
        num_tokens = min(len(tokens), len(predictions), len(confidences))
        
        for i in range(num_tokens):
            pred = predictions[i]
            
            # Handle out of range predictions
            if pred >= len(self.id2label):
                pred = 0  # Default to "O" (outside)
            
            label = self.id2label.get(pred, "O")
            
            if label == "O":
                # End current entity if any
                if current_entity and current_tokens:
                    self._save_entity(
                        entities, current_entity, current_tokens,
                        current_boxes, current_confidences
                    )
                current_entity = None
                current_tokens = []
                current_boxes = []
                current_confidences = []
            elif label.startswith("B-"):
                # Start new entity
                if current_entity and current_tokens:
                    self._save_entity(
                        entities, current_entity, current_tokens,
                        current_boxes, current_confidences
                    )
                current_entity = label[2:]  # Remove "B-" prefix
                current_tokens = [tokens[i]] if i < len(tokens) else []
                current_boxes = [boxes[i]] if i < len(boxes) else []
                current_confidences = [confidences[i]]
            elif label.startswith("I-"):
                # Continue current entity
                entity_type = label[2:]
                if current_entity == entity_type:
                    if i < len(tokens):
                        current_tokens.append(tokens[i])
                    if i < len(boxes):
                        current_boxes.append(boxes[i])
                    current_confidences.append(confidences[i])
        
        # Save final entity if any
        if current_entity and current_tokens:
            self._save_entity(
                entities, current_entity, current_tokens,
                current_boxes, current_confidences
            )
        
        return entities
    
    def _save_entity(
        self,
        entities: Dict[str, Any],
        entity_type: str,
        tokens: List[str],
        boxes: List[List[int]],
        confidences: List[float]
    ) -> None:
        """
        Save extracted entity to entities dictionary.
        
        Args:
            entities: Dictionary to store entities
            entity_type: Type of entity (VENDOR, DATE, TOTAL, etc.)
            tokens: List of tokens forming the entity
            boxes: List of bounding boxes
            confidences: List of confidence scores
        """
        # Combine tokens into value
        value = " ".join(tokens)
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Combined bounding box
        if boxes:
            combined_box = {
                "x0": min(b[0] for b in boxes),
                "y0": min(b[1] for b in boxes),
                "x1": max(b[2] for b in boxes),
                "y1": max(b[3] for b in boxes)
            }
        else:
            combined_box = None
        
        entity_data = {
            "value": value,
            "confidence": avg_confidence,
            "box": combined_box
        }
        
        # Map entity type to field name
        field_mapping = {
            "VENDOR": "vendor_name",
            "DATE": "date",
            "TOTAL": "total_amount",
            "SUBTOTAL": "subtotal",
            "TAX": "tax_amount",
            "ITEM": "line_items"
        }
        
        field_name = field_mapping.get(entity_type)
        if field_name:
            if field_name == "line_items":
                # Append to list
                entities["line_items"].append({
                    "description": value,
                    "confidence": avg_confidence,
                    "box": combined_box
                })
            else:
                # Set single field (keep highest confidence if already set)
                if entities[field_name] is None or avg_confidence > entities[field_name].get("confidence", 0):
                    entities[field_name] = entity_data
