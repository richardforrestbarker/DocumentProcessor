"""
Postprocessing utilities for receipt field extraction.

Includes field parsing, validation, and entity consolidation.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class FieldExtractor:
    """
    Extracts and validates receipt fields from OCR and model predictions.
    """
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize field extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for field acceptance
        """
        self.min_confidence = min_confidence
        
        # Regex patterns for field extraction
        self.patterns = {
            'amount': re.compile(r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'),  # Support thousands separators
            'date': [
                re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'),
                re.compile(r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})'),
                re.compile(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})', re.IGNORECASE)
            ],
            'phone': re.compile(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'),
            'email': re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'),
        }
    
    def extract_amount(self, text: str) -> Optional[Decimal]:
        """
        Extract monetary amount from text.
        
        Args:
            text: Input text
            
        Returns:
            Decimal amount or None
        """
        match = self.patterns['amount'].search(text)
        if match:
            try:
                # Remove currency symbols and thousands separators
                amount_str = match.group(1).replace(',', '')
                return Decimal(amount_str)
            except InvalidOperation:
                logger.warning(f"Failed to parse amount: {text}")
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """
        Extract date from text and normalize to ISO format.
        
        Args:
            text: Input text
            
        Returns:
            ISO format date string (YYYY-MM-DD) or None
        """
        for pattern in self.patterns['date']:
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                try:
                    # Try multiple date formats
                    for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y', '%B %d, %Y', '%b %d, %Y']:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.strftime('%Y-%m-%d')
                        except ValueError:
                            continue
                except Exception as e:
                    logger.warning(f"Failed to parse date '{date_str}': {e}")
        return None
    
    def extract_vendor_name(
        self,
        words: List[Dict[str, Any]],
        predictions: Optional[List[int]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract vendor/merchant name from receipt.
        
        Typically appears at the top of the receipt.
        
        Args:
            words: List of OCR words with boxes
            predictions: Optional model predictions for each word
            
        Returns:
            Dictionary with vendor name, confidence, and box
        """
        if not words:
            return None
        
        # TODO: Use model predictions if available
        # For now, use heuristics: top-most large text
        
        # Sort by y-coordinate (top to bottom)
        sorted_words = sorted(words, key=lambda w: w['box'][1])
        
        # Take first few words as potential vendor name
        vendor_words = sorted_words[:3]
        vendor_text = ' '.join(w['text'] for w in vendor_words)
        
        # Average confidence
        avg_confidence = sum(w['confidence'] for w in vendor_words) / len(vendor_words)
        
        # Combined bounding box
        all_boxes = [w['box'] for w in vendor_words]
        combined_box = {
            'x0': min(b[0] for b in all_boxes),
            'y0': min(b[1] for b in all_boxes),
            'x1': max(b[2] for b in all_boxes),
            'y1': max(b[3] for b in all_boxes)
        }
        
        return {
            'value': vendor_text,
            'confidence': avg_confidence,
            'box': combined_box
        }
    
    def extract_total(
        self,
        words: List[Dict[str, Any]],
        predictions: Optional[List[int]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract total amount from receipt.
        
        Args:
            words: List of OCR words with boxes
            predictions: Optional model predictions
            
        Returns:
            Dictionary with total amount, confidence, and box
        """
        # Look for words like "TOTAL", "GRAND TOTAL", "AMOUNT DUE"
        total_keywords = ['total', 'grand', 'amount', 'due', 'balance']
        
        for i, word in enumerate(words):
            text_lower = word['text'].lower()
            
            # Check if word contains total keyword
            if any(keyword in text_lower for keyword in total_keywords):
                # Look for amount in next few words
                for j in range(i, min(i + 5, len(words))):
                    amount = self.extract_amount(words[j]['text'])
                    if amount:
                        return {
                            'value': str(amount),
                            'confidence': words[j]['confidence'],
                            'box': {
                                'x0': words[j]['box'][0],
                                'y0': words[j]['box'][1],
                                'x1': words[j]['box'][2],
                                'y1': words[j]['box'][3]
                            }
                        }
        
        return None
    
    def extract_line_items(
        self,
        words: List[Dict[str, Any]],
        predictions: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract line items from receipt.
        
        Args:
            words: List of OCR words with boxes
            predictions: Optional model predictions
            
        Returns:
            List of line items with description, quantity, price, etc.
        """
        import re
        
        if not words:
            return []
        
        line_items = []
        
        # Group words into lines based on y-coordinate proximity
        lines = self._group_words_into_lines(words)
        
        # Pattern for price (with or without currency symbol)
        price_pattern = re.compile(r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$')
        quantity_pattern = re.compile(r'^(\d+)x?$|^x(\d+)$', re.IGNORECASE)
        
        for line_words in lines:
            if len(line_words) < 2:
                continue
            
            # Skip header/footer lines (typically contain keywords like TOTAL, SUBTOTAL, etc.)
            line_text = ' '.join(w['text'] for w in line_words).lower()
            skip_keywords = ['total', 'subtotal', 'tax', 'change', 'cash', 'credit', 
                           'card', 'visa', 'mastercard', 'thank', 'receipt', 'store']
            if any(kw in line_text for kw in skip_keywords):
                continue
            
            # Try to identify components of a line item
            description_parts = []
            quantity = None
            unit_price = None
            line_total = None
            
            for w in line_words:
                text = w['text'].strip()
                
                # Check if it's a price
                if price_pattern.match(text):
                    price_value = float(text.replace('$', '').replace(',', ''))
                    if line_total is None:
                        line_total = price_value
                    elif unit_price is None:
                        # First price was unit price, this is total
                        unit_price = line_total
                        line_total = price_value
                # Check if it's a quantity
                else:
                    quantity_match = quantity_pattern.match(text)
                    if quantity_match:
                        qty = quantity_match.group(1) or quantity_match.group(2)
                        quantity = int(qty)
                    else:
                        # Part of description
                        description_parts.append(text)
            
            # Only add if we found a valid line item
            if description_parts and (line_total is not None or unit_price is not None):
                description = ' '.join(description_parts)
                
                # Calculate unit price if we have quantity and total
                if unit_price is None and quantity and line_total:
                    unit_price = round(line_total / quantity, 2)
                elif unit_price is None:
                    unit_price = line_total
                
                # Calculate line total if we only have unit price and quantity
                if line_total is None and quantity and unit_price:
                    line_total = round(unit_price * quantity, 2)
                elif line_total is None:
                    line_total = unit_price
                
                # Get bounding box for the entire line
                all_boxes = [w['box'] for w in line_words]
                combined_box = {
                    'x0': min(b[0] for b in all_boxes),
                    'y0': min(b[1] for b in all_boxes),
                    'x1': max(b[2] for b in all_boxes),
                    'y1': max(b[3] for b in all_boxes)
                }
                
                # Calculate average confidence
                avg_confidence = sum(w['confidence'] for w in line_words) / len(line_words)
                
                line_items.append({
                    'description': description,
                    'quantity': quantity or 1,
                    'unit_price': unit_price,
                    'line_total': line_total,
                    'box': combined_box,
                    'confidence': avg_confidence
                })
        
        return line_items
    
    def _group_words_into_lines(
        self,
        words: List[Dict[str, Any]],
        y_threshold: int = 20
    ) -> List[List[Dict[str, Any]]]:
        """
        Group words into lines based on y-coordinate proximity.
        
        Args:
            words: List of words with boxes
            y_threshold: Maximum y-distance to consider words on same line
            
        Returns:
            List of word groups, each representing a line
        """
        if not words:
            return []
        
        # Sort by y-coordinate (top), then x-coordinate (left)
        sorted_words = sorted(words, key=lambda w: (w['box'][1], w['box'][0]))
        
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]['box'][1]
        
        for word in sorted_words[1:]:
            word_y = word['box'][1]
            
            # If word is on the same line (within threshold)
            if abs(word_y - current_y) <= y_threshold:
                current_line.append(word)
            else:
                # Start new line
                # Sort current line by x-coordinate
                lines.append(sorted(current_line, key=lambda w: w['box'][0]))
                current_line = [word]
                current_y = word_y
        
        # Add last line
        if current_line:
            lines.append(sorted(current_line, key=lambda w: w['box'][0]))
        
        return lines
    
    def verify_totals(
        self,
        subtotal: Optional[Decimal],
        tax: Optional[Decimal],
        total: Optional[Decimal]
    ) -> bool:
        """
        Verify that subtotal + tax = total (within tolerance).
        
        Args:
            subtotal: Subtotal amount
            tax: Tax amount
            total: Total amount
            
        Returns:
            True if totals are consistent
        """
        if not all([subtotal, tax, total]):
            return False
        
        calculated_total = subtotal + tax
        tolerance = Decimal('0.02')  # 2 cent tolerance for rounding
        
        difference = abs(calculated_total - total)
        is_valid = difference <= tolerance
        
        if not is_valid:
            logger.warning(
                f"Total verification failed: {subtotal} + {tax} = {calculated_total}, "
                f"but receipt total is {total} (difference: {difference})"
            )
        
        return is_valid
    
    def consolidate_fields(
        self,
        raw_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Consolidate and clean extracted fields.
        
        Args:
            raw_fields: Raw extracted fields
            
        Returns:
            Cleaned and validated fields
        """
        consolidated = {}
        
        # Filter by confidence threshold
        for field_name, field_data in raw_fields.items():
            if isinstance(field_data, dict) and 'confidence' in field_data:
                if field_data['confidence'] >= self.min_confidence:
                    consolidated[field_name] = field_data
                else:
                    logger.info(
                        f"Field '{field_name}' filtered due to low confidence: "
                        f"{field_data['confidence']:.2f} < {self.min_confidence}"
                    )
        
        return consolidated
