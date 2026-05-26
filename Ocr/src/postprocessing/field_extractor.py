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
    Extracts and validates fields from various financial documents (receipts, invoices, bills).
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
            'invoice_number': re.compile(r'(?:invoice|inv|#)\s*[:\-]?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            'po_number': re.compile(r'(?:po|purchase\s+order)\s*[:\-]?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            'account_number': re.compile(r'(?:account|acct)\s*[:\-]?\s*([A-Z0-9\-]+)', re.IGNORECASE),
            'billing_period': re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:to|-|through)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE),
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
    
    def classify_document_type(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Classify document type based on keywords and patterns.
        
        Args:
            words: List of OCR words
            
        Returns:
            Dictionary with document type, confidence, and box
        """
        if not words:
            return None
        
        full_text = ' '.join(w['text'] for w in words).lower()
        
        # Keywords for different document types
        invoice_keywords = ['invoice', 'bill to', 'ship to', 'payment terms', 'due date', 'po number']
        bill_keywords = ['billing period', 'account number', 'previous balance', 'current charges', 'amount due', 'statement']
        receipt_keywords = ['receipt', 'thank you', 'cashier', 'register', 'transaction', 'tender']
        
        # Count keyword matches
        invoice_score = sum(1 for kw in invoice_keywords if kw in full_text)
        bill_score = sum(1 for kw in bill_keywords if kw in full_text)
        receipt_score = sum(1 for kw in receipt_keywords if kw in full_text)
        
        # Determine document type
        doc_type = None
        confidence = 0.5
        
        if invoice_score > bill_score and invoice_score > receipt_score and invoice_score > 0:
            doc_type = "invoice"
            confidence = min(0.9, 0.5 + (invoice_score * 0.1))
        elif bill_score > receipt_score and bill_score > 0:
            doc_type = "bill"
            confidence = min(0.9, 0.5 + (bill_score * 0.1))
        elif receipt_score > 0:
            doc_type = "receipt"
            confidence = min(0.9, 0.5 + (receipt_score * 0.1))
        else:
            # Default to generic financial document
            doc_type = "financial_document"
            confidence = 0.4
        
        return {
            'value': doc_type,
            'confidence': confidence,
            'box': None
        }
    
    def extract_invoice_number(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract invoice number from document."""
        full_text = ' '.join(w['text'] for w in words)
        match = self.patterns['invoice_number'].search(full_text)
        
        if match:
            invoice_num = match.group(1)
            for w in words:
                if invoice_num in w['text']:
                    return {
                        'value': invoice_num,
                        'confidence': w['confidence'],
                        'box': self._make_box(w['box'])
                    }
        return None
    
    def extract_due_date(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract due date from document."""
        for i, w in enumerate(words):
            if 'due' in w['text'].lower():
                # Extract text chunk once for efficiency
                text_chunk = ' '.join(w['text'] for w in words[i:min(i+8, len(words))])
                date_val = self.extract_date(text_chunk)
                if date_val:
                    # Find the word that contains the date pattern
                    for j in range(i, min(i + 5, len(words))):
                        if any(pattern.search(words[j]['text']) for pattern in self.patterns['date']):
                            return {
                                'value': date_val,
                                'confidence': words[j]['confidence'],
                                'box': self._make_box(words[j]['box'])
                            }
        return None
    
    def extract_payment_terms(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract payment terms from document."""
        full_text = ' '.join(w['text'] for w in words)
        patterns = [
            re.compile(r'(?:net|terms?)\s*(\d+)', re.IGNORECASE),
            re.compile(r'(?:payment\s+terms?)[:\s]*([^,\n]+)', re.IGNORECASE)
        ]
        
        for pattern in patterns:
            match = pattern.search(full_text)
            if match:
                terms = match.group(1).strip()
                for w in words:
                    if terms in w['text'] or w['text'] in terms:
                        return {
                            'value': terms,
                            'confidence': w['confidence'],
                            'box': self._make_box(w['box'])
                        }
        return None
    
    def extract_customer_name(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract customer name from invoice."""
        for i, w in enumerate(words):
            text_lower = w['text'].lower()
            if 'bill to' in text_lower or 'customer' in text_lower:
                # Take next 2-3 words as customer name
                if i + 1 < len(words):
                    customer_words = words[i+1:min(i+4, len(words))]
                    if len(customer_words) > 0:  # Safety check for division by zero
                        customer_name = ' '.join(w['text'] for w in customer_words)
                        avg_conf = sum(w['confidence'] for w in customer_words) / len(customer_words)
                        return {
                            'value': customer_name,
                            'confidence': avg_conf,
                            'box': self._combine_boxes([w['box'] for w in customer_words])
                        }
        return None
    
    def extract_customer_address(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract customer address from invoice."""
        # This is similar to merchant address but looks after "bill to" keyword
        for i, w in enumerate(words):
            if 'bill to' in w['text'].lower() and i + 1 < len(words):
                # Look for address in next few lines
                address_words = words[i+1:min(i+10, len(words))]
                if len(address_words) > 0:  # Safety check for division by zero
                    address_text = ' '.join(w['text'] for w in address_words[:5])
                    num_words = min(5, len(address_words))
                    avg_conf = sum(w['confidence'] for w in address_words[:num_words]) / num_words
                    return {
                        'value': address_text,
                        'confidence': avg_conf,
                        'box': self._combine_boxes([w['box'] for w in address_words[:num_words]])
                    }
        return None
    
    def extract_po_number(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract purchase order number."""
        full_text = ' '.join(w['text'] for w in words)
        match = self.patterns['po_number'].search(full_text)
        
        if match:
            po_num = match.group(1)
            for w in words:
                if po_num in w['text']:
                    return {
                        'value': po_num,
                        'confidence': w['confidence'],
                        'box': self._make_box(w['box'])
                    }
        return None
    
    def extract_account_number(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract account number from bill."""
        full_text = ' '.join(w['text'] for w in words)
        match = self.patterns['account_number'].search(full_text)
        
        if match:
            acct_num = match.group(1)
            for w in words:
                if acct_num in w['text']:
                    return {
                        'value': acct_num,
                        'confidence': w['confidence'],
                        'box': self._make_box(w['box'])
                    }
        return None
    
    def extract_billing_period(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract billing period from bill."""
        for i, w in enumerate(words):
            if 'billing period' in w['text'].lower() or 'statement period' in w['text'].lower():
                # Look for dates in next few words
                for j in range(i, min(i + 10, len(words))):
                    text_chunk = ' '.join(w['text'] for w in words[j:min(j+8, len(words))])
                    # Use compiled pattern for date range
                    match = self.patterns['billing_period'].search(text_chunk)
                    if match:
                        return {
                            'value': f"{match.group(1)} to {match.group(2)}",
                            'confidence': words[j]['confidence'],
                            'box': self._make_box(words[j]['box'])
                        }
        return None
    
    def extract_previous_balance(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract previous balance from bill."""
        return self._extract_amount_near_keyword(words, ['previous balance', 'prev balance', 'balance forward'])
    
    def extract_current_charges(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract current charges from bill."""
        return self._extract_amount_near_keyword(words, ['current charges', 'new charges', 'current amount'])
    
    def extract_amount_due(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract amount due from bill."""
        return self._extract_amount_near_keyword(words, ['amount due', 'total due', 'balance due'])
    
    def extract_payment_method(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract payment method from receipt."""
        payment_types = ['cash', 'credit', 'debit', 'visa', 'mastercard', 'amex', 'discover', 'check']
        for w in words:
            text_lower = w['text'].lower()
            for ptype in payment_types:
                if ptype in text_lower:
                    return {
                        'value': ptype.upper(),
                        'confidence': w['confidence'],
                        'box': self._make_box(w['box'])
                    }
        return None
    
    def extract_cashier_name(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract cashier name from receipt."""
        for i, w in enumerate(words):
            if 'cashier' in w['text'].lower() or 'served by' in w['text'].lower():
                if i + 1 < len(words):
                    return {
                        'value': words[i+1]['text'],
                        'confidence': words[i+1]['confidence'],
                        'box': self._make_box(words[i+1]['box'])
                    }
        return None
    
    def extract_register_number(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract register/terminal number from receipt."""
        for i, w in enumerate(words):
            text_lower = w['text'].lower()
            if 'register' in text_lower or 'terminal' in text_lower or 'pos' in text_lower:
                # Look for number in same or next word
                for j in range(i, min(i + 3, len(words))):
                    match = re.search(r'(\d+)', words[j]['text'])
                    if match:
                        return {
                            'value': match.group(1),
                            'confidence': words[j]['confidence'],
                            'box': self._make_box(words[j]['box'])
                        }
        return None
    
    def extract_discount(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract discount amount."""
        return self._extract_amount_near_keyword(words, ['discount', 'savings', 'coupon'])
    
    def extract_shipping(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract shipping amount."""
        return self._extract_amount_near_keyword(words, ['shipping', 'freight', 'delivery'])
    
    def extract_address(
        self,
        words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract merchant/vendor address from document."""
        # Look for address pattern near top of document (after vendor name)
        # Skip first 3 words (likely vendor name)
        if len(words) < 5:
            return None
        
        # Look for address indicators
        address_keywords = ['street', 'st', 'ave', 'avenue', 'rd', 'road', 'blvd', 'boulevard', 'suite', 'ste']
        for i, w in enumerate(words[3:20], start=3):  # Check first 20 words after vendor
            text_lower = w['text'].lower()
            if any(kw in text_lower for kw in address_keywords):
                # Take surrounding words as address
                start = max(0, i - 2)
                end = min(len(words), i + 5)
                address_words = words[start:end]
                address_text = ' '.join(w['text'] for w in address_words)
                avg_conf = sum(w['confidence'] for w in address_words) / len(address_words)
                return {
                    'value': address_text,
                    'confidence': avg_conf,
                    'box': self._combine_boxes([w['box'] for w in address_words])
                }
        return None
    
    def _extract_amount_near_keyword(
        self,
        words: List[Dict[str, Any]],
        keywords: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Helper to extract amount near specific keywords."""
        for i, w in enumerate(words):
            text_lower = w['text'].lower()
            if any(kw in text_lower for kw in keywords):
                # Look for amount in next few words
                for j in range(max(0, i-2), min(i + 5, len(words))):
                    amount = self.extract_amount(words[j]['text'])
                    if amount:
                        return {
                            'value': str(amount),
                            'confidence': words[j]['confidence'],
                            'box': self._make_box(words[j]['box'])
                        }
        return None
    
    def _make_box(self, box: List[int]) -> Dict[str, int]:
        """Convert box list to dictionary format."""
        return {
            'x0': box[0],
            'y0': box[1],
            'x1': box[2],
            'y1': box[3]
        }
    
    def _combine_boxes(self, boxes: List[List[int]]) -> Dict[str, int]:
        """Combine multiple boxes into one bounding box."""
        if not boxes:
            return {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        return {
            'x0': min(b[0] for b in boxes),
            'y0': min(b[1] for b in boxes),
            'x1': max(b[2] for b in boxes),
            'y1': max(b[3] for b in boxes)
        }
