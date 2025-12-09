# Supported Document Types and Fields

This document describes the document types and fields that the DocumentProcessor inference models can extract.

## Supported Document Types

The system automatically classifies documents into one of the following types with confidence levels:

1. **Receipt** - Retail purchase receipts, transaction receipts
2. **Invoice** - Business invoices, bills for services or products
3. **Bill** - Utility bills, service statements, account statements
4. **Financial Document** - Generic financial documents (fallback category)

## Document Classification

Document type is automatically detected using:
- **Keyword-based heuristics**: Analyzes text for document-specific keywords
- **Model inference** (IDEFICS2): Uses AI to classify based on visual layout and content
- **Confidence scores**: All classifications include confidence levels (0.0-1.0)

### Classification Keywords

| Document Type | Keywords |
|--------------|----------|
| Invoice | invoice, bill to, ship to, payment terms, due date, po number |
| Bill | billing period, account number, previous balance, current charges, amount due, statement |
| Receipt | receipt, thank you, cashier, register, transaction, tender |

## Supported Models and Document Types

All models have commercial-friendly open source licenses (MIT or Apache 2.0).

| Model | Receipt | Invoice | Bill | Financial Doc | License | Notes |
|-------|---------|---------|------|---------------|---------|-------|
| **Donut** | ✅ Optimized | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | MIT | CORD-v2 fine-tuned for receipts |
| **IDEFICS2** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent | Apache 2.0 | Multi-document support via prompting |
| **Phi-3-Vision** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent | MIT | Efficient, 128k context window |
| **InternVL** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent | MIT | Strong OCR capabilities |
| **Qwen2-VL** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent | Apache 2.0 | Efficient and accurate |

**Legend:**
- ✅ Excellent/Optimized: Model designed for or performs excellently on this document type
- ⚠️ Limited: Model can extract some fields but not optimized for this type

## Extracted Fields by Document Type

All fields include:
- `value`: The extracted value
- `confidence`: Confidence score (0.0-1.0)
- `box`: Bounding box coordinates (x0, y0, x1, y1) or null

### Common Fields (All Document Types)

These fields are extracted from all document types:

| Field | Type | Description |
|-------|------|-------------|
| `document_type` | string | Document classification (receipt, invoice, bill, financial_document) |
| `vendor_name` | string | Business/merchant/company name |
| `merchant_address` | string | Business address |
| `date` | string (ISO) | Document or transaction date |
| `total_amount` | string (decimal) | Total amount |
| `subtotal` | string (decimal) | Subtotal before tax |
| `tax_amount` | string (decimal) | Tax amount |
| `currency` | string | Currency code (USD, EUR, etc.) |
| `line_items` | array | Array of line items (see below) |
| `discount` | string (decimal) | Discount amount |
| `shipping` | string (decimal) | Shipping/delivery charges |
| `notes` | string | Additional notes or comments |

### Line Item Structure

Each line item contains:

```json
{
  "description": "Item description",
  "quantity": 2,
  "unit_price": "10.50",
  "line_total": "21.00",
  "confidence": 0.89,
  "box": {"x0": 50, "y0": 300, "x1": 550, "y1": 330}
}
```

### Receipt-Specific Fields

Additional fields for receipts:

| Field | Type | Description |
|-------|------|-------------|
| `payment_method` | string | Payment method (CASH, CREDIT, DEBIT, VISA, etc.) |
| `cashier_name` | string | Cashier or server name |
| `register_number` | string | Register or terminal number |

### Invoice-Specific Fields

Additional fields for invoices:

| Field | Type | Description |
|-------|------|-------------|
| `invoice_number` | string | Invoice number or reference |
| `due_date` | string (ISO) | Payment due date |
| `payment_terms` | string | Payment terms (e.g., "Net 30", "Due on receipt") |
| `customer_name` | string | Customer or "Bill To" name |
| `customer_address` | string | Customer or "Bill To" address |
| `po_number` | string | Purchase order number |

### Bill-Specific Fields

Additional fields for bills:

| Field | Type | Description |
|-------|------|-------------|
| `account_number` | string | Account number |
| `billing_period` | string | Billing period or statement period |
| `previous_balance` | string (decimal) | Previous balance carried forward |
| `current_charges` | string (decimal) | Current period charges |
| `amount_due` | string (decimal) | Total amount due (may differ from total_amount) |

## Model Recommendations

### For Receipts
- **Best**: Donut (naver-clova-ix/donut-base-finetuned-cord-v2)
  - Optimized for receipts with CORD-v2 fine-tuning
  - Fast inference (~2-3 seconds on GPU)
  - Lower memory requirements (~2GB)
- **Alternatives**:
  - **Qwen2-VL-2B**: Lightweight, good performance
  - **Phi-3-Vision**: Efficient with strong accuracy
  - **IDEFICS2**: Better at non-standard formats, higher memory

### For Invoices
- **Best**: InternVL (OpenGVLab/InternVL2-8B)
  - Excellent understanding of invoice structure
  - Strong OCR capabilities for detailed text
  - Extracts invoice-specific fields reliably
- **Alternatives**:
  - **IDEFICS2**: Very good, handles various layouts
  - **Qwen2-VL**: Strong performance, efficient
  - **Phi-3-Vision**: Good balance of speed and accuracy

### For Bills
- **Best**: IDEFICS2 or InternVL
  - Both excel at understanding billing statements
  - Extract account and billing period information
  - Handle tabular data well
- **Alternatives**:
  - **Qwen2-VL**: Good at structured data extraction
  - **Phi-3-Vision**: Efficient processing

### For Mixed Document Types
- **Best**: IDEFICS2, Phi-3-Vision, InternVL, or Qwen2-VL
  - All support multiple document types
  - Automatic document classification
  - Flexible prompting for custom use cases
- **Efficiency Choice**: Phi-3-Vision or Qwen2-VL-2B for lower memory usage
- **Accuracy Choice**: InternVL2-8B or IDEFICS2-8b for highest accuracy
- **Balanced Choice**: Qwen2-VL-7B for good balance of performance and efficiency

## Field Extraction Strategy

The system uses a multi-level extraction strategy:

1. **Model Inference**: Primary extraction using vision-language models
   - All models: Direct generation of structured output from document images

2. **Heuristic Fallback**: Rule-based extraction for missing fields
   - Keyword matching for field identification
   - Regex patterns for dates, amounts, IDs
   - Spatial analysis for field relationships

3. **Confidence Scoring**: All extractions include confidence levels
   - Model-generated: Scores from generation process (default 0.8)
   - Heuristic: Scores based on pattern strength (0.4-0.9)
   - Document classification: Based on keyword count (0.4-0.9)

## Adding Support for New Document Types

To extend support for new document types:

1. **Update Document Classification** (`field_extractor.py`):
   - Add keywords for the new document type
   - Update `classify_document_type()` method

2. **Add Document-Specific Fields** (if needed):
   - Update `InferenceResult.cs` with new fields
   - Add extraction methods to `FieldExtractor` class

3. **Update Model Prompts**:
   - Extend document extraction prompts in model implementations
   - Include new fields in example JSON outputs

4. **Test with Sample Documents**:
   - Validate extraction accuracy with new document type
   - Adjust confidence thresholds as needed

## Confidence Level Interpretation

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| 0.9 - 1.0 | Very High | Trust the extraction |
| 0.8 - 0.9 | High | Generally reliable |
| 0.7 - 0.8 | Medium | Review if critical |
| 0.5 - 0.7 | Low | Manual verification recommended |
| < 0.5 | Very Low | Likely incorrect or uncertain |

Fields with confidence below the configured threshold (default 0.5) may be filtered out automatically.

## Future Enhancements

Potential improvements for future versions:

1. **Additional Document Types**:
   - Purchase orders
   - Delivery notes
   - Credit notes
   - Bank statements
   - Tax forms

2. **Enhanced Field Extraction**:
   - Multi-currency support
   - Payment status tracking
   - Due date calculations
   - Tax category breakdown

3. **Model Improvements**:
   - Fine-tuned models for each document type
   - Ensemble approaches combining multiple models
   - Active learning for continuous improvement

4. **Validation Rules**:
   - Cross-field validation (subtotal + tax = total)
   - Business rule validation
   - Format validation (date, amount formats)
