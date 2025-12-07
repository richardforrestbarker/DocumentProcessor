# Document OCR Service

Python-based document OCR service using PaddleOCR and open-source transformer models (Donut, IDEFICS2, LayoutLMv3) for structured data extraction from multiple financial document types including receipts, invoices, bills, and other financial documents.

## Python Requirements

- **Python version must be 3.12** for all environments.
- **Windows users:** The Ninja build system must **not** be on your PATH, or else pip will use Ninja for building wheels instead of the default backend (setuptools). This can cause build failures for some dependencies. If you encounter build errors, ensure Ninja is not present in your PATH.

## Overview

This service provides OCR and structured field extraction from document images using:
- **PaddleOCR (PP-StructureV3)**: Text detection and recognition
- **Donut (default)**: OCR-free document understanding transformer (MIT license) - optimized for receipts
- **IDEFICS2**: Multimodal vision-language model (Apache 2.0 license) - supports multiple document types
- **LayoutLMv3**: Layout-aware field extraction (requires OCR first)

## Features

- **Multi-document type support**: Receipts, invoices, bills, and general financial documents
- **Automatic document classification**: Identifies document type with confidence scores
- **Extended field extraction**: Document-type-specific fields with confidence levels
- Multi-page document processing
- Token-to-bounding-box mapping
- Configurable model selection (Donut, IDEFICS2, LayoutLMv3)
- GPU acceleration with CPU fallback
- CLI interface for integration with .NET API
- Open-source models with permissive licenses

## Supported Models

| Model | License | OCR Required | Memory | Best For |
|-------|---------|--------------|--------|----------|
| Donut | MIT | No | ~2GB | Fast processing, receipt-specific |
| IDEFICS2 | Apache 2.0 | No | ~16GB (4-bit: ~6GB) | High accuracy, multi-document types |
| Phi-3-Vision | MIT | No | ~7GB | Efficient, balanced performance |
| InternVL | MIT | No | ~8GB (2B: ~4GB) | High accuracy, strong OCR |
| Qwen2-VL | Apache 2.0 | No | ~7GB (2B: ~4GB) | Strong performance, efficient |
| LayoutLMv3 | - | Yes | ~2GB | Token classification, custom training |

**Model Capabilities:**
- **Donut** (naver-clova-ix/donut-base-finetuned-cord-v2): Best for receipts (CORD-v2 fine-tuned). Document type is inferred as "receipt".
- **IDEFICS2** (HuggingFaceM4/idefics2-8b): Supports all document types (receipts, invoices, bills, financial documents). Uses advanced prompting to extract document-specific fields.
- **Phi-3-Vision** (microsoft/Phi-3-vision-128k-instruct): Microsoft's lightweight vision-language model with 128k context window. Good balance of efficiency and accuracy for all document types.
- **InternVL** (OpenGVLab/InternVL2-8B, InternVL2-4B, InternVL2-2B): Powerful vision-language model with strong OCR and document understanding. Available in multiple sizes.
- **Qwen2-VL** (Qwen/Qwen2-VL-7B-Instruct, Qwen2-VL-2B-Instruct): Alibaba's efficient vision-language model with strong performance on document tasks.
- **LayoutLMv3** (microsoft/layoutlmv3-base): Requires OCR preprocessing. Can be fine-tuned for specific document types.

## Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended for performance)
- 8GB+ RAM (16GB+ recommended with GPU)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Installing OCR Dependencies

#### PaddleOCR (Recommended - Primary OCR Engine)

**On Linux/macOS:**
```bash
# Install PaddlePaddle (CPU)
pip install paddlepaddle

# Or with GPU support (CUDA 11.8)
pip install paddlepaddle-gpu

# Install PaddleOCR
pip install paddleocr
```

**On Windows:**
```bash
# Install PaddlePaddle (CPU)
pip install paddlepaddle

# Install PaddleOCR
pip install paddleocr

# Note: GPU support on Windows requires specific CUDA version
# See: https://www.paddlepaddle.org.cn/install/quick
```

PaddleOCR will automatically download required models on first use.

#### ImageMagick (Required - Image Preprocessing)

ImageMagick is required for the image preprocessing pipeline. It provides optimal image processing for best OCR accuracy.

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install imagemagick
```

**On macOS:**
```bash
brew install imagemagick
```

**On Windows:**
1. Download the installer from: https://imagemagick.org/script/download.php
2. Run the installer and select "Install development headers and libraries for C and C++"
3. Add ImageMagick to your PATH environment variable (the installer can do this automatically)

**Verify installation:**
```bash
magick --version
```

#### Tesseract (Fallback OCR Engine)

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
pip install pytesseract
```

**On macOS:**
```bash
brew install tesseract
pip install pytesseract
```

**On Windows:**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and note the installation path (e.g., `C:\Program Files\Tesseract-OCR`)
3. Add Tesseract to your PATH environment variable
4. Install the Python wrapper:
   ```bash
   pip install pytesseract
   ```

### Downloading Models

Models are automatically downloaded from HuggingFace on first use. You can also pre-download them:

#### Donut (Default - Recommended)

Donut is an OCR-free document understanding model with MIT license.

```bash
# Using HuggingFace CLI
huggingface-cli download naver-clova-ix/donut-base-finetuned-cord-v2 --local-dir ./models/donut-cord-v2

# Or using Python
from transformers import DonutProcessor, VisionEncoderDecoderModel
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

#### IDEFICS2

IDEFICS2 is a multimodal vision-language model with Apache 2.0 license. Requires more GPU memory.

```bash
# Using HuggingFace CLI
huggingface-cli download HuggingFaceM4/idefics2-8b --local-dir ./models/idefics2-8b

# For 4-bit quantized version (lower memory)
huggingface-cli download HuggingFaceM4/idefics2-8b-AWQ --local-dir ./models/idefics2-8b-awq
```

#### LayoutLMv3 (Legacy)

LayoutLMv3 requires OCR to be run first.

```bash
huggingface-cli download microsoft/layoutlmv3-base --local-dir ./models/layoutlmv3-base
3. Use the local path when running the CLI:
   ```bash
   python cli.py process --image receipt.jpg --model ./models/layoutlmv3-base
   ```

### Verifying Installation

```bash
# Check all dependencies
python -c "
import torch
import transformers
import paddleocr
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print('PaddleOCR: OK')
"

# Test with version command
python cli.py version
```

## Usage

### Command-line Interface

Process a single document (uses Donut by default):

```bash
python cli.py process --image path/to/document.jpg --output result.json
```

Process multiple pages:

```bash
python cli.py process --image page1.jpg --image page2.jpg --output result.json
```

Use a specific model type:

```bash
# Use Donut (default, OCR-free, MIT license)
python cli.py process --image document.jpg --output result.json --model-type donut

# Use IDEFICS2 (multimodal, Apache 2.0 license, requires more GPU memory)
python cli.py process --image document.jpg --output result.json --model-type idefics2 --device cuda

# Use LayoutLMv3 (requires OCR first)
python cli.py process --image document.jpg --output result.json --model-type layoutlmv3 --model microsoft/layoutlmv3-base
```

Configure OCR engine and device:

```bash
python cli.py process --image document.jpg --output result.json --ocr-engine paddle --device cuda
```

### Debug Mode

Debug mode saves intermediary images for each processing step, allowing you to validate that each stage of the pipeline is functioning correctly:

```bash
python cli.py process \
  --image document.jpg \
  --output result.json \
  --debug \
  --debug-output-dir ./my_debug_output
```

When debug mode is enabled, the following files are created in the debug output directory:

| Step | File | Description |
|------|------|-------------|
| 1 | `step_01_source_page01.png` | Original source image |
| 2 | `step_02_grayscale_page01.png` | Grayscale converted image |
| 3 | `step_03_denoised_page01.png` | Denoised image (if --denoise enabled) |
| 4 | `step_04_deskewed_page01.png` | Deskewed image (if --deskew enabled) |
| 5 | `step_05_contrast_enhanced_page01.png` | Contrast enhanced image |
| 6 | `step_06_preprocessed_final_page01.png` | Final preprocessed image sent to OCR |
| 7 | `step_07_paddle_bboxes_page01.png` | Image with OCR bounding boxes drawn (color-coded by confidence) |
| 8 | `step_08_result_bboxes_page01.png` | Image with extracted field bounding boxes drawn |
| - | `debug_summary.json` | Summary JSON with list of files and final result |

The debug output helps diagnose issues in the processing pipeline:
- **Low OCR accuracy?** Check grayscale and preprocessing steps
- **Missing text?** Examine OCR bounding boxes visualization
- **Incorrect field extraction?** Review result bounding boxes to see what fields were identified

### Python API

```python
from src.receipt_processor import ReceiptProcessor

processor = ReceiptProcessor(
    model_name="naver-clova-ix/donut-base-finetuned-cord-v2",
    model_type="donut",
    ocr_engine="paddle",
    device="cuda"
)

result = processor.process_receipt(["document1.jpg", "document2.jpg"])
print(result.to_json())
```

## Configuration

Configuration file: `config/config.yaml`

```yaml
model:
  name_or_path: "naver-clova-ix/donut-base-finetuned-cord-v2"
  type: "donut"  # donut, idefics2, or layoutlmv3
  device: "auto"  # auto, cuda, cpu
  
ocr:
  engine: "paddle"  # paddle, tesseract
  detection_mode: "word"  # word, line
  
preprocessing:
  target_dpi: 300
  denoise: true
  deskew: true
  enhance_contrast: true
  # ImageMagick preprocessing parameters
  fuzz_percent: 30           # Background removal tolerance (0-100)
  deskew_threshold: 40       # Deskew sensitivity (0-100)
  contrast_type: sigmoidal   # sigmoidal, linear, or none
  contrast_strength: 3       # Sigmoidal strength (1-10 typical)
  contrast_midpoint: 120     # Sigmoidal midpoint (0-200%, >100 brightens)
  
postprocessing:
  min_confidence: 0.5
  verify_totals: true
```

### Preprocessing Parameters

The image preprocessing pipeline supports configurable parameters via CLI or config:

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Fuzz % | `--fuzz-percent` | 30 | Tolerance for background removal. Higher values remove more colors similar to white. |
| Deskew Threshold | `--deskew-threshold` | 40 | Skew detection sensitivity. Lower values are more aggressive. |
| Contrast Type | `--contrast-type` | sigmoidal | Enhancement algorithm (see below) |
| Contrast Strength | `--contrast-strength` | 3 | Intensity for sigmoidal contrast (1-10 typical) |
| Contrast Midpoint | `--contrast-midpoint` | 120 | Midpoint for sigmoidal (>100 brightens, <100 darkens) |

#### Contrast Types

| Type | Description | Best For |
|------|-------------|----------|
| `sigmoidal` | Non-linear S-curve contrast using `-sigmoidal-contrast strength x midpoint%`. Preserves highlight and shadow detail while boosting midtones. | Most images, especially photos |
| `linear` | Simple histogram stretch using `-auto-level`. Stretches the darkest pixel to black and lightest to white. | High-contrast documents |
| `none` | Skip contrast enhancement entirely. | Already processed images |

**Sigmoidal Parameters:**
- `contrast_strength` (1-10): Controls the steepness of the S-curve. Higher = more contrast.
- `contrast_midpoint` (0-200%): The brightness level around which contrast is centered.
  - Values > 100% brighten the image overall
  - Values < 100% darken the image overall
  - 50% targets middle tones (traditional midpoint)

## Output Format

All extracted fields include confidence levels. The output format varies based on document type:

```json
{
  "job_id": "unique-job-id",
  "status": "done",
  "document_type": {
    "value": "invoice",
    "confidence": 0.88,
    "box": null
  },
  "pages": [
    {
      "page_number": 1,
      "raw_ocr_text": "Full text from OCR...",
      "words": [
        {
          "text": "INVOICE",
          "box": {"x0": 100, "y0": 50, "x1": 200, "y1": 80},
          "confidence": 0.98
        }
      ]
    }
  ],
  "vendor_name": {
    "value": "Company Name",
    "confidence": 0.95,
    "box": {"x0": 50, "y0": 20, "x1": 300, "y1": 80}
  },
  "date": {
    "value": "2024-01-15",
    "confidence": 0.92,
    "box": {"x0": 400, "y0": 30, "x1": 550, "y1": 70}
  },
  "invoice_number": {
    "value": "INV-2024-001",
    "confidence": 0.87,
    "box": {"x0": 50, "y0": 100, "x1": 200, "y1": 130}
  },
  "due_date": {
    "value": "2024-02-15",
    "confidence": 0.85,
    "box": {"x0": 400, "y0": 100, "x1": 550, "y1": 130}
  },
  "customer_name": {
    "value": "Client Company",
    "confidence": 0.86,
    "box": {"x0": 50, "y0": 150, "x1": 250, "y1": 180}
  },
  "total_amount": {
    "value": "1250.00",
    "confidence": 0.96,
    "box": {"x0": 420, "y0": 600, "x1": 520, "y1": 650}
  },
  "subtotal": {
    "value": "1150.00",
    "confidence": 0.94,
    "box": {"x0": 420, "y0": 550, "x1": 520, "y1": 580}
  },
  "tax_amount": {
    "value": "100.00",
    "confidence": 0.93,
    "box": {"x0": 420, "y0": 575, "x1": 520, "y1": 605}
  },
  "line_items": [
    {
      "description": "Consulting Services",
      "quantity": 10,
      "unit_price": "100.00",
      "line_total": "1000.00",
      "box": {"x0": 50, "y0": 300, "x1": 550, "y1": 340},
      "confidence": 0.89
    }
  ]
}
```

### Document-Specific Fields

**Common Fields (all document types):**
- `document_type`: Document classification (receipt, invoice, bill, financial_document)
- `vendor_name`: Business or merchant name
- `merchant_address`: Business address
- `date`: Document date
- `total_amount`: Total amount
- `subtotal`: Subtotal before tax
- `tax_amount`: Tax amount
- `currency`: Currency code (e.g., USD)
- `line_items`: Array of line items with description, quantity, prices
- `discount`: Discount amount
- `shipping`: Shipping/delivery charges
- `notes`: Additional notes

**Invoice-Specific Fields:**
- `invoice_number`: Invoice or reference number
- `due_date`: Payment due date
- `payment_terms`: Payment terms (e.g., "Net 30")
- `customer_name`: Customer or "Bill To" name
- `customer_address`: Customer address
- `po_number`: Purchase order number

**Bill-Specific Fields:**
- `account_number`: Account number
- `billing_period`: Billing period or statement period
- `previous_balance`: Previous balance carried forward
- `current_charges`: Current period charges
- `amount_due`: Total amount due

**Receipt-Specific Fields:**
- `payment_method`: Payment method (cash, credit, debit, etc.)
- `cashier_name`: Cashier or server name
- `register_number`: Register or terminal number

## Architecture

### Pipeline Stages

1. **Image Preprocessing** (via ImageMagick):
   - Deskew (rotation correction)
   - Contrast enhancement
   - Grayscale conversion
   - Remove background
   - Denoise
   - Convert to TIFF
   - Fix resolution (300 DPI) - last step to avoid large intermediate files
2. **Text Detection**: PaddleOCR detector finds text regions
3. **OCR**: PaddleOCR recognizer extracts text with bounding boxes
4. **Tokenization**: Split text into model tokens, map to boxes
5. **Model Inference**: LayoutLMv3 identifies field types and entities
6. **Postprocessing**: Parse values, verify totals, merge multi-page results

### Manual Image Preprocessing

You can run the preprocessing steps manually using ImageMagick before calling the CLI. This is useful for debugging or customizing the preprocessing pipeline.

Shell scripts are provided in the `scripts/` directory for each preprocessing step:

```bash
# Run all preprocessing steps at once
./scripts/preprocess_all.sh input.jpg output.tiff

# Or run steps individually:

# Step 1: Deskew (straighten the image)
./scripts/deskew.sh input.jpg step1.tiff

# Step 2: Enhance contrast
./scripts/enhance_contrast.sh step1.tiff step2.tiff

# Step 3: Convert to grayscale
./scripts/grayscale.sh step2.tiff step3.tiff

# Step 4: Remove background
./scripts/remove_background.sh step3.tiff step4.tiff

# Step 5: Denoise
./scripts/denoise.sh step4.tiff step5.tiff

# Step 6: Convert to TIFF (optimal format for Tesseract)
./scripts/convert_to_tiff.sh step5.tiff step6.tiff

# Step 7: Fix resolution to 300 DPI
./scripts/fix_resolution.sh step6.tiff final.tiff 300
```

#### Direct ImageMagick Commands

If you prefer to run ImageMagick commands directly without the scripts. Parameters shown with default values:

```bash
# Step 1: Deskew (threshold: 40%)
magick input.jpg -deskew 40% -background white step1.tiff

# Step 2: Enhance contrast (sigmoidal: strength 3, midpoint 120%)
magick step1.tiff -auto-level -sigmoidal-contrast 3x120% step2.tiff

# Step 3: Grayscale
magick step2.tiff -colorspace Gray step3.tiff

# Step 4: Remove background (fuzz: 30%)
magick step3.tiff -fuzz 30% -transparent white -background white -alpha remove -auto-level step4.tiff

# Step 5: Denoise
magick step4.tiff -enhance step5.tiff

# Step 6: Convert to TIFF
magick step5.tiff -compress lzw step6.tiff

# Step 7: Fix resolution to 300 DPI
magick step6.tiff -resample 300 -units PixelsPerInch final.tiff
```

#### All-in-One Command

Run all preprocessing steps in a single ImageMagick command (with default parameter values):

```bash
magick input.jpg \
    -deskew 40% -background white \
    -auto-level -sigmoidal-contrast 3x120% \
    -colorspace Gray \
    -fuzz 30% -transparent white -background white -alpha remove -auto-level \
    -enhance \
    -compress lzw \
    -resample 300 -units PixelsPerInch \
    output.tiff
```

#### Image Size Limits

Tesseract has a maximum image dimension limit of 32767 pixels. The preprocessing pipeline automatically handles this by:

1. Checking the image dimensions before resampling
2. Calculating what the dimensions would be after resampling to target DPI
3. If the resampled image would exceed the limit, reducing DPI in increments of 50
4. Minimum DPI is 100; if even 100 DPI would exceed limits, resolution adjustment is skipped

Note: DPI is less important than contrast between text and background for OCR accuracy. Black text on white backgrounds gives the best results.

### Token-to-Box Mapping

Each word from OCR is tokenized using the model's tokenizer. Sub-tokens inherit the parent word's bounding box:

```
Word: "TOTAL"  Box: [100, 200, 200, 250]
Tokens: ["TO", "##TAL"]
Mapping: 
  - "TO" → [100, 200, 200, 250]
  - "##TAL" → [100, 200, 200, 250]
```

## Testing

The test suite includes both unit tests and integration tests.

### Test Categories

1. **Unit Tests** (`tests/test_cli_unit.py`) - 52 tests
   - CLI argument parsing and validation
   - Device selection logic  
   - Bounding box normalization
   - Heuristic field extraction
   - Output formatting and JSON structure
   - Error handling
   - Preprocessing functions
   - These tests mock OCR/model calls and don't require full dependencies

2. **Integration Tests** (`tests/test_cli_integration.py`) - 21 tests
   - PaddleOCR text detection and recognition
   - Tesseract OCR fallback
   - LayoutLMv3 model loading and inference
   - Full pipeline end-to-end processing
   - Multi-page receipt handling
   - These tests run the actual models and require full dependencies

### Running Tests

```bash
# Run all tests (unit tests will pass, integration tests skip if deps missing)
python -m pytest tests/

# Run only unit tests (no dependencies required beyond numpy, Pillow)
python -m pytest tests/test_cli_unit.py -v

# Run integration tests (requires paddleocr, pytesseract, transformers)
python -m pytest tests/test_cli_integration.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html

# Run excluding slow tests (model loading)
python -m pytest tests/ -m "not slow"

# Run specific test class
python -m pytest tests/test_cli_unit.py::TestNormalizeBoxes -v

# Run specific test
python -m pytest tests/test_cli_unit.py::TestCLIArguments::test_version_command -v
```

### Test Dependencies

**Minimal (unit tests only):**
```bash
pip install pytest pytest-cov numpy Pillow
```

**Full (all tests including integration):**
```bash
pip install -r requirements.txt
```

### Test Coverage

Run with coverage to see which code is tested:

```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Development

### Fine-tuning LayoutLMv3

See `docs/fine_tuning.md` for instructions on:
- Preparing training data
- Labeling documents
- Training the model
- Evaluating performance

### Adding New Models

1. Implement model interface in `src/models/base.py`
2. Add model-specific code in `src/models/your_model.py`
3. Register in `src/models/__init__.py`
4. Update configuration options

## Performance

Typical performance on a document (1-2 pages, 300 DPI):

| Hardware | OCR Time | Model Inference | Total |
|----------|----------|----------------|-------|
| CPU only | 2-4s | 8-15s | 10-20s |
| GPU (CUDA) | 1-2s | 1-3s | 2-5s |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:
```bash
python cli.py process --image receipt.jpg --device cpu
```

### Low Accuracy

1. Check image quality (300+ DPI recommended)
2. Ensure good lighting and minimal skew
3. Try preprocessing options:
   ```bash
   python cli.py process --image document.jpg --denoise --deskew
   ```
4. Consider fine-tuning on your specific document formats

## License

[Specify license]

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing to this project.
