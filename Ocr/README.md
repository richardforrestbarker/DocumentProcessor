# Document OCR Service

Python-based document OCR service using PaddleOCR and commercially-licensed vision-language models (Donut, IDEFICS2, Phi-3-Vision, InternVL, Qwen2-VL) for structured data extraction from multiple financial document types including receipts, invoices, bills, and other financial documents.

## Python Requirements

- **Python version must be 3.12** for all environments.
- **Windows users:** The Ninja build system must **not** be on your PATH, or else pip will use Ninja for building wheels instead of the default backend (setuptools). This can cause build failures for some dependencies. If you encounter build errors, ensure Ninja is not present in your PATH.

## Overview

This service provides OCR and structured field extraction from document images using:
- **PaddleOCR (PP-StructureV3)**: Text detection and recognition
- **Donut (default)**: OCR-free document understanding transformer (MIT license) - optimized for receipts
- **IDEFICS2**: Multimodal vision-language model (Apache 2.0 license) - supports multiple document types
- **Phi-3-Vision**: Lightweight vision-language model (MIT license) - efficient and balanced
- **InternVL**: High-accuracy vision-language model (MIT license) - strong OCR capabilities
- **Qwen2-VL**: Efficient vision-language model (Apache 2.0 license) - strong performance

## Features

- **Multi-document type support**: Receipts, invoices, bills, and general financial documents
- **Automatic document classification**: Identifies document type with confidence scores
- **Extended field extraction**: Document-type-specific fields with confidence levels
- Multi-page document processing
- Token-to-bounding-box mapping
- Configurable model selection (all commercially licensed)
- GPU acceleration with CPU fallback
- CLI interface for integration with .NET API
- Open-source models with permissive licenses

## Supported Models

All models have commercial-friendly open source licenses (MIT or Apache 2.0).

| Model | License | OCR Required | Memory | Best For |
|-------|---------|--------------|--------|----------|
| Donut | MIT | No | ~2GB | Fast processing, receipt-specific |
| IDEFICS2 | Apache 2.0 | No | ~16GB (4-bit: ~6GB) | High accuracy, multi-document types |
| Phi-3-Vision | MIT | No | ~7GB | Efficient, balanced performance |
| InternVL | MIT | No | ~8GB (2B: ~4GB) | High accuracy, strong OCR |
| Qwen2-VL | Apache 2.0 | No | ~7GB (2B: ~4GB) | Strong performance, efficient |

**Model Capabilities:**
- **Donut** (naver-clova-ix/donut-base-finetuned-cord-v2): Best for receipts (CORD-v2 fine-tuned). Document type is inferred as "receipt". MIT license.
- **IDEFICS2** (HuggingFaceM4/idefics2-8b): Supports all document types (receipts, invoices, bills, financial documents). Uses advanced prompting to extract document-specific fields. Apache 2.0 license.
- **Phi-3-Vision** (microsoft/Phi-3-vision-128k-instruct): Microsoft's lightweight vision-language model with 128k context window. Good balance of efficiency and accuracy for all document types. MIT license.
- **InternVL** (OpenGVLab/InternVL2-8B, InternVL2-4B, InternVL2-2B): Powerful vision-language model with strong OCR and document understanding. Available in multiple sizes. MIT license.
- **Qwen2-VL** (Qwen/Qwen2-VL-7B-Instruct, Qwen2-VL-2B-Instruct): Alibaba's efficient vision-language model with strong performance on document tasks. Apache 2.0 license.

## Setup

### Prerequisites

- **Python 3.12** (required for all environments)
- **CUDA-capable GPU** (optional, but **highly recommended** for 2-4x faster performance)
- **8GB+ RAM** (16GB+ recommended for GPU acceleration)
- **NVIDIA GPU with 6-8GB+ VRAM** (for GPU acceleration)

### Installation

#### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

#### Step 2: Install PyTorch (Choose GPU or CPU)

**Option A: GPU Installation (Recommended - 2-4x faster)**

Before installing, verify you have CUDA installed:

```bash
# Check if CUDA is available
nvidia-smi

# Check CUDA version
nvcc --version
```

If CUDA is installed, install PyTorch with CUDA support:

```bash
# For CUDA 12.4+ (including CUDA 13.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option B: CPU-Only Installation (Slower, but works without GPU)**

```bash
# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify PyTorch Installation:**

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
- **GPU**: `CUDA available: True`
- **CPU**: `CUDA available: False`

#### Step 3: Install Remaining Dependencies

```bash
# Install all other dependencies
pip install -r requirements.txt
```

**Note for Windows users:** The Ninja build system must **not** be on your PATH, or pip will use Ninja instead of setuptools, causing build failures for some dependencies.

### Installing OCR Dependencies

#### PaddleOCR (Recommended - Primary OCR Engine)

PaddleOCR installation depends on whether you want GPU or CPU support.

**GPU Version (Recommended with CUDA):**

```bash
# For CUDA 12.x+
pip install paddlepaddle-gpu

# Install PaddleOCR
pip install paddleocr
```

**CPU Version:**

```bash
# Install PaddlePaddle (CPU)
pip install paddlepaddle

# Install PaddleOCR
pip install paddleocr
```

**Note**: PaddleOCR will automatically download required models on first use (~300MB).

**Verify PaddleOCR:**
```bash
python -c "import paddleocr; print('PaddleOCR installed successfully')"
```

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



### Verifying Installation

Use the automated verification script:

```bash
# Quick verification (checks dependencies and cache)
python verify_models.py --quick

# Full verification (loads all models)
python verify_models.py

# With GPU testing
python verify_models.py --gpu

# Save results to JSON
python verify_models.py --json results.json
```

Or manually verify:

```bash
# Check all dependencies
python -c "
import torch
import transformers
import paddleocr
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print('PaddleOCR: OK')
"

# Test with CLI version command
python cli.py version
```

**Expected output with GPU:**
```
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

**Expected output with CPU:**
```
PyTorch: 2.6.0+cpu
CUDA available: False
```

### Installing/Switching Between GPU and CPU Versions

#### Switching from CPU to GPU

If you initially installed the CPU version but later want GPU acceleration:

```powershell
# Run the automated installation script (Windows)
.\install_pytorch_cuda.ps1
```

Or manually:

```bash
# Uninstall CPU version
pip uninstall -y torch torchvision torchaudio

# Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Switching from GPU to CPU

If you need to switch back to CPU-only:

```bash
# Uninstall GPU version
pip uninstall -y torch torchvision torchaudio

# Install CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### CUDA Installation Guide

If you don't have CUDA installed and want GPU acceleration:

1. **Check GPU compatibility**: Verify you have an NVIDIA GPU
   ```bash
   nvidia-smi
   ```

2. **Download CUDA Toolkit**: 
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Download CUDA Toolkit 12.4+ or 13.x
   - Follow the installation wizard

3. **Verify CUDA installation**:
   ```bash
   nvcc --version
   ```

4. **Install PyTorch with CUDA** (see Step 2 above)

5. **Run verification**:
   ```bash
   python verify_models.py --gpu
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

# Use Phi-3-Vision (efficient, MIT license)
python cli.py process --image document.jpg --output result.json --model microsoft/Phi-3-vision-128k-instruct

# Use InternVL (high accuracy, MIT license)
python cli.py process --image document.jpg --output result.json --model OpenGVLab/InternVL2-8B

# Use Qwen2-VL (balanced, Apache 2.0 license)
python cli.py process --image document.jpg --output result.json --model Qwen/Qwen2-VL-7B-Instruct
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

## Configuration

Configuration file: `config/config.yaml`

```yaml
model:
  name_or_path: "naver-clova-ix/donut-base-finetuned-cord-v2"
  type: "donut"  # donut, idefics2, phi3-vision, internvl, or qwen2-vl
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
5. **Model Inference**: Vision-language models identify field types and entities
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
   - Vision-language model loading and inference
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

## Model Verification

### Automated Verification with PyTorch Check

The verification script now automatically detects your hardware and recommends the correct PyTorch version:

```bash
# Interactive verification - checks hardware and offers to install correct PyTorch
python verify_models.py --quick

# The script will:
# 1. Detect if you have NVIDIA GPU + CUDA Toolkit
# 2. Check if PyTorch matches your hardware (GPU vs CPU)
# 3. Offer to install the correct version if there's a mismatch
# 4. Verify all dependencies and models
```

**Example Output:**
```
======================================================================
PyTorch Installation Check
======================================================================
✓ NVIDIA GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU
✓ CUDA Toolkit detected: version 13.2
✓ PyTorch with CUDA support is correctly installed
```

or if CPU-only is detected:
```
⚠ PyTorch is installed with CPU-only support, but you have CUDA hardware!
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  CUDA: 13.2
  Installing CUDA-enabled PyTorch will give you 2-4x faster performance.

Would you like to install PyTorch with CUDA support now? (y/N):
```

### Verification Options

```bash
# Quick check with hardware detection and install offer
python verify_models.py --quick

# Force install CUDA version (if hardware detected)
python verify_models.py --install-cuda

# Skip PyTorch check and just verify dependencies
python verify_models.py --quick --skip-pytorch-check

# Full verification - loads and tests all models on GPU
python verify_models.py --gpu

# Save detailed results to JSON
python verify_models.py --json results.json
```

### Quick Verification

To verify that all models and dependencies are installed correctly, use the provided verification scripts:

#### Smoke Test (Quick)
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Run smoke test
python smoke_test.py
```

The smoke test verifies:
- Core dependencies (PyTorch, Transformers, Pillow)
- OCR engines (PaddleOCR, Tesseract)
- At least one model can load

#### Full Verification
```bash
# Quick mode - checks dependencies and cache only
python verify_models.py --quick

# Full mode - loads and tests all cached models
python verify_models.py

# Save results to JSON
python verify_models.py --json results.json

# Test with GPU
python verify_models.py --gpu
```

#### Test Individual Models
```bash
python test_single_model.py donut
python test_single_model.py idefics2
python test_single_model.py phi3_vision
python test_single_model.py internvl
python test_single_model.py qwen2_vl
```

### Verification Results

See `VERIFICATION_REPORT.md` for the latest verification results, which includes:
- ✅ Core dependencies status
- ✅ OCR engine availability
- ✅ Model cache status
- ✅ Model loading test results
- ✅ CLI functionality tests

### Expected Output

✅ **All tests passing:**
```
======================================================================
✅ VERIFICATION PASSED
All required components are working correctly!
======================================================================
```

For detailed results, check:
- `VERIFICATION_REPORT.md` - Human-readable summary
- `quick_verification.json` - Machine-readable results

## License

[Specify license]

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing to this project.
