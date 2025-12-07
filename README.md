# DocumentProcessor - Document OCR Processing Component

DocumentProcessor is a reusable .NET component that provides advanced document OCR (Optical Character Recognition) capabilities with machine learning-based field extraction. It enables applications to process multiple types of financial document images (receipts, invoices, bills, and other financial documents) and extract structured data with high accuracy.

## Features

- **Multi-Document Type Support**: Process receipts, invoices, bills, and general financial documents
- **Automatic Document Classification**: Identifies document type with confidence levels
- **Document OCR Processing**: Extract text and structured data from document images
- **Extended Field Extraction**: Document-type-specific fields with confidence scores
  - **Receipts**: vendor, date, total, tax, line items, payment method, cashier, register number
  - **Invoices**: invoice number, due date, payment terms, customer info, PO number, billing details
  - **Bills**: account number, billing period, previous balance, current charges, amount due
  - **Common Fields**: vendor name, address, date, totals, taxes, discounts, shipping, line items
- **Machine Learning Field Extraction**: Identify and extract specific fields using transformer models
- **Multi-Phase Pipeline**: Separate preprocessing, OCR, and inference stages for optimal control
- **RESTful API**: Easy integration via HTTP endpoints
- **Blazor WebAssembly Component**: Ready-to-use UI component for document processing
- **Python OCR Service**: High-accuracy OCR using PaddleOCR, Tesseract, and transformer models
- **Configurable Preprocessing**: Adjustable image enhancement for optimal OCR results

## Technology Stack

- **.NET 10.0**: Backend API and component library
- **ASP.NET Core**: RESTful API services
- **Blazor WebAssembly**: Interactive document processing UI component
- **Python 3.12**: OCR and machine learning pipeline
- **PaddleOCR / Tesseract**: Text detection and recognition
- **Transformer Models**: Multiple vision-language models for field extraction
  - Donut (MIT), IDEFICS2 (Apache 2.0), Phi-3-Vision (MIT)
  - InternVL (MIT), Qwen2-VL (Apache 2.0), LayoutLMv3
- **ImageMagick**: Image preprocessing pipeline

## Prerequisites

### .NET Development

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) or later

### Python Requirements for OCR Service

- **Python version must be 3.12** for all environments
- **Windows users:** The Ninja build system must **not** be on your PATH, as it can cause build failures for some Python dependencies. If you encounter build errors, ensure Ninja is not present in your PATH (e.g., from Visual Studio installations)

### System Dependencies

- **ImageMagick**: Required for image preprocessing
  - Ubuntu/Debian: `sudo apt-get install imagemagick`
  - macOS: `brew install imagemagick`
  - Windows: Download from [ImageMagick Downloads](https://imagemagick.org/script/download.php)
  
- **Tesseract OCR** (optional, fallback engine):
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
  - macOS: `brew install tesseract`
  - Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Building the Project

### 1. Clone the Repository

```bash
git clone https://github.com/richardforrestbarker/DocumentProcessor.git
cd DocumentProcessor
```

### 2. Build the .NET Solution

```bash
# Restore NuGet packages and build
dotnet restore
dotnet build
```

### 3. Set Up the Python OCR Service

```bash
cd Ocr

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Return to root
cd ..
```

For detailed Python setup instructions, see the [OCR Service README](Ocr/README.md).

## Project Structure

```
DocumentProcessor/
├── Api/                      # ASP.NET Core API service
│   ├── DocumentController.cs # Document processing endpoints
│   └── OcrServiceExtensions.cs # Service registration
├── Data/                     # Shared data models and interfaces
│   ├── IDocumentProcessor.cs # Document processor interface
│   ├── OcrConfiguration.cs   # OCR configuration model
│   └── Messages/             # Request/response DTOs
├── Wasm/                     # Blazor WebAssembly components
│   ├── DocumentProcessing.razor # Main document processing component
│   └── ClientSideDocumentProcessor.cs # Client-side implementation
├── Example/                  # Example Blazor web application
│   ├── Components/Pages/     # Blazor pages
│   │   └── Home.razor        # Home page with DocumentProcessingView
│   ├── Program.cs            # Application startup
│   └── appsettings.json      # Configuration including API URL
├── Ocr/                      # Python OCR service
│   ├── cli.py                # Command-line interface
│   ├── src/                  # Python source code
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # OCR service documentation
├── Tests/                    # Unit and integration tests
└── ServiceDefaults/          # Shared service configuration
```

## Using DocumentProcessor in Your Application

The DocumentProcessor component can be integrated into your .NET application in multiple ways:

### Option 1: As a Service (API Integration)

Run the API service and make HTTP requests to the document processing endpoints:

```bash
dotnet run --project Api/Api.csproj
```

The API will be available at `https://localhost:5001` (or `http://localhost:5000`).

### Option 2: As a NuGet Package Reference

Add the Data and Api projects to your solution and reference them:

```xml
<ItemGroup>
  <ProjectReference Include="..\DocumentProcessor\Data\Data.csproj" />
  <ProjectReference Include="..\DocumentProcessor\Api\Api.csproj" />
</ItemGroup>
```

Then register the services in your `Program.cs`:

```csharp
using Api.Ocr;

var builder = WebApplication.CreateBuilder(args);

// Add DocumentProcessor OCR services
builder.Services.AddOcrDocumentProcessing(builder.Configuration);

var app = builder.Build();
```

### Option 3: Blazor WebAssembly Component

To use the interactive document processing UI in your Blazor application:

1. Reference the Wasm project:

```xml
<ItemGroup>
  <ProjectReference Include="..\DocumentProcessor\Wasm\Wasm.csproj" />
</ItemGroup>
```

2. Add the component to your page:

```razor
@page "/process-document"
<DocumentProcessing />
```

## Running the Example Application

The repository includes a complete example Blazor web application that demonstrates how to use the DocumentProcessor component with client-side rendering. The Example application includes the DocumentProcessingView component and is pre-configured to work with the API.

### Prerequisites

Before running the Example application, ensure you have:
1. Built the solution (see "Building the Project" section above)
2. Set up the Python OCR service (see "Set Up the Python OCR Service" section above)

### Running the Example

The Example application requires two processes to be running:

#### 1. Start the API Service

In a terminal window, start the API service:

```bash
dotnet run --project Api/Api.csproj
```

The API will be available at `https://localhost:7415`.

#### 2. Start the Example Application

In another terminal window, start the Example application:

```bash
dotnet run --project Example/Example.csproj
```

The Example application will be available at `https://localhost:7256`. Open this URL in your browser to access the document processing interface.

### Using the Example Application

1. The home page displays the DocumentProcessingView component
2. Upload a document image (receipt, invoice, or form)
3. Adjust preprocessing settings (deskew, denoise, contrast) and preview the results
4. Continue to OCR to extract text
5. Continue to Inference to extract structured fields
6. Accept the final result when satisfied

The Example application demonstrates:
- Client-side rendering with Blazor WebAssembly
- Integration with both the Api and Wasm projects
- Proper configuration of appsettings.json for API communication
- Usage of the DocumentProcessingView component
- Complete document processing workflow

## API Endpoints

The DocumentController provides the following endpoints:

### POST /api/document/preprocess
Run preprocessing on an image without DPI resampling. Returns base64-encoded preprocessed image.

**Request Body:**
```json
{
  "imageBase64": "base64-encoded-image",
  "filename": "document.jpg",
  "jobId": "optional-job-id",
  "denoise": false,
  "deskew": true,
  "fuzzPercent": 30,
  "deskewThreshold": 40,
  "contrastType": "sigmoidal",
  "contrastStrength": 3.0,
  "contrastMidpoint": 120
}
```

### POST /api/document/ocr
Run OCR on a preprocessed image with DPI resampling and safety checks.

**Request Body:**
```json
{
  "imageBase64": "base64-encoded-preprocessed-image",
  "jobId": "optional-job-id",
  "ocrEngine": "paddle",
  "targetDpi": 300,
  "device": "auto"
}
```

### POST /api/document/inference
Run model inference on OCR results to extract structured fields.

**Request Body:**
```json
{
  "ocrResult": { /* OCR result object */ },
  "imageBase64": "base64-encoded-image",
  "jobId": "optional-job-id",
  "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
  "modelType": "donut",
  "device": "auto"
}
```

### GET /api/document/status/{jobId}
Get the status of a document processing job.

## Configuration

### OCR Configuration

Configure OCR settings in your `appsettings.json`:

```json
{
  "Ocr": {
    "model_name_or_path": "microsoft/layoutlmv3-base",
    "device": "auto",
    "ocr_engine": "paddle",
    "detection_mode": "word",
    "box_normalization_scale": 1000,
    "python_service_path": "./Ocr/cli.py",
    "temp_storage_path": "./temp/documents",
    "max_file_size": 10485760,
    "temp_file_ttl_hours": 1,
    "enable_gpu": true,
    "min_confidence_threshold": 0.8
  }
}
```

#### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_name_or_path` | HuggingFace model name or local path | `microsoft/layoutlmv3-base` |
| `device` | Compute device: `auto`, `cuda`, `cpu` | `auto` |
| `ocr_engine` | OCR engine: `paddle` or `tesseract` | `paddle` |
| `detection_mode` | Detection mode: `word` or `line` | `word` |
| `box_normalization_scale` | Bounding box scale for models | `1000` |
| `python_service_path` | Path to Python CLI script | `./Ocr/cli.py` |
| `temp_storage_path` | Temporary file storage location | `./temp/documents` |
| `max_file_size` | Maximum upload size in bytes | `10485760` (10MB) |
| `temp_file_ttl_hours` | Temporary file retention time | `1` hour |
| `enable_gpu` | Enable GPU acceleration | `true` |
| `min_confidence_threshold` | Minimum field confidence (0.0-1.0) | `0.8` |

### API Integration Configuration

If your application needs to integrate with external barcode APIs, configure them in the `Application.Integrations` section of `appsettings.json`. The DocumentProcessor component itself focuses on OCR and document processing, but can be extended to support barcode lookups.

## Extension Points

The DocumentProcessor component is designed to be extensible:

### Custom Document Processors

Implement the `IDocumentProcessor` interface to create custom processing pipelines:

```csharp
public interface IDocumentProcessor
{
    Task<PreprocessingResult> PreprocessImageAsync(PreprocessingRequest request);
    Task<OcrResult> RunOcrAsync(OcrRequest request);
    Task<InferenceResult> RunInferenceAsync(InferenceRequest request);
    Task<JobStatus?> GetJobStatusAsync(string jobId);
}
```

### Service Registration

Register your custom implementation in `Program.cs`:

```csharp
// Use the built-in implementation
builder.Services.AddOcrDocumentProcessing(builder.Configuration);

// Or register a custom implementation
builder.Services.AddSingleton<IDocumentProcessor, MyCustomDocumentProcessor>();
```

## Document OCR Pipeline

The DocumentProcessor includes an advanced OCR pipeline that uses machine learning to extract structured data from document images (receipts, invoices, forms). It combines optical character recognition with layout-aware transformer models (LayoutLMv3, Donut, IDEFICS2) to accurately identify and extract fields like vendor names, dates, amounts, and line items.

### Architecture

The DocumentProcessor uses a hybrid architecture:

1. **API Service (C#/.NET)**: Handles HTTP requests, validation, and orchestrates the Python OCR pipeline
2. **Python OCR Service**: Performs image processing, OCR, and ML-based field extraction
3. **Blazor Component**: Provides interactive UI for document processing with live preview
4. **GPU Acceleration**: Supports CUDA-enabled GPUs for faster processing (falls back to CPU)

```
                    ┌─────────────────────────┐
                    │   Blazor Component      │
                    │  (DocumentProcessing)   │
                    └───────────┬─────────────┘
                                │ HTTP POST
                                ▼
                    ┌─────────────────────────┐
                    │   .NET API Service      │
                    │  (DocumentController)   │
                    │  - Request validation   │
                    │  - Base64 handling      │
                    └───────────┬─────────────┘
                                │ Process.Start
                                │ (Python subprocess)
                                ▼
                    ┌─────────────────────────┐
                    │   Python OCR CLI        │
                    │   (Ocr/cli.py)          │
                    │  - Image preprocessing  │
                    │  - PaddleOCR / Tesseract│
                    │  - Model inference      │
                    │  - Field extraction     │
                    └─────────────────────────┘
```

### OCR Pipeline Stages

1. **Image Preprocessing** (using ImageMagick CLI via shell scripts)
   - Deskewing (rotation correction)
   - Contrast enhancement
   - Grayscale conversion
   - Remove background
   - Denoising
   - Convert to TIFF format (optimal for Tesseract)
   - Fix resolution to 300 DPI

2. **Text Detection & OCR**
   - PaddleOCR (primary, high accuracy)
   - Tesseract (fallback)
   - Word-level bounding boxes with confidence scores

3. **Layout Analysis**
   - LayoutLMv3 model for document understanding
   - Token-to-box mapping (normalized 0-1000 scale)
   - Visual and textual feature fusion

4. **Field Extraction**
   - Vendor name detection
   - Date parsing (multiple formats)
   - Amount extraction (total, subtotal, tax)
   - Line item grouping
   - Currency detection

### Python OCR Service Setup

The Python OCR service is located in the `Ocr/` directory. See [Ocr/README.md](Ocr/README.md) for complete setup instructions.

#### Quick Start

```bash
cd Ocr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test the installation
python cli.py version
```

#### GPU Support

For CUDA GPU acceleration on Linux:
```bash
pip install paddlepaddle-gpu
```

For CPU-only or macOS:
```bash
pip install paddlepaddle
```

### Using the Document Processor

#### Via API

**Preprocess a document:**
```bash
curl -X POST http://localhost:5000/api/document/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "imageBase64": "base64-encoded-image-data",
    "filename": "document.jpg",
    "deskew": true,
    "denoise": false
  }'
```

**Run OCR:**
```bash
curl -X POST http://localhost:5000/api/document/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "imageBase64": "base64-encoded-preprocessed-image",
    "ocrEngine": "paddle",
    "targetDpi": 300
  }'
```

**Extract fields:**
```bash
curl -X POST http://localhost:5000/api/document/inference \
  -H "Content-Type: application/json" \
  -d '{
    "ocrResult": {...},
    "imageBase64": "base64-encoded-image",
    "modelType": "donut"
  }'
```

#### Via Python CLI

Process a single document:
```bash
cd Ocr
python cli.py process --image document.jpg --output result.json
```

Process with preprocessing options:
```bash
python cli.py process \
  --image document.jpg \
  --output result.json \
  --ocr-engine paddle \
  --device cuda \
  --denoise \
  --deskew
```

Process multi-page document:
```bash
python cli.py process \
  --image page1.jpg \
  --image page2.jpg \
  --output result.json
```

Debug mode (saves intermediate images):
```bash
python cli.py process \
  --image document.jpg \
  --output result.json \
  --debug \
  --debug-output-dir ./debug_output
```

#### Separate Pipeline Commands

The CLI supports separate commands for each phase of the OCR pipeline, which is useful for the document processing live view feature:

**Preprocess only (without DPI resampling):**
```bash
python cli.py preprocess \
  --image receipt.jpg \
  --output-format base64 \
  --deskew \
  --denoise \
  --fuzz-percent 30 \
  --contrast-type sigmoidal
```

**OCR only (with DPI resampling):**
```bash
python cli.py ocr \
  --image preprocessed.png \
  --ocr-engine paddle \
  --target-dpi 300 \
  --output ocr_result.json
```

**Inference only (on OCR results):**
```bash
python cli.py inference \
  --ocr-result ocr_result.json \
  --image preprocessed.png \
  --model naver-clova-ix/donut-base-finetuned-cord-v2 \
  --model-type donut
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image, -i` | Path to receipt image (can specify multiple) | Required |
| `--output, -o` | Output JSON file path | stdout |
| `--model, -m` | LayoutLMv3 model name or path | microsoft/layoutlmv3-base |
| `--ocr-engine` | OCR engine: `paddle` or `tesseract` | paddle |
| `--device` | Inference device: `auto`, `cuda`, `cpu` | auto |
| `--denoise` | Apply denoising preprocessing | false |
| `--deskew` | Apply deskew correction | false |
| `--job-id` | Custom job identifier | auto-generated |
| `--debug` | Enable debug mode: save intermediary images for each processing step | false |
| `--debug-output-dir` | Directory to save debug output files | ./debug_output |

### OCR Output Format

The OCR system returns structured JSON with the following schema. All fields include confidence levels:

```json
{
  "job_id": "abc123",
  "status": "done",
  "document_type": {
    "value": "invoice",
    "confidence": 0.92,
    "box": null
  },
  "pages": [
    {
      "page_number": 1,
      "raw_ocr_text": "COMPANY NAME\n123 Main St...",
      "words": [
        {
          "text": "COMPANY",
          "box": { "x0": 100, "y0": 50, "x1": 200, "y1": 80 },
          "confidence": 0.98
        }
      ]
    }
  ],
  "vendor_name": {
    "value": "COMPANY NAME",
    "confidence": 0.95,
    "box": { "x0": 100, "y0": 50, "x1": 300, "y1": 80 }
  },
  "date": {
    "value": "2024-01-15",
    "confidence": 0.92,
    "box": { "x0": 400, "y0": 50, "x1": 550, "y1": 80 }
  },
  "total_amount": {
    "value": "1250.00",
    "confidence": 0.96,
    "box": { "x0": 400, "y0": 500, "x1": 500, "y1": 530 }
  },
  "subtotal": {
    "value": "1150.00",
    "confidence": 0.94,
    "box": { "x0": 400, "y0": 450, "x1": 500, "y1": 480 }
  },
  "tax_amount": {
    "value": "100.00",
    "confidence": 0.93,
    "box": { "x0": 400, "y0": 475, "x1": 500, "y1": 505 }
  },
  "currency": {
    "value": "USD",
    "confidence": 0.90,
    "box": null
  },
  "invoice_number": {
    "value": "INV-12345",
    "confidence": 0.88,
    "box": { "x0": 50, "y0": 100, "x1": 150, "y1": 130 }
  },
  "due_date": {
    "value": "2024-02-15",
    "confidence": 0.85,
    "box": { "x0": 450, "y0": 100, "x1": 550, "y1": 130 }
  },
  "customer_name": {
    "value": "Client Company",
    "confidence": 0.87,
    "box": { "x0": 50, "y0": 150, "x1": 250, "y1": 180 }
  },
  "line_items": [
    {
      "description": "Consulting Services",
      "quantity": 10,
      "unit_price": "100.00",
      "line_total": "1000.00",
      "confidence": 0.89,
      "box": { "x0": 50, "y0": 300, "x1": 550, "y1": 330 }
    }
  ]
}
```

**Document-Specific Fields:**

- **Receipts**: `payment_method`, `cashier_name`, `register_number`
- **Invoices**: `invoice_number`, `due_date`, `payment_terms`, `customer_name`, `customer_address`, `po_number`
- **Bills**: `account_number`, `billing_period`, `previous_balance`, `current_charges`, `amount_due`
- **All Documents**: `document_type`, `vendor_name`, `merchant_address`, `date`, `total_amount`, `subtotal`, `tax_amount`, `currency`, `line_items`, `discount`, `shipping`, `notes`

### Bounding Box Format

All bounding boxes are normalized to a 0-1000 scale for consistency with LayoutLM models:

- `x0`: Left edge (0-1000)
- `y0`: Top edge (0-1000)
- `x1`: Right edge (0-1000)
- `y1`: Bottom edge (0-1000)

### Output Format

The OCR pipeline returns structured JSON with extracted fields:

```json
{
  "job_id": "abc123",
  "status": "done",
  "pages": [
    {
      "page_number": 1,
      "raw_ocr_text": "STORE NAME\n123 Main St...",
      "words": [
        {
          "text": "STORE",
          "box": { "x0": 100, "y0": 50, "x1": 200, "y1": 80 },
          "confidence": 0.98
        }
      ]
    }
  ],
  "vendor_name": {
    "value": "STORE NAME",
    "confidence": 0.95,
    "box": { "x0": 100, "y0": 50, "x1": 300, "y1": 80 }
  },
  "date": {
    "value": "2024-01-15",
    "confidence": 0.92
  },
  "total_amount": {
    "value": "25.99",
    "confidence": 0.96
  },
  "line_items": [
    {
      "description": "Product 1",
      "quantity": 1.0,
      "unit_price": 12.99,
      "line_total": 12.99,
      "confidence": 0.89
    }
  ]
}
```

## Running Tests

### .NET Tests

```bash
dotnet test
```

### Python Tests

```bash
cd Ocr

# Run all tests
python -m pytest tests/

# Run unit tests only (fast, no dependencies)
python -m pytest tests/test_cli_unit.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Troubleshooting

**No OCR results returned:**
- Ensure Python dependencies are installed: `pip install -r Ocr/requirements.txt`
- Check that PaddleOCR or Tesseract is working: `python -c "from paddleocr import PaddleOCR; print('OK')"`
- Verify image is readable and in supported format (JPEG, PNG, TIFF)

**GPU not being used:**
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure paddlepaddle-gpu is installed instead of paddlepaddle
- Set `enable_gpu: true` in `appsettings.json`

**Low accuracy:**
- Try enabling preprocessing options: `--denoise --deskew`
- Ensure image is high resolution (300 DPI recommended)
- Use well-lit, non-blurry images
- Adjust preprocessing parameters (contrast, fuzz percentage)

**Process timeout:**
- First run downloads models (~500MB), subsequent runs are faster
- Increase timeout in configuration if using CPU
- Consider using GPU acceleration for better performance

### Blazor Component Usage

The `DocumentProcessing.razor` component provides an interactive UI for document processing with live preview:

1. **Upload a document image**: Select a file from your device
2. **Adjust preprocessing settings**: Modify deskew, denoise, contrast parameters
3. **Preview results**: See preprocessed image before running OCR
4. **Run OCR**: Extract text with bounding boxes
5. **Extract fields**: Get structured data (vendor, date, amounts, line items)

The component handles the entire workflow through the three-phase pipeline (preprocess → OCR → inference).

## Advanced Topics

### Building the Python OCR Service

The Python OCR service can be built as a standalone package or containerized:

```bash
cd Ocr

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Test CLI
python cli.py version
python cli.py process --help

# Build Docker image (optional)
docker build -t document-processor-ocr .
```

### Supported ML Models

The component supports multiple transformer models for field extraction:

| Model | Type | License | Best For |
|-------|------|---------|----------|
| Donut | OCR-free | MIT | Fast processing, receipt-specific |
| IDEFICS2 | Multimodal | Apache 2.0 | High accuracy, flexible |
| LayoutLMv3 | Token classification | - | Custom fine-tuning |

See [Ocr/README.md](Ocr/README.md) for model-specific configuration.

## Performance

Typical performance on a document (1-2 pages, 300 DPI):

| Hardware | Preprocessing | OCR | Inference | Total |
|----------|--------------|-----|-----------|-------|
| CPU only | 1-2s | 2-4s | 8-15s | 11-21s |
| GPU (CUDA) | 1-2s | 1-2s | 1-3s | 3-7s |

## Example Integration

Here's a complete example of integrating DocumentProcessor into an ASP.NET Core application:

```csharp
// Program.cs
using Api.Ocr;

var builder = WebApplication.CreateBuilder(args);

// Add DocumentProcessor services
builder.Services.AddOcrDocumentProcessing(builder.Configuration);

// Add controllers
builder.Services.AddControllers();

var app = builder.Build();

app.MapControllers();
app.Run();
```

```csharp
// YourController.cs
[ApiController]
[Route("api/[controller]")]
public class MyDocumentController : ControllerBase
{
    private readonly IDocumentProcessor _processor;
    
    public MyDocumentController(IDocumentProcessor processor)
    {
        _processor = processor;
    }
    
    [HttpPost("process")]
    public async Task<IActionResult> ProcessDocument([FromBody] ProcessRequest request)
    {
        // Preprocess
        var preprocessed = await _processor.PreprocessImageAsync(new PreprocessingRequest
        {
            ImageBase64 = request.ImageBase64,
            Deskew = true,
            Denoise = true
        });
        
        // OCR
        var ocrResult = await _processor.RunOcrAsync(new OcrRequest
        {
            ImageBase64 = preprocessed.PreprocessedImageBase64,
            OcrEngine = "paddle"
        });
        
        // Extract fields
        var inference = await _processor.RunInferenceAsync(new InferenceRequest
        {
            OcrResult = ocrResult,
            ImageBase64 = preprocessed.PreprocessedImageBase64,
            ModelType = "donut"
        });
        
        return Ok(inference);
    }
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please see the contribution guidelines for this project.

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.
