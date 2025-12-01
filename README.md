# Bardcode - Home Supply Management System

Bardcode is a home supply management system that helps you track and manage household items using barcode scanning. The system allows you to scan product barcodes, retrieve product information, and maintain an inventory of your home supplies.

## Features

- **Barcode Scanning**: Scan product barcodes using your device's camera
- **Product Information Retrieval**: Automatically fetch product details from external APIs
- **Inventory Management**: Track and manage your home supplies
- **Web-Based Interface**: Access your inventory from any device with a modern web browser

## Technology Stack

- **.NET 9.0**: Backend API and hosting
- **Blazor WebAssembly**: Progressive web application frontend
- **Entity Framework Core**: Database access with SQLite
- **ASP.NET Core**: RESTful API services
- **.NET Aspire**: Distributed application orchestration

## Prerequisites

Before building and running the project, ensure you have the following installed:

- [.NET 9.0 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) or later
- [LibMan CLI](https://learn.microsoft.com/en-us/aspnet/core/client-side/libman/libman-cli) (for managing client-side libraries)
- [Entity Framework Core tools](https://learn.microsoft.com/en-us/ef/core/cli/dotnet) (for database migrations)

### Python Requirements for OCR (Bardcoded.Ocr)

- **Python version must be 3.12** for all environments.
- **Windows users:** The Ninja build system must **not** be on your PATH, or else pip will use Ninja for building wheels instead of the default backend (setuptools). This will cause build failures for some dependencies. If you encounter build errors, ensure Ninja is not present in your PATH. For instance, you might find it installed with Visual Studio.

### Installing LibMan CLI

To install the LibMan CLI tool globally, run:

```bash
dotnet tool install -g Microsoft.Web.LibraryManager.Cli
```

### Installing Entity Framework Core Tools

To install the EF Core CLI tools globally, run:

```bash
dotnet tool install -g dotnet-ef
```

Alternatively, the EF Core tools are included as a project reference and can be used via `dotnet ef` without global installation.

## Building the Project

Follow these steps to build the project:

### 1. Clone the Repository

```bash
git clone https://github.com/richardforrestbarker/Bardcode.git
cd Bardcode
```

### 2. Restore NuGet Packages

```bash
dotnet restore
```

### 3. Restore Client-Side Libraries with LibMan

The Blazor WebAssembly project uses LibMan to manage client-side libraries (jQuery, Bootstrap, Quagga.js, and Popper.js). Restore these libraries by running:

```bash
cd Bardcoded.Wasm
libman restore
cd ..
```

This will download the required client libraries specified in `Bardcoded.Wasm/libman.json` to the `wwwroot/lib` directory.

> **Note**: If you encounter version resolution errors with libman, you may need to update the library versions in `Bardcoded.Wasm/libman.json` to match what's currently available on the cdnjs provider.

### 4. Build the Solution

```bash
dotnet build
```

## Database Setup

The project uses Entity Framework Core with SQLite for data storage. Follow these steps to set up the database:

### Creating Migrations

To create a new migration, use the following command:

```bash
dotnet ef migrations add <MigrationName> --project Api/Api.csproj --startup-project Api/Api.csproj
```

Replace `<MigrationName>` with a descriptive name for your migration (e.g., "InitialCreate" or version number like "1.0.0.0").

### Applying Migrations

To apply migrations and update the SQLite database, run:

```bash
dotnet ef database update --project Api/Api.csproj --startup-project Api/Api.csproj
```

This command will create the database file and apply all pending migrations.

## Running the Application

The application uses .NET Aspire for orchestration, which manages multiple services including the API service, web frontend, and Redis cache.

### Using .NET Aspire (Recommended)

To run the entire application with all services:

```bash
dotnet run --project Bardcoded.AppHost/Bardcoded.AppHost.csproj
```

This will start:
- The API service (Api)
- The Blazor WebAssembly frontend (Bardcoded.Wasm)
- A Redis cache instance
- The Aspire dashboard for monitoring

### Running Individual Services

Alternatively, you can run services individually:

**API Service:**
```bash
dotnet run --project Api/Api.csproj
```

**Web Frontend:**
```bash
dotnet run --project Bardcoded.Wasm/Bardcoded.Wasm.csproj
```

## Running Tests

To run the test suite:

```bash
dotnet test
```

## Project Structure

- **Api**: RESTful API backend service
- **Bardcoded.Wasm**: Blazor WebAssembly frontend application
- **Data**: Shared data models and DTOs
- **Bardcoded.AppHost**: .NET Aspire orchestration host
- **Bardcoded.ServiceDefaults**: Shared service configuration
- **Bardcoded.Tests**: Unit and integration tests

## Configuration

Configuration files are located in:
- `Api/appsettings.json` - API service configuration
- `Bardcoded.Wasm/wwwroot/appsettings.json` - Frontend configuration

### Barcode Data Providers

Bardcode uses external APIs to retrieve product information from barcodes. The system supports multiple barcode data providers that are configured in `Api/appsettings.json` under the `Application.Integrations` section.

#### Supported Providers

**Setting the API Key (Security Best Practice):**

**IMPORTANT:** Never store API keys directly in configuration files as this is a security risk. Instead, set the API key using an environment variable or command-line argument.

**Option 1: Environment Variable (Recommended)**

Set the environment variable using the hierarchical configuration key format:

```bash
# Linux/macOS
export Application__Integrations__0__key="your-api-key-here"

# Windows (Command Prompt)
set Application__Integrations__0__key=your-api-key-here

# Windows (PowerShell)
$env:Application__Integrations__0__key="your-api-key-here"
```

**Note:** The index `0` corresponds to the first provider in the `Integrations` array. Adjust the index based on the provider's position in your configuration.

**Option 2: Command-Line Argument**

When running the application, pass the API key as a command-line argument:

```bash
dotnet run --project Api/Api.csproj --Application:Integrations:0:key="your-api-key-here"
```

**Option 3: User Secrets (Development Only)**

For local development, use the .NET user secrets feature:

```bash
cd Api
dotnet user-secrets set "Application:Integrations:0:key" "your-api-key-here"
```

The application includes three barcode data providers, each with different features, costs, and requirements:

##### 1. UPC Database

**Website:** https://upcdatabase.org/

**Features:**
- Supports UPC barcodes
- Requires API key authentication
- Provides product titles, descriptions, and images

**Account Setup:**
1. Create an account at https://upcdatabase.org/
2. Navigate to your account settings to generate an API key
3. Copy your API key

**Rate Limits & Pricing:**
- Free tier: 100 requests per day
- Paid plans available with higher limits
- See https://upcdatabase.org/api for current pricing and rate limits

**License:**
- Review the terms of service at https://upcdatabase.org/terms

**Configuration:**

The UPC Database provider is pre-configured in `Api/appsettings.json` with an empty `key` field:

```json
{
  "$type": "UpcDatabaseApiProvider",
  "url": "https://api.upcdatabase.org",
  "path": "product/{barcode}",
  "key": "",
  "allowedBarcodeTypes": [ "UPC" ]
}
```

##### 2. Open Food Facts

**Website:** https://world.openfoodfacts.org/

**Features:**
- Free and open database
- No API key required
- Supports EAN-13, EAN-8, UPC-A, UPC-E barcodes
- Primarily focused on food products
- Community-driven database

**Account Setup:**
- No account or API key required
- Optional: Create an account to contribute product data

**Rate Limits & Pricing:**
- Completely free
- Fair use policy: Please be respectful of the API and avoid excessive requests
- See https://world.openfoodfacts.org/data for API documentation

**License:**
- Open Database License (ODbL)
- Data is freely available
- Read more at https://world.openfoodfacts.org/terms-of-use

**Configuration:**

The default configuration in `Api/appsettings.json` works without modification:

```json
{
  "$type": "OpenFoodFactsApiProvider",
  "url": "https://world.openfoodfacts.org",
  "path": "api/v2/product/{barcode}.json",
  "key": "",
  "allowedBarcodeTypes": [ "EAN-13", "EAN-8", "UPC-A", "UPC-E" ]
}
```

No API key is needed for Open Food Facts.

##### 3. Barcode Lookup

**Website:** https://www.barcodelookup.com/

**Features:**
- Comprehensive barcode database
- Supports UPC-A, UPC-E, EAN-13, EAN-8, ISBN-10, ISBN-13
- Provides detailed product information including features, images, and metadata
- Commercial-grade API

**Account Setup:**
1. Create an account at https://www.barcodelookup.com/
2. Sign up for an API plan at https://www.barcodelookup.com/api
3. Copy your API key from your account dashboard

**Rate Limits & Pricing:**
- Free tier: Limited requests per month (check current limits)
- Paid plans: Various tiers with different rate limits
- See https://www.barcodelookup.com/api#plans for current pricing
- Rate limit documentation: https://www.barcodelookup.com/api#rate-limiting

**License:**
- Commercial API with terms of service
- Review the API License Agreement at https://www.barcodelookup.com/api#license
- End User License Agreement: https://www.barcodelookup.com/eula

**Configuration:**

The Barcode Lookup provider is pre-configured in `Api/appsettings.json` with an empty `key` field:

```json
{
  "$type": "BarcodeLookupApiProvider",
  "url": "https://api.barcodelookup.com",
  "path": "v3/products?barcode={barcode}&key=",
  "key": "",
  "allowedBarcodeTypes": [ "UPC-A", "UPC-E", "EAN-13", "EAN-8", "ISBN-10", "ISBN-13" ]
}
```

**Setting the API Key (Security Best Practice):**

**IMPORTANT:** Never store API keys directly in configuration files as this is a security risk. Instead, set the API key using an environment variable or command-line argument.

**Option 1: Environment Variable (Recommended)**

Set the environment variable using the hierarchical configuration key format:

```bash
# Linux/macOS
export Application__Integrations__2__key="your-api-key-here"

# Windows (Command Prompt)
set Application__Integrations__2__key=your-api-key-here

# Windows (PowerShell)
$env:Application__Integrations__2__key="your-api-key-here"
```

**Note:** The index `2` corresponds to the third provider in the `Integrations` array (Barcode Lookup is third by default). Adjust the index based on the provider's position in your configuration.

**Option 2: Command-Line Argument**

When running the application, pass the API key as a command-line argument:

```bash
dotnet run --project Api/Api.csproj --Application:Integrations:2:key="your-api-key-here"
```

**Option 3: User Secrets (Development Only)**

For local development, use the .NET user secrets feature:

```bash
cd Api
dotnet user-secrets set "Application:Integrations:2:key" "your-api-key-here"
```

#### Provider Priority

The system queries providers in the order they appear in the configuration file. Once a provider successfully returns product data, subsequent providers are not queried. You can reorder the providers in `appsettings.json` to change the priority.

#### Disabling Providers

To disable a provider, you can either:
- Remove it from the `Application.Integrations` array in `appsettings.json`
- Leave the `key` field empty (for providers that require authentication)

#### Example Complete Configuration

Here's an example of the complete configuration in `appsettings.json` with all three providers. **Note:** The `key` fields should remain empty in the configuration file.

```json
"Application": {
  "Integrations": [
    {
      "$type": "OpenFoodFactsApiProvider",
      "url": "https://world.openfoodfacts.org",
      "path": "api/v2/product/{barcode}.json",
      "key": "",
      "allowedBarcodeTypes": [ "EAN-13", "EAN-8", "UPC-A", "UPC-E" ]
    },
    {
      "$type": "UpcDatabaseApiProvider",
      "url": "https://api.upcdatabase.org",
      "path": "product/{barcode}",
      "key": "",
      "allowedBarcodeTypes": [ "UPC" ]
    },
    {
      "$type": "BarcodeLookupApiProvider",
      "url": "https://api.barcodelookup.com",
      "path": "v3/products?barcode={barcode}&key=",
      "key": "",
      "allowedBarcodeTypes": [ "UPC-A", "UPC-E", "EAN-13", "EAN-8", "ISBN-10", "ISBN-13" ]
    }
  ],
  "Features": {
    "FetchFromApis": true,
    "UseDatabase": true,
    "UseCache": false
  }
}
```

**Setting API Keys Securely:**

Set the API keys using environment variables instead of hardcoding them in the configuration:

```bash
# Set UPC Database API key (index 1)
export Application__Integrations__1__key="your-upcdatabase-key"

# Set Barcode Lookup API key (index 2)
export Application__Integrations__2__key="your-barcodelookup-key"
```

In this example, Open Food Facts is queried first (free and no authentication required), followed by UPC Database, and finally Barcode Lookup.

## Receipt OCR Feature

Bardcode includes an advanced Receipt OCR feature that uses machine learning to extract structured data from receipt images. This feature combines optical character recognition (OCR) with a layout-aware model (LayoutLMv3) to accurately identify and extract fields like vendor names, dates, amounts, and line items.

### OCR Architecture

The Receipt OCR system uses a hybrid architecture:

1. **API Service (C#/.NET)**: Handles file uploads, job management, and orchestrates the Python OCR pipeline
2. **Python OCR Service**: Performs actual image processing, OCR, and field extraction
3. **GPU Acceleration**: Supports CUDA-enabled GPUs for faster processing (falls back to CPU)

```
                    ┌─────────────────────────┐
                    │   Blazor Client         │
                    │  (Image Upload UI)      │
                    └───────────┬─────────────┘
                                │ HTTP POST
                                ▼
                    ┌─────────────────────────┐
                    │   .NET API Service      │
                    │  (ReceiptsController)   │
                    │  - File validation      │
                    │  - Job management       │
                    └───────────┬─────────────┘
                                │ Background Thread
                                │ (Process.Start)
                                ▼
                    ┌─────────────────────────┐
                    │   Python OCR CLI        │
                    │   (Bardcoded.Ocr)       │
                    │  - Image preprocessing  │
                    │  - PaddleOCR / Tesseract│
                    │  - LayoutLMv3 inference │
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

### Setting Up OCR

#### Prerequisites

1. **Python 3.10+**: Required for the OCR service
2. **GPU Support** (optional but recommended): NVIDIA GPU with CUDA support

#### Installing Python Dependencies

Navigate to the OCR project directory and install dependencies:

```bash
cd Bardcoded.Ocr
pip install -r requirements.txt
```

For GPU support on Linux:
```bash
pip install paddlepaddle-gpu
```

For CPU-only or macOS:
```bash
pip install paddlepaddle
```

#### Required System Dependencies

**Tesseract OCR** (fallback engine):

Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

macOS:
```bash
brew install tesseract
```

Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

**ImageMagick** (required for image preprocessing):

ImageMagick CLI is used for optimal image preprocessing to improve OCR accuracy.

Ubuntu/Debian:
```bash
sudo apt-get install imagemagick
```

macOS:
```bash
brew install imagemagick
```

Windows:
1. Download the ImageMagick installer from [ImageMagick Downloads](https://imagemagick.org/script/download.php#windows)
2. During installation, ensure you check the option "Install development headers and libraries for C and C++"
3. Add ImageMagick to your system PATH

Verify installation:
```bash
magick --version
```

See `Bardcoded.Ocr/README.md` for details on using the preprocessing scripts manually.

### Using the OCR Feature

#### Via API

**Upload a receipt:**
```bash
curl -X POST http://localhost:5000/api/receipts/upload \
  -F "files=@receipt.jpg" \
  -F "merchantId=store-123"
```

Response:
```json
{
  "jobId": "abc123-def456",
  "status": "processing",
  "statusUrl": "/api/receipts/status/abc123-def456",
  "resultUrl": "/api/receipts/result/abc123-def456"
}
```

**Check status:**
```bash
curl http://localhost:5000/api/receipts/status/abc123-def456
```

**Get results:**
```bash
curl http://localhost:5000/api/receipts/result/abc123-def456
```

#### Via Python CLI

Process a single receipt:
```bash
cd Bardcoded.Ocr
python cli.py process --image receipt.jpg
```

Process with output file:
```bash
python cli.py process --image receipt.jpg --output result.json
```

Process with options:
```bash
python cli.py process \
  --image receipt.jpg \
  --ocr-engine paddle \
  --device cuda \
  --denoise \
  --deskew \
  --model microsoft/layoutlmv3-base

  # or 

   python cli.py process --image tests/test-receipts/receipt-3.jpg --output tests/test-receipts/result.json --ocr-engine tesseract --model microsoft/layoutlmv3-base --debug --debug-output-dir ./tests/test-receipts/receipt-3/ --denoise --deskew



```

Process multi-page receipt:
```bash
python cli.py process \
  --image page1.jpg \
  --image page2.jpg \
  --output result.json
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

The OCR system returns structured JSON with the following schema:

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
    "confidence": 0.92,
    "box": { "x0": 400, "y0": 50, "x1": 550, "y1": 80 }
  },
  "total_amount": {
    "value": "25.99",
    "confidence": 0.96,
    "box": { "x0": 400, "y0": 500, "x1": 500, "y1": 530 }
  },
  "subtotal": {
    "value": "23.85",
    "confidence": 0.94,
    "box": { "x0": 400, "y0": 450, "x1": 500, "y1": 480 }
  },
  "tax_amount": {
    "value": "2.14",
    "confidence": 0.93,
    "box": { "x0": 400, "y0": 475, "x1": 500, "y1": 505 }
  },
  "currency": {
    "value": "USD",
    "confidence": 0.90,
    "box": null
  },
  "line_items": [
    {
      "description": "Product 1",
      "quantity": 1.0,
      "unit_price": 12.99,
      "line_total": 12.99,
      "confidence": 0.89,
      "box": { "x0": 50, "y0": 200, "x1": 550, "y1": 230 }
    }
  ]
}
```

### Bounding Box Format

All bounding boxes are normalized to a 0-1000 scale for consistency with LayoutLM models:

- `x0`: Left edge (0-1000)
- `y0`: Top edge (0-1000)
- `x1`: Right edge (0-1000)
- `y1`: Bottom edge (0-1000)

### Configuration

OCR settings are configured in `Api/appsettings.json`:

```json
{
  "Ocr": {
    "model_name_or_path": "microsoft/layoutlmv3-base",
    "device": "auto",
    "ocr_engine": "paddle",
    "detection_mode": "word",
    "box_normalization_scale": 1000,
    "python_service_path": "./Bardcoded.Ocr/cli.py",
    "temp_storage_path": "./temp/receipts",
    "max_file_size": 10485760,
    "temp_file_ttl_hours": 24,
    "enable_gpu": true,
    "min_confidence_threshold": 0.5
  }
}
```

### Troubleshooting OCR

**No OCR results returned:**
- Ensure Python dependencies are installed: `pip install -r Bardcoded.Ocr/requirements.txt`
- Check that PaddleOCR or Tesseract is working: `python -c "from paddleocr import PaddleOCR; print('OK')"`
- Verify image is readable and in supported format (JPEG, PNG)

**GPU not being used:**
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure paddlepaddle-gpu is installed instead of paddlepaddle
- Set `enable_gpu: true` in configuration

**Low accuracy:**
- Try enabling preprocessing: `--denoise --deskew`
- Ensure image is high resolution (300 DPI recommended)
- Use well-lit, non-blurry images

**Process timeout:**
- First run downloads models (~500MB), subsequent runs are faster
- Increase timeout in configuration if using CPU

### Document Processing Live View

The Document Processing Live View feature provides an interactive UI for processing document images with real-time preview of preprocessing effects. This allows users to optimize preprocessing settings before running OCR.

#### Features

- **Side-by-side image comparison**: View original and preprocessed images together
- **Live preview**: Settings changes trigger automatic preview updates after 5 seconds
- **Progress indicators**: Timer circle shows countdown; spinner shows API processing
- **Phased workflow**: Separate preprocessing and OCR phases with navigation
- **Adjustable settings**: Control deskew, denoise, contrast, and other preprocessing parameters

#### API Endpoints

The document processing API is isolated and can be found in `Api/Ocr/`:

**POST /api/document/preprocess**
Run preprocessing on an image without DPI resampling. Returns base64 encoded preprocessed image.

Request body:
```json
{
  "imageBase64": "base64-encoded-image-data",
  "filename": "receipt.jpg",
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

**POST /api/document/ocr**
Run OCR on a preprocessed image. Includes DPI resampling and safety checks.

Request body:
```json
{
  "imageBase64": "base64-encoded-preprocessed-image",
  "jobId": "optional-job-id",
  "ocrEngine": "paddle",
  "targetDpi": 300,
  "device": "auto"
}
```

**POST /api/document/inference**
Run model inference on OCR results to extract structured fields.

Request body:
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

**GET /api/document/status/{jobId}**
Get the status of a document processing job.

#### Using the Live View Component

Navigate to `/document-processing` in the web application to access the document processing live view.

1. **Select an image**: Upload a receipt or document image
2. **Adjust settings**: Modify preprocessing parameters (deskew, denoise, contrast, etc.)
3. **Wait for preview**: After 5 seconds of no changes, preprocessing runs automatically
4. **Accept preprocessing**: Click "Done - Continue to OCR" when satisfied with the preview
5. **Review OCR results**: View extracted text and fields
6. **Accept or retry**: Click "Accept Result" or "Go Back" to adjust settings

### Building the OCR Project

The OCR project is a Python project included in the Visual Studio solution:

```bash
# Install dependencies
cd Bardcoded.Ocr
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Run directly
python cli.py version
python cli.py process --help
```

### OCR Project Structure

```
Bardcoded.Ocr/
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
├── Bardcoded.Ocr.pyproj     # Visual Studio project file
├── Dockerfile               # Container configuration
├── config/
│   └── config.yaml          # Default configuration
├── docs/
│   └── fine_tuning.md       # Model training guide
└── src/
    ├── __init__.py
    ├── config.py            # Configuration management
    ├── receipt_processor.py # Main pipeline orchestrator
    ├── models/
    │   ├── base.py          # Base model interface
    │   └── layoutlmv3.py    # LayoutLMv3 implementation
    ├── ocr/
    │   └── ocr_engine.py    # OCR engine abstraction
    ├── preprocessing/
    │   └── image_preprocessor.py
    └── postprocessing/
        └── field_extractor.py
```

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]
