# Document Processor E2E Tests

End-to-end Playwright tests for the Document Processor Blazor WebAssembly frontend.

## Prerequisites

1. .NET 10 SDK
2. Playwright browsers

## Setup

### 1. Install Playwright Browsers

After building the project for the first time, install the Playwright browsers:

**Windows (PowerShell):**
```powershell
cd Tests.E2E
dotnet build
pwsh bin/Debug/net10.0/playwright.ps1 install
```

**Linux/macOS:**
```bash
cd Tests.E2E
dotnet build
./bin/Debug/net10.0/playwright.sh install
```

### 2. Prepare Test Data

The tests expect test receipt images in the `Ocr\tests\test-receipts` directory. Make sure you have at least one test image available:

- `PXL_20260226_190517040.jpg`
- `PXL_20260501_220901195.jpg`
- `PXL_20260525_203852138.jpg`

You can also add your own test images to `Tests.E2E\TestData\` directory.

## Running the Tests

### Run All Tests

```bash
dotnet test Tests.E2E\Tests.E2E.csproj
```

### Run Specific Test Categories

```bash
# Run only page load tests
dotnet test Tests.E2E\Tests.E2E.csproj --filter "FullyQualifiedName~PageLoad"

# Run only file upload tests
dotnet test Tests.E2E\Tests.E2E.csproj --filter "FullyQualifiedName~FileUpload"

# Run only preprocessing tests
dotnet test Tests.E2E\Tests.E2E.csproj --filter "FullyQualifiedName~Preprocessing"
```

### Run with Headed Browser (Visual Mode)

Set environment variable to see the browser:

**Windows:**
```powershell
$env:HEADED="1"
dotnet test Tests.E2E\Tests.E2E.csproj
```

**Linux/macOS:**
```bash
HEADED=1 dotnet test Tests.E2E/Tests.E2E.csproj
```

### Run with Slow Motion

To slow down test execution for debugging:

**Windows:**
```powershell
$env:PWDEBUG="1"
dotnet test Tests.E2E\Tests.E2E.csproj
```

**Linux/macOS:**
```bash
PWDEBUG=1 dotnet test Tests.E2E/Tests.E2E.csproj
```

## Test Structure

The test suite is organized into the following categories:

### 1. Page Load Tests
- Verify home page loads successfully
- Check DocumentProcessingView component renders correctly

### 2. File Upload Tests
- Upload valid image files
- Handle invalid file types
- Handle large files
- Display uploaded images

### 3. Preprocessing Tests
- Complete preprocessing with default settings
- Access and modify preprocessing settings
- Reject and redo preprocessing

### 4. OCR Tests
- Extract text from images
- Show progress indicators during OCR
- Handle OCR errors gracefully

### 5. Inference Tests
- Extract structured fields (vendor, date, total, line items)
- Complete full workflow (upload → preprocess → OCR → inference → accept)

### 6. UI Interaction Tests
- Magnifier lens on image hover
- Dismissible error messages
- Phase instruction changes

### 7. Responsive Design Tests
- Mobile viewport (375x667)
- Tablet viewport (768x1024)
- Desktop viewport (default)

### 8. Accessibility Tests
- Proper labels for form inputs
- Button accessibility
- Keyboard navigation

### 9. Performance Tests
- Page load time < 10 seconds
- Image upload and display < 5 seconds

## Test Configuration

### Application URL

By default, tests run against `http://localhost:5000`. To change the base URL, modify the `_baseUrl` field in `DocumentProcessorE2ETests.cs`:

```csharp
private string _baseUrl = "http://localhost:5000";
```

Or set an environment variable:

```bash
export PLAYWRIGHT_BASE_URL=http://localhost:8080
```

### Timeouts

Default timeouts are set in the `Setup()` method:
- General timeout: 60 seconds
- Navigation timeout: 30 seconds

Adjust these values if tests are failing due to slow operations.

## Running with the Application

### Option 1: Manual Start

1. Start the Document Processor application:
   ```bash
   cd Example
   dotnet run
   ```

2. In another terminal, run the tests:
   ```bash
   dotnet test Tests.E2E\Tests.E2E.csproj
   ```

### Option 2: Automated (TODO)

The `DocumentProcessorWebApplicationFactory` can be extended to automatically start the application server before tests run.

## Debugging Tests

### Take Screenshots on Failure

Playwright automatically captures screenshots on test failures in the `bin/Debug/net10.0/playwright-results` directory.

### Video Recording

Enable video recording by setting:

```csharp
[Parallelizable(ParallelScope.Self)]
[TestFixture]
public class DocumentProcessorE2ETests : PageTest
{
    public override BrowserNewContextOptions ContextOptions()
    {
        return new BrowserNewContextOptions
        {
            RecordVideoDir = "videos/",
            RecordVideoSize = new() { Width = 1280, Height = 720 }
        };
    }
}
```

### Trace Viewer

Generate traces for detailed debugging:

```bash
dotnet test Tests.E2E\Tests.E2E.csproj -- Playwright.BrowserType.LaunchOptions.Trace=on
```

View traces with:

```bash
pwsh bin/Debug/net10.0/playwright.ps1 show-trace trace.zip
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup .NET
        uses: actions/setup-dotnet@v3
        with:
          dotnet-version: '10.0.x'

      - name: Install dependencies
        run: dotnet restore

      - name: Build
        run: dotnet build --no-restore

      - name: Install Playwright browsers
        run: pwsh Tests.E2E/bin/Debug/net10.0/playwright.ps1 install --with-deps

      - name: Start application
        run: |
          cd Example
          dotnet run &
          sleep 10

      - name: Run E2E tests
        run: dotnet test Tests.E2E/Tests.E2E.csproj --no-build --verbosity normal

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-results
          path: Tests.E2E/bin/Debug/net10.0/playwright-results/
```

## Known Issues

1. **Backend-dependent tests**: Some tests are marked with `[Ignore]` because they require the full backend (Python OCR CLI) to be running. To enable these:
   - Ensure the Python OCR CLI is installed and working
   - Start the API backend
   - Remove the `[Ignore]` attribute from the tests

2. **PaddleOCR on Windows**: If PaddleOCR fails, tests will fall back to Tesseract. See `Ocr\README.md` for setup instructions.

3. **Slow CI/CD**: E2E tests can be slow in CI/CD environments. Consider running them only on the main branch or before releases.

## Adding New Tests

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use helper methods for common operations (e.g., `NavigateToHomePage()`, `UploadTestImage()`)
3. Add appropriate waits for async operations
4. Include accessibility and responsive design checks where relevant
5. Mark backend-dependent tests with `[Ignore]` if they can't run standalone

## Troubleshooting

### "Browser not found" Error

Run the Playwright install command:
```bash
pwsh bin/Debug/net10.0/playwright.ps1 install
```

### "Connection refused" Error

Ensure the application is running at the configured base URL.

### "Element not found" Timeout

Increase the timeout in the test or check if the selector has changed:
```csharp
await Page.WaitForSelectorAsync("selector", new() { Timeout = 60000 });
```

### Tests Passing Locally but Failing in CI

- Check browser versions (CI might use different versions)
- Add longer timeouts for CI environments
- Verify test data is available in CI
- Check environment-specific issues (Windows vs Linux)

## Resources

- [Playwright for .NET Documentation](https://playwright.dev/dotnet/)
- [NUnit Documentation](https://docs.nunit.org/)
- [Blazor Testing Best Practices](https://learn.microsoft.com/en-us/aspnet/core/blazor/test)
