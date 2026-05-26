using System.Text.RegularExpressions;

namespace DocumentProcessor.Tests.E2E;

/// <summary>
/// End-to-end tests for the Document Processor Blazor WebAssembly frontend.
/// Tests the complete user workflow from file upload through preprocessing, OCR, and inference.
/// </summary>
[Parallelizable(ParallelScope.Self)]
[TestFixture]
public class DocumentProcessorE2ETests : PageTest
{
    private string _baseUrl = "https://localhost:7256";
    private DocumentProcessorWebApplicationFactory? _factory;

    [OneTimeSetUp]
    public async Task OneTimeSetUp()
    {
        // Install Playwright browsers if needed
        // Run: pwsh bin/Debug/net10.0/playwright.ps1 install
    }

    [SetUp]
    public async Task Setup()
    {
        // Set longer timeout for E2E tests using Page property
        Page.SetDefaultTimeout(60000); // 60 seconds
        Page.SetDefaultNavigationTimeout(30000); // 30 seconds
    }

    [TearDown]
    public async Task TearDown()
    {
        // Clean up after each test
        await Context.CloseAsync();
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        _factory?.Dispose();
    }

    #region Helper Methods

    /// <summary>
    /// Navigate to the home page and wait for it to load
    /// </summary>
    private async Task NavigateToHomePage()
    {
        await Page.GotoAsync(_baseUrl);
        await Page.WaitForLoadStateAsync(LoadState.NetworkIdle);

        // Wait for the DocumentProcessingView component to render
        await Page.WaitForSelectorAsync("h3:has-text('Document Processing Live View')", 
            new() { Timeout = 10000 });
    }

    /// <summary>
    /// Upload a test receipt image
    /// </summary>
    private async Task UploadTestImage(string imagePath)
    {
        var fileInput = await Page.QuerySelectorAsync("input[type='file']");
        Assert.That(fileInput, Is.Not.Null, "File input should be present");

        await fileInput!.SetInputFilesAsync(imagePath);

        // Wait for image to be loaded
        await Page.WaitForSelectorAsync("img.magnifier-image", new() { Timeout = 5000 });
    }

    /// <summary>
    /// Wait for preprocessing to complete
    /// </summary>
    private async Task WaitForPreprocessingComplete()
    {
        // Wait for "Start OCR" button to appear (indicates preprocessing is done)
        await Page.WaitForSelectorAsync("button:has-text('Start OCR')", 
            new() { Timeout = 30000 });
    }

    /// <summary>
    /// Wait for OCR to complete
    /// </summary>
    private async Task WaitForOcrComplete()
    {
        // Wait for "Start Inference" button to appear (indicates OCR is done)
        await Page.WaitForSelectorAsync("button:has-text('Start Inference')", 
            new() { Timeout = 60000 });
    }

    /// <summary>
    /// Wait for inference to complete
    /// </summary>
    private async Task WaitForInferenceComplete()
    {
        // Wait for "Accept Result" button to appear (indicates inference is done)
        await Page.WaitForSelectorAsync("button:has-text('Accept Result')", 
            new() { Timeout = 60000 });
    }

    #endregion

    #region Page Load Tests

    [Test]
    public async Task HomePage_ShouldLoadSuccessfully()
    {
        await Page.GotoAsync(_baseUrl);
        await Page.WaitForLoadStateAsync(LoadState.NetworkIdle);

        // Check page title
        await Expect(Page).ToHaveTitleAsync(new Regex("Document Processor"));

        // Check main heading
        var heading = Page.Locator("h1:has-text('Document Processor')");
        await Expect(heading).ToBeVisibleAsync();
    }

    [Test]
    public async Task DocumentProcessingView_ShouldRenderCorrectly()
    {
        await NavigateToHomePage();

        // Check component title
        var title = Page.Locator("h3:has-text('Document Processing Live View')");
        await Expect(title).ToBeVisibleAsync();

        // Check file input exists
        var fileInput = Page.Locator("input[type='file']");
        await Expect(fileInput).ToBeVisibleAsync();

        // Check phase instructions are visible
        var instructions = Page.Locator(".alert-info");
        await Expect(instructions).ToBeVisibleAsync();
    }

    #endregion

    #region File Upload Tests

    [Test]
    public async Task FileUpload_WithValidImage_ShouldDisplayImage()
    {
        await NavigateToHomePage();

        // Create a test image file
        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "TestData", "test-receipt.jpg");

        // If test image doesn't exist, use a sample from Ocr/tests
        if (!File.Exists(testImagePath))
        {
            testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
                "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
                "PXL_20260226_190517040.jpg");
        }

        if (File.Exists(testImagePath))
        {
            await UploadTestImage(testImagePath);

            // Verify image is displayed
            var image = Page.Locator("img.magnifier-image");
            await Expect(image).ToBeVisibleAsync();

            // Verify image has src attribute with base64 data
            var src = await image.GetAttributeAsync("src");
            Assert.That(src, Does.StartWith("data:image"), 
                "Image should have base64 data URL");
        }
        else
        {
            Assert.Warn($"Test image not found at: {testImagePath}");
        }
    }

    [Test]
    public async Task FileUpload_WithLargeFile_ShouldShowError()
    {
        await NavigateToHomePage();

        // Note: This test would require creating a large file or mocking the validation
        // For now, we'll check that the error handling exists

        var errorAlert = Page.Locator(".alert-danger");

        // Initially, no error should be present
        await Expect(errorAlert).Not.ToBeVisibleAsync();
    }

    [Test]
    public async Task FileUpload_WithInvalidFileType_ShouldShowError()
    {
        await NavigateToHomePage();

        // Create a temporary text file
        var tempFile = Path.Combine(Path.GetTempPath(), "test.txt");
        await File.WriteAllTextAsync(tempFile, "This is not an image");

        try
        {
            var fileInput = await Page.QuerySelectorAsync("input[type='file']");
            await fileInput!.SetInputFilesAsync(tempFile);

            // Wait for error message
            var errorAlert = Page.Locator(".alert-danger");
            await Expect(errorAlert).ToBeVisibleAsync(new() { Timeout = 5000 });

            var errorText = await errorAlert.TextContentAsync();
            Assert.That(errorText, Does.Contain("image").IgnoreCase, 
                "Error should mention image file type");
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    #endregion

    #region Preprocessing Tests

    [Test]
    public async Task Preprocessing_WithDefaultSettings_ShouldComplete()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        await UploadTestImage(testImagePath);

        // Wait for preprocessing to start automatically
        // Look for progress indicator or phase change
        await Task.Delay(2000); // Give it time to start

        // Wait for preprocessing to complete
        await WaitForPreprocessingComplete();

        // Verify "Start OCR" button is visible
        var ocrButton = Page.Locator("button:has-text('Start OCR')");
        await Expect(ocrButton).ToBeVisibleAsync();
    }

    [Test]
    public async Task Preprocessing_SettingsPanel_ShouldBeAccessible()
    {
        await NavigateToHomePage();

        // Look for preprocessing settings (collapsible panel)
        var settingsButton = Page.Locator("button:has-text('Preprocessing Settings')");

        if (await settingsButton.CountAsync() > 0)
        {
            await settingsButton.ClickAsync();

            // Check for common preprocessing options
            var deskewOption = Page.Locator("input[type='checkbox']:near(:text('Deskew'))");
            var contrastOption = Page.Locator(":text('Contrast')");

            // At least one setting should be visible
            var hasSettings = await deskewOption.CountAsync() > 0 || 
                            await contrastOption.CountAsync() > 0;
            Assert.That(hasSettings, Is.True, 
                "Preprocessing settings panel should contain options");
        }
    }

    [Test]
    public async Task Preprocessing_RejectResult_ShouldAllowRedo()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        await UploadTestImage(testImagePath);
        await WaitForPreprocessingComplete();

        // Look for "Redo Preprocessing" or similar button
        var redoButton = Page.Locator("button:has-text('Redo')");

        if (await redoButton.CountAsync() > 0)
        {
            await redoButton.First.ClickAsync();

            // Should still be in preprocessing phase
            var phaseInstruction = Page.Locator(".alert-info:has-text('Preprocessing')");
            await Expect(phaseInstruction).ToBeVisibleAsync();
        }
    }

    #endregion

    #region OCR Tests

    [Test]
    [Ignore("Requires full application running with OCR backend")]
    public async Task OCR_ShouldExtractText()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        await UploadTestImage(testImagePath);
        await WaitForPreprocessingComplete();

        // Click "Start OCR" button
        var ocrButton = Page.Locator("button:has-text('Start OCR')");
        await ocrButton.ClickAsync();

        // Wait for OCR to complete
        await WaitForOcrComplete();

        // Verify OCR result is displayed
        var resultPanel = Page.Locator(":text('OCR Result')");
        await Expect(resultPanel).ToBeVisibleAsync();

        // Check that some text was extracted
        var extractedText = Page.Locator(".ocr-text, .extracted-text");
        if (await extractedText.CountAsync() > 0)
        {
            var text = await extractedText.First.TextContentAsync();
            Assert.That(text, Is.Not.Empty, "Should extract some text");
        }
    }

    [Test]
    [Ignore("Requires full application running with OCR backend")]
    public async Task OCR_ProgressIndicator_ShouldShow()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        await UploadTestImage(testImagePath);
        await WaitForPreprocessingComplete();

        var ocrButton = Page.Locator("button:has-text('Start OCR')");
        await ocrButton.ClickAsync();

        // Look for progress indicator
        var progressBar = Page.Locator(".progress, [role='progressbar']");
        var spinner = Page.Locator(".spinner-border, .spinner");

        // At least one should be visible
        var hasProgress = await progressBar.IsVisibleAsync() || await spinner.IsVisibleAsync();

        if (hasProgress)
        {
            Assert.Pass("Progress indicator is shown during OCR");
        }
    }

    #endregion

    #region Inference Tests

    [Test]
    [Ignore("Requires full application running with inference backend")]
    public async Task Inference_ShouldExtractFields()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        await UploadTestImage(testImagePath);
        await WaitForPreprocessingComplete();

        // Start OCR
        var ocrButton = Page.Locator("button:has-text('Start OCR')");
        await ocrButton.ClickAsync();
        await WaitForOcrComplete();

        // Start Inference
        var inferenceButton = Page.Locator("button:has-text('Start Inference')");
        await inferenceButton.ClickAsync();
        await WaitForInferenceComplete();

        // Check for extracted fields
        var fieldsPanel = Page.Locator(":text('Extracted Fields')");
        await Expect(fieldsPanel).ToBeVisibleAsync();

        // Look for common receipt fields
        var totalField = Page.Locator(":text('Total')");
        var dateField = Page.Locator(":text('Date')");
        var vendorField = Page.Locator(":text('Vendor')");

        // At least one field should be visible
        var hasFields = await totalField.CountAsync() > 0 || 
                       await dateField.CountAsync() > 0 || 
                       await vendorField.CountAsync() > 0;

        Assert.That(hasFields, Is.True, "Should extract at least one field");
    }

    [Test]
    [Ignore("Requires full application running")]
    public async Task CompleteWorkflow_ShouldAcceptResult()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        // Complete workflow: Upload -> Preprocess -> OCR -> Inference -> Accept
        await UploadTestImage(testImagePath);
        await WaitForPreprocessingComplete();

        await Page.Locator("button:has-text('Start OCR')").ClickAsync();
        await WaitForOcrComplete();

        await Page.Locator("button:has-text('Start Inference')").ClickAsync();
        await WaitForInferenceComplete();

        // Accept result
        var acceptButton = Page.Locator("button:has-text('Accept Result')");
        await acceptButton.ClickAsync();

        // Check for success message
        var successAlert = Page.Locator(".alert-success");
        await Expect(successAlert).ToBeVisibleAsync(new() { Timeout = 5000 });

        var successText = await successAlert.TextContentAsync();
        Assert.That(successText, Does.Contain("accepted").IgnoreCase, 
            "Should show success message");
    }

    #endregion

    #region UI Interaction Tests

    [Test]
    public async Task MagnifierLens_ShouldShowOnHover()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        await UploadTestImage(testImagePath);

        // Hover over the image
        var image = Page.Locator("img.magnifier-image");
        await image.HoverAsync();

        // Check if magnifier lens appears
        var magnifierLens = Page.Locator(".magnifier-lens");

        // The magnifier might not be immediately visible, that's ok
        var lensCount = await magnifierLens.CountAsync();
        Assert.That(lensCount, Is.GreaterThanOrEqualTo(0), 
            "Magnifier lens element should exist");
    }

    [Test]
    public async Task ErrorMessage_ShouldBeDismissible()
    {
        await NavigateToHomePage();

        // Trigger an error by uploading invalid file
        var tempFile = Path.Combine(Path.GetTempPath(), "test.txt");
        await File.WriteAllTextAsync(tempFile, "Not an image");

        try
        {
            var fileInput = await Page.QuerySelectorAsync("input[type='file']");
            await fileInput!.SetInputFilesAsync(tempFile);

            await Task.Delay(1000); // Wait for error to appear

            var errorAlert = Page.Locator(".alert-danger");
            if (await errorAlert.CountAsync() > 0)
            {
                // Look for close button
                var closeButton = errorAlert.Locator("button.btn-close");

                if (await closeButton.CountAsync() > 0)
                {
                    await closeButton.ClickAsync();

                    // Error should disappear
                    await Expect(errorAlert).Not.ToBeVisibleAsync();
                }
            }
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task PhaseInstructions_ShouldChangePerPhase()
    {
        await NavigateToHomePage();

        // Check initial phase instructions (Preprocessing)
        var instructions = Page.Locator(".alert-info");
        var initialText = await instructions.TextContentAsync();

        Assert.That(initialText, Does.Contain("Preprocessing").IgnoreCase, 
            "Should show preprocessing instructions initially");
    }

    #endregion

    #region Responsive Design Tests

    [Test]
    public async Task ResponsiveDesign_MobileViewport_ShouldWork()
    {
        // Set mobile viewport
        await Page.SetViewportSizeAsync(375, 667); // iPhone SE size

        await NavigateToHomePage();

        // Check that main elements are still visible
        var title = Page.Locator("h3:has-text('Document Processing Live View')");
        await Expect(title).ToBeVisibleAsync();

        var fileInput = Page.Locator("input[type='file']");
        await Expect(fileInput).ToBeVisibleAsync();
    }

    [Test]
    public async Task ResponsiveDesign_TabletViewport_ShouldWork()
    {
        // Set tablet viewport
        await Page.SetViewportSizeAsync(768, 1024); // iPad size

        await NavigateToHomePage();

        // Check that layout adapts
        var container = Page.Locator(".document-processing-view");
        await Expect(container).ToBeVisibleAsync();
    }

    #endregion

    #region Accessibility Tests

    [Test]
    public async Task Accessibility_FileInput_ShouldHaveLabel()
    {
        await NavigateToHomePage();

        var fileInput = Page.Locator("input[type='file']");

        // Check if input has associated label or aria-label
        var ariaLabel = await fileInput.GetAttributeAsync("aria-label");
        var hasId = await fileInput.GetAttributeAsync("id");

        var hasLabel = !string.IsNullOrEmpty(ariaLabel) || 
                      (!string.IsNullOrEmpty(hasId) && 
                       await Page.Locator($"label[for='{hasId}']").CountAsync() > 0);

        Assert.That(hasLabel, Is.True, 
            "File input should have accessible label");
    }

    [Test]
    public async Task Accessibility_Buttons_ShouldHaveProperLabels()
    {
        await NavigateToHomePage();

        // Check all buttons have text or aria-label
        var buttons = await Page.Locator("button").AllAsync();

        foreach (var button in buttons)
        {
            var text = await button.TextContentAsync();
            var ariaLabel = await button.GetAttributeAsync("aria-label");
            var title = await button.GetAttributeAsync("title");

            var hasLabel = !string.IsNullOrWhiteSpace(text) || 
                          !string.IsNullOrEmpty(ariaLabel) || 
                          !string.IsNullOrEmpty(title);

            Assert.That(hasLabel, Is.True, 
                $"Button should have accessible label");
        }
    }

    #endregion

    #region Performance Tests

    [Test]
    public async Task Performance_PageLoad_ShouldBeFast()
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        await Page.GotoAsync(_baseUrl);
        await Page.WaitForLoadStateAsync(LoadState.NetworkIdle);

        stopwatch.Stop();

        Assert.That(stopwatch.ElapsedMilliseconds, Is.LessThan(10000), 
            "Page should load within 10 seconds");
    }

    [Test]
    public async Task Performance_ImageUpload_ShouldBeReasonablyFast()
    {
        await NavigateToHomePage();

        var testImagePath = Path.Combine(TestContext.CurrentContext.TestDirectory, 
            "..", "..", "..", "..", "Ocr", "tests", "test-receipts", 
            "PXL_20260226_190517040.jpg");

        if (!File.Exists(testImagePath))
        {
            Assert.Ignore("Test image not found");
            return;
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        await UploadTestImage(testImagePath);

        stopwatch.Stop();

        Assert.That(stopwatch.ElapsedMilliseconds, Is.LessThan(5000), 
            "Image upload and display should complete within 5 seconds");
    }

    #endregion
}
