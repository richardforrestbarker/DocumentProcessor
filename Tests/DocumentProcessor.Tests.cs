using Xunit;
using DocumentProcessor.Data.Messages;
using DocumentProcessor.Data.Ocr.Messages;
using DocumentProcessor.Wasm;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Components.Forms;
using DocumentProcessor.Data.Ocr;
using System.IO;
using System.Threading;
using Clients;
using System.Net.Http;
using DocumentProcessor.Wasm.Components;
using DocumentProcessor.Clients;

namespace DocumentProcessor.Tests
{
    public class PreprocessingRequestTests
    {
        [Fact]
        public void PreprocessingRequest_Properties_AreSetCorrectly()
        {
            var req = new PreprocessingRequest
            {
                ImageBase64 = "imgbase64",
                JobId = "job123",
                Denoise = true,
                Deskew = false,
                FuzzPercent = 25,
                DeskewThreshold = 15,
                ContrastType = "linear",
                ContrastStrength = 2.0,
                ContrastMidpoint = 80
            };
            Assert.Equal("imgbase64", req.ImageBase64);
            Assert.Equal("job123", req.JobId);
            Assert.True(req.Denoise);
            Assert.False(req.Deskew);
            Assert.Equal(25, req.FuzzPercent);
            Assert.Equal(15, req.DeskewThreshold);
            Assert.Equal("linear", req.ContrastType);
            Assert.Equal(2.0, req.ContrastStrength);
            Assert.Equal(80, req.ContrastMidpoint);
        }
    }

    public class OcrWordTests
    {
        [Fact]
        public void OcrWord_Properties_AreSetCorrectly()
        {
            var box = new BoundingBox { X0 = 1, Y0 = 2, X1 = 3, Y1 = 4 };
            var word = new OcrWord { Text = "hello", Box = box, Confidence = 0.99 };
            Assert.Equal("hello", word.Text);
            Assert.Equal(box, word.Box);
            Assert.Equal(0.99, word.Confidence);
            Assert.Equal(1, word.Box.X0);
            Assert.Equal(2, word.Box.Y0);
            Assert.Equal(3, word.Box.X1);
            Assert.Equal(4, word.Box.Y1);
        }
    }

    public class OcrResultTests
    {
        [Fact]
        public void OcrResult_Properties_AreSetCorrectly()
        {
            var words = new List<OcrWord>
            {
                new OcrWord { Text = "one", Box = new BoundingBox { X0 = 0, Y0 = 0, X1 = 10, Y1 = 10 }, Confidence = 1.0 },
                new OcrWord { Text = "two", Box = new BoundingBox { X0 = 10, Y0 = 10, X1 = 20, Y1 = 20 }, Confidence = 0.8 }
            };
            var result = new OcrResult
            {
                JobId = "job456",
                Status = "done",
                Words = words,
                RawOcrText = "one two",
                ImageWidth = 100,
                ImageHeight = 200,
                Error = null
            };
            Assert.Equal("job456", result.JobId);
            Assert.Equal("done", result.Status);
            Assert.Equal(words, result.Words);
            Assert.Equal("one two", result.RawOcrText);
            Assert.Equal(100, result.ImageWidth);
            Assert.Equal(200, result.ImageHeight);
            Assert.Null(result.Error);
        }
    }

    public class InferenceRequestTests
    {
        [Fact]
        public void InferenceRequest_Properties_AreSetCorrectly()
        {
            var ocrResult = new OcrResult
            {
                JobId = "job789",
                Status = "done",
                Words = new List<OcrWord>(),
                RawOcrText = "text",
                ImageWidth = 50,
                ImageHeight = 60
            };
            var req = new InferenceRequest
            {
                OcrResult = ocrResult,
                ImageBase64 = "img",
                JobId = "job789",
                Model = "model",
                Device = "cpu"
            };
            Assert.Equal(ocrResult, req.OcrResult);
            Assert.Equal("img", req.ImageBase64);
            Assert.Equal("job789", req.JobId);
            Assert.Equal("model", req.Model);
            Assert.Equal("cpu", req.Device);
        }
    }

    public class BoundingBoxTests
    {
        [Fact]
        public void BoundingBox_Properties_AreSetCorrectly()
        {
            var box = new BoundingBox { X0 = 5, Y0 = 6, X1 = 7, Y1 = 8 };
            Assert.Equal(5, box.X0);
            Assert.Equal(6, box.Y0);
            Assert.Equal(7, box.X1);
            Assert.Equal(8, box.Y1);
        }
    }

    public class DocumentProcessingViewLogicTests
    {
        private class TestLogger : ILogger<DocumentProcessingView>
        {
            public List<string> Logs = new();
            public IDisposable BeginScope<TState>(TState state) => null;
            public bool IsEnabled(LogLevel level) => true;
            public void Log<TState>(LogLevel level, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
            {
                Logs.Add(formatter(state, exception));
            }
        }

        private class TestDocumentProcessor : IDocumentProcessor
        {
            public PreprocessingRequest LastPreprocessingRequest;
            public OcrRequest LastOcrRequest;
            public InferenceRequest LastInferenceRequest;
            public Task<PreprocessingResult> PreprocessImageAsync(PreprocessingRequest request)
            {
                LastPreprocessingRequest = request;
                return Task.FromResult(new PreprocessingResult { JobId = request.JobId, Status = "done", ImageBase64 = "preprocessed" });
            }
            public Task<OcrResult> RunOcrAsync(OcrRequest request)
            {
                LastOcrRequest = request;
                return Task.FromResult(new OcrResult { JobId = request.JobId, Status = "done", Words = new List<OcrWord>(), RawOcrText = "text", ImageWidth = 1, ImageHeight = 1 });
            }
            public Task<InferenceResult> RunInferenceAsync(InferenceRequest request)
            {
                LastInferenceRequest = request;
                return Task.FromResult(new InferenceResult { JobId = request.JobId, Status = "done" });
            }
            public Task<JobStatus?> GetJobStatusAsync(string jobId) => Task.FromResult<JobStatus?>(null);
        }

        [Fact]
        public void OnInitialized_WithInitialImage_TriggersPreprocessing()
        {
            var logger = new TestLogger();
            var processor = new TestDocumentProcessor();
            var view = new DocumentProcessingView();
            typeof(DocumentProcessingView).GetProperty("Logger", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).SetValue(view, logger);
            typeof(DocumentProcessingView).GetProperty("DocumentProcessingClient", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).SetValue(view, processor);
            typeof(DocumentProcessingView).GetProperty("InitialImageBase64").SetValue(view, "imgbase64");
            var onInit = typeof(DocumentProcessingView).GetMethod("OnInitialized", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            onInit.Invoke(view, null);
            Assert.Equal("imgbase64", typeof(DocumentProcessingView).GetField("sourceImageBase64", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(view));
        }

        [Fact]
        public async Task HandleFileSelection_RejectsLargeFile()
        {
            var logger = new TestLogger();
            var processor = new TestDocumentProcessor();
            var view = new DocumentProcessingView();
            typeof(DocumentProcessingView).GetProperty("Logger", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).SetValue(view, logger);
            typeof(DocumentProcessingView).GetProperty("DocumentProcessingClient", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).SetValue(view, processor);
            var file = new TestFile(11 * 1024 * 1024, "image/png");
            var args = new InputFileChangeEventArgs(new TestFileList(file));
            var method = typeof(DocumentProcessingView).GetMethod("HandleFileSelection", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            var task = (Task)method.Invoke(view, new object[] { args });
            await task;
            var errorMessage = typeof(DocumentProcessingView).GetField("errorMessage", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(view) as string;
            Assert.Contains("File size exceeds", errorMessage);
        }

        [Fact]
        public async Task HandleFileSelection_RejectsNonImageFile()
        {
            var logger = new TestLogger();
            var processor = new TestDocumentProcessor();
            var view = new DocumentProcessingView();
            typeof(DocumentProcessingView).GetProperty("Logger", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).SetValue(view, logger);
            typeof(DocumentProcessingView).GetProperty("DocumentProcessingClient", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).SetValue(view, processor);
            var file = new TestFile(1024, "application/pdf");
            var args = new InputFileChangeEventArgs(new TestFileList(file));
            var method = typeof(DocumentProcessingView).GetMethod("HandleFileSelection", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            var task = (Task)method.Invoke(view, new object[] { args });
            await task;
            var errorMessage = typeof(DocumentProcessingView).GetField("errorMessage", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(view) as string;
            Assert.Contains("Please select an image file", errorMessage);
        }

        // Test fakes for InputFile
        private class TestFile : IBrowserFile
        {
            public TestFile(long size, string contentType) { Size = size; ContentType = contentType; }
            public string Name => "test.png";
            public DateTimeOffset LastModified => DateTimeOffset.Now;
            public long Size { get; }
            public string ContentType { get; }
            public Stream OpenReadStream(long maxAllowedSize = 512000, CancellationToken cancellationToken = default) => new MemoryStream(new byte[Size]);
        }
        private class TestFileList : IReadOnlyList<IBrowserFile>
        {
            private readonly IBrowserFile _file;
            public TestFileList(IBrowserFile file) { _file = file; }
            public IBrowserFile this[int index] => _file;
            public int Count => 1;
            public IEnumerator<IBrowserFile> GetEnumerator() { yield return _file; }
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
        }
    }

    public class WasmExtensionsTests
    {
        [Fact]
        public void AddDocumentProcessorWasmAsync_ConfiguresHttpClient()
        {
            // This test would require a Blazor WebAssemblyHostBuilder test fake.
            // For now, we can check that the extension method exists and is callable.
            var method = typeof(Extensions).GetMethod("AddDocumentProcessorWasmAsync");
            Assert.NotNull(method);
        }
    }

    public class ClientSideDocumentProcessorTests
    {
        [Fact]
        public async Task PreprocessImageAsync_ReturnsErrorOnException()
        {
            var processor = new ClientSideDocumentProcessor(new FailingHttpClientFactory(), new FailingLoggerFactory());
            var result = await processor.PreprocessImageAsync(new PreprocessingRequest { ImageBase64 = "", JobId = "job" });
            Assert.Equal("Error", result.Status);
            Assert.NotNull(result.Error);
        }

        private class FailingHttpClientFactory : IHttpClientFactory
        {
            public HttpClient CreateClient(string name) => throw new Exception("fail");
        }
        private class FailingLoggerFactory : ILoggerFactory
        {
            public void AddProvider(ILoggerProvider provider) { }
            public ILogger CreateLogger(string categoryName) => new FailingLogger();
            public void Dispose() { }
        }
        private class FailingLogger : ILogger
        {
            public IDisposable BeginScope<TState>(TState state) => null;
            public bool IsEnabled(LogLevel level) => true;
            public void Log<TState>(LogLevel level, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter) { }
        }
    }

    public class OcrConfigurationTests
    {
        [Fact]
        public void OcrConfiguration_DefaultValues_AreSetCorrectly()
        {
            var config = new DocumentProcessor.Data.OcrConfiguration
            {
                PythonServicePath = "./Ocr/cli.py",
                PythonWorkingDirectory = "./Ocr",
                TempStoragePath = "./temp/documents",
                // MaxFileSize is required; the class sets a default of 10MB
                MaxFileSize = 10 * 1024 * 1024,
                PythonVenvPath = "./Ocr/venv"
            };
            Assert.Equal("", config.ModelNameOrPath);
            Assert.Equal("auto", config.Device);
            Assert.Equal("paddle", config.OcrEngine);
            Assert.Equal("word", config.DetectionMode);
            Assert.Equal(1000, config.BoxNormalizationScale);
            Assert.Equal("./Ocr/cli.py", config.PythonServicePath);
            Assert.Equal("./Ocr", config.PythonWorkingDirectory);
            Assert.Equal("./Ocr/venv", config.PythonVenvPath);
            Assert.Equal("./temp/documents", config.TempStoragePath);
            Assert.Equal(10 * 1024 * 1024, config.MaxFileSize);
            Assert.Equal(24, config.TempFileTtlHours);
            Assert.True(config.EnableGpu);
            Assert.Equal(0.5, config.MinConfidenceThreshold);
        }

        [Fact]
        public void OcrConfiguration_PythonVenvPath_CanBeSet()
        {
            var config = new DocumentProcessor.Data.OcrConfiguration
            {
                PythonServicePath = "./Ocr/cli.py",
                PythonWorkingDirectory = "./Ocr",
                TempStoragePath = "./temp/documents",
                MaxFileSize = 10 * 1024 * 1024,
                PythonVenvPath = "/custom/path/to/venv"
            };
            Assert.Equal("/custom/path/to/venv", config.PythonVenvPath);
        }
    }
}
