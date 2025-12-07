using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;
using DocumentProcessor.Data.Ocr.Messages;
using DocumentProcessor.Data.Messages;
using DocumentProcessor.Data.Ocr;
using DocumentProcessor.Data;
using Microsoft.Extensions.Logging;

namespace DocumentProcessor.Api.Ocr
{

    /// <summary>
    /// Document processor implementation that calls Python CLI.
    /// </summary>
    public class ServiceSideDocumentProcessor : IDocumentProcessor
    {
        private readonly ILogger<ServiceSideDocumentProcessor> _logger;
        private readonly OcrConfiguration _config;
        private readonly ConcurrentDictionary<string, JobStatus> _jobs = new();

        public ServiceSideDocumentProcessor(ILogger<ServiceSideDocumentProcessor> logger, OcrConfiguration config)
        {
            _logger = logger;
            _config = config;
        }

        public async Task<PreprocessingResult> PreprocessImageAsync(PreprocessingRequest request)
        {
            var jobId = request.JobId ?? Guid.NewGuid().ToString();
            
            _jobs[jobId] = new JobStatus
            {
                JobId = jobId,
                Status = "processing",
                Phase = "preprocessing",
                Progress = 0,
                Message = "Starting preprocessing"
            };

            var result = new PreprocessingResult
            {
                JobId = jobId,
                Status = "processing"
            };

            try
            {
                // Save base64 image to temp file
                var tempDir = Path.Combine(_config.TempStoragePath, jobId);
                Directory.CreateDirectory(tempDir);
                
                var extension = GetImageExtension(request.Filename);
                var inputPath = Path.Combine(tempDir, $"input{extension}");
                var outputPath = Path.Combine(tempDir, "preprocess_result.json");
                var imageBytes = Convert.FromBase64String(request.ImageBase64);
                await File.WriteAllBytesAsync(inputPath, imageBytes);

                _logger.LogInformation("Preprocess input saved: {InputPath} ({Bytes} bytes)", inputPath, imageBytes.Length);

                // Build CLI arguments
                var pythonPath = FindPythonExecutable();
                var cliPath = GetCliPath();
                
                var args = new List<string>
                {
                    cliPath,
                    "preprocess",
                    "--image", inputPath,
                    "--output", outputPath,
                    "--output-format", "base64",
                    "--job-id", jobId
                };

                if (request.Denoise)
                    args.Add("--denoise");
                
                if (request.Deskew)
                    args.Add("--deskew");
                
                args.AddRange(new[]
                {
                    "--fuzz-percent", request.FuzzPercent.ToString(),
                    "--deskew-threshold", request.DeskewThreshold.ToString(),
                    "--contrast-type", request.ContrastType,
                    "--contrast-strength", request.ContrastStrength.ToString(),
                    "--contrast-midpoint", request.ContrastMidpoint.ToString()
                });

                // Threshold options
                if (request.ApplyThreshold)
                {
                    args.Add("--apply-threshold");
                    args.AddRange(new[] { "--threshold-percent", request.ThresholdPercent.ToString() });
                }

                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = "processing",
                    Phase = "preprocessing",
                    Progress = 50,
                    Message = "Running preprocessing pipeline"
                };

                // Run CLI
                await RunPythonCliAsync(pythonPath, args, jobId);
                
                // Read result from file
                if (!File.Exists(outputPath))
                {
                    throw new FileNotFoundException("Preprocessing output not found", outputPath);
                }

                var outputJson = await File.ReadAllTextAsync(outputPath);
                var cliResult = JsonSerializer.Deserialize<JsonElement>(outputJson);

                if (cliResult.TryGetProperty("status", out var statusProp) && 
                    statusProp.GetString() == "done")
                {
                    result.Status = "done";
                    result.ImageBase64 = cliResult.TryGetProperty("image_base64", out var imgProp) 
                        ? imgProp.GetString() 
                        : null;
                    result.ImageFormat = cliResult.TryGetProperty("image_format", out var fmtProp) 
                        ? fmtProp.GetString() 
                        : "png";
                    result.Width = cliResult.TryGetProperty("width", out var wProp) 
                        ? wProp.GetInt32() 
                        : 0;
                    result.Height = cliResult.TryGetProperty("height", out var hProp) 
                        ? hProp.GetInt32() 
                        : 0;

                    _logger.LogInformation("Preprocessing completed: {Width}x{Height}, format {Format}", result.Width, result.Height, result.ImageFormat);
                }
                else
                {
                    result.Status = "failed";
                    result.Error = cliResult.TryGetProperty("error", out var errProp) 
                        ? errProp.GetString() 
                        : "Unknown error";
                }

                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = result.Status,
                    Phase = "preprocessing",
                    Progress = 100,
                    Message = result.Status == "done" ? "Preprocessing complete" : result.Error
                };

                // Clean up temp files
                try
                {
                    Directory.Delete(tempDir, true);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to clean up temp directory for job {JobId}", jobId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during preprocessing for job {JobId}", jobId);
                result.Status = "failed";
                result.Error = ex.Message;
                
                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = "failed",
                    Phase = "preprocessing",
                    Progress = 0,
                    Error = ex.Message
                };
            }

            return result;
        }

        public async Task<OcrResult> RunOcrAsync(OcrRequest request)
        {
            var jobId = request.JobId ?? Guid.NewGuid().ToString();
            
            _jobs[jobId] = new JobStatus
            {
                JobId = jobId,
                Status = "processing",
                Phase = "ocr",
                Progress = 0,
                Message = "Starting OCR"
            };

            var result = new OcrResult
            {
                JobId = jobId,
                Status = "processing"
            };

            try
            {
                // Save base64 image to temp file
                var tempDir = Path.Combine(_config.TempStoragePath, jobId);
                Directory.CreateDirectory(tempDir);
                
                var inputPath = Path.Combine(tempDir, "preprocessed.png");
                var outputPath = Path.Combine(tempDir, "ocr_result.json");
                var imageBytes = Convert.FromBase64String(request.ImageBase64);
                await File.WriteAllBytesAsync(inputPath, imageBytes);

                _logger.LogInformation("OCR input saved: {InputPath} ({Bytes} bytes)", inputPath, imageBytes.Length);

                // Build CLI arguments
                var pythonPath = FindPythonExecutable();
                var cliPath = GetCliPath();
                
                var args = new List<string>
                {
                    cliPath,
                    "ocr",
                    "--image", inputPath,
                    "--output", outputPath,
                    "--job-id", jobId,
                    "--ocr-engine", request.OcrEngine,
                    "--target-dpi", request.TargetDpi.ToString(),
                    "--device", request.Device
                };

                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = "processing",
                    Phase = "ocr",
                    Progress = 50,
                    Message = "Running OCR engine"
                };

                // Run CLI
                await RunPythonCliAsync(pythonPath, args, jobId);
                
                // Read result from file
                if (!File.Exists(outputPath))
                {
                    throw new FileNotFoundException("OCR output not found", outputPath);
                }

                var outputJson = await File.ReadAllTextAsync(outputPath);
                var cliResult = JsonSerializer.Deserialize<JsonElement>(outputJson);
                
                if (cliResult.TryGetProperty("status", out var statusProp) && 
                    statusProp.GetString() == "done")
                {
                    result.Status = "done";
                    result.RawOcrText = cliResult.TryGetProperty("raw_ocr_text", out var textProp) 
                        ? textProp.GetString() 
                        : "";
                    result.ImageWidth = cliResult.TryGetProperty("image_width", out var wProp) 
                        ? wProp.GetInt32() 
                        : 0;
                    result.ImageHeight = cliResult.TryGetProperty("image_height", out var hProp) 
                        ? hProp.GetInt32() 
                        : 0;

                    // Parse words
                    if (cliResult.TryGetProperty("words", out var wordsProp))
                    {
                        foreach (var wordElem in wordsProp.EnumerateArray())
                        {
                            var box = new BoundingBox();
                            if (wordElem.TryGetProperty("box", out var boxProp))
                            {
                                box.X0 = boxProp.TryGetProperty("x0", out var x0) ? x0.GetInt32() : 0;
                                box.Y0 = boxProp.TryGetProperty("y0", out var y0) ? y0.GetInt32() : 0;
                                box.X1 = boxProp.TryGetProperty("x1", out var x1) ? x1.GetInt32() : 0;
                                box.Y1 = boxProp.TryGetProperty("y1", out var y1) ? y1.GetInt32() : 0;
                            }

                            result.Words.Add(new OcrWord
                            {
                                Text = wordElem.TryGetProperty("text", out var t) ? t.GetString() ?? "" : "",
                                Box = box,
                                Confidence = wordElem.TryGetProperty("confidence", out var c) ? c.GetDouble() : 0
                            });
                        }
                    }

                    _logger.LogInformation("OCR completed: {Width}x{Height}, words {WordCount}", result.ImageWidth, result.ImageHeight, result.Words.Count);
                }
                else
                {
                    result.Status = "failed";
                    result.Error = cliResult.TryGetProperty("error", out var errProp) 
                        ? errProp.GetString() 
                        : "Unknown error";
                }

                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = result.Status,
                    Phase = "ocr",
                    Progress = 100,
                    Message = result.Status == "done" ? "OCR complete" : result.Error
                };

                // Clean up temp files
                try
                {
                    Directory.Delete(tempDir, true);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to clean up temp directory for job {JobId}", jobId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during OCR for job {JobId}", jobId);
                result.Status = "failed";
                result.Error = ex.Message;
                
                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = "failed",
                    Phase = "ocr",
                    Progress = 0,
                    Error = ex.Message
                };
            }

            return result;
        }

        public async Task<InferenceResult> RunInferenceAsync(InferenceRequest request)
        {
            var jobId = request.JobId ?? Guid.NewGuid().ToString();
            
            _jobs[jobId] = new JobStatus
            {
                JobId = jobId,
                Status = "processing",
                Phase = "inference",
                Progress = 0,
                Message = "Starting model inference"
            };

            var result = new InferenceResult
            {
                JobId = jobId,
                Status = "processing"
            };

            try
            {
                // Save base64 image and OCR result to temp files
                var tempDir = Path.Combine(_config.TempStoragePath, jobId);
                Directory.CreateDirectory(tempDir);
                
                var imagePath = Path.Combine(tempDir, "image.png");
                var ocrResultPath = Path.Combine(tempDir, "ocr_result.json");
                var outputPath = Path.Combine(tempDir, "inference_result.json");
                var imageBytes = Convert.FromBase64String(request.ImageBase64);
                await File.WriteAllBytesAsync(imagePath, imageBytes);

                var ocrResultJson = JsonSerializer.Serialize(request.OcrResult);
                await File.WriteAllTextAsync(ocrResultPath, ocrResultJson);

                _logger.LogInformation("Inference inputs saved: {ImagePath}, {OcrPath}", imagePath, ocrResultPath);

                // Build CLI arguments
                var pythonPath = FindPythonExecutable();
                var cliPath = GetCliPath();
                
                var args = new List<string>
                {
                    cliPath,
                    "inference",
                    "--ocr-result", ocrResultPath,
                    "--image", imagePath,
                    "--output", outputPath,
                    "--job-id", jobId,
                    "--model", request.Model,
                    "--device", request.Device
                };

                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = "processing",
                    Phase = "inference",
                    Progress = 50,
                    Message = "Running model inference"
                };

                // Run CLI
                await RunPythonCliAsync(pythonPath, args, jobId);
                
                // Read result from file
                if (!File.Exists(outputPath))
                {
                    throw new FileNotFoundException("Inference output not found", outputPath);
                }

                var outputJson = await File.ReadAllTextAsync(outputPath);
                var cliResult = JsonSerializer.Deserialize<JsonElement>(outputJson);
                
                if (cliResult.TryGetProperty("status", out var statusProp) && 
                    statusProp.GetString() == "done")
                {
                    result.Status = "done";
                    result.VendorName = ParseExtractedField(cliResult, "vendor_name");
                    result.MerchantAddress = ParseExtractedField(cliResult, "merchant_address");
                    result.Date = ParseExtractedField(cliResult, "date");
                    result.TotalAmount = ParseExtractedField(cliResult, "total_amount");
                    result.Subtotal = ParseExtractedField(cliResult, "subtotal");
                    result.TaxAmount = ParseExtractedField(cliResult, "tax_amount");
                    result.Currency = ParseExtractedField(cliResult, "currency");

                    // Parse line items
                    if (cliResult.TryGetProperty("line_items", out var itemsProp))
                    {
                        foreach (var itemElem in itemsProp.EnumerateArray())
                        {
                            result.LineItems.Add(new LineItem
                            {
                                Description = itemElem.TryGetProperty("description", out var d) ? d.GetString() : null,
                                Quantity = itemElem.TryGetProperty("quantity", out var q) ? (decimal?)q.GetDouble() : null,
                                UnitPrice = itemElem.TryGetProperty("unit_price", out var u) ? (decimal?)u.GetDouble() : null,
                                LineTotal = itemElem.TryGetProperty("line_total", out var l) ? (decimal?)l.GetDouble() : null,
                                Confidence = itemElem.TryGetProperty("confidence", out var c) ? c.GetDouble() : 0
                            });
                        }
                    }

                    _logger.LogInformation("Inference completed for job {JobId} with {ItemCount} items", jobId, result.LineItems.Count);
                }
                else
                {
                    result.Status = "failed";
                    result.Error = cliResult.TryGetProperty("error", out var errProp) 
                        ? errProp.GetString() 
                        : "Unknown error";
                }

                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = result.Status,
                    Phase = "inference",
                    Progress = 100,
                    Message = result.Status == "done" ? "Inference complete" : result.Error
                };

                // Clean up temp files
                try
                {
                    Directory.Delete(tempDir, true);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to clean up temp directory for job {JobId}", jobId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during inference for job {JobId}", jobId);
                result.Status = "failed";
                result.Error = ex.Message;
                
                _jobs[jobId] = new JobStatus
                {
                    JobId = jobId,
                    Status = "failed",
                    Phase = "inference",
                    Progress = 0,
                    Error = ex.Message
                };
            }

            return result;
        }

        public Task<JobStatus?> GetJobStatusAsync(string jobId)
        {
            _jobs.TryGetValue(jobId, out var status);
            return Task.FromResult(status);
        }

        private static ExtractedField? ParseExtractedField(JsonElement root, string fieldName)
        {
            if (!root.TryGetProperty(fieldName, out var fieldProp) || fieldProp.ValueKind == JsonValueKind.Null)
            {
                return null;
            }

            BoundingBox? box = null;
            if (fieldProp.TryGetProperty("box", out var boxProp) && boxProp.ValueKind == JsonValueKind.Object)
            {
                box = new BoundingBox
                {
                    X0 = boxProp.TryGetProperty("x0", out var x0) ? x0.GetInt32() : 0,
                    Y0 = boxProp.TryGetProperty("y0", out var y0) ? y0.GetInt32() : 0,
                    X1 = boxProp.TryGetProperty("x1", out var x1) ? x1.GetInt32() : 0,
                    Y1 = boxProp.TryGetProperty("y1", out var y1) ? y1.GetInt32() : 0
                };
            }

            return new ExtractedField
            {
                Value = fieldProp.TryGetProperty("value", out var valProp) ? valProp.GetString() ?? "" : "",
                Confidence = fieldProp.TryGetProperty("confidence", out var confProp) ? confProp.GetDouble() : 0,
                Box = box
            };
        }

        private static string GetImageExtension(string? filename)
        {
            if (string.IsNullOrEmpty(filename))
                return ".png";
            
            var ext = Path.GetExtension(filename).ToLowerInvariant();
            return string.IsNullOrEmpty(ext) ? ".png" : ext;
        }

        private async Task<string> RunPythonCliAsync(string pythonPath, List<string> args, string? jobId = null)
        {
            var processStartInfo = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = string.Join(" ", args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a)),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _logger.LogInformation("Executing: {FileName} {Arguments}", processStartInfo.FileName, processStartInfo.Arguments);

            using var process = new Process { StartInfo = processStartInfo };
            if (!process.Start())
            {
                throw new InvalidOperationException("Failed to start CLI process");
            }

            var outputLines = new System.Collections.Generic.List<string>();
            var errorBuilder = new System.Text.StringBuilder();

            // Read stdout line by line and parse JSON events
            var outputTask = Task.Run(async () =>
            {
                while (!process.StandardOutput.EndOfStream)
                {
                    var line = await process.StandardOutput.ReadLineAsync();
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        outputLines.Add(line);
                        
                        // Try to parse as JSON event
                        if (jobId != null)
                        {
                            TryParseAndUpdateJobStatus(line, jobId);
                        }
                    }
                }
            });

            var errorTask = Task.Run(async () =>
            {
                while (!process.StandardError.EndOfStream)
                {
                    var line = await process.StandardError.ReadLineAsync();
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        errorBuilder.AppendLine(line);
                    }
                }
            });

            // Wait for process with timeout (60 seconds for faster live view feedback)
            var completed = process.WaitForExit(60000);

            await outputTask;
            await errorTask;

            var output = string.Join(Environment.NewLine, outputLines);
            var error = errorBuilder.ToString();

            if (!completed)
            {
                process.Kill();
                throw new TimeoutException("CLI process timed out after 60 seconds");
            }

            if (!string.IsNullOrWhiteSpace(error))
            {
                _logger.LogWarning("CLI stderr: {Error}", error);
            }

            _logger.LogInformation("CLI completed with exit code {ExitCode}", process.ExitCode);

            if (process.ExitCode != 0)
            {
                throw new Exception($"CLI failed with exit code {process.ExitCode}: {error}");
            }

            return output;
        }

        private void TryParseAndUpdateJobStatus(string line, string jobId)
        {
            // Quick check: JSON lines start with '{'
            if (!line.TrimStart().StartsWith('{'))
            {
                return;
            }

            try
            {
                var json = JsonSerializer.Deserialize<JsonElement>(line);
                if (json.TryGetProperty("event", out var eventProp))
                {
                    var eventType = eventProp.GetString();
                    UpdateJobStatusFromEvent(jobId, eventType, json);
                }
            }
            catch (JsonException)
            {
                // Not a JSON line, ignore it
            }
        }

        private void UpdateJobStatusFromEvent(string jobId, string? eventType, JsonElement eventData)
        {
            if (string.IsNullOrEmpty(eventType))
                return;

            var status = _jobs.GetOrAdd(jobId, new JobStatus
            {
                JobId = jobId,
                Status = "processing",
                Phase = "unknown",
                Progress = 0
            });

            switch (eventType)
            {
                case "preprocess_start":
                    status.Status = "processing";
                    status.Phase = "preprocessing";
                    status.Progress = 10;
                    status.Message = "Starting preprocessing";
                    break;
                case "preprocessing_image":
                    status.Status = "processing";
                    status.Phase = "preprocessing";
                    status.Progress = 50;
                    status.Message = "Applying image filters";
                    break;
                case "preprocess_complete":
                    status.Status = "processing";
                    status.Phase = "preprocessing";
                    status.Progress = 90;
                    status.Message = "Preprocessing complete";
                    break;
                case "preprocess_error":
                    status.Status = "failed";
                    status.Phase = "preprocessing";
                    status.Error = eventData.TryGetProperty("error", out var err) ? err.GetString() : "Unknown error";
                    break;
                case "ocr_start":
                    status.Status = "processing";
                    status.Phase = "ocr";
                    status.Progress = 10;
                    status.Message = "Starting OCR";
                    break;
                case "resampling_image":
                    status.Status = "processing";
                    status.Phase = "ocr";
                    status.Progress = 20;
                    status.Message = "Resampling image";
                    break;
                case "running_ocr_engine":
                    status.Status = "processing";
                    status.Phase = "ocr";
                    status.Progress = 40;
                    status.Message = eventData.TryGetProperty("message", out var ocrMsg) ? ocrMsg.GetString() : "Running OCR";
                    break;
                case "ocr_complete":
                    status.Status = "processing";
                    status.Phase = "ocr";
                    status.Progress = 90;
                    status.Message = "OCR complete";
                    break;
                case "ocr_error":
                    status.Status = "failed";
                    status.Phase = "ocr";
                    status.Error = eventData.TryGetProperty("error", out var ocrErr) ? ocrErr.GetString() : "Unknown error";
                    break;
                case "inference_start":
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 10;
                    status.Message = "Starting inference";
                    break;
                case "loading_model":
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 20;
                    status.Message = "Loading ML model";
                    break;
                case "running_inference":
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 40;
                    status.Message = "Running inference";
                    break;
                case "model_entities":
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 60;
                    status.Message = "Model predictions complete";
                    break;
                case "extracting_fields":
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 70;
                    status.Message = "Extracting fields";
                    break;
                case "inference_complete":
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 90;
                    status.Message = "Inference complete";
                    break;
                case "inference_error":
                    status.Status = "failed";
                    status.Phase = "inference";
                    status.Error = eventData.TryGetProperty("error", out var infErr) ? infErr.GetString() : "Unknown error";
                    break;
                case "model_error":
                    // Model error is a warning, not a failure
                    status.Status = "processing";
                    status.Phase = "inference";
                    status.Progress = 50;
                    status.Message = "Model unavailable, using heuristics";
                    break;
            }

            _jobs[jobId] = status;
            _logger.LogDebug("Updated job {JobId} status: {Phase} - {Message}", jobId, status.Phase, status.Message);
        }

        private string FindPythonExecutable()
        {
            // If a virtual environment path is configured, use the Python interpreter from that venv
            if (!string.IsNullOrWhiteSpace(_config.PythonVenvPath))
            {
                // Get full path (handles both relative and absolute paths)
                var venvPath = Path.GetFullPath(_config.PythonVenvPath);
                
                // Determine the Python executable path based on OS
                // Windows: venv/Scripts/python.exe
                // Linux/Mac: venv/bin/python
                var pythonPath = OperatingSystem.IsWindows()
                    ? Path.Combine(venvPath, "Scripts", "python.exe")
                    : Path.Combine(venvPath, "bin", "python");
                
                // If the venv Python exists, use it
                if (File.Exists(pythonPath))
                {
                    _logger.LogInformation("Using Python from virtual environment: {PythonPath}", pythonPath);
                    return pythonPath;
                }
                
                _logger.LogWarning("Python virtual environment not found at {VenvPath}, falling back to system Python", venvPath);
            }
            
            // Fallback to system Python if venv is not configured or not found
            var candidates = new[] { "python3", "python", "python3.11", "python3.10" };
            
            foreach (var candidate in candidates)
            {
                try
                {
                    var psi = new ProcessStartInfo
                    {
                        FileName = candidate,
                        Arguments = "--version",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var process = Process.Start(psi);
                    if (process != null)
                    {
                        process.WaitForExit(5000);
                        if (process.ExitCode == 0)
                        {
                            _logger.LogInformation("Using system Python: {Candidate}", candidate);
                            return candidate;
                        }
                    }
                }
                catch
                {
                    // Continue to next candidate
                }
            }
            
            _logger.LogWarning("No Python executable found, defaulting to 'python3'");
            return "python3";
        }

        private string GetCliPath()
        {
            var cliPath = _config.PythonServicePath;
            
            if (!Path.IsPathRooted(cliPath))
            {
                cliPath = Path.Combine(Directory.GetCurrentDirectory(), cliPath);
            }
            
            cliPath = Path.GetFullPath(cliPath);
            
            if (!File.Exists(cliPath))
            {
                _logger.LogWarning("CLI path not found at {CliPath}, trying alternate locations", cliPath);
                
                var alternates = new[]
                {
                    Path.Combine(Directory.GetCurrentDirectory(), "Bardcoded.Ocr", "cli.py"),
                    Path.Combine(Directory.GetCurrentDirectory(), "..", "Bardcoded.Ocr", "cli.py"),
                    Path.Combine(AppContext.BaseDirectory, "Bardcoded.Ocr", "cli.py"),
                    Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Bardcoded.Ocr", "cli.py")
                };
                
                foreach (var alt in alternates)
                {
                    var normalized = Path.GetFullPath(alt);
                    if (File.Exists(normalized))
                    {
                        _logger.LogInformation("Found CLI at alternate location: {CliPath}", normalized);
                        return normalized;
                    }
                }
            }
            
            return cliPath;
        }
    }
}
