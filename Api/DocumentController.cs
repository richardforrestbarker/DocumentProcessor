using DocumentProcessor.Api.Ocr;
using DocumentProcessor.Data.Ocr.Messages;
using DocumentProcessor.Data.Messages;
using DocumentProcessor.Data.Ocr;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace DocumentProcessor.Api.Controllers
{
    /// <summary>
    /// Controller for document OCR processing live view pipeline.
    /// Provides separate endpoints for preprocessing, OCR, and inference phases.
    /// </summary>
    [Route("api/document")]
    [ApiController]
    [Produces("application/json")]
    public class DocumentController : ControllerBase
    {
        private readonly ILogger<DocumentController> _logger;
        private readonly IDocumentProcessor _documentProcessor;

        public DocumentController(
            ILogger<DocumentController> logger,
            IDocumentProcessor documentProcessor)
        {
            _logger = logger;
            _documentProcessor = documentProcessor;
        }

        /// <summary>
        /// Run preprocessing on an image.
        /// Returns the preprocessed image as base64 for live preview.
        /// </summary>
        /// <param name="request">Preprocessing request with image and settings</param>
        /// <returns>Preprocessed image as base64</returns>
        [HttpPost("preprocess")]
        [ProducesResponseType(typeof(PreprocessingResult), 200)]
        [ProducesResponseType(400)]
        public async Task<IActionResult> Preprocess([FromBody] PreprocessingRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.ImageBase64))
                {
                    return BadRequest(new { error = "ImageBase64 is required" });
                }

                _logger.LogInformation("Starting preprocessing for job {JobId}", request.JobId);
                
                var result = await _documentProcessor.PreprocessImageAsync(request);
                
                if (result.Status == "failed")
                {
                    return StatusCode(500, result);
                }

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during preprocessing");
                return StatusCode(500, new { error = "Internal server error during preprocessing" });
            }
        }

        /// <summary>
        /// Run OCR on a preprocessed image.
        /// Includes resampling and DPI safety checks.
        /// </summary>
        /// <param name="request">OCR request with preprocessed image</param>
        /// <returns>OCR results with words and bounding boxes</returns>
        [HttpPost("ocr")]
        [ProducesResponseType(typeof(OcrResult), 200)]
        [ProducesResponseType(400)]
        public async Task<IActionResult> RunOcr([FromBody] OcrRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.ImageBase64))
                {
                    return BadRequest(new { error = "ImageBase64 is required" });
                }

                _logger.LogInformation("Starting OCR for job {JobId}", request.JobId);
                
                var result = await _documentProcessor.RunOcrAsync(request);
                
                if (result.Status == "failed")
                {
                    return StatusCode(500, result);
                }

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during OCR");
                return StatusCode(500, new { error = "Internal server error during OCR" });
            }
        }

        /// <summary>
        /// Run model inference on OCR results.
        /// Extracts structured fields from the receipt.
        /// </summary>
        /// <param name="request">Inference request with OCR results and image</param>
        /// <returns>Extracted receipt fields</returns>
        [HttpPost("inference")]
        [ProducesResponseType(typeof(InferenceResult), 200)]
        [ProducesResponseType(400)]
        public async Task<IActionResult> RunInference([FromBody] InferenceRequest request)
        {
            try
            {
                if (request.OcrResult == null)
                {
                    return BadRequest(new { error = "OcrResult is required" });
                }

                if (string.IsNullOrEmpty(request.ImageBase64))
                {
                    return BadRequest(new { error = "ImageBase64 is required" });
                }

                _logger.LogInformation("Starting inference for job {JobId}", request.JobId);
                
                var result = await _documentProcessor.RunInferenceAsync(request);
                
                if (result.Status == "failed")
                {
                    return StatusCode(500, result);
                }

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during inference");
                return StatusCode(500, new { error = "Internal server error during inference" });
            }
        }

        /// <summary>
        /// Get the status of a document processing job.
        /// </summary>
        /// <param name="jobId">Job identifier</param>
        /// <returns>Current job status</returns>
        [HttpGet("status/{jobId}")]
        [ProducesResponseType(typeof(JobStatus), 200)]
        [ProducesResponseType(404)]
        public async Task<IActionResult> GetStatus(string jobId)
        {
            try
            {
                var status = await _documentProcessor.GetJobStatusAsync(jobId);
                
                if (status == null)
                {
                    return NotFound(new { error = $"Job {jobId} not found" });
                }

                return Ok(status);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting status for job {JobId}", jobId);
                return StatusCode(500, new { error = "Internal server error" });
            }
        }
    }
}
