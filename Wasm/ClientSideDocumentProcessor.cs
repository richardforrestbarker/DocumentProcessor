using DocumentProcessor.Data.Ocr;
using DocumentProcessor.Data.Ocr.Messages;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Validation;
using System.Net.Http.Json;
using System.Reflection.Metadata;
using System.Text.Json;

namespace DocumentProcessor.Clients
{
    public class ClientSideDocumentProcessor : IDocumentProcessor
    {
        HttpClient _httpClient;
        ILogger log;
        public ClientSideDocumentProcessor(IHttpClientFactory clientFactory, ILoggerFactory loggerFactory)
        {
            _httpClient = clientFactory.CreateClient("document-processing");
            log = loggerFactory.CreateLogger<ClientSideDocumentProcessor>();
        }

        /// <summary>
        /// Gets the status of a job by its ID using the http client to call the server side of this API.
        /// </summary>
        /// <param name="jobId"></param>
        /// <returns></returns>
        public async Task<JobStatus?> GetJobStatusAsync(string jobId)
        {
            try
            {
                var response = await _httpClient.GetAsync($"/api/document/status/{jobId}");
                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadFromJsonAsync<JobStatus>();
                }
                else if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                {
                    return null;
                }
                else
                {
                    log.LogWarning("Failed to get job status for {JobId}: {StatusCode}", jobId, response.StatusCode);
                    return null;
                }
            }
            catch (Exception ex)
            {
                log.LogError(ex, "Error getting job status for {JobId}", jobId);
                return null;
            }
        }

        /// <summary>
        /// Calls the /api/document/preprocess endpoint to preprocess an image on the client side usibng the http client.
        /// </summary>
        /// <param name="request"></param>
        /// <returns></returns>
        public async Task<PreprocessingResult> PreprocessImageAsync(PreprocessingRequest request)
        {
            try
            {
                var response = await _httpClient.PostAsJsonAsync("/api/document/preprocess", request);
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<PreprocessingResult>() ?? throw new JsonException("Failed to parse the result of the preprocess call.");
            }
            catch (Exception ex)
            {
                return new PreprocessingResult
                {
                    JobId = request.JobId ?? string.Empty,
                    Status = "Error",
                    Error = ex.Message
                };
            } finally
            {
                log.LogDebug("Finished Preprocess request.");
            }
        }

        public async Task<InferenceResult> RunInferenceAsync(InferenceRequest request)
        {
            try
            {
                var response = await _httpClient.PostAsJsonAsync("/api/document/inference", request);
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<InferenceResult>() ?? throw new JsonException("Failed to parse the result of the preprocess call.");
            }
            catch (Exception ex)
            {
                return new InferenceResult
                {
                    JobId = request.JobId ?? string.Empty,
                    Status = "Error",
                    Error = ex.Message
                };
            }
            finally
            {
                log.LogDebug("Finished Preprocess request.");
            }
        }

        public async Task<OcrResult> RunOcrAsync(OcrRequest request)
        {
            try
            {
                var response = await _httpClient.PostAsJsonAsync("/api/document/ocr", request);
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<OcrResult>() ?? throw new JsonException("Failed to parse the result of the preprocess call.");
            }
            catch (Exception ex)
            {
                return new OcrResult
                {
                    JobId = request.JobId ?? string.Empty,
                    Status = "Error",
                    Error = ex.Message
                };
            }
            finally
            {
                log.LogDebug("Finished ocr request.");
            }
        }
    }
}
