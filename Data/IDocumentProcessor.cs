using Data.Ocr.Messages;

namespace Data.Ocr
{
    /// <summary>
    /// Interface for document processing service for the OCR pipeline live view.
    /// </summary>
    public interface IDocumentProcessor
    {
        /// <summary>
        /// Run preprocessing on an image and return the result as base64.
        /// </summary>
        Task<PreprocessingResult> PreprocessImageAsync(PreprocessingRequest request);
        
        /// <summary>
        /// Run OCR on a preprocessed image.
        /// </summary>
        Task<OcrResult> RunOcrAsync(OcrRequest request);
        
        /// <summary>
        /// Run model inference on OCR results.
        /// </summary>
        Task<InferenceResult> RunInferenceAsync(InferenceRequest request);
        
        /// <summary>
        /// Get the status of a job.
        /// </summary>
        Task<JobStatus?> GetJobStatusAsync(string jobId);
    }
}
