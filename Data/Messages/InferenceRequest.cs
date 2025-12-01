namespace DocumentProcessor.Data.Ocr.Messages
{
    /// <summary>
    /// Request for running model inference.
    /// </summary>
    public class InferenceRequest
    {
        /// <summary>
        /// Path to OCR result JSON or the OcrResult object.
        /// </summary>
        public required OcrResult OcrResult { get; set; }
        
        /// <summary>
        /// Base64 encoded image for visual features.
        /// </summary>
        public required string ImageBase64 { get; set; }
        
        /// <summary>
        /// Job identifier for tracking.
        /// </summary>
        public string? JobId { get; set; }
        
        /// <summary>
        /// Model name or path.
        /// </summary>
        public string Model { get; set; } = "naver-clova-ix/donut-base-finetuned-cord-v2";
        
        /// <summary>
        /// Model type: 'donut', 'idefics2', or 'layoutlmv3'.
        /// </summary>
        public string ModelType { get; set; } = "donut";
        
        /// <summary>
        /// Device for inference: 'auto', 'cuda', or 'cpu'.
        /// </summary>
        public string Device { get; set; } = "auto";
    }
}
