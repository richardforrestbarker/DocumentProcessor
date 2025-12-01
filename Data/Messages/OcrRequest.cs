namespace Data.Ocr.Messages
{
    /// <summary>
    /// Request for running OCR.
    /// </summary>
    public class OcrRequest
    {
        /// <summary>
        /// Path to preprocessed image file or base64 encoded image.
        /// </summary>
        public required string ImageBase64 { get; set; }
        
        /// <summary>
        /// Job identifier for tracking.
        /// </summary>
        public string? JobId { get; set; }
        
        /// <summary>
        /// OCR engine to use: 'paddle' or 'tesseract'.
        /// </summary>
        public string OcrEngine { get; set; } = "paddle";
        
        /// <summary>
        /// Target DPI for resampling.
        /// </summary>
        public int TargetDpi { get; set; } = 300;
        
        /// <summary>
        /// Device for OCR: 'auto', 'cuda', or 'cpu'.
        /// </summary>
        public string Device { get; set; } = "auto";
    }
}
