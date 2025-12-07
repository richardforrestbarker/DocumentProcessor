namespace DocumentProcessor.Data.Ocr.Messages
{
    /// <summary>
    /// Request for preprocessing an image.
    /// </summary>
    public class PreprocessingRequest
    {
        /// <summary>
        /// Base64 encoded source image.
        /// </summary>
        public required string ImageBase64 { get; set; }
        
        /// <summary>
        /// Original filename for format detection.
        /// </summary>
        public string? Filename { get; set; }
        
        /// <summary>
        /// Job identifier for tracking.
        /// </summary>
        public string? JobId { get; set; }
        
        /// <summary>
        /// Apply denoising.
        /// </summary>
        public bool Denoise { get; set; } = false;
        
        /// <summary>
        /// Apply deskewing.
        /// </summary>
        public bool Deskew { get; set; } = true;
        
        /// <summary>
        /// Fuzz percentage for background removal (0-100).
        /// </summary>
        public int FuzzPercent { get; set; } = 30;
        
        /// <summary>
        /// Deskew threshold percentage (0-100).
        /// </summary>
        public int DeskewThreshold { get; set; } = 40;
        
        /// <summary>
        /// Contrast enhancement type: 'sigmoidal', 'linear', or 'none'.
        /// </summary>
        public string ContrastType { get; set; } = "sigmoidal";
        
        /// <summary>
        /// Contrast strength for sigmoidal type (1-10).
        /// </summary>
        public double ContrastStrength { get; set; } = 3.0;
        
        /// <summary>
        /// Contrast midpoint percentage (0-200).
        /// </summary>
        public int ContrastMidpoint { get; set; } = 120;
        
        /// <summary>
        /// Apply thresholding after contrast step.
        /// </summary>
        public bool ApplyThreshold { get; set; } = false;
        
        /// <summary>
        /// Threshold percentage (0-100) used when ApplyThreshold is true.
        /// </summary>
        public int ThresholdPercent { get; set; } = 50;
    }
}
