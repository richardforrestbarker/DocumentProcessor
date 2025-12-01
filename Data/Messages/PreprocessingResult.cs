namespace DocumentProcessor.Data.Ocr.Messages
{
    /// <summary>
    /// Result from preprocessing.
    /// </summary>
    public class PreprocessingResult
    {
        public required string JobId { get; set; }
        public required string Status { get; set; }
        public string? ImageBase64 { get; set; }
        public string? ImageFormat { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public string? Error { get; set; }
    }
}
