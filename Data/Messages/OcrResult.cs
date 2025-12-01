using DocumentProcessor.Data.Messages;

namespace DocumentProcessor.Data.Ocr.Messages
{
    /// <summary>
    /// Result from OCR.
    /// </summary>
    public record OcrResult
    {
        public required string JobId { get; set; } = string.Empty;
        public required string Status { get; set; } = "Unstarted";
        public List<OcrWord> Words { get; set; } = new();
        public string? RawOcrText { get; set; }
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public string? Error { get; set; }
    }
}
