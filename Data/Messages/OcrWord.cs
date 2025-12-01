using System.Text.Json.Serialization;

namespace DocumentProcessor.Data.Messages
{
    /// <summary>
    /// Represents a single word from OCR with its bounding box and confidence
    /// </summary>
    public record OcrWord
    {
        [JsonPropertyName("text")]
        public required string Text { get; set; } = string.Empty;

        [JsonPropertyName("box")]
        public required BoundingBox Box { get; set; } = new BoundingBox();

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; } = 0;
    }
}
