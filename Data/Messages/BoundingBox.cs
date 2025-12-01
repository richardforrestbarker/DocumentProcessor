using System.Text.Json.Serialization;

namespace DocumentProcessor.Data.Messages
{
    /// <summary>
    /// Represents a normalized bounding box in model coordinate space (0-1000 scale)
    /// </summary>
    public class BoundingBox
    {
        [JsonPropertyName("x0")]
        public int X0 { get; set; }

        [JsonPropertyName("y0")]
        public int Y0 { get; set; }

        [JsonPropertyName("x1")]
        public int X1 { get; set; }

        [JsonPropertyName("y1")]
        public int Y1 { get; set; }
    }
}
