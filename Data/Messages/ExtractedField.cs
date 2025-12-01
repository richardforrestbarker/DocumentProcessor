using System.Text.Json.Serialization;

namespace Data.Messages
{
    /// <summary>
    /// Represents a field extracted from a receipt with confidence and bounding box
    /// </summary>
    public class ExtractedField
    {
        [JsonPropertyName("value")]
        public string? Value { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }

        [JsonPropertyName("box")]
        public BoundingBox? Box { get; set; }
    }
}
