using System.Text.Json.Serialization;

namespace Data.Messages
{
    /// <summary>
    /// Represents a line item from a receipt
    /// </summary>
    public class LineItem
    {
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("quantity")]
        public decimal? Quantity { get; set; }

        [JsonPropertyName("unit_price")]
        public decimal? UnitPrice { get; set; }

        [JsonPropertyName("line_total")]
        public decimal? LineTotal { get; set; }

        [JsonPropertyName("box")]
        public BoundingBox? Box { get; set; }

        [JsonPropertyName("confidence")]
        public double Confidence { get; set; }
    }
}
