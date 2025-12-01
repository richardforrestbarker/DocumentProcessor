using Data.Messages;

namespace Data.Ocr.Messages
{
    /// <summary>
    /// Result from model inference.
    /// </summary>
    public class InferenceResult
    {
        public required string JobId { get; set; }
        public required string Status { get; set; }
        public ExtractedField? VendorName { get; set; }
        public ExtractedField? MerchantAddress { get; set; }
        public ExtractedField? Date { get; set; }
        public ExtractedField? TotalAmount { get; set; }
        public ExtractedField? Subtotal { get; set; }
        public ExtractedField? TaxAmount { get; set; }
        public ExtractedField? Currency { get; set; }
        public List<LineItem> LineItems { get; set; } = new();
        public string? Error { get; set; }
    }
}
