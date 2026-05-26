using DocumentProcessor.Data.Messages;

namespace DocumentProcessor.Data.Ocr.Messages
{
    /// <summary>
    /// Result from model inference.
    /// </summary>
    public class InferenceResult
    {
        public required string JobId { get; set; }
        public required string Status { get; set; }
        
        // Document classification
        public ExtractedField? DocumentType { get; set; }
        
        // Common fields across all document types
        public ExtractedField? VendorName { get; set; }
        public ExtractedField? MerchantAddress { get; set; }
        public ExtractedField? Date { get; set; }
        public ExtractedField? TotalAmount { get; set; }
        public ExtractedField? Subtotal { get; set; }
        public ExtractedField? TaxAmount { get; set; }
        public ExtractedField? Currency { get; set; }
        public List<LineItem> LineItems { get; set; } = new();
        
        // Invoice-specific fields
        public ExtractedField? InvoiceNumber { get; set; }
        public ExtractedField? DueDate { get; set; }
        public ExtractedField? PaymentTerms { get; set; }
        public ExtractedField? CustomerName { get; set; }
        public ExtractedField? CustomerAddress { get; set; }
        public ExtractedField? PoNumber { get; set; }
        
        // Bill-specific fields
        public ExtractedField? AccountNumber { get; set; }
        public ExtractedField? BillingPeriod { get; set; }
        public ExtractedField? PreviousBalance { get; set; }
        public ExtractedField? CurrentCharges { get; set; }
        public ExtractedField? AmountDue { get; set; }
        
        // Receipt-specific fields
        public ExtractedField? PaymentMethod { get; set; }
        public ExtractedField? CashierName { get; set; }
        public ExtractedField? RegisterNumber { get; set; }
        
        // General financial document fields
        public ExtractedField? Discount { get; set; }
        public ExtractedField? Shipping { get; set; }
        public ExtractedField? Notes { get; set; }
        
        public string? Error { get; set; }
    }
}
