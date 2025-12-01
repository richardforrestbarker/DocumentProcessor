namespace Data.Ocr.Messages
{
    /// <summary>
    /// Job status for tracking long-running operations.
    /// </summary>
    public class JobStatus
    {
        public required string JobId { get; set; }
        public required string Status { get; set; }
        public string? Phase { get; set; }
        public int Progress { get; set; }
        public string? Message { get; set; }
        public string? Error { get; set; }
    }
}
