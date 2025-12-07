using System.Text.Json.Serialization;

namespace DocumentProcessor.Data
{
    /// <summary>
    /// Configuration for the OCR and document processing pipeline
    /// </summary>
    public class OcrConfiguration
    {
        /// <summary>
        /// Model name or path (e.g., "microsoft/layoutlmv3-base")
        /// </summary>
        [JsonPropertyName("model_name_or_path")]
        public string ModelNameOrPath { get; set; } = "microsoft/layoutlmv3-base";

        /// <summary>
        /// Device to use for inference: "auto", "cuda", "cpu"
        /// </summary>
        [JsonPropertyName("device")]
        public string Device { get; set; } = "auto";

        /// <summary>
        /// OCR engine to use: "paddle" or "tesseract"
        /// </summary>
        [JsonPropertyName("ocr_engine")]
        public string OcrEngine { get; set; } = "paddle";

        /// <summary>
        /// Detection mode: "word" or "line"
        /// </summary>
        [JsonPropertyName("detection_mode")]
        public string DetectionMode { get; set; } = "word";

        /// <summary>
        /// Box normalization scale (typically 1000 for LayoutLM)
        /// </summary>
        [JsonPropertyName("box_normalization_scale")]
        public int BoxNormalizationScale { get; set; } = 1000;

        /// <summary>
        /// Path to the Python OCR service executable
        /// </summary>
        [JsonPropertyName("python_service_path")]
        public string PythonServicePath { get; set; } = "./Ocr/cli.py";

        /// <summary>
        /// Optional working directory to use when invoking the Python CLI.
        /// If not set, defaults to the directory containing the PythonServicePath.
        /// </summary>
        [JsonPropertyName("python_working_directory")]
        public string? PythonWorkingDirectory { get; set; }

        /// <summary>
        /// Temporary storage path for uploaded images
        /// </summary>
        [JsonPropertyName("temp_storage_path")]
        public string TempStoragePath { get; set; } = "./temp/documents";

        /// <summary>
        /// Maximum image file size in bytes (default: 10MB)
        /// </summary>
        [JsonPropertyName("max_file_size")]
        public long MaxFileSize { get; set; } = 10 * 1024 * 1024;

        /// <summary>
        /// Time-to-live for temporary files in hours
        /// </summary>
        [JsonPropertyName("temp_file_ttl_hours")]
        public int TempFileTtlHours { get; set; } = 24;

        /// <summary>
        /// Enable GPU acceleration if available
        /// </summary>
        [JsonPropertyName("enable_gpu")]
        public bool EnableGpu { get; set; } = true;

        /// <summary>
        /// Minimum confidence threshold for field extraction (0.0 to 1.0)
        /// </summary>
        [JsonPropertyName("min_confidence_threshold")]
        public double MinConfidenceThreshold { get; set; } = 0.5;

        /// <summary>
        /// Path to Python virtual environment (e.g., "./Ocr/venv")
        /// The Python interpreter will be located at bin/python (Linux/Mac) or Scripts/python.exe (Windows)
        /// </summary>
        [JsonPropertyName("python_venv_path")]
        public string? PythonVenvPath { get; set; } = "./Ocr/venv";
    }
}
