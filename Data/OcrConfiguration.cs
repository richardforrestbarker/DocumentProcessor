using System.Text.Json.Serialization;

namespace DocumentProcessor.Data
{
    /// <summary>
    /// Default Configuration for the OCR and document processing pipeline
    /// </summary>
    public class OcrConfiguration
    {
        /// <summary>
        /// Model name path
        /// </summary>
        public string ModelNameOrPath { get; set; } = "";

        /// <summary>
        /// Device to use for inference: "auto", "cuda", "cpu"
        /// </summary>
       
        public string Device { get; set; } = "auto";

        /// <summary>
        /// OCR engine to use: "paddle" or "tesseract"
        /// </summary>
        public string OcrEngine { get; set; } = "paddle";

        /// <summary>
        /// Detection mode: "word" or "line"
        /// </summary>
        public string DetectionMode { get; set; } = "word";

        /// <summary>
        /// Box normalization scale (typically 1000 for LayoutLM)
        /// </summary>
        public int BoxNormalizationScale { get; set; } = 1000;

        /// <summary>
        /// Path to the Python OCR service executable
        /// </summary>
        public required string PythonServicePath { get; set; }

        /// <summary>
        /// Optional working directory to use when invoking the Python CLI.
        /// If not set, defaults to the directory containing the PythonServicePath.
        /// </summary>
        public required string PythonWorkingDirectory { get; set; }

        /// <summary>
        /// Temporary storage path for uploaded images
        /// </summary>
        public required string TempStoragePath { get; set; }

        /// <summary>
        /// Maximum image file size in bytes (default: 10MB)
        /// </summary>
        public required long MaxFileSize { get; set; } = 10 * 1024 * 1024;

        /// <summary>
        /// Time-to-live for temporary files in hours
        /// </summary>
        public int TempFileTtlHours { get; set; } = 24;

        /// <summary>
        /// Enable GPU acceleration if available
        /// </summary>
        public bool EnableGpu { get; set; } = true;

        /// <summary>
        /// Minimum confidence threshold for field extraction (0.0 to 1.0)
        /// </summary>
        public double MinConfidenceThreshold { get; set; } = 0.5;

        /// <summary>
        /// Path to Python virtual environment (e.g., "./Ocr/venv")
        /// The Python interpreter will be located at bin/python (Linux/Mac) or Scripts/python.exe (Windows)
        /// </summary>
 
        public required string PythonVenvPath { get; set; }
    }
}
