namespace DocumentProcessor.Wasm.Preprocessing
{
    public sealed class BrowserPreprocessingSettings
    {
        public bool Deskew { get; set; } = true;
        public int DeskewThreshold { get; set; } = 40;
        public bool Denoise { get; set; } = false;
        public int FuzzPercent { get; set; } = 30;
        public string ContrastType { get; set; } = "sigmoidal";
        public double ContrastStrength { get; set; } = 3.0;
        public int ContrastMidpoint { get; set; } = 120;
        public bool ApplyThreshold { get; set; } = false;
        public int ThresholdPercent { get; set; } = 50;
        public int TargetDpi { get; set; } = 300;
    }

    public sealed class BrowserPreprocessingResult
    {
        public required string ImageBase64 { get; init; }
        public required string ImageFormat { get; init; }
        public required string ImageMimeType { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
    }
}
