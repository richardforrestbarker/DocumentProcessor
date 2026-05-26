namespace DocumentProcessor.Wasm.Preprocessing
{
    public interface IBrowserImagePreprocessor
    {
        Task<BrowserPreprocessingResult> PreprocessAsync(string imageBase64, BrowserPreprocessingSettings settings, CancellationToken cancellationToken = default);
        Task<string> ResampleForOcrAsync(string imageBase64, int targetDpi, CancellationToken cancellationToken = default);
    }
}
