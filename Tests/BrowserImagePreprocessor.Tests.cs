using DocumentProcessor.Wasm.Preprocessing;
using System;
using System.IO;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace DocumentProcessor.Tests
{
    public class BrowserImagePreprocessorTests
    {
        [Fact]
        public async Task PreprocessAsync_ReturnsImageResult_WithDimensions()
        {
            var preprocessor = new BrowserImagePreprocessor();
            var input = BuildPngBase64(18, 12, new Rgba32(240, 240, 240, 255), new Rgba32(30, 30, 30, 255));

            var result = await preprocessor.PreprocessAsync(input, new BrowserPreprocessingSettings
            {
                Deskew = false,
                ApplyThreshold = false,
                Denoise = false
            });

            Assert.Equal(18, result.Width);
            Assert.Equal(12, result.Height);
            Assert.False(string.IsNullOrWhiteSpace(result.ImageBase64));
            Assert.Contains(result.ImageFormat, new[] { "tiff", "png" });
            Assert.Contains(result.ImageMimeType, new[] { "image/tiff", "image/png" });
        }

        [Fact]
        public void ResampleToTargetDpi_ResamplesToRequestedDpi_WhenSafe()
        {
            using var image = new Image<Rgba32>(20, 10, new Rgba32(255, 255, 255, 255));
            BrowserImagePreprocessor.ResampleToTargetDpi(image, currentDpi: 72, targetDpi: 300);
            Assert.True(image.Width > 20);
            Assert.True(image.Height > 10);
            Assert.Equal(300, (int)Math.Round(image.Metadata.HorizontalResolution));
            Assert.Equal(300, (int)Math.Round(image.Metadata.VerticalResolution));
        }

        private static string BuildPngBase64(int width, int height, Rgba32 background, Rgba32 foreground, float horizontalDpi = 300, float verticalDpi = 300)
        {
            using var image = new Image<Rgba32>(width, height, background);
            for (var y = 2; y < height - 2; y++)
            {
                for (var x = 2; x < width - 2; x++)
                {
                    image[x, y] = foreground;
                }
            }

            image.Metadata.HorizontalResolution = horizontalDpi;
            image.Metadata.VerticalResolution = verticalDpi;

            using var ms = new MemoryStream();
            image.Save(ms, new PngEncoder());
            return Convert.ToBase64String(ms.ToArray());
        }
    }
}
