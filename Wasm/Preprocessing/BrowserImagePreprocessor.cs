using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats.Tiff;
using SixLabors.ImageSharp.Formats.Tiff.Constants;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DocumentProcessor.Wasm.Preprocessing
{
    public sealed class BrowserImagePreprocessor : IBrowserImagePreprocessor
    {
        private const int TesseractMaxDimension = 32767;
        private const long PillowMaxPixelsEquivalent = 178_956_970L;

        public async Task<BrowserPreprocessingResult> PreprocessAsync(string imageBase64, BrowserPreprocessingSettings settings, CancellationToken cancellationToken = default)
        {
            var inputBytes = Convert.FromBase64String(imageBase64);
            using var preprocessStream = new MemoryStream(inputBytes);
            using var image = await Image.LoadAsync<Rgba32>(preprocessStream, cancellationToken);

            EnsureSafeImageDimensions(image.Width, image.Height);

            if (settings.Deskew)
            {
                var angle = EstimateSkewAngle(image, settings.DeskewThreshold);
                if (Math.Abs(angle) > 0.01f)
                {
                    image.Mutate(ctx => ctx.Rotate(-angle));
                }
            }

            image.Mutate(ctx => ctx.Grayscale());
            ApplyBackgroundRemoval(image, settings.FuzzPercent);
            ApplyAutoLevel(image);

            if (!string.Equals(settings.ContrastType, "none", StringComparison.OrdinalIgnoreCase))
            {
                ApplyContrast(image, settings.ContrastType, settings.ContrastStrength, settings.ContrastMidpoint);
            }

            if (settings.ApplyThreshold)
            {
                ApplyThreshold(image, settings.ThresholdPercent);
            }

            if (settings.Denoise)
            {
                image.Mutate(ctx => ctx.GaussianBlur(0.5f));
            }

            var encodedImage = await EncodeTiffLzwOrPngAsync(image, cancellationToken);

            return new BrowserPreprocessingResult
            {
                ImageBase64 = Convert.ToBase64String(encodedImage.Bytes),
                ImageFormat = encodedImage.Format,
                ImageMimeType = encodedImage.MimeType,
                Width = image.Width,
                Height = image.Height
            };
        }

        public async Task<string> ResampleForOcrAsync(string imageBase64, int targetDpi, CancellationToken cancellationToken = default)
        {
            var inputBytes = Convert.FromBase64String(imageBase64);
            using var resampleStream = new MemoryStream(inputBytes);
            using var image = await Image.LoadAsync<Rgba32>(resampleStream, cancellationToken);

            var currentDpi = image.Metadata.HorizontalResolution > 0 ? (int)Math.Round(image.Metadata.HorizontalResolution) : 72;
            ResampleToTargetDpi(image, currentDpi, targetDpi);

            var output = await EncodeTiffLzwOrPngAsync(image, cancellationToken);
            return Convert.ToBase64String(output.Bytes);
        }

        private static float EstimateSkewAngle(Image<Rgba32> img, int deskewThreshold)
        {
            var threshold = (byte)Math.Clamp(255.0 - (Math.Clamp(deskewThreshold, 0, 100) / 100.0) * 255.0, 0, 255);
            float bestAngle = 0f;
            var bestScore = double.MinValue;

            for (float angle = -5f; angle <= 5f; angle += 0.25f)
            {
                using var clone = img.Clone(ctx => ctx.Rotate(angle));
                var score = ProjectionVarianceScore(clone, threshold);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestAngle = angle;
                }
            }

            return bestAngle;
        }

        private static double ProjectionVarianceScore(Image<Rgba32> img, byte threshold)
        {
            var sums = new double[img.Height];
            for (var y = 0; y < img.Height; y++)
            {
                var dark = 0;
                for (var x = 0; x < img.Width; x++)
                {
                    if (img[x, y].R < threshold)
                    {
                        dark++;
                    }
                }

                sums[y] = dark;
            }

            var mean = sums.Average();
            return sums.Select(v => (v - mean) * (v - mean)).Average();
        }

        private static void ApplyBackgroundRemoval(Image<Rgba32> img, int fuzzPercent)
        {
            var tolerance = (int)Math.Round(Math.Clamp(fuzzPercent, 0, 100) * 2.55);

            for (var y = 0; y < img.Height; y++)
            {
                for (var x = 0; x < img.Width; x++)
                {
                    var p = img[x, y];
                    var nearWhite = (255 - p.R) <= tolerance && (255 - p.G) <= tolerance && (255 - p.B) <= tolerance;
                    if (nearWhite)
                    {
                        img[x, y] = new Rgba32(255, 255, 255, 255);
                    }
                }
            }
        }

        private static void ApplyAutoLevel(Image<Rgba32> img)
        {
            byte min = 255;
            byte max = 0;

            for (var y = 0; y < img.Height; y++)
            {
                for (var x = 0; x < img.Width; x++)
                {
                    var v = img[x, y].R;
                    if (v < min)
                    {
                        min = v;
                    }

                    if (v > max)
                    {
                        max = v;
                    }
                }
            }

            if (max <= min)
            {
                return;
            }

            var scale = 255f / (max - min);

            for (var y = 0; y < img.Height; y++)
            {
                for (var x = 0; x < img.Width; x++)
                {
                    var v = img[x, y].R;
                    var nv = (byte)Math.Clamp((v - min) * scale, 0, 255);
                    img[x, y] = new Rgba32(nv, nv, nv, 255);
                }
            }
        }

        private static void ApplyContrast(Image<Rgba32> img, string type, double strength, int midpointPercent)
        {
            ApplyAutoLevel(img);

            if (string.Equals(type, "linear", StringComparison.OrdinalIgnoreCase))
            {
                ApplyLinearContrast(img, strength);
                return;
            }

            var mid = Math.Clamp(midpointPercent / 200.0, 0.0, 1.0);
            var gain = Math.Clamp(strength, 0.1, 20.0);

            for (var y = 0; y < img.Height; y++)
            {
                for (var x = 0; x < img.Width; x++)
                {
                    var input = img[x, y].R / 255.0;
                    var output = 1.0 / (1.0 + Math.Exp(-gain * (input - mid)));
                    var nv = (byte)Math.Clamp((int)Math.Round(output * 255.0), 0, 255);
                    img[x, y] = new Rgba32(nv, nv, nv, 255);
                }
            }
        }

        private static void ApplyLinearContrast(Image<Rgba32> img, double strength)
        {
            var factor = Math.Clamp(strength, 0.1, 10.0);

            for (var y = 0; y < img.Height; y++)
            {
                for (var x = 0; x < img.Width; x++)
                {
                    var normalized = img[x, y].R / 255.0;
                    var centered = normalized - 0.5;
                    var stretched = centered * factor;
                    var nv = (byte)Math.Clamp((int)Math.Round((stretched + 0.5) * 255.0), 0, 255);
                    img[x, y] = new Rgba32(nv, nv, nv, 255);
                }
            }
        }

        private static void ApplyThreshold(Image<Rgba32> img, int thresholdPercent)
        {
            var threshold = (byte)Math.Clamp((Math.Clamp(thresholdPercent, 0, 100) / 100.0) * 255.0, 0, 255);

            for (var y = 0; y < img.Height; y++)
            {
                for (var x = 0; x < img.Width; x++)
                {
                    var value = img[x, y].R >= threshold ? (byte)255 : (byte)0;
                    img[x, y] = new Rgba32(value, value, value, 255);
                }
            }
        }

        private static async Task<(byte[] Bytes, string Format, string MimeType)> EncodeTiffLzwOrPngAsync(Image<Rgba32> image, CancellationToken cancellationToken)
        {
            try
            {
                await using var ms = new MemoryStream();
                await image.SaveAsync(ms, new TiffEncoder { Compression = TiffCompression.Lzw }, cancellationToken);
                return (ms.ToArray(), "tiff", "image/tiff");
            }
            catch
            {
                await using var ms = new MemoryStream();
                await image.SaveAsync(ms, new PngEncoder(), cancellationToken);
                return (ms.ToArray(), "png", "image/png");
            }
        }

        public static void ResampleToTargetDpi(Image<Rgba32> image, int currentDpi, int targetDpi)
        {
            currentDpi = currentDpi <= 0 ? 72 : currentDpi;
            targetDpi = targetDpi <= 0 ? 300 : targetDpi;

            var scale = targetDpi / (double)currentDpi;
            var newWidth = (int)Math.Round(image.Width * scale);
            var newHeight = (int)Math.Round(image.Height * scale);
            var totalPixels = (long)newWidth * newHeight;

            if (newWidth <= 0 || newHeight <= 0)
            {
                return;
            }

            if (newWidth > TesseractMaxDimension || newHeight > TesseractMaxDimension || totalPixels > PillowMaxPixelsEquivalent)
            {
                return;
            }

            image.Mutate(ctx => ctx.Resize(newWidth, newHeight, KnownResamplers.Bicubic));
            image.Metadata.HorizontalResolution = targetDpi;
            image.Metadata.VerticalResolution = targetDpi;
        }

        private static void EnsureSafeImageDimensions(int width, int height)
        {
            if (width <= 0 || height <= 0)
            {
                throw new InvalidOperationException("Invalid image dimensions.");
            }

            var pixels = (long)width * height;
            if (width > TesseractMaxDimension || height > TesseractMaxDimension || pixels > PillowMaxPixelsEquivalent)
            {
                throw new InvalidOperationException("Image dimensions exceed safe limits.");
            }
        }
    }
}
