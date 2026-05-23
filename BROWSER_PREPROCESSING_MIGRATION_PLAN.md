# Browser-Side Image Preprocessing Migration Plan (Blazor WebAssembly)

## Goal
Move all **image manipulation before text extraction** from server-side Python/ImageMagick to Blazor WebAssembly C# so preprocessing happens in the browser, reducing API CPU load and removing ImageMagick dependency.

## Current Manipulated Attributes (Before Text Extraction)
From the current preprocessing + OCR flow, these attributes are manipulated before OCR text extraction:

1. **Deskew enabled/disabled** (`Deskew`)
2. **Deskew sensitivity** (`DeskewThreshold`)
3. **Grayscale conversion** (always applied)
4. **Background removal tolerance** (`FuzzPercent`)
5. **Auto-level histogram stretch** (always in background removal; also used in contrast)
6. **Contrast mode** (`ContrastType`: `sigmoidal` | `linear` | `none`)
7. **Contrast strength** (`ContrastStrength`)
8. **Contrast midpoint** (`ContrastMidpoint`)
9. **Threshold enabled/disabled** (`ApplyThreshold`)
10. **Threshold cutoff** (`ThresholdPercent`)
11. **Denoise enabled/disabled** (`Denoise`)
12. **Format conversion** (to TIFF with LZW today)
13. **DPI resampling before OCR** (`TargetDpi`) with max-dimension safety checks

## Proposed Browser Architecture
1. Add a browser preprocessor service in `Wasm` (C# only):
   - `Wasm/Preprocessing/BrowserImagePreprocessor.cs`
   - `Wasm/Preprocessing/BrowserPreprocessingModels.cs`
2. Keep existing `PreprocessingRequest` contract for UI controls.
3. During preprocessing phase in `DocumentProcessingView.razor`, call browser preprocessor directly.
4. Send resulting preprocessed image base64 to `/api/document/ocr`.
5. Temporarily keep server `/api/document/preprocess` behind a feature flag for rollback.

## Pipeline to Run in Browser
Run in this order for parity with current behavior:
1. Deskew (optional)
2. Grayscale
3. Background removal (+ auto-level)
4. Contrast enhancement (optional)
5. Threshold (optional)
6. Denoise (optional)
7. Encode output (TIFF LZW preferred, PNG fallback)
8. OCR-stage DPI resample safety check + resize (before OCR call)

---

## Exact C# Code for Each Manipulated Attribute

### 1) Settings model (maps current request knobs)
```csharp
public sealed class BrowserPreprocessingSettings
{
    public bool Deskew { get; set; } = true;
    public int DeskewThreshold { get; set; } = 40; // 0-100
    public bool Denoise { get; set; } = false;
    public int FuzzPercent { get; set; } = 30; // 0-100
    public string ContrastType { get; set; } = "sigmoidal"; // sigmoidal|linear|none
    public double ContrastStrength { get; set; } = 3.0;
    public int ContrastMidpoint { get; set; } = 120; // 0-200
    public bool ApplyThreshold { get; set; } = false;
    public int ThresholdPercent { get; set; } = 50; // 0-100
    public int TargetDpi { get; set; } = 300;
}
```

### 2) Core processor skeleton
```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats.Tiff;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public sealed class BrowserImagePreprocessor
{
    private const int TesseractMaxDimension = 32767;
    private const long PillowMaxPixelsEquivalent = 178_956_970L;

    public async Task<byte[]> PreprocessAsync(byte[] inputBytes, BrowserPreprocessingSettings s)
    {
        using var image = await Image.LoadAsync<Rgba32>(inputBytes);

        if (s.Deskew)
        {
            var angle = EstimateSkewAngle(image, s.DeskewThreshold);
            image.Mutate(ctx => ctx.Rotate(-angle));
        }

        image.Mutate(ctx => ctx.Grayscale());

        ApplyBackgroundRemoval(image, s.FuzzPercent);
        ApplyAutoLevel(image);

        if (!string.Equals(s.ContrastType, "none", StringComparison.OrdinalIgnoreCase))
        {
            ApplyContrast(image, s.ContrastType, s.ContrastStrength, s.ContrastMidpoint);
        }

        if (s.ApplyThreshold)
        {
            ApplyThreshold(image, s.ThresholdPercent);
        }

        if (s.Denoise)
        {
            image.Mutate(ctx => ctx.MedianBlur(1));
        }

        return await EncodeTiffLzwOrPngAsync(image);
    }
```

### 3) Deskew + DeskewThreshold
```csharp
    private static float EstimateSkewAngle(Image<Rgba32> img, int deskewThreshold)
    {
        // Keep semantics aligned with current UI guidance:
        // lower threshold => more aggressive skew detection.
        // Invert 0-100 into 255-0 for binarization cutoff.
        byte binThreshold = (byte)Math.Clamp(255.0 - (deskewThreshold / 100.0) * 255.0, 0, 255);

        float bestAngle = 0f;
        double bestScore = double.MinValue;

        for (float angle = -5f; angle <= 5f; angle += 0.25f)
        {
            using var clone = img.Clone(c => c.Rotate(angle));
            var score = ProjectionVarianceScore(clone, binThreshold);
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
        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            int dark = 0;
            for (int x = 0; x < row.Length; x++)
            {
                if (row[x].R < threshold) dark++;
            }
            sums[y] = dark;
        }

        var mean = sums.Average();
        return sums.Select(v => (v - mean) * (v - mean)).Average();
    }
```

### 4) Background removal + FuzzPercent
```csharp
    private static void ApplyBackgroundRemoval(Image<Rgba32> img, int fuzzPercent)
    {
        // Fuzz defines how close to white a pixel can be before forced to white.
        int tolerance = (int)Math.Round(Math.Clamp(fuzzPercent, 0, 100) * 2.55);

        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                ref var p = ref row[x];
                bool nearWhite = (255 - p.R) <= tolerance && (255 - p.G) <= tolerance && (255 - p.B) <= tolerance;
                if (nearWhite)
                {
                    p = new Rgba32(255, 255, 255, 255);
                }
            }
        }
    }
```

### 5) Auto-level (histogram stretch)
```csharp
    private static void ApplyAutoLevel(Image<Rgba32> img)
    {
        byte min = 255;
        byte max = 0;

        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                byte v = row[x].R;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        if (max <= min) return;
        float scale = 255f / (max - min);

        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                byte v = row[x].R;
                byte nv = (byte)Math.Clamp((v - min) * scale, 0, 255);
                row[x] = new Rgba32(nv, nv, nv, 255);
            }
        }
    }
```

### 6) ContrastType + ContrastStrength + ContrastMidpoint
```csharp
    private static void ApplyContrast(Image<Rgba32> img, string type, double strength, int midpointPercent)
    {
        ApplyAutoLevel(img); // preserve current behavior

        if (string.Equals(type, "linear", StringComparison.OrdinalIgnoreCase))
        {
            ApplyLinearContrast(img, strength);
            return;
        }

        // Sigmoidal mapping: output = 1 / (1 + exp(-gain*(input-mid)))
        // midpointPercent maps 0-200 -> 0.0-1.0 (100 => 0.5 center).
        double mid = Math.Clamp(midpointPercent / 200.0, 0.0, 1.0);
        double gain = Math.Clamp(strength, 0.1, 20.0);

        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                double input = row[x].R / 255.0;
                double output = 1.0 / (1.0 + Math.Exp(-gain * (input - mid)));
                byte nv = (byte)Math.Clamp((int)Math.Round(output * 255.0), 0, 255);
                row[x] = new Rgba32(nv, nv, nv, 255);
            }
        }
    }

    private static void ApplyLinearContrast(Image<Rgba32> img, double strength)
    {
        // strength=1 keeps identity; >1 increases separation around mid-gray.
        double factor = Math.Clamp(strength, 0.1, 10.0);

        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                double normalized = row[x].R / 255.0;
                double centered = normalized - 0.5;
                double stretched = centered * factor;
                byte nv = (byte)Math.Clamp((int)Math.Round((stretched + 0.5) * 255.0), 0, 255);
                row[x] = new Rgba32(nv, nv, nv, 255);
            }
        }
    }
```

### 7) ApplyThreshold + ThresholdPercent
```csharp
    private static void ApplyThreshold(Image<Rgba32> img, int thresholdPercent)
    {
        byte t = (byte)Math.Clamp((thresholdPercent / 100.0) * 255.0, 0, 255);

        for (int y = 0; y < img.Height; y++)
        {
            var row = img.GetPixelRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                byte v = row[x].R >= t ? (byte)255 : (byte)0;
                row[x] = new Rgba32(v, v, v, 255);
            }
        }
    }
```

### 8) Denoise
```csharp
// inside PreprocessAsync after threshold
if (s.Denoise)
{
    image.Mutate(ctx => ctx.MedianBlur(1));
}
```

### 9) TIFF conversion (LZW) with PNG fallback
```csharp
    private static async Task<byte[]> EncodeTiffLzwOrPngAsync(Image<Rgba32> image)
    {
        try
        {
            await using var ms = new MemoryStream();
            await image.SaveAsync(ms, new TiffEncoder { Compression = TiffCompression.Lzw });
            return ms.ToArray();
        }
        catch
        {
            await using var ms = new MemoryStream();
            await image.SaveAsync(ms, new PngEncoder());
            return ms.ToArray();
        }
    }
```

### 10) DPI resampling + safety limits (TargetDpi)
```csharp
    public static void ResampleToTargetDpi(Image<Rgba32> image, int currentDpi, int targetDpi)
    {
        currentDpi = currentDpi <= 0 ? 72 : currentDpi;
        double scale = targetDpi / (double)currentDpi;

        int newW = (int)Math.Round(image.Width * scale);
        int newH = (int)Math.Round(image.Height * scale);
        long totalPixels = (long)newW * newH;

        if (newW > TesseractMaxDimension || newH > TesseractMaxDimension || totalPixels > PillowMaxPixelsEquivalent)
        {
            return; // keep original size when unsafe
        }

        image.Mutate(ctx => ctx.Resize(newW, newH, KnownResamplers.Bicubic));
        image.Metadata.HorizontalResolution = targetDpi;
        image.Metadata.VerticalResolution = targetDpi;
    }
}
```

---

## Integration Plan (No Implementation Yet)
1. Add `IBrowserImagePreprocessor` service and register in `Wasm/Extensions.cs`.
2. Update `DocumentProcessingView.razor` preprocessing phase:
   - Build `BrowserPreprocessingSettings` from existing `settings`.
   - Run browser preprocessor locally.
   - Render preview from local output bytes (base64).
3. Keep `PreprocessingRequest` DTO for compatibility until API preprocess endpoint is retired.
4. Update `RunOcr` request path to submit browser-processed image directly.
5. Introduce `UseBrowserPreprocessing` feature flag in `Wasm/wwwroot/appsettings.json`.
6. Add telemetry counters:
   - Preprocess duration (browser)
   - Payload size to OCR endpoint
   - Server preprocess endpoint call count (target: 0)
7. Gradually disable server preprocess endpoint after parity metrics are met.

## Validation Plan
1. **Parity checks:** same input image + same settings should produce OCR text with comparable confidence/word count.
2. **Visual checks:** compare current vs browser output for deskew, thresholding, and noise cleanup.
3. **Performance checks:** measure API CPU time and request/response volume before/after migration.
4. **Safety checks:** enforce max upload size and image dimension guardrails in browser and API.

## Risks and Mitigations
1. **Deskew parity drift:** keep threshold mapping and angle search deterministic; add regression image set.
2. **WASM memory pressure on large images:** downscale preview copies and stream decode where possible.
3. **TIFF codec compatibility in browser runtime:** fallback to PNG and keep OCR endpoint format-agnostic.
4. **Behavior differences from ImageMagick auto-level/contrast:** lock formulas and compare against baseline fixtures.

## Rollout Phases
1. **Phase 1:** Add browser preprocessor behind flag; keep server preprocess default.
2. **Phase 2:** Enable browser preprocess for internal users; compare OCR quality and API load.
3. **Phase 3:** Make browser preprocess default; keep server endpoint as rollback.
4. **Phase 4:** Remove ImageMagick dependency and retire server preprocessing path.

## Expected Outcome
- Preprocessing CPU shifts from server to browser.
- `/api/document/preprocess` usage drops to near-zero.
- ImageMagick installation/runtime dependency can be removed.
- OCR/inference remain server-hosted while receiving cleaner, user-tuned images from clients.
