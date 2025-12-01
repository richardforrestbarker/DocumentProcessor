#!/bin/bash
# Enhance image contrast using ImageMagick.
# Uses a combination of auto-level and sigmoidal contrast for optimal text visibility.
#
# Usage: ./enhance_contrast.sh <input_image> <output_image> [contrast_strength]
#
# Example:
#   ./enhance_contrast.sh receipt.tiff receipt_contrast.tiff 3
#
# ImageMagick commands:
#   # Auto-level stretches the histogram
#   # Sigmoidal contrast applies non-linear contrast enhancement
#   magick <input> -auto-level -sigmoidal-contrast 3x50% <output>

set -e

# Check if ImageMagick is installed
if ! command -v magick &> /dev/null; then
    echo "Error: ImageMagick is not installed."
    echo "Please install ImageMagick:"
    echo "  Ubuntu/Debian: sudo apt-get install imagemagick"
    echo "  macOS: brew install imagemagick"
    echo "  Windows: Download from https://imagemagick.org/script/download.php"
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_image> <output_image> [contrast_strength]"
    echo "Example: $0 receipt.tiff receipt_contrast.tiff 3"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
CONTRAST="${3:-3}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Enhance contrast:
# -auto-level: Stretch the histogram to full range
# -sigmoidal-contrast: Apply S-curve contrast enhancement
#   Format: strength x midpoint%
#   strength=3 is moderate, midpoint=50% targets middle tones
magick "$INPUT" \
    -auto-level \
    -sigmoidal-contrast "${CONTRAST}x50%" \
    "$OUTPUT"

echo "Enhanced contrast (strength: ${CONTRAST}): $OUTPUT"
