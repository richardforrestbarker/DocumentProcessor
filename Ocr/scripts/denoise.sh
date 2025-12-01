#!/bin/bash
# Apply denoising using ImageMagick.
# Uses ImageMagick's enhance filter which reduces noise while preserving edges.
#
# Usage: ./denoise.sh <input_image> <output_image>
#
# Example:
#   ./denoise.sh receipt.tiff receipt_denoised.tiff
#
# ImageMagick command:
#   magick <input> -enhance <output>

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

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_image> <output_image>"
    echo "Example: $0 receipt.tiff receipt_denoised.tiff"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Apply enhance filter for noise reduction
# -enhance: Apply a digital filter to reduce noise
magick "$INPUT" -enhance "$OUTPUT"

echo "Applied denoising: $OUTPUT"
