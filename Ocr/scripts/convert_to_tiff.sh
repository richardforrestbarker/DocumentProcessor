#!/bin/bash
# Convert image to TIFF format for optimal Tesseract OCR processing.
# TIFF is the preferred format for Tesseract as it provides lossless compression.
#
# Usage: ./convert_to_tiff.sh <input_image> <output_image>
#
# Example:
#   ./convert_to_tiff.sh receipt.jpg receipt.tiff
#
# ImageMagick command:
#   magick <input> -compress lzw <output.tiff>

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
    echo "Example: $0 receipt.jpg receipt.tiff"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Convert to TIFF with LZW compression for smaller file size while maintaining quality
magick "$INPUT" -compress lzw "$OUTPUT"

echo "Converted $INPUT to TIFF format: $OUTPUT"
