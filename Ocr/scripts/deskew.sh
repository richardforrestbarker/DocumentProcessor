#!/bin/bash
# Detect and correct image skew using ImageMagick.
# Uses ImageMagick's deskew function which analyzes the image and rotates it to straighten text lines.
#
# Usage: ./deskew.sh <input_image> <output_image> [threshold]
#
# Example:
#   ./deskew.sh receipt.tiff receipt_deskewed.tiff 40
#
# The threshold parameter (0-100) controls the sensitivity of deskew detection.
# Lower values are more aggressive. Default is 40%.
#
# ImageMagick command:
#   magick <input> -deskew 40% -background white +repage <output>

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
    echo "Usage: $0 <input_image> <output_image> [threshold]"
    echo "Example: $0 receipt.tiff receipt_deskewed.tiff 40"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
THRESHOLD="${3:-40}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Apply deskew with threshold
# -deskew: Straighten the image
# -background white: Fill any new pixels from rotation with white
# +repage: Reset the virtual canvas to eliminate negative offsets (fixes TIFF issues)
magick "$INPUT" \
    -deskew "${THRESHOLD}%" \
    -background white \
    +repage \
    "$OUTPUT"

echo "Applied deskew correction (threshold: ${THRESHOLD}%): $OUTPUT"
