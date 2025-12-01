#!/bin/bash
# Remove background from image to isolate text.
# Uses ImageMagick's background removal capabilities to clean up the image.
#
# Usage: ./remove_background.sh <input_image> <output_image> [fuzz_percent]
#
# Example:
#   ./remove_background.sh receipt.tiff receipt_nobg.tiff 10
#
# ImageMagick commands:
#   # Remove white background with fuzz tolerance
#   magick <input> -fuzz 10% -transparent white -background white -alpha remove -auto-level <output>

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
    echo "Usage: $0 <input_image> <output_image> [fuzz_percent]"
    echo "Example: $0 receipt.tiff receipt_nobg.tiff 10"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
FUZZ_PERCENT="${3:-10}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' does not exist"
    exit 1
fi

# Remove white/light background:
# -fuzz: Tolerance for color matching (10% handles slight variations)
# -transparent white: Make white pixels transparent
# -background white: Set background color for flatten
# -alpha remove: Remove alpha channel by flattening to background
# -auto-level: Stretch histogram to improve contrast
magick "$INPUT" \
    -fuzz "${FUZZ_PERCENT}%" \
    -transparent white \
    -background white \
    -alpha remove \
    -auto-level \
    "$OUTPUT"

echo "Removed background from image: $OUTPUT"
