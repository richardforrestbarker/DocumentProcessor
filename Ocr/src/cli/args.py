"""
CLI argument parsing module.

Handles command-line argument parsing for the Receipt OCR CLI.
"""

import argparse
from typing import Tuple


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add common device argument to a parser."""
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for inference (default: auto)"
    )


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Explicit log level override"
    )


def _add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common preprocessing arguments to a parser."""
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply denoising preprocessing"
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Apply deskewing preprocessing"
    )
    parser.add_argument(
        "--fuzz-percent",
        type=int,
        default=30,
        metavar="0-100",
        help="Fuzz percentage for background removal (default: 30). Higher values remove more background colors."
    )
    parser.add_argument(
        "--deskew-threshold",
        type=int,
        default=40,
        metavar="0-100",
        help="Deskew threshold percentage (default: 40). Lower values are more aggressive at detecting skew."
    )
    parser.add_argument(
        "--contrast-type",
        choices=["sigmoidal", "linear", "none"],
        default="sigmoidal",
        help="""Contrast enhancement type (default: sigmoidal).
Types:
  sigmoidal: Non-linear S-curve contrast (best for most images)
  linear: Simple histogram stretch using -auto-level only
  none: Skip contrast enhancement"""
    )
    parser.add_argument(
        "--contrast-strength",
        type=float,
        default=3,
        metavar="1-10",
        help="Contrast strength for sigmoidal type (default: 3). Higher values increase contrast more aggressively."
    )
    parser.add_argument(
        "--contrast-midpoint",
        type=int,
        default=120,
        metavar="0-200",
        help="Contrast midpoint percentage for sigmoidal type (default: 120). Values >100 brighten, <100 darken."
    )
    parser.add_argument(
        "--apply-threshold",
        action="store_true",
        help="Apply ImageMagick threshold after contrast step"
    )
    parser.add_argument(
        "--threshold-percent",
        type=int,
        default=50,
        metavar="0-100",
        help="Threshold percentage (default: 50). Used when --apply-threshold is set."
    )


def _add_ocr_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OCR-related arguments including resampling and DPI safety checks."""
    parser.add_argument(
        "--ocr-engine",
        choices=["paddle", "tesseract"],
        default="paddle",
        help="OCR engine to use (default: paddle)"
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=300,
        help="Target DPI for image resampling (default: 300)"
    )
    _add_device_argument(parser)


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model inference arguments."""
    parser.add_argument(
        "--model",
        "-m",
        default="naver-clova-ix/donut-base-finetuned-cord-v2",
        help="Model name or path (default: naver-clova-ix/donut-base-finetuned-cord-v2)"
    )
    _add_device_argument(parser)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Process receipt images with OCR and structured extraction",
        prog="receipt-ocr"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command (full pipeline)
    process_parser = subparsers.add_parser("process", help="Process receipt images (full pipeline)")
    process_parser.add_argument(
        "--image",
        "-i",
        action="append",
        required=True,
        dest="images",
        help="Path to receipt image (can be specified multiple times for multi-page receipts)"
    )
    process_parser.add_argument(
        "--output",
        "-o",
        help="Path to write JSON output (prints to stdout if not specified)"
    )
    process_parser.add_argument(
        "--model",
        "-m",
        default="naver-clova-ix/donut-base-finetuned-cord-v2",
        help="Model name or path (default: naver-clova-ix/donut-base-finetuned-cord-v2)"
    )
    process_parser.add_argument(
        "--ocr-engine",
        choices=["paddle", "tesseract"],
        default="paddle",
        help="OCR engine to use (default: paddle)"
    )
    process_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for inference (default: auto)"
    )
    process_parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply denoising preprocessing"
    )
    process_parser.add_argument(
        "--deskew",
        action="store_true",
        help="Apply deskewing preprocessing"
    )
    process_parser.add_argument(
        "--job-id",
        help="Job identifier for tracking"
    )
    process_parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip LayoutLM model inference (use only heuristic extraction)"
    )
    _add_logging_arguments(process_parser)
    process_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save intermediary images for each processing step (denoise, deskew, grayscale, OCR bounding boxes, result bounding boxes)"
    )
    process_parser.add_argument(
        "--debug-output-dir",
        help="Directory to save debug output files (default: ./debug_output)"
    )
    process_parser.add_argument(
        "--fuzz-percent",
        type=int,
        default=30,
        metavar="0-100",
        help="Fuzz percentage for background removal (default: 30). Higher values remove more background colors."
    )
    process_parser.add_argument(
        "--deskew-threshold",
        type=int,
        default=40,
        metavar="0-100",
        help="Deskew threshold percentage (default: 40). Lower values are more aggressive at detecting skew."
    )
    process_parser.add_argument(
        "--contrast-type",
        choices=["sigmoidal", "linear", "none"],
        default="sigmoidal",
        help="""Contrast enhancement type (default: sigmoidal).
Types:
  sigmoidal: Non-linear S-curve contrast (best for most images)
  linear: Simple histogram stretch using -auto-level only
  none: Skip contrast enhancement"""
    )
    process_parser.add_argument(
        "--contrast-strength",
        type=float,
        default=3,
        metavar="1-10",
        help="Contrast strength for sigmoidal type (default: 3). Higher values increase contrast more aggressively."
    )
    process_parser.add_argument(
        "--contrast-midpoint",
        type=int,
        default=120,
        metavar="0-200",
        help="Contrast midpoint percentage for sigmoidal type (default: 120). Values >100 brighten, <100 darken."
    )
    process_parser.add_argument(
        "--apply-threshold",
        action="store_true",
        help="Apply ImageMagick threshold after contrast step"
    )
    process_parser.add_argument(
        "--threshold-percent",
        type=int,
        default=50,
        metavar="0-100",
        help="Threshold percentage (default: 50). Used when --apply-threshold is set."
    )
    
    # Preprocess command (preprocessing only, before resampling)
    preprocess_parser = subparsers.add_parser(
        "preprocess", 
        help="Run preprocessing only (deskew, contrast, grayscale, background removal, denoise). Does NOT include resampling/DPI adjustment."
    )
    preprocess_parser.add_argument(
        "--image",
        "-i",
        required=True,
        dest="input_image",
        help="Path to source image"
    )
    preprocess_parser.add_argument(
        "--output",
        "-o",
        help="Path to write preprocessed result JSON. If not specified, outputs base64 encoded image to stdout"
    )
    preprocess_parser.add_argument(
        "--output-format",
        choices=["file", "base64"],
        default="base64",
        help="Output format: 'file' saves to --output path, 'base64' outputs base64 encoded string (default: base64)"
    )
    preprocess_parser.add_argument(
        "--job-id",
        help="Job identifier for tracking"
    )
    _add_logging_arguments(preprocess_parser)
    _add_preprocessing_arguments(preprocess_parser)
    
    # OCR command (resampling + OCR)
    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Run OCR on a preprocessed image (includes resampling and DPI safety checks)"
    )
    ocr_parser.add_argument(
        "--image",
        "-i",
        required=True,
        dest="input_image",
        help="Path to preprocessed image"
    )
    ocr_parser.add_argument(
        "--output",
        "-o",
        help="Path to write JSON output (prints to stdout if not specified)"
    )
    ocr_parser.add_argument(
        "--job-id",
        help="Job identifier for tracking"
    )
    _add_logging_arguments(ocr_parser)
    _add_ocr_arguments(ocr_parser)
    
    # Inference command (model inference)
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run model inference on OCR results to extract structured fields"
    )
    inference_parser.add_argument(
        "--ocr-result",
        required=True,
        dest="ocr_result_path",
        help="Path to OCR result JSON file from the 'ocr' command"
    )
    inference_parser.add_argument(
        "--image",
        "-i",
        required=True,
        dest="input_image",
        help="Path to original or preprocessed image for visual features"
    )
    inference_parser.add_argument(
        "--output",
        "-o",
        help="Path to write JSON output (prints to stdout if not specified)"
    )
    inference_parser.add_argument(
        "--job-id",
        help="Job identifier for tracking"
    )
    _add_logging_arguments(inference_parser)
    _add_inference_arguments(inference_parser)
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    return parser


def parse_args(args=None) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    """
    Parse command-line arguments.
    
    Args:
        args: Optional list of arguments (uses sys.argv if None)
        
    Returns:
        Tuple of (parsed arguments namespace, parser instance)
    """
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)
    return parsed_args, parser
