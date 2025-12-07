#!/usr/bin/env python3
"""
Receipt OCR CLI

Command-line interface for processing receipt images with OCR and structured extraction.

This is the main entry point. The CLI logic is split into modular components:
- src/cli/args.py: Argument parsing
- src/cli/commands.py: Command implementations
- src/cli/utils.py: Utility functions

Commands:
- process: Full pipeline (preprocess + OCR + inference)
- preprocess: Run preprocessing only (before resampling)
- ocr: Run OCR on preprocessed image (includes resampling)
- inference: Run model inference on OCR results
- version: Show version information
"""

import json
import sys
import logging

# Import required functions for tests
from src.receipt_processor import (
    get_device,
    load_image,
    preprocess_image,
    normalize_boxes,
    extract_fields_heuristic,
    process_receipt,
    run_ocr,
    run_model_inference_standalone,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Bardcoded Python OCR CLI")


def main():
    """Main entry point for CLI."""
    # Import from modular components
    from src.cli.args import parse_args
    from src.cli.commands import (
        version_command, 
        preprocess_command,
        ocr_command,
        inference_command
    )
    
    args, parser = parse_args()
    
    if args.command == "preprocess":
        try:
            result = preprocess_command(
                input_image=args.input_image,
                output_path=args.output,
                output_format=args.output_format,
                job_id=args.job_id,
                verbose=args.verbose,
                log_level=args.log_level,
                denoise=args.denoise,
                deskew=args.deskew,
                fuzz_percent=args.fuzz_percent,
                deskew_threshold=args.deskew_threshold,
                contrast_type=args.contrast_type,
                contrast_strength=args.contrast_strength,
                contrast_midpoint=args.contrast_midpoint,
                apply_threshold=getattr(args, 'apply_threshold', False),
                threshold_percent=getattr(args, 'threshold_percent', 50)
            )
            
            if not args.output:
                print(json.dumps(result, indent=2))
            
            sys.exit(0 if result.get("status") == "done" else 1)
        except Exception as e:
            print(f"Error during preprocessing: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "ocr":
        try:
            result = ocr_command(
                input_image=args.input_image,
                output_path=args.output,
                job_id=args.job_id,
                verbose=args.verbose,
                log_level=args.log_level,
                ocr_engine=args.ocr_engine,
                target_dpi=args.target_dpi,
                device=args.device
            )
            
            if not args.output:
                print(json.dumps(result, indent=2))
            
            sys.exit(0 if result.get("status") == "done" else 1)
        except Exception as e:
            print(f"Error during OCR: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "inference":
        try:
            result = inference_command(
                ocr_result_path=args.ocr_result_path,
                input_image=args.input_image,
                output_path=args.output,
                job_id=args.job_id,
                verbose=args.verbose,
                log_level=args.log_level,
                model=args.model,
                device=args.device
            )
            
            if not args.output:
                print(json.dumps(result, indent=2))
            
            sys.exit(0 if result.get("status") == "done" else 1)
        except Exception as e:
            print(f"Error during inference: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "version":
        version_command()
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
