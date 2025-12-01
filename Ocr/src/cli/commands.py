"""
CLI command implementations.

Implements the main commands for the Receipt OCR CLI.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from .utils import check_dependencies, get_device, load_image, setup_logging, get_image_dimensions

logger = logging.getLogger(__name__)

# Version information
VERSION = "1.0.0"


def process_command(
    image_paths: List[str],
    output_path: Optional[str] = None,
    model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
    model_type: str = "donut",
    ocr_engine: str = "paddle",
    device: str = "auto",
    denoise: bool = False,
    deskew: bool = False,
    job_id: Optional[str] = None,
    skip_model: bool = False,
    verbose: bool = False,
    debug: bool = False,
    debug_output_dir: Optional[str] = None,
    fuzz_percent: int = 30,
    deskew_threshold: int = 40,
    contrast_type: str = "sigmoidal",
    contrast_strength: float = 3,
    contrast_midpoint: int = 120
) -> dict:
    """
    Process receipt images and extract structured data.
    
    Args:
        image_paths: List of paths to receipt image files
        output_path: Optional path to write JSON output
        model_name: Model name or path for the selected model type
        model_type: Model type to use ('donut', 'idefics2', or 'layoutlmv3')
        ocr_engine: OCR engine to use ('paddle' or 'tesseract')
        device: Device for inference ('auto', 'cuda', or 'cpu')
        denoise: Apply denoising preprocessing
        deskew: Apply deskewing preprocessing
        job_id: Optional job identifier
        skip_model: Skip model inference and use only heuristics
        verbose: Enable verbose logging
        debug: Enable debug mode to save intermediary images
        debug_output_dir: Directory for debug output files
        fuzz_percent: Fuzz percentage for background removal (0-100)
        deskew_threshold: Deskew threshold percentage (0-100)
        contrast_type: Contrast enhancement type ('sigmoidal', 'linear', or 'none')
        contrast_strength: Contrast strength for sigmoidal type (1-10 typical)
        contrast_midpoint: Contrast midpoint percentage for sigmoidal type
        
    Returns:
        Dictionary containing extracted receipt data
    """
    setup_logging(verbose)
    check_dependencies()
    
    logger.info(f"Processing {len(image_paths)} image(s)...")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Model: {model_name}")
    logger.info(f"OCR Engine: {ocr_engine}")
    if debug:
        logger.info("Debug mode enabled - saving intermediary images")
    
    # Resolve device
    actual_device = get_device(device)
    logger.info(f"Using device: {actual_device}")
    
    # Import processing modules
    from ..preprocessing.image_preprocessor import ImagePreprocessor
    from ..ocr.ocr_engine import create_ocr_engine
    from ..postprocessing.field_extractor import FieldExtractor
    
    # Initialize debug output manager if debug mode is enabled
    debug_manager = None
    if debug:
        from .debug_output import DebugOutputManager
        effective_job_id = job_id or f"job-{hash(tuple(image_paths)) % 100000:05d}"
        debug_output_directory = debug_output_dir or "./debug_output"
        debug_manager = DebugOutputManager(output_dir=debug_output_directory, job_id=effective_job_id)
    
    # Initialize components with debug support
    preprocessor = ImagePreprocessor(
        denoise=denoise,
        deskew=deskew,
        enhance_contrast=True,
        debug_manager=debug_manager,
        fuzz_percent=fuzz_percent,
        deskew_threshold=deskew_threshold,
        contrast_type=contrast_type,
        contrast_strength=contrast_strength,
        contrast_midpoint=contrast_midpoint
    )
    ocr = create_ocr_engine(ocr_engine, use_gpu=(actual_device == "cuda"))
    field_extractor = FieldExtractor()
    
    # Initialize result
    result = {
        "job_id": job_id or f"job-{hash(tuple(image_paths)) % 100000:05d}",
        "status": "done",
        "pages": [],
        "vendor_name": None,
        "merchant_address": None,
        "date": None,
        "total_amount": None,
        "subtotal": None,
        "tax_amount": None,
        "currency": None,
        "line_items": []
    }
    
    all_words = []
    source_images = []  # Store source images for debug output
    
    try:
        for page_num, image_path in enumerate(image_paths):
            logger.info(f"Processing page {page_num + 1}: {image_path}")
            
            
            # Preprocess image (with debug output if enabled)
            (processed_image, img_width, img_height) = preprocessor.preprocess(image_path, page_num=page_num + 1)
            
            # Run OCR
            words = ocr.detect_and_recognize(processed_image)
            logger.info(f"OCR detected {len(words)} text regions")
            
            # Save OCR bounding boxes if debug mode is enabled
            if debug and debug_manager:
                debug_manager.save_ocr_bounding_boxes(
                    processed_image, words, page_num + 1, ocr_engine
                )
            
            # Normalize boxes to 0-1000 scale
            normalized_words = normalize_boxes(words, img_width, img_height)
            
            # Build raw OCR text
            raw_text = ' '.join(w['text'] for w in words)
            
            # Add page result
            result["pages"].append({
                "page_number": page_num + 1,
                "raw_ocr_text": raw_text,
                "words": [
                    {
                        "text": w['text'],
                        "box": {
                            "x0": w['box'][0],
                            "y0": w['box'][1],
                            "x1": w['box'][2],
                            "y1": w['box'][3]
                        },
                        "confidence": w['confidence']
                    }
                    for w in normalized_words
                ]
            })
            
            all_words.extend(normalized_words)
        
        # Try model inference if not skipped
        model_predictions = None
        if not skip_model:
            try:
                from ..models import get_model
                
                logger.info(f"Running {model_type} model inference...")
                model = get_model(
                    model_type,
                    model_name_or_path=model_name,
                    device=actual_device
                )
                model.load()
                
                # Get first page image for visual features
                first_image = load_image(image_paths[0])
                
                # Prepare tokens and boxes for models that need them
                tokens = [w['text'] for w in all_words] if all_words else []
                boxes = [w['box'] for w in all_words] if all_words else []
                
                # Run prediction
                # Donut and IDEFICS2 can work without OCR words
                # LayoutLMv3 requires OCR words
                if model_type == "layoutlmv3" and not all_words:
                    logger.warning("LayoutLMv3 requires OCR words. Skipping model inference.")
                else:
                    model_result = model.predict_from_words(
                        words=tokens,
                        boxes=boxes,
                        image=first_image
                    )
                    
                    if model_result.get("entities"):
                        model_predictions = model_result["entities"]
                        logger.info(f"Model extracted entities: {list(model_predictions.keys())}")
                
            except Exception as e:
                logger.warning(f"{model_type} inference failed: {e}. Using heuristic extraction.")
        
        # Extract fields using heuristics (and model predictions if available)
        if all_words:
            # Extract vendor name
            vendor = field_extractor.extract_vendor_name(all_words, model_predictions)
            if vendor:
                result["vendor_name"] = vendor
            
            # Extract date
            date = extract_date_field(all_words)
            if date:
                result["date"] = date
            
            # Extract total
            total = field_extractor.extract_total(all_words, model_predictions)
            if total:
                result["total_amount"] = total
            
            # Extract subtotal
            subtotal = extract_subtotal_field(all_words)
            if subtotal:
                result["subtotal"] = subtotal
            
            # Extract tax
            tax = extract_tax_field(all_words)
            if tax:
                result["tax_amount"] = tax
            
            # Detect currency
            currency = detect_currency(all_words)
            if currency:
                result["currency"] = currency
            
            # Extract line items
            line_items = field_extractor.extract_line_items(all_words, model_predictions)
            result["line_items"] = line_items
        
        # Save result bounding boxes for debug mode
        if debug and debug_manager and source_images:
            for page_num, source_image in enumerate(source_images):
                debug_manager.save_result_bounding_boxes(source_image, result, page_num + 1)
            debug_manager.save_debug_summary(result)
    
    except Exception as e:
        logger.error(f"Error processing receipt: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    # Write output if specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {output_path}")
    
    return result


def version_command() -> None:
    """Display version information."""
    print(f"Receipt OCR Service v{VERSION}")
    print("Models: Donut (default), IDEFICS2, LayoutLMv3")
    print("OCR: PaddleOCR, Tesseract (fallback)")
    
    # Check available dependencies
    deps = []
    
    try:
        import paddleocr
        deps.append("PaddleOCR: Available")
    except ImportError:
        deps.append("PaddleOCR: Not installed")
    
    try:
        import pytesseract
        deps.append("Tesseract: Available")
    except ImportError:
        deps.append("Tesseract: Not installed")
    
    try:
        import torch
        cuda_status = "Available" if torch.cuda.is_available() else "Not available"
        deps.append(f"PyTorch: {torch.__version__}")
        deps.append(f"CUDA: {cuda_status}")
    except ImportError:
        deps.append("PyTorch: Not installed")
    
    try:
        import transformers
        deps.append(f"Transformers: {transformers.__version__}")
    except ImportError:
        deps.append("Transformers: Not installed")
    
    print("\nDependencies:")
    for dep in deps:
        print(f"  - {dep}")
    
    print("\nSupported Models:")
    print("  - Donut (naver-clova-ix/donut-base-finetuned-cord-v2) - MIT license")
    print("  - IDEFICS2 (HuggingFaceM4/idefics2-8b) - Apache 2.0 license")
    print("  - LayoutLMv3 (microsoft/layoutlmv3-base) - requires OCR")


def normalize_boxes(
    words: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    scale: int = 1000
) -> List[Dict[str, Any]]:
    """
    Normalize bounding boxes to 0-1000 scale for LayoutLM.
    
    Args:
        words: List of words with boxes
        image_width: Image width
        image_height: Image height
        scale: Normalization scale (default 1000)
        
    Returns:
        Words with normalized boxes
    """
    normalized = []
    for word in words:
        box = word['box']
        normalized_box = [
            int(box[0] * scale / image_width),
            int(box[1] * scale / image_height),
            int(box[2] * scale / image_width),
            int(box[3] * scale / image_height)
        ]
        # Clamp values to valid range
        normalized_box = [max(0, min(scale, x)) for x in normalized_box]
        
        normalized.append({
            'text': word['text'],
            'box': normalized_box,
            'confidence': word['confidence']
        })
    
    return normalized


def extract_date_field(words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract date from words using regex patterns."""
    import re
    
    full_text = ' '.join(w['text'] for w in words)
    
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Find the word containing the date
            for w in words:
                if date_str in w['text'] or w['text'] in date_str:
                    return {
                        "value": date_str,
                        "confidence": w['confidence'],
                        "box": {
                            "x0": w['box'][0],
                            "y0": w['box'][1],
                            "x1": w['box'][2],
                            "y1": w['box'][3]
                        }
                    }
    
    return None


def extract_subtotal_field(words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract subtotal amount from words."""
    import re
    
    amount_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    subtotal_keywords = ['subtotal', 'sub total', 'sub-total']
    
    for i, w in enumerate(words):
        text_lower = w['text'].lower()
        if any(kw in text_lower for kw in subtotal_keywords):
            for j in range(max(0, i-2), min(len(words), i+5)):
                match = re.search(amount_pattern, words[j]['text'])
                if match:
                    return {
                        "value": match.group(1).replace(',', ''),
                        "confidence": words[j]['confidence'],
                        "box": {
                            "x0": words[j]['box'][0],
                            "y0": words[j]['box'][1],
                            "x1": words[j]['box'][2],
                            "y1": words[j]['box'][3]
                        }
                    }
    
    return None


def extract_tax_field(words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract tax amount from words."""
    import re
    
    amount_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    tax_keywords = ['tax', 'vat', 'gst', 'hst']
    
    for i, w in enumerate(words):
        text_lower = w['text'].lower()
        if any(kw in text_lower for kw in tax_keywords):
            for j in range(max(0, i-2), min(len(words), i+5)):
                match = re.search(amount_pattern, words[j]['text'])
                if match:
                    return {
                        "value": match.group(1).replace(',', ''),
                        "confidence": words[j]['confidence'],
                        "box": {
                            "x0": words[j]['box'][0],
                            "y0": words[j]['box'][1],
                            "x1": words[j]['box'][2],
                            "y1": words[j]['box'][3]
                        }
                    }
    
    return None


def detect_currency(words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detect currency from words."""
    full_text = ' '.join(w['text'] for w in words)
    
    if '$' in full_text or 'USD' in full_text:
        return {"value": "USD", "confidence": 0.9, "box": None}
    elif '€' in full_text or 'EUR' in full_text:
        return {"value": "EUR", "confidence": 0.9, "box": None}
    elif '£' in full_text or 'GBP' in full_text:
        return {"value": "GBP", "confidence": 0.9, "box": None}
    elif '¥' in full_text or 'JPY' in full_text or 'CNY' in full_text:
        return {"value": "JPY/CNY", "confidence": 0.8, "box": None}
    elif 'CAD' in full_text:
        return {"value": "CAD", "confidence": 0.9, "box": None}
    elif 'AUD' in full_text:
        return {"value": "AUD", "confidence": 0.9, "box": None}
    
    return None


def preprocess_command(
    input_image: str,
    output_path: Optional[str] = None,
    output_format: str = "base64",
    job_id: Optional[str] = None,
    verbose: bool = False,
    denoise: bool = False,
    deskew: bool = False,
    fuzz_percent: int = 30,
    deskew_threshold: int = 40,
    contrast_type: str = "sigmoidal",
    contrast_strength: float = 3,
    contrast_midpoint: int = 120
) -> dict:
    """
    Run preprocessing only on an image (before resampling step).
    
    This command runs: deskew, contrast enhancement, grayscale, background removal, denoise.
    It does NOT include resampling/DPI adjustment - those are part of the OCR phase.
    
    Args:
        input_image: Path to source image
        output_path: Path to write preprocessed image (for file output)
        output_format: 'file' or 'base64'
        job_id: Optional job identifier
        verbose: Enable verbose logging
        denoise: Apply denoising preprocessing
        deskew: Apply deskewing preprocessing
        fuzz_percent: Fuzz percentage for background removal
        deskew_threshold: Deskew threshold percentage
        contrast_type: Contrast enhancement type
        contrast_strength: Contrast strength for sigmoidal
        contrast_midpoint: Contrast midpoint for sigmoidal
        
    Returns:
        Dictionary containing preprocessing result with base64 image or file path
    """
    import base64
    import tempfile
    import os
    from io import BytesIO
    
    setup_logging(verbose)
    check_dependencies()
    
    logger.info(f"Preprocessing image: {input_image}")
    
    from ..preprocessing.image_preprocessor import ImagePreprocessor
    
    # Create preprocessor without DPI resampling (target_dpi=None or skip_resample=True)
    preprocessor = ImagePreprocessor(
        target_dpi=None,  # Skip DPI resampling - that's for OCR phase
        denoise=denoise,
        deskew=deskew,
        enhance_contrast=(contrast_type != "none"),
        fuzz_percent=fuzz_percent,
        deskew_threshold=deskew_threshold,
        contrast_type=contrast_type,
        contrast_strength=contrast_strength,
        contrast_midpoint=contrast_midpoint
    )
    
    effective_job_id = job_id or f"preprocess-{hash(input_image) % 100000:05d}"
    
    result = {
        "job_id": effective_job_id,
        "status": "done",
        "input_image": input_image,
        "preprocessing_settings": {
            "denoise": denoise,
            "deskew": deskew,
            "fuzz_percent": fuzz_percent,
            "deskew_threshold": deskew_threshold,
            "contrast_type": contrast_type,
            "contrast_strength": contrast_strength,
            "contrast_midpoint": contrast_midpoint
        }
    }
    
    try:
        # Run preprocessing (without resampling)
        processed_image, width, height = preprocessor.preprocess(input_image, page_num=1)
        
        result["width"] = width
        result["height"] = height
        
        from PIL import Image
        import numpy as np
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(processed_image.astype(np.uint8))
        
        if output_format == "base64":
            # Encode as base64
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            result["image_base64"] = base64_data
            result["image_format"] = "png"
            logger.info(f"Preprocessing complete. Base64 encoded image generated.")
            
        elif output_format == "file":
            if output_path:
                pil_image.save(output_path)
                result["output_path"] = output_path
                logger.info(f"Preprocessing complete. Image saved to: {output_path}")
            else:
                # Save to temp file
                fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_image.save(temp_path)
                result["output_path"] = temp_path
                logger.info(f"Preprocessing complete. Image saved to: {temp_path}")
                
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


def ocr_command(
    input_image: str,
    output_path: Optional[str] = None,
    job_id: Optional[str] = None,
    verbose: bool = False,
    ocr_engine: str = "paddle",
    target_dpi: int = 300,
    device: str = "auto"
) -> dict:
    """
    Run OCR on a preprocessed image (includes resampling and DPI safety checks).
    
    This command is meant to be run on the output of the preprocess command.
    It includes resampling to target DPI and safety checks before running OCR.
    
    Args:
        input_image: Path to preprocessed image
        output_path: Path to write JSON output
        job_id: Optional job identifier
        verbose: Enable verbose logging
        ocr_engine: OCR engine to use ('paddle' or 'tesseract')
        target_dpi: Target DPI for resampling
        device: Device for OCR ('auto', 'cuda', or 'cpu')
        
    Returns:
        Dictionary containing OCR results with words and bounding boxes
    """
    setup_logging(verbose)
    check_dependencies()
    
    logger.info(f"Running OCR on image: {input_image}")
    logger.info(f"OCR Engine: {ocr_engine}")
    
    actual_device = get_device(device)
    logger.info(f"Using device: {actual_device}")
    
    from ..ocr.ocr_engine import create_ocr_engine
    from ..preprocessing.image_preprocessor import ImagePreprocessor
    
    effective_job_id = job_id or f"ocr-{hash(input_image) % 100000:05d}"
    
    result = {
        "job_id": effective_job_id,
        "status": "done",
        "input_image": input_image,
        "ocr_engine": ocr_engine,
        "words": [],
        "raw_ocr_text": ""
    }
    
    try:
        # Create a minimal preprocessor just for resampling/DPI adjustment
        # This is separate from the preprocessing phase
        preprocessor = ImagePreprocessor(
            target_dpi=target_dpi,
            denoise=False,  # Already done in preprocess phase
            deskew=False,   # Already done in preprocess phase
            enhance_contrast=False  # Already done in preprocess phase
        )
        
        # Load and resample the image (DPI safety checks are in _find_safe_dpi)
        processed_image, width, height = preprocessor.preprocess(input_image, page_num=1)
        
        # Create OCR engine
        ocr = create_ocr_engine(ocr_engine, use_gpu=(actual_device == "cuda"))
        
        # Run OCR
        words = ocr.detect_and_recognize(processed_image)
        logger.info(f"OCR detected {len(words)} text regions")
        
        # Normalize boxes to 0-1000 scale
        normalized_words = normalize_boxes(words, width, height)
        
        # Build raw OCR text
        raw_text = ' '.join(w['text'] for w in words)
        
        result["words"] = [
            {
                "text": w['text'],
                "box": {
                    "x0": w['box'][0],
                    "y0": w['box'][1],
                    "x1": w['box'][2],
                    "y1": w['box'][3]
                },
                "confidence": w['confidence']
            }
            for w in normalized_words
        ]
        result["raw_ocr_text"] = raw_text
        result["image_width"] = width
        result["image_height"] = height
        
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    # Write output if specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {output_path}")
    
    return result


def inference_command(
    ocr_result_path: str,
    input_image: str,
    output_path: Optional[str] = None,
    job_id: Optional[str] = None,
    verbose: bool = False,
    model: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
    model_type: str = "donut",
    device: str = "auto"
) -> dict:
    """
    Run model inference on OCR results to extract structured fields.
    
    Args:
        ocr_result_path: Path to OCR result JSON file from 'ocr' command
        input_image: Path to original or preprocessed image for visual features
        output_path: Path to write JSON output
        job_id: Optional job identifier
        verbose: Enable verbose logging
        model: Model name or path
        model_type: Model type ('donut', 'idefics2', 'layoutlmv3')
        device: Device for inference
        
    Returns:
        Dictionary containing extracted receipt fields
    """
    setup_logging(verbose)
    check_dependencies()
    
    logger.info(f"Running inference with model: {model_type} ({model})")
    
    actual_device = get_device(device)
    logger.info(f"Using device: {actual_device}")
    
    from ..postprocessing.field_extractor import FieldExtractor
    
    # Load OCR results
    with open(ocr_result_path, 'r') as f:
        ocr_result = json.load(f)
    
    effective_job_id = job_id or ocr_result.get("job_id") or f"inference-{hash(input_image) % 100000:05d}"
    
    result = {
        "job_id": effective_job_id,
        "status": "done",
        "input_image": input_image,
        "ocr_result_path": ocr_result_path,
        "model_type": model_type,
        "vendor_name": None,
        "merchant_address": None,
        "date": None,
        "total_amount": None,
        "subtotal": None,
        "tax_amount": None,
        "currency": None,
        "line_items": []
    }
    
    try:
        # Load the image for visual features
        first_image = load_image(input_image)
        
        # Extract words from OCR result
        all_words = ocr_result.get("words", [])
        
        # Convert word format if needed
        normalized_words = []
        for w in all_words:
            box = w.get("box", {})
            normalized_words.append({
                'text': w.get('text', ''),
                'box': [box.get('x0', 0), box.get('y0', 0), box.get('x1', 0), box.get('y1', 0)],
                'confidence': w.get('confidence', 0)
            })
        
        field_extractor = FieldExtractor()
        model_predictions = None
        
        # Try model inference
        try:
            from ..models import get_model
            
            logger.info(f"Running {model_type} model inference...")
            model_obj = get_model(
                model_type,
                model_name_or_path=model,
                device=actual_device
            )
            model_obj.load()
            
            # Prepare tokens and boxes for models that need them
            tokens = [w['text'] for w in normalized_words] if normalized_words else []
            boxes = [w['box'] for w in normalized_words] if normalized_words else []
            
            # Run prediction
            if model_type == "layoutlmv3" and not normalized_words:
                logger.warning("LayoutLMv3 requires OCR words. Skipping model inference.")
            else:
                model_result = model_obj.predict_from_words(
                    words=tokens,
                    boxes=boxes,
                    image=first_image
                )
                
                if model_result.get("entities"):
                    model_predictions = model_result["entities"]
                    logger.info(f"Model extracted entities: {list(model_predictions.keys())}")
                    
        except Exception as e:
            logger.warning(f"{model_type} inference failed: {e}. Using heuristic extraction.")
        
        # Extract fields using heuristics (and model predictions if available)
        if normalized_words:
            # Extract vendor name
            vendor = field_extractor.extract_vendor_name(normalized_words, model_predictions)
            if vendor:
                result["vendor_name"] = vendor
            
            # Extract date
            date = extract_date_field(normalized_words)
            if date:
                result["date"] = date
            
            # Extract total
            total = field_extractor.extract_total(normalized_words, model_predictions)
            if total:
                result["total_amount"] = total
            
            # Extract subtotal
            subtotal = extract_subtotal_field(normalized_words)
            if subtotal:
                result["subtotal"] = subtotal
            
            # Extract tax
            tax = extract_tax_field(normalized_words)
            if tax:
                result["tax_amount"] = tax
            
            # Detect currency
            currency = detect_currency(normalized_words)
            if currency:
                result["currency"] = currency
            
            # Extract line items
            line_items = field_extractor.extract_line_items(normalized_words, model_predictions)
            result["line_items"] = line_items
            
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
    
    # Write output if specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {output_path}")
    
    return result
