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


def version_command() -> None:
    """Display version information."""
    print(f"Document OCR Service v{VERSION}")
    print("Models: Donut (default), IDEFICS2, Phi-3-Vision, InternVL, Qwen2-VL")
    print("All models have commercial-friendly licenses (MIT or Apache 2.0)")
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
    
    print("\nSupported Models (all commercially licensed):")
    print("  - Donut (naver-clova-ix/donut-base-finetuned-cord-v2) - MIT license, receipt-optimized")
    print("  - IDEFICS2 (HuggingFaceM4/idefics2-8b) - Apache 2.0 license, multi-document support")
    print("  - Phi-3-Vision (microsoft/Phi-3-vision-128k-instruct) - MIT license, efficient")
    print("  - InternVL (OpenGVLab/InternVL2-8B) - MIT license, high accuracy")
    print("  - Qwen2-VL (Qwen/Qwen2-VL-7B-Instruct) - Apache 2.0 license, strong performance")


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
                            "y0": w['box'][1],
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
    log_level: Optional[str] = None,
    denoise: bool = False,
    deskew: bool = False,
    fuzz_percent: int = 30,
    deskew_threshold: int = 40,
    contrast_type: str = "sigmoidal",
    contrast_strength: float = 3,
    contrast_midpoint: int = 120,
    apply_threshold: bool = False,
    threshold_percent: int = 50
) -> dict:
    """
    Run preprocessing only on an image (before resampling step).
    Pipeline ends at TIFF conversion, does NOT include DPI resampling.
    """
    import base64
    import tempfile
    import os
    from io import BytesIO

    setup_logging(verbose, log_level)
    check_dependencies()

    logger.info(json.dumps({
        "event": "preprocess_start",
        "input_image": input_image,
        "job_id": job_id,
        "output_path": output_path,
        "output_format": output_format,
        "settings": {
            "denoise": denoise,
            "deskew": deskew,
            "fuzz_percent": fuzz_percent,
            "deskew_threshold": deskew_threshold,
            "contrast_type": contrast_type,
            "contrast_strength": contrast_strength,
            "contrast_midpoint": contrast_midpoint,
            "apply_threshold": apply_threshold,
            "threshold_percent": threshold_percent
        }
    }))

    from ..preprocessing.image_preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor(
        target_dpi=None,
        denoise=denoise,
        deskew=deskew,
        enhance_contrast=(contrast_type != "none"),
        fuzz_percent=fuzz_percent,
        deskew_threshold=deskew_threshold,
        contrast_type=contrast_type,
        contrast_strength=contrast_strength,
        contrast_midpoint=contrast_midpoint
    )

    # Set threshold options on the preprocessor if supported
    if hasattr(preprocessor, 'apply_threshold'):
        setattr(preprocessor, 'apply_threshold', apply_threshold)
    if hasattr(preprocessor, 'threshold_percent'):
        setattr(preprocessor, 'threshold_percent', threshold_percent)

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
            "contrast_midpoint": contrast_midpoint,
            "apply_threshold": apply_threshold,
            "threshold_percent": threshold_percent
        }
    }

    try:
        logger.info(json.dumps({
            "event": "preprocessing_image",
            "job_id": effective_job_id,
            "message": "Applying image preprocessing filters"
        }))
        processed_image, width, height = preprocessor.preprocess(input_image, page_num=1)
        result["width"] = width
        result["height"] = height

        from PIL import Image
        import numpy as np
        pil_image = Image.fromarray(processed_image.astype(np.uint8))
        if output_format == "base64":
            from io import BytesIO
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            result["image_base64"] = base64_data
            result["image_format"] = "png"
        elif output_path:
            pil_image.save(output_path)
            result["output_path"] = output_path

        logger.info(json.dumps({
            "event": "preprocess_complete",
            "job_id": effective_job_id,
            "width": width,
            "height": height,
            "output_format": output_format,
            "output_path": result.get("output_path")
        }))
    except Exception as e:
        logger.error(json.dumps({
            "event": "preprocess_error",
            "job_id": effective_job_id,
            "error": str(e)
        }))
        result["status"] = "failed"
        result["error"] = str(e)

    # Always honor output_path for JSON result if provided
    if output_path and output_format == "base64":
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {output_path}")

    return result


def ocr_command(
    input_image: str,
    output_path: Optional[str] = None,
    job_id: Optional[str] = None,
    verbose: bool = False,
    log_level: Optional[str] = None,
    ocr_engine: str = "paddle",
    target_dpi: int = 300,
    device: str = "auto"
) -> dict:
    """
    Run OCR on a preprocessed image (now only calls DPI resampling, not full preprocessing).
    """
    setup_logging(verbose, log_level)
    check_dependencies()

    logger.info(json.dumps({
        "event": "ocr_start",
        "input_image": input_image,
        "job_id": job_id,
        "ocr_engine": ocr_engine,
        "target_dpi": target_dpi,
        "device": device
    }))

    actual_device = get_device(device)
    from ..ocr.ocr_engine import create_ocr_engine
    from ..preprocessing.image_preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor(target_dpi=target_dpi)

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
        logger.info(json.dumps({
            "event": "resampling_image",
            "job_id": effective_job_id,
            "message": "Resampling image to target DPI"
        }))
        processed_image, width, height = preprocessor.resampleToDpi(input_image, target_dpi)
        
        logger.info(json.dumps({
            "event": "running_ocr_engine",
            "job_id": effective_job_id,
            "message": f"Running {ocr_engine} OCR engine"
        }))
        ocr = create_ocr_engine(ocr_engine, use_gpu=(actual_device == "cuda"))
        words = ocr.detect_and_recognize(processed_image)
        normalized_words = normalize_boxes(words, width, height)
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

        logger.info(json.dumps({
            "event": "ocr_complete",
            "job_id": effective_job_id,
            "image_width": width,
            "image_height": height,
            "word_count": len(result["words"])
        }))
    except Exception as e:
        logger.error(json.dumps({
            "event": "ocr_error",
            "job_id": effective_job_id,
            "error": str(e)
        }))
        result["status"] = "failed"
        result["error"] = str(e)
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
    log_level: Optional[str] = None,
    model: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
    device: str = "auto"
) -> dict:
    """
    Run model inference on OCR results to extract structured fields.
    """
    setup_logging(verbose, log_level)
    check_dependencies()
    
    logger.info(json.dumps({
        "event": "inference_start",
        "input_image": input_image,
        "job_id": job_id,
        "ocr_result_path": ocr_result_path,
        "model": model,
        "device": device
    }))
    
    # Restrict supported models
    supported_models = {
        "naver-clova-ix/donut-base-finetuned-cord-v2",
        "HuggingFaceM4/idefics2-8b",
    }
    if model not in supported_models:
        err = {
            "event": "model_error",
            "job_id": job_id,
            "error": f"Unsupported model '{model}'. Supported models are: {', '.join(sorted(supported_models))}"
        }
        logger.warning(json.dumps(err))
        return {
            "job_id": job_id or f"inference-{hash(input_image) % 100000:05d}",
            "status": "failed",
            "input_image": input_image,
            "ocr_result_path": ocr_result_path,
            "model": model,
            "error": err["error"]
        }

    actual_device = get_device(device)
    from ..postprocessing.field_extractor import FieldExtractor
    from ..models import get_model, LayoutLMv3Model
    
    with open(ocr_result_path, 'r') as f:
        ocr_result = json.load(f)
    
    effective_job_id = job_id or ocr_result.get("job_id") or f"inference-{hash(input_image) % 100000:05d}"
    
    result = {
        "job_id": effective_job_id,
        "status": "done",
        "input_image": input_image,
        "ocr_result_path": ocr_result_path,
        "model": model,
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
        first_image = load_image(input_image)
        all_words = ocr_result.get("words", [])
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
        
        try:
            logger.info(json.dumps({
                "event": "loading_model",
                "job_id": effective_job_id,
                "message": f"Loading model {model}"
            }))
            model_obj = get_model(model_name_or_path=model, device=actual_device)
            model_obj.load()
            tokens = [w['text'] for w in normalized_words] if normalized_words else []
            boxes = [w['box'] for w in normalized_words] if normalized_words else []
            # If LayoutLMv3 and words are missing, warn user and skip model inference
            if isinstance(model_obj, LayoutLMv3Model) and not normalized_words:
                logger.warning("LayoutLMv3 requires OCR words. Skipping model inference.")
            else:
                logger.info(json.dumps({
                    "event": "running_inference",
                    "job_id": effective_job_id,
                    "message": "Running model inference"
                }))
                model_result = model_obj.predict_from_words(
                    words=tokens,
                    boxes=boxes,
                    image=first_image
                )
                if model_result.get("entities"):
                    model_predictions = model_result["entities"]
                    logger.info(json.dumps({
                        "event": "model_entities",
                        "job_id": effective_job_id,
                        "entities": list(model_predictions.keys())
                    }))
        except Exception as e:
            logger.warning(json.dumps({
                "event": "model_error",
                "job_id": effective_job_id,
                "error": str(e)
            }))
        
        if normalized_words:
            logger.info(json.dumps({
                "event": "extracting_fields",
                "job_id": effective_job_id,
                "message": "Extracting structured fields"
            }))
            vendor = field_extractor.extract_vendor_name(normalized_words, model_predictions)
            if vendor:
                result["vendor_name"] = vendor
            date = extract_date_field(normalized_words)
            if date:
                result["date"] = date
            total = field_extractor.extract_total(normalized_words, model_predictions)
            if total:
                result["total_amount"] = total
            subtotal = extract_subtotal_field(normalized_words)
            if subtotal:
                result["subtotal"] = subtotal
            tax = extract_tax_field(normalized_words)
            if tax:
                result["tax_amount"] = tax
            currency = detect_currency(normalized_words)
            if currency:
                result["currency"] = currency
            line_items = field_extractor.extract_line_items(normalized_words, model_predictions)
            result["line_items"] = line_items
        
        logger.info(json.dumps({
            "event": "inference_complete",
            "job_id": effective_job_id,
            "has_vendor": result["vendor_name"] is not None,
            "has_total": result["total_amount"] is not None,
            "line_items": len(result["line_items"])
        }))
    except Exception as e:
        logger.error(json.dumps({
            "event": "inference_error",
            "job_id": effective_job_id,
            "error": str(e)
        }))
        result["status"] = "failed"
        result["error"] = str(e)
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {output_path}")
    
    return result
