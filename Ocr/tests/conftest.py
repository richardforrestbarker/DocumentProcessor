"""
Pytest configuration and shared fixtures for OCR tests.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


@pytest.fixture
def sample_receipt_image():
    """Create a simple synthetic receipt image for testing."""
    # Create a white image with black text-like patterns
    img = Image.new('RGB', (400, 600), color='white')
    return np.array(img)


@pytest.fixture
def sample_receipt_image_path(sample_receipt_image, tmp_path):
    """Save sample receipt image to a temporary file and return path."""
    img_path = tmp_path / "receipt.jpg"
    Image.fromarray(sample_receipt_image).save(img_path)
    return str(img_path)


@pytest.fixture
def sample_words():
    """Sample OCR word results for testing."""
    return [
        {'text': 'GROCERY', 'box': [100, 50, 300, 100], 'confidence': 0.98},
        {'text': 'STORE', 'box': [100, 100, 250, 150], 'confidence': 0.96},
        {'text': 'Date:', 'box': [50, 200, 120, 230], 'confidence': 0.95},
        {'text': '01/15/2024', 'box': [130, 200, 280, 230], 'confidence': 0.94},
        {'text': 'Milk', 'box': [50, 300, 150, 330], 'confidence': 0.97},
        {'text': '$3.99', 'box': [300, 300, 380, 330], 'confidence': 0.96},
        {'text': 'Bread', 'box': [50, 350, 150, 380], 'confidence': 0.95},
        {'text': '$2.50', 'box': [300, 350, 380, 380], 'confidence': 0.94},
        {'text': 'Subtotal', 'box': [50, 450, 180, 480], 'confidence': 0.97},
        {'text': '$6.49', 'box': [300, 450, 380, 480], 'confidence': 0.95},
        {'text': 'Tax', 'box': [50, 490, 100, 520], 'confidence': 0.96},
        {'text': '$0.52', 'box': [300, 490, 380, 520], 'confidence': 0.94},
        {'text': 'Total', 'box': [50, 540, 130, 570], 'confidence': 0.98},
        {'text': '$7.01', 'box': [300, 540, 380, 570], 'confidence': 0.97},
    ]


@pytest.fixture
def sample_normalized_words():
    """Sample normalized words (0-1000 scale) for testing."""
    return [
        {'text': 'GROCERY', 'box': [250, 83, 750, 166], 'confidence': 0.98},
        {'text': 'STORE', 'box': [250, 166, 625, 250], 'confidence': 0.96},
        {'text': 'Total', 'box': [125, 900, 325, 950], 'confidence': 0.98},
        {'text': '$7.01', 'box': [750, 900, 950, 950], 'confidence': 0.97},
    ]


@pytest.fixture
def sample_ocr_result():
    """Sample complete OCR result for testing."""
    return {
        "job_id": "test-job-001",
        "status": "done",
        "pages": [
            {
                "page_number": 1,
                "raw_ocr_text": "GROCERY STORE Date: 01/15/2024 Milk $3.99 Total $7.01",
                "words": [
                    {"text": "GROCERY", "box": {"x0": 250, "y0": 83, "x1": 750, "y1": 166}, "confidence": 0.98},
                    {"text": "Total", "box": {"x0": 125, "y0": 900, "x1": 325, "y1": 950}, "confidence": 0.98},
                ]
            }
        ],
        "vendor_name": {"value": "GROCERY STORE", "confidence": 0.97, "box": {"x0": 250, "y0": 83, "x1": 750, "y1": 250}},
        "date": {"value": "01/15/2024", "confidence": 0.94, "box": {"x0": 130, "y0": 200, "x1": 280, "y1": 230}},
        "total_amount": {"value": "7.01", "confidence": 0.97, "box": {"x0": 300, "y0": 540, "x1": 380, "y1": 570}},
        "subtotal": None,
        "tax_amount": None,
        "currency": {"value": "USD", "confidence": 0.9, "box": None},
        "line_items": []
    }


@pytest.fixture
def sample_ocr_result_file(sample_ocr_result, tmp_path):
    """Save sample OCR result to a temporary file and return path."""
    result_path = tmp_path / "ocr_result.json"
    with open(result_path, 'w') as f:
        json.dump(sample_ocr_result, f)
    return str(result_path)


@pytest.fixture
def sample_ocr_command_result():
    """Sample result from the ocr command."""
    return {
        "job_id": "test-ocr-001",
        "status": "done",
        "input_image": "/path/to/image.png",
        "ocr_engine": "paddle",
        "words": [
            {"text": "GROCERY", "box": {"x0": 250, "y0": 83, "x1": 750, "y1": 166}, "confidence": 0.98},
            {"text": "Total", "box": {"x0": 125, "y0": 900, "x1": 325, "y1": 950}, "confidence": 0.98},
            {"text": "$7.01", "box": {"x0": 750, "y0": 900, "x1": 950, "y1": 950}, "confidence": 0.97},
        ],
        "raw_ocr_text": "GROCERY Total $7.01",
        "image_width": 400,
        "image_height": 600
    }


@pytest.fixture
def sample_ocr_command_result_file(sample_ocr_command_result, tmp_path):
    """Save sample OCR command result to a temporary file and return path."""
    result_path = tmp_path / "ocr_command_result.json"
    with open(result_path, 'w') as f:
        json.dump(sample_ocr_command_result, f)
    return str(result_path)


@pytest.fixture
def mock_paddleocr():
    """Mock PaddleOCR for testing without actual model."""
    try:
        with patch('paddleocr.PaddleOCR') as mock:
            mock_instance = MagicMock()
            mock_instance.ocr.return_value = [[
                [[[100, 50], [300, 50], [300, 100], [100, 100]], ('GROCERY STORE', 0.98)],
                [[[50, 200], [280, 200], [280, 230], [50, 230]], ('01/15/2024', 0.94)],
                [[[50, 540], [130, 540], [130, 570], [50, 570]], ('Total', 0.98)],
                [[[300, 540], [380, 540], [380, 570], [300, 570]], ('$7.01', 0.97)],
            ]]
            mock.return_value = mock_instance
            yield mock
    except ModuleNotFoundError:
        # If paddleocr is not installed, we can't patch it
        # Just yield a mock that won't be used
        yield MagicMock()


@pytest.fixture
def mock_tesseract():
    """Mock pytesseract for testing without actual Tesseract."""
    try:
        with patch('pytesseract.image_to_data') as mock:
            mock.return_value = {
                'text': ['GROCERY', 'STORE', 'Total', '$7.01', '', ''],
                'conf': [98, 96, 98, 97, -1, -1],
                'left': [100, 100, 50, 300, 0, 0],
                'top': [50, 100, 540, 540, 0, 0],
                'width': [200, 150, 80, 80, 0, 0],
                'height': [50, 50, 30, 30, 0, 0],
            }
            yield mock
    except ModuleNotFoundError:
        yield MagicMock()


@pytest.fixture
def mock_layoutlm():
    """Mock LayoutLMv3 model for testing without actual model."""
    try:
        with patch('transformers.AutoProcessor') as mock_processor, \
             patch('transformers.AutoModelForTokenClassification') as mock_model:
            
            # Mock processor
            processor_instance = MagicMock()
            processor_instance.return_value = {
                'input_ids': MagicMock(),
                'attention_mask': MagicMock(),
                'bbox': MagicMock(),
            }
            mock_processor.from_pretrained.return_value = processor_instance
            
            # Mock model
            model_instance = MagicMock()
            model_instance.config.id2label = {0: 'O', 1: 'B-VENDOR', 2: 'B-TOTAL'}
            mock_model.from_pretrained.return_value = model_instance
            
            yield mock_processor, mock_model
    except ModuleNotFoundError:
        yield MagicMock(), MagicMock()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def multi_page_receipt_paths(sample_receipt_image, tmp_path):
    """Create multiple sample receipt images for multi-page testing."""
    paths = []
    for i in range(3):
        img_path = tmp_path / f"receipt_page_{i+1}.jpg"
        Image.fromarray(sample_receipt_image).save(img_path)
        paths.append(str(img_path))
    return paths
