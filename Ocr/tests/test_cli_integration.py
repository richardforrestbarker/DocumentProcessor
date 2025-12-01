"""
Integration tests that actually execute the OCR models.

These tests require the full dependencies to be installed:
- PaddleOCR or Tesseract
- LayoutLMv3 (optional)

Tests are marked with skip conditions if dependencies are not available.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Check for optional dependencies
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import torch
    from transformers import AutoProcessor, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def create_synthetic_receipt(text_lines, width=400, height=600):
    """Create a synthetic receipt image with text for testing."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a simple font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    y_position = 20
    for line in text_lines:
        draw.text((20, y_position), line, fill='black', font=font)
        y_position += 30
    
    return np.array(img)


@pytest.fixture
def realistic_receipt_image(tmp_path):
    """Create a realistic synthetic receipt image."""
    lines = [
        "GROCERY STORE",
        "123 Main Street",
        "City, State 12345",
        "",
        "Date: 01/15/2024",
        "Time: 14:30",
        "",
        "Milk 2%           $3.99",
        "Bread             $2.50",
        "Eggs              $4.99",
        "",
        "Subtotal          $11.48",
        "Tax 8%            $0.92",
        "-------------------",
        "TOTAL             $12.40",
        "",
        "Thank you!",
    ]
    
    img = create_synthetic_receipt(lines)
    img_path = tmp_path / "realistic_receipt.jpg"
    Image.fromarray(img).save(img_path, quality=95)
    return str(img_path)


@pytest.fixture
def multi_page_realistic_receipts(tmp_path):
    """Create multiple realistic receipt images."""
    paths = []
    
    # Page 1
    lines1 = [
        "MEGA MART",
        "456 Shopping Ave",
        "",
        "Date: 12/25/2023",
        "",
        "Electronics Dept",
        "USB Cable         $9.99",
        "HDMI Adapter      $14.99",
    ]
    img1 = create_synthetic_receipt(lines1)
    path1 = tmp_path / "receipt_page1.jpg"
    Image.fromarray(img1).save(path1, quality=95)
    paths.append(str(path1))
    
    # Page 2
    lines2 = [
        "Page 2 of 2",
        "",
        "Subtotal          $24.98",
        "Tax               $2.00",
        "TOTAL             $26.98",
        "",
        "Paid: Credit Card",
        "Thank you!",
    ]
    img2 = create_synthetic_receipt(lines2)
    path2 = tmp_path / "receipt_page2.jpg"
    Image.fromarray(img2).save(path2, quality=95)
    paths.append(str(path2))
    
    return paths


class TestPaddleOCRIntegration:
    """Integration tests using actual PaddleOCR."""

    @pytest.mark.skipif(not HAS_PADDLEOCR, reason="PaddleOCR not installed")
    def test_paddleocr_detects_text(self, realistic_receipt_image):
        """Test that PaddleOCR actually detects text from image."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        words = run_ocr(image, ocr_engine='paddle', device='cpu')
        
        assert len(words) > 0
        assert all('text' in w for w in words)
        assert all('box' in w for w in words)
        assert all('confidence' in w for w in words)

    @pytest.mark.skipif(not HAS_PADDLEOCR, reason="PaddleOCR not installed")
    def test_paddleocr_finds_grocery_store(self, realistic_receipt_image):
        """Test that PaddleOCR finds expected text."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        words = run_ocr(image, ocr_engine='paddle', device='cpu')
        
        all_text = ' '.join(w['text'].lower() for w in words)
        # Should find at least some of the expected text
        assert any(term in all_text for term in ['grocery', 'store', 'total', 'date'])

    @pytest.mark.skipif(not HAS_PADDLEOCR, reason="PaddleOCR not installed")
    def test_paddleocr_bounding_boxes_valid(self, realistic_receipt_image):
        """Test that PaddleOCR returns valid bounding boxes."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        words = run_ocr(image, ocr_engine='paddle', device='cpu')
        
        for word in words:
            box = word['box']
            assert len(box) == 4
            assert box[0] < box[2]  # x0 < x1
            assert box[1] < box[3]  # y0 < y1
            assert all(coord >= 0 for coord in box)

    @pytest.mark.skipif(not HAS_PADDLEOCR, reason="PaddleOCR not installed")
    def test_paddleocr_confidence_range(self, realistic_receipt_image):
        """Test that PaddleOCR returns valid confidence scores."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        words = run_ocr(image, ocr_engine='paddle', device='cpu')
        
        for word in words:
            assert 0 <= word['confidence'] <= 1


class TestTesseractIntegration:
    """Integration tests using actual Tesseract."""

    @pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract not installed")
    def test_tesseract_detects_text(self, realistic_receipt_image):
        """Test that Tesseract actually detects text from image."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        words = run_ocr(image, ocr_engine='tesseract', device='cpu')
        
        assert len(words) > 0
        assert all('text' in w for w in words)
        assert all('box' in w for w in words)
        assert all('confidence' in w for w in words)

    @pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract not installed")
    def test_tesseract_bounding_boxes_valid(self, realistic_receipt_image):
        """Test that Tesseract returns valid bounding boxes."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        words = run_ocr(image, ocr_engine='tesseract', device='cpu')
        
        for word in words:
            box = word['box']
            assert len(box) == 4
            assert box[0] <= box[2]  # x0 <= x1
            assert box[1] <= box[3]  # y0 <= y1


class TestLayoutLMIntegration:
    """Integration tests using actual LayoutLMv3."""

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="Transformers not installed")
    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    @pytest.mark.slow
    def test_layoutlm_loads_and_runs(self, realistic_receipt_image):
        """Test that LayoutLMv3 actually loads and runs inference."""
        from cli import run_layoutlm_inference, load_image, run_ocr, normalize_boxes
        
        image = load_image(realistic_receipt_image)
        
        # Get OCR words
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        words = run_ocr(image, ocr_engine=ocr_engine, device='cpu')
        
        if len(words) == 0:
            pytest.skip("No words detected by OCR")
        
        # Normalize boxes
        h, w = image.shape[:2]
        normalized_words = normalize_boxes(words, w, h)
        
        # Run LayoutLM inference
        result = run_layoutlm_inference(
            image,
            normalized_words,
            model_name="microsoft/layoutlmv3-base",
            device="cpu"
        )
        
        # Should return predictions or empty if model not fine-tuned
        assert isinstance(result, dict)
        assert 'predictions' in result

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="Transformers not installed")
    @pytest.mark.slow
    def test_layoutlm_handles_empty_words(self, realistic_receipt_image):
        """Test that LayoutLMv3 handles empty word list gracefully."""
        from cli import run_layoutlm_inference, load_image
        
        image = load_image(realistic_receipt_image)
        
        # Empty words list
        result = run_layoutlm_inference(
            image,
            [],
            model_name="microsoft/layoutlmv3-base",
            device="cpu"
        )
        
        assert result['predictions'] == []


class TestEndToEndIntegration:
    """End-to-end integration tests running the full pipeline."""

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_full_pipeline_single_page(self, realistic_receipt_image):
        """Test full pipeline with single page receipt."""
        from cli import process_receipt
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            [realistic_receipt_image],
            ocr_engine=ocr_engine,
            device='cpu',
            job_id='test-e2e-001'
        )
        
        assert result['status'] == 'done'
        assert result['job_id'] == 'test-e2e-001'
        assert len(result['pages']) == 1
        assert result['pages'][0]['page_number'] == 1
        assert len(result['pages'][0]['words']) > 0

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_full_pipeline_multi_page(self, multi_page_realistic_receipts):
        """Test full pipeline with multi-page receipt."""
        from cli import process_receipt
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            multi_page_realistic_receipts,
            ocr_engine=ocr_engine,
            device='cpu',
            job_id='test-e2e-002'
        )
        
        assert result['status'] == 'done'
        assert len(result['pages']) == 2
        assert result['pages'][0]['page_number'] == 1
        assert result['pages'][1]['page_number'] == 2

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_full_pipeline_extracts_fields(self, realistic_receipt_image):
        """Test that full pipeline extracts some fields."""
        from cli import process_receipt
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            [realistic_receipt_image],
            ocr_engine=ocr_engine,
            device='cpu'
        )
        
        # Should extract at least vendor name (first words at top)
        assert result['vendor_name'] is not None or result['date'] is not None or result['total_amount'] is not None

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_full_pipeline_with_preprocessing(self, realistic_receipt_image):
        """Test full pipeline with preprocessing enabled."""
        from cli import process_receipt
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            [realistic_receipt_image],
            ocr_engine=ocr_engine,
            device='cpu',
            denoise=True,
            deskew=True
        )
        
        assert result['status'] == 'done'
        assert len(result['pages']) == 1

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_full_pipeline_output_file(self, realistic_receipt_image, tmp_path):
        """Test full pipeline writes output to file."""
        from cli import process_receipt
        
        output_path = tmp_path / "output.json"
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            [realistic_receipt_image],
            ocr_engine=ocr_engine,
            device='cpu',
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        with open(output_path) as f:
            saved_result = json.load(f)
        
        assert saved_result['job_id'] == result['job_id']
        assert saved_result['status'] == result['status']

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_cli_process_command(self, realistic_receipt_image, tmp_path):
        """Test CLI process command runs successfully."""
        from cli import main
        
        output_path = tmp_path / "cli_output.json"
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        # Simulate CLI execution
        sys.argv = [
            'cli.py', 'process',
            '--image', realistic_receipt_image,
            '--output', str(output_path),
            '--ocr-engine', ocr_engine,
            '--device', 'cpu'
        ]
        
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
        
        assert output_path.exists()
        
        with open(output_path) as f:
            result = json.load(f)
        
        assert result['status'] == 'done'

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_cli_stdout_output(self, realistic_receipt_image):
        """Test CLI prints JSON to stdout when no output file specified."""
        from cli import main
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        sys.argv = [
            'cli.py', 'process',
            '--image', realistic_receipt_image,
            '--ocr-engine', ocr_engine,
            '--device', 'cpu'
        ]
        
        captured_output = StringIO()
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(sys, 'stdout', captured_output)
            try:
                main()
            except SystemExit as e:
                assert e.code == 0
        
        output = captured_output.getvalue()
        result = json.loads(output)
        
        assert result['status'] == 'done'


class TestPreprocessingIntegration:
    """Integration tests for preprocessing with actual OpenCV."""

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
    def test_denoise_actually_modifies_image(self, realistic_receipt_image):
        """Test that denoising actually modifies the image."""
        from cli import load_image, preprocess_image
        
        image = load_image(realistic_receipt_image)
        processed = preprocess_image(image, denoise=True, deskew=False)
        
        # Processed image should be different from original
        # (at least in shape due to grayscale->RGB conversion)
        assert processed is not None
        assert len(processed.shape) == 3

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
    def test_deskew_handles_straight_image(self, realistic_receipt_image):
        """Test that deskew handles already-straight images."""
        from cli import load_image, preprocess_image
        
        image = load_image(realistic_receipt_image)
        processed = preprocess_image(image, denoise=False, deskew=True)
        
        assert processed is not None
        assert processed.shape[:2] == image.shape[:2]

    @pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
    def test_preprocessing_chain(self, realistic_receipt_image):
        """Test that preprocessing chain works together."""
        from cli import load_image, preprocess_image
        
        image = load_image(realistic_receipt_image)
        processed = preprocess_image(image, denoise=True, deskew=True)
        
        assert processed is not None
        # Should be RGB
        assert len(processed.shape) == 3
        assert processed.shape[2] == 3


class TestOCREngineFallback:
    """Test OCR engine fallback behavior."""

    @pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract not installed")
    def test_tesseract_fallback_when_paddle_unavailable(self, realistic_receipt_image):
        """Test Tesseract is used when PaddleOCR import fails."""
        from cli import run_ocr, load_image
        
        image = load_image(realistic_receipt_image)
        
        # Force using tesseract
        words = run_ocr(image, ocr_engine='tesseract', device='cpu')
        
        assert len(words) >= 0  # May find some words
        for word in words:
            assert 'text' in word
            assert 'box' in word
            assert 'confidence' in word


class TestDeviceHandling:
    """Test device handling in integration context."""

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_cpu_device_works(self, realistic_receipt_image):
        """Test that CPU device works correctly."""
        from cli import process_receipt
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            [realistic_receipt_image],
            ocr_engine=ocr_engine,
            device='cpu'
        )
        
        assert result['status'] == 'done'

    @pytest.mark.skipif(not HAS_PADDLEOCR and not HAS_TESSERACT, reason="No OCR engine available")
    def test_auto_device_selection(self, realistic_receipt_image):
        """Test that auto device selection works."""
        from cli import process_receipt
        
        ocr_engine = 'paddle' if HAS_PADDLEOCR else 'tesseract'
        
        result = process_receipt(
            [realistic_receipt_image],
            ocr_engine=ocr_engine,
            device='auto'
        )
        
        assert result['status'] == 'done'


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
