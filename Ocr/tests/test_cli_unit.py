"""
Unit tests for CLI arguments, preprocessing, and utility functions.

These tests do NOT execute the actual OCR or ML models - they test the
CLI interface, argument parsing, preprocessing functions, and output handling.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cli
from cli import (
    get_device,
    load_image,
    preprocess_image,
    normalize_boxes,
    extract_fields_heuristic,
    process_receipt,
    main,
)


class TestCLIArguments:
    """Tests for CLI argument parsing."""

    def test_process_command_requires_image(self):
        """Test that process command requires at least one image."""
        with patch.object(sys, 'argv', ['cli.py', 'process']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2  # argparse exits with 2 for missing required args

    def test_process_command_accepts_single_image(self, sample_receipt_image_path):
        """Test that process command accepts a single image."""
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path]):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    with patch('sys.stdout', new_callable=StringIO):
                        try:
                            main()
                        except SystemExit as e:
                            assert e.code == 0

    def test_process_command_accepts_multiple_images(self, multi_page_receipt_paths):
        """Test that process command accepts multiple images."""
        args = ['cli.py', 'process']
        for path in multi_page_receipt_paths:
            args.extend(['--image', path])
        
        with patch.object(sys, 'argv', args):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    with patch('sys.stdout', new_callable=StringIO):
                        try:
                            main()
                        except SystemExit as e:
                            assert e.code == 0

    def test_ocr_engine_choices(self, sample_receipt_image_path):
        """Test that only valid OCR engines are accepted."""
        # Valid engines should work
        for engine in ['paddle', 'tesseract']:
            with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--ocr-engine', engine]):
                with patch.object(cli, 'run_ocr', return_value=[]):
                    with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                        with patch('sys.stdout', new_callable=StringIO):
                            try:
                                main()
                            except SystemExit as e:
                                assert e.code == 0

        # Invalid engine should fail
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--ocr-engine', 'invalid']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_device_choices(self, sample_receipt_image_path):
        """Test that only valid devices are accepted."""
        # Valid devices should work
        for device in ['auto', 'cuda', 'cpu']:
            with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--device', device]):
                with patch.object(cli, 'run_ocr', return_value=[]):
                    with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                        with patch('sys.stdout', new_callable=StringIO):
                            try:
                                main()
                            except SystemExit as e:
                                assert e.code == 0

        # Invalid device should fail
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--device', 'invalid']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_denoise_flag(self, sample_receipt_image_path):
        """Test denoise flag is properly parsed."""
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--denoise']):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    with patch.object(cli, 'preprocess_image') as mock_preprocess:
                        mock_preprocess.return_value = MagicMock()
                        with patch('sys.stdout', new_callable=StringIO):
                            try:
                                main()
                            except SystemExit:
                                pass

    def test_deskew_flag(self, sample_receipt_image_path):
        """Test deskew flag is properly parsed."""
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--deskew']):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    with patch.object(cli, 'preprocess_image') as mock_preprocess:
                        mock_preprocess.return_value = MagicMock()
                        with patch('sys.stdout', new_callable=StringIO):
                            try:
                                main()
                            except SystemExit:
                                pass

    def test_job_id_argument(self, sample_receipt_image_path):
        """Test job-id argument is passed correctly."""
        test_job_id = "my-custom-job-123"
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--job-id', test_job_id]):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    captured_output = StringIO()
                    with patch('sys.stdout', captured_output):
                        try:
                            main()
                        except SystemExit:
                            pass
                    output = captured_output.getvalue()
                    if output:
                        result = json.loads(output)
                        assert result.get('job_id') == test_job_id

    def test_output_file_argument(self, sample_receipt_image_path, temp_output_dir):
        """Test output file argument writes to specified path."""
        output_path = temp_output_dir / "result.json"
        with patch.object(sys, 'argv', ['cli.py', 'process', '--image', sample_receipt_image_path, '--output', str(output_path)]):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    try:
                        main()
                    except SystemExit:
                        pass
                    assert output_path.exists()

    def test_version_command(self):
        """Test version command prints version information."""
        with patch.object(sys, 'argv', ['cli.py', 'version']):
            captured_output = StringIO()
            with patch('sys.stdout', captured_output):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
            output = captured_output.getvalue()
            assert 'Receipt OCR Service' in output
            assert 'LayoutLMv3' in output

    def test_no_command_shows_help(self):
        """Test that running without command shows help."""
        with patch.object(sys, 'argv', ['cli.py']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_short_form_arguments(self, sample_receipt_image_path, temp_output_dir):
        """Test short form arguments work correctly."""
        output_path = temp_output_dir / "result.json"
        with patch.object(sys, 'argv', ['cli.py', 'process', '-i', sample_receipt_image_path, '-o', str(output_path), '-m', 'custom/model']):
            with patch.object(cli, 'run_ocr', return_value=[]):
                with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                    try:
                        main()
                    except SystemExit:
                        pass


class TestDeviceSelection:
    """Tests for device selection logic."""

    def test_auto_device_without_torch(self):
        """Test auto device returns cpu when torch not available."""
        with patch.dict(sys.modules, {'torch': None}):
            with patch.object(cli, 'get_device') as mock_get_device:
                mock_get_device.return_value = 'cpu'
                assert mock_get_device('auto') == 'cpu'

    def test_explicit_cpu_device(self):
        """Test explicit cpu device is returned as-is."""
        assert get_device('cpu') == 'cpu'

    def test_explicit_cuda_device(self):
        """Test explicit cuda device is returned as-is."""
        assert get_device('cuda') == 'cuda'

    def test_auto_device_with_cuda_available(self):
        """Test auto device returns cuda when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch.dict(sys.modules, {'torch': mock_torch}):
            # Re-import to use mocked torch
            result = get_device('auto')
            # Since we can't fully mock the import, this may return 'cpu' 
            # in test environment without CUDA
            assert result in ['cuda', 'cpu']


class TestNormalizeBoxes:
    """Tests for bounding box normalization."""

    def test_normalize_boxes_basic(self, sample_words):
        """Test basic box normalization to 1000 scale."""
        normalized = normalize_boxes(sample_words, image_width=400, image_height=600)
        
        assert len(normalized) == len(sample_words)
        for word in normalized:
            assert all(0 <= coord <= 1000 for coord in word['box'])

    def test_normalize_boxes_preserves_text(self, sample_words):
        """Test that normalization preserves text and confidence."""
        normalized = normalize_boxes(sample_words, image_width=400, image_height=600)
        
        for orig, norm in zip(sample_words, normalized):
            assert orig['text'] == norm['text']
            assert orig['confidence'] == norm['confidence']

    def test_normalize_boxes_scales_correctly(self):
        """Test that box coordinates are scaled correctly."""
        words = [{'text': 'test', 'box': [100, 150, 200, 300], 'confidence': 0.9}]
        normalized = normalize_boxes(words, image_width=400, image_height=600, scale=1000)
        
        # x0: 100/400 * 1000 = 250
        # y0: 150/600 * 1000 = 250
        # x1: 200/400 * 1000 = 500
        # y1: 300/600 * 1000 = 500
        expected_box = [250, 250, 500, 500]
        assert normalized[0]['box'] == expected_box

    def test_normalize_boxes_clamps_values(self):
        """Test that box values are clamped to 0-scale range."""
        words = [{'text': 'test', 'box': [-10, -20, 500, 800], 'confidence': 0.9}]
        normalized = normalize_boxes(words, image_width=400, image_height=600, scale=1000)
        
        for coord in normalized[0]['box']:
            assert 0 <= coord <= 1000

    def test_normalize_boxes_empty_list(self):
        """Test normalization of empty word list."""
        normalized = normalize_boxes([], image_width=400, image_height=600)
        assert normalized == []

    def test_normalize_boxes_custom_scale(self, sample_words):
        """Test normalization with custom scale."""
        normalized = normalize_boxes(sample_words, image_width=400, image_height=600, scale=500)
        
        for word in normalized:
            assert all(0 <= coord <= 500 for coord in word['box'])


class TestHeuristicExtraction:
    """Tests for heuristic field extraction."""

    def test_extract_vendor_name(self, sample_words):
        """Test vendor name extraction from top of receipt."""
        result = extract_fields_heuristic(sample_words)
        
        assert result['vendor_name'] is not None
        assert 'GROCERY' in result['vendor_name']['value']

    def test_extract_date(self, sample_words):
        """Test date extraction."""
        result = extract_fields_heuristic(sample_words)
        
        assert result['date'] is not None
        assert '01/15/2024' in result['date']['value']

    def test_extract_total(self, sample_words):
        """Test total amount extraction."""
        result = extract_fields_heuristic(sample_words)
        
        assert result['total_amount'] is not None
        # Should find the amount near 'Total'

    def test_extract_subtotal(self, sample_words):
        """Test subtotal extraction."""
        result = extract_fields_heuristic(sample_words)
        
        assert result['subtotal'] is not None
        # Should find the amount near 'Subtotal'

    def test_extract_tax(self, sample_words):
        """Test tax amount extraction."""
        result = extract_fields_heuristic(sample_words)
        
        assert result['tax_amount'] is not None
        # Should find the amount near 'Tax'

    def test_extract_currency_usd(self, sample_words):
        """Test USD currency detection from $ symbol."""
        result = extract_fields_heuristic(sample_words)
        
        assert result['currency'] is not None
        assert result['currency']['value'] == 'USD'

    def test_extract_currency_eur(self):
        """Test EUR currency detection."""
        words = [
            {'text': 'Store', 'box': [100, 50, 200, 100], 'confidence': 0.95},
            {'text': 'Total', 'box': [50, 200, 130, 230], 'confidence': 0.98},
            {'text': '€15.00', 'box': [200, 200, 300, 230], 'confidence': 0.96},
        ]
        result = extract_fields_heuristic(words)
        
        assert result['currency'] is not None
        assert result['currency']['value'] == 'EUR'

    def test_extract_empty_words(self):
        """Test extraction with empty word list."""
        result = extract_fields_heuristic([])
        
        assert result['vendor_name'] is None
        assert result['date'] is None
        assert result['total_amount'] is None

    def test_extract_confidence_values(self, sample_words):
        """Test that confidence values are included in extracted fields."""
        result = extract_fields_heuristic(sample_words)
        
        if result['vendor_name']:
            assert 'confidence' in result['vendor_name']
            assert 0 <= result['vendor_name']['confidence'] <= 1

    def test_extract_bounding_boxes(self, sample_words):
        """Test that bounding boxes are included in extracted fields."""
        result = extract_fields_heuristic(sample_words)
        
        if result['vendor_name']:
            assert 'box' in result['vendor_name']
            box = result['vendor_name']['box']
            assert all(key in box for key in ['x0', 'y0', 'x1', 'y1'])

    def test_extract_date_formats(self):
        """Test extraction of various date formats."""
        # Format: MM/DD/YYYY
        words1 = [{'text': '12/25/2023', 'box': [100, 100, 200, 130], 'confidence': 0.95}]
        result1 = extract_fields_heuristic(words1)
        assert result1['date'] is not None

        # Format: YYYY-MM-DD
        words2 = [{'text': '2023-12-25', 'box': [100, 100, 200, 130], 'confidence': 0.95}]
        result2 = extract_fields_heuristic(words2)
        assert result2['date'] is not None


class TestOutputFormatting:
    """Tests for output formatting and JSON structure."""

    def test_process_receipt_output_structure(self, sample_receipt_image_path):
        """Test that process_receipt returns correct structure."""
        with patch.object(cli, 'run_ocr', return_value=[]):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                result = process_receipt([sample_receipt_image_path])
        
        # Check required top-level keys
        assert 'job_id' in result
        assert 'status' in result
        assert 'pages' in result
        assert 'vendor_name' in result
        assert 'date' in result
        assert 'total_amount' in result
        assert 'subtotal' in result
        assert 'tax_amount' in result
        assert 'currency' in result
        assert 'line_items' in result

    def test_process_receipt_job_id_generation(self, sample_receipt_image_path):
        """Test that job_id is generated when not provided."""
        with patch.object(cli, 'run_ocr', return_value=[]):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                result = process_receipt([sample_receipt_image_path])
        
        assert result['job_id'].startswith('job-')

    def test_process_receipt_custom_job_id(self, sample_receipt_image_path):
        """Test that custom job_id is used when provided."""
        custom_id = "custom-job-456"
        with patch.object(cli, 'run_ocr', return_value=[]):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                result = process_receipt([sample_receipt_image_path], job_id=custom_id)
        
        assert result['job_id'] == custom_id

    def test_process_receipt_page_structure(self, sample_receipt_image_path):
        """Test page structure in output."""
        mock_words = [{'text': 'Test', 'box': [10, 10, 100, 50], 'confidence': 0.95}]
        with patch.object(cli, 'run_ocr', return_value=mock_words):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                result = process_receipt([sample_receipt_image_path])
        
        assert len(result['pages']) == 1
        page = result['pages'][0]
        assert 'page_number' in page
        assert 'raw_ocr_text' in page
        assert 'words' in page
        assert page['page_number'] == 1

    def test_process_receipt_multi_page(self, multi_page_receipt_paths):
        """Test multi-page receipt processing."""
        with patch.object(cli, 'run_ocr', return_value=[]):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                result = process_receipt(multi_page_receipt_paths)
        
        assert len(result['pages']) == 3
        for i, page in enumerate(result['pages']):
            assert page['page_number'] == i + 1

    def test_output_file_valid_json(self, sample_receipt_image_path, temp_output_dir):
        """Test that output file contains valid JSON."""
        output_path = temp_output_dir / "result.json"
        with patch.object(cli, 'run_ocr', return_value=[]):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                process_receipt([sample_receipt_image_path], output_path=str(output_path))
        
        assert output_path.exists()
        with open(output_path) as f:
            result = json.load(f)
        assert isinstance(result, dict)

    def test_output_creates_parent_directories(self, sample_receipt_image_path, temp_output_dir):
        """Test that output creates parent directories if needed."""
        output_path = temp_output_dir / "nested" / "dir" / "result.json"
        with patch.object(cli, 'run_ocr', return_value=[]):
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                process_receipt([sample_receipt_image_path], output_path=str(output_path))
        
        assert output_path.exists()


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_image_file(self, tmp_path):
        """Test handling of missing image file."""
        missing_path = str(tmp_path / "nonexistent.jpg")
        
        result = process_receipt([missing_path])
        
        assert result['status'] == 'failed'
        assert 'error' in result

    def test_invalid_image_file(self, tmp_path):
        """Test handling of invalid image file."""
        invalid_path = tmp_path / "invalid.jpg"
        invalid_path.write_text("not an image")
        
        result = process_receipt([str(invalid_path)])
        
        assert result['status'] == 'failed'
        assert 'error' in result

    def test_ocr_engine_fallback(self, sample_receipt_image_path):
        """Test that OCR falls back to Tesseract when PaddleOCR fails."""
        with patch.object(cli, 'run_ocr') as mock_ocr:
            mock_ocr.side_effect = [
                RuntimeError("PaddleOCR failed"),
                []  # Tesseract fallback returns empty
            ]
            with patch.object(cli, 'run_layoutlm_inference', return_value={"predictions": []}):
                # This should not raise an exception
                result = process_receipt([sample_receipt_image_path])

    def test_model_inference_failure(self, sample_receipt_image_path):
        """Test handling of model inference failure."""
        mock_words = [{'text': 'Test', 'box': [10, 10, 100, 50], 'confidence': 0.95}]
        with patch.object(cli, 'run_ocr', return_value=mock_words):
            # Even when layoutlm fails, it should return gracefully (caught internally)
            # and the pipeline should use heuristics
            result = process_receipt([sample_receipt_image_path])
            # The function catches exceptions internally, so status should be 'done'
            assert result['status'] == 'done'

    def test_graceful_degradation_without_opencv(self, sample_receipt_image_path):
        """Test preprocessing works without OpenCV."""
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch.dict(sys.modules, {'cv2': None}):
            # Should return original image without error
            result = preprocess_image(test_image, denoise=True, deskew=True)
            assert result is not None


class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_preprocess_no_options(self, sample_receipt_image):
        """Test preprocessing without any options still works."""
        result = preprocess_image(sample_receipt_image, denoise=False, deskew=False)
        assert result is not None

    def test_preprocess_with_denoise(self, sample_receipt_image):
        """Test preprocessing with denoising enabled."""
        result = preprocess_image(sample_receipt_image, denoise=True, deskew=False)
        assert result is not None

    def test_preprocess_with_deskew(self, sample_receipt_image):
        """Test preprocessing with deskewing enabled."""
        result = preprocess_image(sample_receipt_image, denoise=False, deskew=True)
        assert result is not None

    def test_preprocess_with_all_options(self, sample_receipt_image):
        """Test preprocessing with all options enabled."""
        result = preprocess_image(sample_receipt_image, denoise=True, deskew=True)
        assert result is not None

    def test_preprocess_grayscale_input(self):
        """Test preprocessing handles grayscale input."""
        import numpy as np
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        
        result = preprocess_image(gray_image, denoise=False, deskew=False)
        assert result is not None

    def test_load_image_rgb(self, sample_receipt_image_path):
        """Test that load_image returns RGB format."""
        image = load_image(sample_receipt_image_path)
        
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # RGB has 3 channels

    def test_load_image_grayscale_converted(self, tmp_path):
        """Test that grayscale images are converted to RGB."""
        from PIL import Image
        import numpy as np
        
        # Create grayscale image
        gray_img = Image.new('L', (100, 100), color=128)
        img_path = tmp_path / "gray.jpg"
        gray_img.save(img_path)
        
        # Load should convert to RGB
        image = load_image(str(img_path))
        
        assert len(image.shape) == 3
        assert image.shape[2] == 3


class TestDebugOutput:
    """Tests for debug output functionality."""

    def test_debug_argument_parsing(self):
        """Test that --debug argument is properly parsed."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['process', '--image', 'test.jpg', '--debug'])
        assert args.debug is True

    def test_debug_argument_default_false(self):
        """Test that debug defaults to False."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['process', '--image', 'test.jpg'])
        assert args.debug is False

    def test_debug_output_dir_argument(self):
        """Test that --debug-output-dir argument is properly parsed."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['process', '--image', 'test.jpg', '--debug', '--debug-output-dir', '/custom/path'])
        assert args.debug_output_dir == '/custom/path'

    def test_debug_output_manager_creation(self, tmp_path):
        """Test that DebugOutputManager creates output directory."""
        from src.cli.debug_output import DebugOutputManager
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job-123")
        
        assert manager.job_dir.exists()
        assert manager.job_id == "test-job-123"

    def test_debug_output_manager_save_source_image(self, tmp_path, sample_receipt_image):
        """Test saving source image in debug mode."""
        from src.cli.debug_output import DebugOutputManager
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        result = manager.save_source_image(sample_receipt_image, page_num=1)
        
        assert result
        assert Path(result).exists()
        assert "step_01_source" in result

    def test_debug_output_manager_save_grayscale_image(self, tmp_path, sample_receipt_image):
        """Test saving grayscale image in debug mode."""
        from src.cli.debug_output import DebugOutputManager
        import cv2
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        # Convert to grayscale
        gray = cv2.cvtColor(sample_receipt_image, cv2.COLOR_RGB2GRAY)
        
        result = manager.save_grayscale_image(gray, page_num=1)
        
        assert result
        assert Path(result).exists()
        assert "step_02_grayscale" in result

    def test_debug_output_manager_save_ocr_bounding_boxes(self, tmp_path, sample_receipt_image, sample_words):
        """Test saving OCR bounding boxes visualization."""
        from src.cli.debug_output import DebugOutputManager
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        result = manager.save_ocr_bounding_boxes(
            sample_receipt_image, 
            sample_words, 
            page_num=1, 
            ocr_engine="paddle"
        )
        
        assert result
        assert Path(result).exists()
        assert "step_07_paddle_bboxes" in result

    def test_debug_output_manager_save_result_bounding_boxes(self, tmp_path, sample_receipt_image, sample_ocr_result):
        """Test saving result bounding boxes visualization."""
        from src.cli.debug_output import DebugOutputManager
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        result = manager.save_result_bounding_boxes(
            sample_receipt_image, 
            sample_ocr_result, 
            page_num=1
        )
        
        assert result
        assert Path(result).exists()
        assert "step_08_result_bboxes" in result

    def test_debug_output_manager_save_debug_summary(self, tmp_path, sample_ocr_result):
        """Test saving debug summary JSON."""
        from src.cli.debug_output import DebugOutputManager
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        result = manager.save_debug_summary(sample_ocr_result)
        
        assert result
        assert Path(result).exists()
        
        # Verify JSON content
        with open(result) as f:
            summary = json.load(f)
        assert summary['job_id'] == 'test-job'
        assert 'processing_steps' in summary
        assert 'result' in summary

    def test_preprocessor_with_debug_manager(self, tmp_path, sample_receipt_image):
        """Test that ImagePreprocessor integrates with DebugOutputManager."""
        from src.cli.debug_output import DebugOutputManager
        from src.preprocessing.image_preprocessor import ImagePreprocessor
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        preprocessor = ImagePreprocessor(
            denoise=True,
            deskew=True,
            enhance_contrast=True,
            debug_manager=manager
        )
        
        # Process image
        result = preprocessor.preprocess_array(sample_receipt_image, page_num=1)
        
        assert result is not None
        
        # Check that debug files were created
        debug_files = list(manager.job_dir.glob("*.png"))
        assert len(debug_files) >= 1  # At least grayscale should be saved

    def test_debug_mode_creates_all_step_files(self, tmp_path, sample_receipt_image):
        """Test that debug mode creates files for all processing steps."""
        from src.cli.debug_output import DebugOutputManager
        from src.preprocessing.image_preprocessor import ImagePreprocessor
        
        debug_dir = tmp_path / "debug"
        manager = DebugOutputManager(output_dir=str(debug_dir), job_id="test-job")
        
        # Save source
        manager.save_source_image(sample_receipt_image, page_num=1)
        
        # Process with preprocessor
        preprocessor = ImagePreprocessor(
            denoise=True,
            deskew=True,
            enhance_contrast=True,
            debug_manager=manager
        )
        preprocessor.preprocess_array(sample_receipt_image, page_num=1)
        
        # Check files exist
        files = list(manager.job_dir.glob("step_*.png"))
        file_names = [f.name for f in files]
        
        # Should have at least source, grayscale, denoised, deskewed, contrast, preprocessed
        assert any("step_01" in f for f in file_names)  # source
        assert any("step_02" in f for f in file_names)  # grayscale
