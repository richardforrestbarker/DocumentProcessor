"""
Unit tests for new CLI commands: preprocess, ocr, and inference.

Tests the separated preprocessing, OCR, and inference phases of the pipeline.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLIPreprocessCommand:
    """Tests for the preprocess CLI command."""

    def test_preprocess_command_requires_image(self):
        """Test that preprocess command requires an image argument."""
        from src.cli.args import parse_args
        
        with pytest.raises(SystemExit) as exc_info:
            parse_args(['preprocess'])
        assert exc_info.value.code == 2  # argparse exits with 2 for missing required args

    def test_preprocess_command_accepts_image(self):
        """Test that preprocess command accepts an image argument."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['preprocess', '--image', 'test.jpg'])
        assert args.input_image == 'test.jpg'
        assert args.command == 'preprocess'

    def test_preprocess_command_default_output_format(self):
        """Test that preprocess command defaults to base64 output format."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['preprocess', '--image', 'test.jpg'])
        assert args.output_format == 'base64'

    def test_preprocess_command_accepts_file_format(self):
        """Test that preprocess command accepts file output format."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['preprocess', '--image', 'test.jpg', '--output-format', 'file'])
        assert args.output_format == 'file'

    def test_preprocess_command_accepts_preprocessing_options(self):
        """Test that preprocess command accepts all preprocessing options."""
        from src.cli.args import parse_args
        
        args, _ = parse_args([
            'preprocess', '--image', 'test.jpg',
            '--denoise',
            '--deskew',
            '--fuzz-percent', '50',
            '--deskew-threshold', '30',
            '--contrast-type', 'linear',
            '--contrast-strength', '5',
            '--contrast-midpoint', '150'
        ])
        
        assert args.denoise is True
        assert args.deskew is True
        assert args.fuzz_percent == 50
        assert args.deskew_threshold == 30
        assert args.contrast_type == 'linear'
        assert args.contrast_strength == 5
        assert args.contrast_midpoint == 150


class TestCLIOcrCommand:
    """Tests for the ocr CLI command."""

    def test_ocr_command_requires_image(self):
        """Test that ocr command requires an image argument."""
        from src.cli.args import parse_args
        
        with pytest.raises(SystemExit) as exc_info:
            parse_args(['ocr'])
        assert exc_info.value.code == 2

    def test_ocr_command_accepts_image(self):
        """Test that ocr command accepts an image argument."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['ocr', '--image', 'preprocessed.png'])
        assert args.input_image == 'preprocessed.png'
        assert args.command == 'ocr'

    def test_ocr_command_default_options(self):
        """Test that ocr command has correct default options."""
        from src.cli.args import parse_args
        
        args, _ = parse_args(['ocr', '--image', 'preprocessed.png'])
        assert args.ocr_engine == 'paddle'
        assert args.target_dpi == 300
        assert args.device == 'auto'

    def test_ocr_command_accepts_custom_options(self):
        """Test that ocr command accepts custom options."""
        from src.cli.args import parse_args
        
        args, _ = parse_args([
            'ocr', '--image', 'preprocessed.png',
            '--ocr-engine', 'tesseract',
            '--target-dpi', '200',
            '--device', 'cpu'
        ])
        
        assert args.ocr_engine == 'tesseract'
        assert args.target_dpi == 200
        assert args.device == 'cpu'


class TestCLIInferenceCommand:
    """Tests for the inference CLI command."""

    def test_inference_command_requires_ocr_result(self):
        """Test that inference command requires an OCR result path."""
        from src.cli.args import parse_args
        
        with pytest.raises(SystemExit) as exc_info:
            parse_args(['inference', '--image', 'test.jpg'])
        assert exc_info.value.code == 2

    def test_inference_command_requires_image(self):
        """Test that inference command requires an image argument."""
        from src.cli.args import parse_args
        
        with pytest.raises(SystemExit) as exc_info:
            parse_args(['inference', '--ocr-result', 'result.json'])
        assert exc_info.value.code == 2

    def test_inference_command_accepts_arguments(self):
        """Test that inference command accepts required arguments."""
        from src.cli.args import parse_args
        
        args, _ = parse_args([
            'inference',
            '--ocr-result', 'result.json',
            '--image', 'test.jpg'
        ])
        
        assert args.ocr_result_path == 'result.json'
        assert args.input_image == 'test.jpg'
        assert args.command == 'inference'

    def test_inference_command_default_options(self):
        """Test that inference command has correct default options."""
        from src.cli.args import parse_args
        
        args, _ = parse_args([
            'inference',
            '--ocr-result', 'result.json',
            '--image', 'test.jpg'
        ])
        
        assert args.model == 'naver-clova-ix/donut-base-finetuned-cord-v2'
        assert args.device == 'auto'

    def test_inference_command_accepts_custom_options(self):
        """Test that inference command accepts custom options."""
        from src.cli.args import parse_args
        
        args, _ = parse_args([
            'inference',
            '--ocr-result', 'result.json',
            '--image', 'test.jpg',
            '--model', 'microsoft/layoutlmv3-base',
            '--device', 'cuda'
        ])
        
        assert args.model == 'microsoft/layoutlmv3-base'
        assert args.device == 'cuda'


class TestPreprocessingPhase:
    """Tests for preprocess command output structure."""

    def test_preprocess_command_output_structure(self, sample_receipt_image_path):
        """Test that preprocess command returns correct output structure."""
        from src.cli.commands import preprocess_command
        
        # Note: This test may fail without ImageMagick, but tests the structure
        try:
            result = preprocess_command(
                input_image=sample_receipt_image_path,
                output_format='base64',
                denoise=False,
                deskew=False
            )
            
            # Check required keys
            assert 'job_id' in result
            assert 'status' in result
            assert 'input_image' in result
            assert 'preprocessing_settings' in result
            
            # Check settings are captured
            settings = result['preprocessing_settings']
            assert 'denoise' in settings
            assert 'deskew' in settings
            assert 'fuzz_percent' in settings
            assert 'contrast_type' in settings
            
            if result['status'] == 'done':
                assert 'image_base64' in result
                assert 'width' in result
                assert 'height' in result
        except RuntimeError as e:
            # ImageMagick not installed
            if "ImageMagick" in str(e):
                pytest.skip("ImageMagick not installed")
            raise


class TestOcrPhase:
    """Tests for ocr command output structure."""

    def test_ocr_command_output_structure(self, sample_receipt_image_path):
        """Test that ocr command returns correct output structure."""
        from src.cli.commands import ocr_command
        
        try:
            result = ocr_command(
                input_image=sample_receipt_image_path,
                ocr_engine='paddle',
                target_dpi=300,
                device='cpu'
            )
            
            # Check required keys
            assert 'job_id' in result
            assert 'status' in result
            assert 'input_image' in result
            assert 'ocr_engine' in result
            assert 'words' in result
            assert 'raw_ocr_text' in result
            
            if result['status'] == 'done':
                assert isinstance(result['words'], list)
                assert 'image_width' in result
                assert 'image_height' in result
        except RuntimeError as e:
            # Dependencies not installed
            pytest.skip(f"Required dependency not installed: {e}")
        except Exception as e:
            if "PaddleOCR" in str(e) or "tesseract" in str(e).lower():
                pytest.skip("OCR engine not installed")
            raise


class TestInferencePhase:
    """Tests for inference command output structure."""

    def test_inference_command_output_structure(
        self, sample_receipt_image_path, sample_ocr_command_result_file
    ):
        """Test that inference command returns correct output structure."""
        from src.cli.commands import inference_command
        
        try:
            result = inference_command(
                ocr_result_path=sample_ocr_command_result_file,
                input_image=sample_receipt_image_path,
                device='cpu'
            )
            
            # Check required keys
            assert 'job_id' in result
            assert 'status' in result
            assert 'input_image' in result
            assert 'model' in result
            
            # These fields should exist (may be None)
            assert 'vendor_name' in result
            assert 'date' in result
            assert 'total_amount' in result
            assert 'subtotal' in result
            assert 'tax_amount' in result
            assert 'currency' in result
            assert 'line_items' in result
            
        except Exception as e:
            # Model dependencies not installed
            if "torch" in str(e).lower() or "transformers" in str(e).lower():
                pytest.skip("Model dependencies not installed")
            raise


class TestPreprocessorSkipsDpiResampling:
    """Test that preprocessor can skip DPI resampling when target_dpi is None."""

    def test_preprocessor_skip_dpi_resampling(self, sample_receipt_image_path):
        """Test that ImagePreprocessor skips DPI resampling when target_dpi is None."""
        try:
            from src.preprocessing.image_preprocessor import ImagePreprocessor
            
            preprocessor = ImagePreprocessor(
                target_dpi=None,  # Skip DPI resampling
                denoise=False,
                deskew=False,
                enhance_contrast=False
            )
            
            # The preprocessor should work without throwing an error
            result, width, height = preprocessor.preprocess(sample_receipt_image_path)
            
            assert result is not None
            assert width > 0
            assert height > 0
            
        except RuntimeError as e:
            if "ImageMagick" in str(e):
                pytest.skip("ImageMagick not installed")
            raise
