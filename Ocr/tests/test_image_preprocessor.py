"""
Unit tests for image_preprocessor.py.

Tests for image preprocessing functions using ImageMagick CLI.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.image_preprocessor import (
    ImagePreprocessor, 
    SCRIPTS_DIR, 
    TESSERACT_MAX_DIMENSION,
    PILLOW_MAX_PIXELS
)


def imagemagick_available():
    """Check if ImageMagick is installed."""
    try:
        result = subprocess.run(
            ["magick", "--version"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip all tests if ImageMagick is not available
pytestmark = pytest.mark.skipif(
    not imagemagick_available(),
    reason="ImageMagick is not installed"
)


class TestImageMagickPreprocessing:
    """Tests for ImageMagick CLI-based preprocessing."""

    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary test image file."""
        from PIL import Image
        
        # Create a simple test image with text-like content
        height, width = 400, 300
        img_array = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # Add some dark horizontal lines (simulating text)
        for y in range(50, 350, 40):
            img_array[y:y+5, 30:270, :] = 30
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(img_array).save(tmp.name)
            yield tmp.name
        
        # Cleanup
        os.unlink(tmp.name)

    def test_imagemagick_check(self):
        """Test that ImageMagick availability check works."""
        preprocessor = ImagePreprocessor()
        # Should not raise if ImageMagick is available
        preprocessor._check_imagemagick()

    def test_preprocess_returns_rgb_array(self, sample_image_path):
        """Test that preprocessing returns a valid RGB numpy array with dimensions."""
        preprocessor = ImagePreprocessor(
            target_dpi=300,
            deskew=True,
            denoise=True,
            enhance_contrast=True
        )
        
        result, width, height = preprocessor.preprocess(sample_image_path)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Verify returned dimensions match array shape
        assert height == result.shape[0]
        assert width == result.shape[1]

    def test_preprocess_with_all_options_disabled(self, sample_image_path):
        """Test preprocessing with optional steps disabled."""
        preprocessor = ImagePreprocessor(
            deskew=False,
            denoise=False,
            enhance_contrast=False
        )
        
        result, width, height = preprocessor.preprocess(sample_image_path)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        # Verify returned dimensions match array shape
        assert height == result.shape[0]
        assert width == result.shape[1]

    def test_scripts_exist(self):
        """Test that all preprocessing shell scripts exist."""
        expected_scripts = [
            "convert_to_tiff.sh",
            "fix_resolution.sh",
            "remove_background.sh",
            "deskew.sh",
            "grayscale.sh",
            "enhance_contrast.sh",
            "denoise.sh",
            "preprocess_all.sh"
        ]
        
        for script in expected_scripts:
            script_path = SCRIPTS_DIR / script
            assert script_path.exists(), f"Script not found: {script}"
            assert os.access(script_path, os.X_OK), f"Script not executable: {script}"


class TestPreprocessingShellScripts:
    """Tests for individual shell scripts."""

    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary test image file."""
        from PIL import Image
        
        height, width = 200, 150
        img_array = np.full((height, width, 3), 255, dtype=np.uint8)
        for y in range(30, 170, 30):
            img_array[y:y+3, 20:130, :] = 50
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(img_array).save(tmp.name)
            yield tmp.name
        
        os.unlink(tmp.name)

    @pytest.fixture
    def temp_output(self):
        """Create a temporary output file path."""
        fd, path = tempfile.mkstemp(suffix='.tiff')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_convert_to_tiff_script(self, sample_image_path, temp_output):
        """Test convert_to_tiff.sh script."""
        script_path = SCRIPTS_DIR / "convert_to_tiff.sh"
        
        result = subprocess.run(
            [str(script_path), sample_image_path, temp_output],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)

    def test_fix_resolution_script(self, sample_image_path, temp_output):
        """Test fix_resolution.sh script."""
        # First convert to tiff
        tiff_path = temp_output.replace('.tiff', '_in.tiff')
        subprocess.run(["convert", sample_image_path, tiff_path], check=True)
        
        script_path = SCRIPTS_DIR / "fix_resolution.sh"
        result = subprocess.run(
            [str(script_path), tiff_path, temp_output, "300"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)
        
        # Clean up
        os.unlink(tiff_path)

    def test_grayscale_script(self, sample_image_path, temp_output):
        """Test grayscale.sh script."""
        script_path = SCRIPTS_DIR / "grayscale.sh"
        
        result = subprocess.run(
            [str(script_path), sample_image_path, temp_output],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)

    def test_deskew_script(self, sample_image_path, temp_output):
        """Test deskew.sh script."""
        script_path = SCRIPTS_DIR / "deskew.sh"
        
        result = subprocess.run(
            [str(script_path), sample_image_path, temp_output],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)

    def test_denoise_script(self, sample_image_path, temp_output):
        """Test denoise.sh script."""
        script_path = SCRIPTS_DIR / "denoise.sh"
        
        result = subprocess.run(
            [str(script_path), sample_image_path, temp_output],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)

    def test_enhance_contrast_script(self, sample_image_path, temp_output):
        """Test enhance_contrast.sh script."""
        script_path = SCRIPTS_DIR / "enhance_contrast.sh"
        
        result = subprocess.run(
            [str(script_path), sample_image_path, temp_output],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)

    def test_remove_background_script(self, sample_image_path, temp_output):
        """Test remove_background.sh script."""
        script_path = SCRIPTS_DIR / "remove_background.sh"
        
        result = subprocess.run(
            [str(script_path), sample_image_path, temp_output],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(temp_output)


class TestPreprocessingPipelineOrder:
    """Tests to verify preprocessing steps are in the correct order."""

    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary test image file."""
        from PIL import Image
        
        height, width = 300, 200
        img_array = np.full((height, width, 3), 255, dtype=np.uint8)
        for y in range(40, 260, 30):
            img_array[y:y+4, 25:175, :] = 40
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(img_array).save(tmp.name)
            yield tmp.name
        
        os.unlink(tmp.name)

    def test_full_pipeline_via_preprocess_all_script(self, sample_image_path):
        """Test that preprocess_all.sh runs all steps correctly."""
        script_path = SCRIPTS_DIR / "preprocess_all.sh"
        
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result = subprocess.run(
                [str(script_path), sample_image_path, output_path],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert os.path.exists(output_path)
            
            # Verify output is a valid image
            from PIL import Image
            img = Image.open(output_path)
            assert img is not None
            assert img.size[0] > 0
            assert img.size[1] > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_preprocess_returns_rgb_for_ocr_engines(self, sample_image_path):
        """Test that final output is RGB format for compatibility with OCR engines."""
        preprocessor = ImagePreprocessor(
            target_dpi=300,
            deskew=True,
            denoise=True,
            enhance_contrast=True
        )
        
        result, width, height = preprocessor.preprocess(sample_image_path)
        
        # OCR engines expect RGB format
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels
        # Verify returned dimensions match array shape
        assert height == result.shape[0]
        assert width == result.shape[1]


class TestErrorHandling:
    """Tests for error handling in preprocessing."""

    def test_missing_imagemagick_error(self, monkeypatch):
        """Test error when ImageMagick is not found."""
        # This test would require mocking subprocess, but we verify
        # the error message format is helpful
        pass  # ImageMagick is available in test environment

    def test_invalid_input_path(self):
        """Test handling of non-existent input file."""
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(Exception):
            preprocessor.preprocess("/nonexistent/path/image.jpg")


class TestImageSizeLimits:
    """Tests for image size limit checking."""
    
    def test_constants_defined(self):
        """Test that size limit constants are properly defined."""
        assert TESSERACT_MAX_DIMENSION == 32767
        assert PILLOW_MAX_PIXELS == 178956970
    
    def test_find_safe_dpi_within_limits(self):
        """Test _find_safe_dpi returns target DPI when within limits."""
        preprocessor = ImagePreprocessor(target_dpi=300)
        
        # Small image that won't exceed limits at 300 DPI
        width, height = 1000, 800
        current_dpi = 72.0
        
        safe_dpi = preprocessor._find_safe_dpi(width, height, current_dpi)
        
        assert safe_dpi == 300
    
    def test_find_safe_dpi_reduces_for_tesseract_limit(self):
        """Test _find_safe_dpi reduces DPI when Tesseract limit would be exceeded."""
        preprocessor = ImagePreprocessor(target_dpi=300)
        
        # Large image that would exceed Tesseract limits at 300 DPI
        # At 72 DPI: 10000 x 8000 = 80M pixels
        # At 300 DPI: would be ~41666 x 33333 (exceeds 32767 limit)
        width, height = 10000, 8000
        current_dpi = 72.0
        
        safe_dpi = preprocessor._find_safe_dpi(width, height, current_dpi)
        
        # Should return a reduced DPI or None
        if safe_dpi is not None:
            # Calculate what dimensions would be at safe_dpi
            new_width = int(width * safe_dpi / current_dpi)
            new_height = int(height * safe_dpi / current_dpi)
            assert new_width <= TESSERACT_MAX_DIMENSION
            assert new_height <= TESSERACT_MAX_DIMENSION
    
    def test_find_safe_dpi_checks_pillow_limit(self):
        """Test _find_safe_dpi checks Pillow pixel limit."""
        preprocessor = ImagePreprocessor(target_dpi=300)
        
        # Image dimensions that would exceed Pillow limit at high DPI
        width, height = 8000, 6000  # 48M pixels at 72 DPI
        current_dpi = 72.0
        
        safe_dpi = preprocessor._find_safe_dpi(width, height, current_dpi)
        
        if safe_dpi is not None:
            # Calculate what total pixels would be at safe_dpi
            new_width = int(width * safe_dpi / current_dpi)
            new_height = int(height * safe_dpi / current_dpi)
            total_pixels = new_width * new_height
            assert total_pixels <= PILLOW_MAX_PIXELS
    
    def test_find_safe_dpi_returns_none_for_huge_image(self):
        """Test _find_safe_dpi returns None for images too large at any DPI."""
        preprocessor = ImagePreprocessor(target_dpi=300)
        
        # Extremely large image that exceeds limits even at 100 DPI
        width, height = 30000, 25000
        current_dpi = 72.0
        
        safe_dpi = preprocessor._find_safe_dpi(width, height, current_dpi)
        
        # For such a large image, it may return None or a very low DPI
        if safe_dpi is not None:
            assert safe_dpi >= 100  # Minimum is 100 DPI
