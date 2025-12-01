"""
Image preprocessing utilities for receipt OCR.

Uses ImageMagick CLI for optimal image processing and Tesseract compatibility.

Preprocessing pipeline order for best OCR accuracy:
1. Deskew (rotation correction)
2. Contrast enhancement
3. Grayscale conversion
4. Remove background
5. Denoise
6. Convert to TIFF format
7. Fix resolution (300 DPI) - last step to avoid large intermediate files

Note: Resolution is set last to avoid creating large intermediate files.
Tesseract has a maximum image size limit of 32767 pixels per dimension.
The preprocessor will automatically reduce DPI if the resampled image would exceed this limit.

Shell scripts are located in the scripts/ directory and can be run manually for debugging.
"""

import logging
import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..cli.debug_output import DebugOutputManager

logger = logging.getLogger(__name__)

# Get the scripts directory path
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# Tesseract maximum image dimension (32767 pixels)
TESSERACT_MAX_DIMENSION = 32767

# Pillow maximum pixel count (178956970 pixels) - to prevent DOS attacks
PILLOW_MAX_PIXELS = 178956970


class ImagePreprocessor:
    """
    Preprocesses receipt images to improve OCR accuracy.
    
    Uses ImageMagick CLI for image processing operations.
    
    Preprocessing pipeline order:
    1. Deskew - correct rotation/skew
    2. Contrast enhancement - improve text visibility
    3. Grayscale - convert to grayscale
    4. Remove background - isolate text from background noise
    5. Denoise - reduce noise while preserving edges
    6. Convert to TIFF - optimal format for Tesseract OCR
    7. Fix resolution - ensure optimal DPI for best OCR results (last step)
    
    Note: Resolution is set as the last step to avoid creating large intermediate files.
    The preprocessor automatically reduces DPI if the resampled image would exceed
    Tesseract's maximum dimension limit (32767 pixels).
    
    Configurable parameters:
    - fuzz_percent: Tolerance for background removal (0-100%)
    - deskew_threshold: Sensitivity for skew detection (0-100%)
    - contrast_type: 'sigmoidal', 'linear', or 'none'
    - contrast_strength: Intensity for sigmoidal contrast (1-10 typical)
    - contrast_midpoint: Midpoint for sigmoidal contrast (0-200%, >100 brightens)
    """
    
    def __init__(
        self,
        target_dpi: Optional[int] = 300,
        denoise: bool = True,
        deskew: bool = True,
        enhance_contrast: bool = True,
        debug_manager: Optional['DebugOutputManager'] = None,
        fuzz_percent: int = 30,
        deskew_threshold: int = 40,
        contrast_type: str = "sigmoidal",
        contrast_strength: float = 3,
        contrast_midpoint: int = 120
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_dpi: Target DPI for resolution normalization (default 300). 
                        Set to None to skip DPI resampling (for preprocess-only phase).
            denoise: Whether to apply denoising
            deskew: Whether to correct skew
            enhance_contrast: Whether to enhance contrast
            debug_manager: Optional DebugOutputManager for saving intermediate steps
            fuzz_percent: Fuzz percentage for background removal (0-100, default 30)
            deskew_threshold: Deskew threshold percentage (0-100, default 40)
            contrast_type: Contrast type - 'sigmoidal', 'linear', or 'none' (default 'sigmoidal')
            contrast_strength: Contrast strength for sigmoidal (1-10 typical, default 3)
            contrast_midpoint: Contrast midpoint for sigmoidal (0-200%, default 120)
        """
        self.target_dpi = target_dpi
        self.denoise = denoise
        self.deskew = deskew
        self.enhance_contrast = enhance_contrast
        self.debug_manager = debug_manager
        self.fuzz_percent = fuzz_percent
        self.deskew_threshold = deskew_threshold
        self.contrast_type = contrast_type
        self.contrast_strength = contrast_strength
        self.contrast_midpoint = contrast_midpoint
        
        # Verify ImageMagick is installed
        self._check_imagemagick()
    
    def _check_imagemagick(self):
        """Check if ImageMagick is installed and available."""
        try:
            result = subprocess.run(
                ["magick", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.debug(f"ImageMagick version: {result.stdout.split('\n')[0]}")
                return
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            pass
        
        raise RuntimeError(
            "ImageMagick is not installed. Please install it:\n"
            "  Ubuntu/Debian: sudo apt-get install imagemagick\n"
            "  macOS: brew install imagemagick\n"
            "  Windows: Download from https://imagemagick.org/script/download.php"
        )
    
    def _run_script(self, script_name: str, *args) -> bool:
        """
        Run a preprocessing shell script.
        
        Args:
            script_name: Name of the script (without path)
            *args: Arguments to pass to the script
            
        Returns:
            True if successful, False otherwise
        """
        script_path = SCRIPTS_DIR / script_name
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        try:
            cmd = [str(script_path)] + [str(arg) for arg in args]
            logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"Script {script_name} failed: {result.stderr}")
                return False
            
            logger.debug(f"Script output: {result.stdout.strip()}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Script {script_name} timed out")
            return False
        except Exception as e:
            logger.error(f"Error running script {script_name}: {e}")
            return False
    
    def _run_imagemagick_cmd(self, args: list) -> bool:
        """
        Run an ImageMagick command directly.
        
        Uses 'magick' command (ImageMagick 7+).
        
        Args:
            args: Arguments to pass to the ImageMagick command
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ["magick"] + args
            logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"ImageMagick command failed: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("ImageMagick command timed out")
            return False
        except Exception as e:
            logger.error(f"Error running ImageMagick: {e}")
            return False
    
    def _get_image_info(self, image_path: str) -> Tuple[int, int, float, float]:
        """
        Get image dimensions and DPI using ImageMagick identify.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (width, height, x_dpi, y_dpi)
        """
        try:
            result = subprocess.run(
                ["magick", "identify", "-format", "%w %h %x %y", image_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) >= 4:
                    width = int(parts[0])
                    height = int(parts[1])
                    # DPI might have units like "72 PixelsPerInch", extract just the number
                    x_dpi = float(parts[2].split()[0]) if parts[2] else 72.0
                    y_dpi = float(parts[3].split()[0]) if parts[3] else 72.0
                    return width, height, x_dpi, y_dpi
        except Exception as e:
            logger.warning(f"Failed to get image info: {e}")
        
        # Default values if identification fails
        return 0, 0, 72.0, 72.0
    
    def _calculate_resampled_dimensions(
        self, 
        width: int, 
        height: int, 
        current_dpi: float, 
        target_dpi: int
    ) -> Tuple[int, int]:
        """
        Calculate what the image dimensions would be after resampling.
        
        Args:
            width: Current width in pixels
            height: Current height in pixels
            current_dpi: Current DPI
            target_dpi: Target DPI
            
        Returns:
            Tuple of (new_width, new_height)
        """
        if current_dpi <= 0:
            current_dpi = 72.0  # Default DPI
        
        scale_factor = target_dpi / current_dpi
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return new_width, new_height
    
    def _find_safe_dpi(
        self, 
        width: int, 
        height: int, 
        current_dpi: float
    ) -> Optional[int]:
        """
        Find a safe DPI that won't exceed Tesseract's or Pillow's size limits.
        
        Tries target_dpi first, then reduces in increments of 50 down to 100 DPI.
        Returns None if even 100 DPI would exceed limits.
        
        Checks:
        - Tesseract max dimension: 32767 pixels per dimension
        - Pillow max pixels: 178956970 total pixels (to prevent DOS attacks)
        
        Args:
            width: Current width in pixels
            height: Current height in pixels
            current_dpi: Current DPI
            
        Returns:
            Safe DPI value, or None if no safe DPI exists
        """
        # Try DPI values from target down to 100 in increments of 50
        for test_dpi in range(self.target_dpi, 99, -50):
            new_width, new_height = self._calculate_resampled_dimensions(
                width, height, current_dpi, test_dpi
            )
            
            # Check both Tesseract dimension limit and Pillow pixel limit
            total_pixels = new_width * new_height
            within_tesseract_limit = (
                new_width <= TESSERACT_MAX_DIMENSION and 
                new_height <= TESSERACT_MAX_DIMENSION
            )
            within_pillow_limit = total_pixels <= PILLOW_MAX_PIXELS
            
            if within_tesseract_limit and within_pillow_limit:
                if test_dpi != self.target_dpi:
                    logger.info(
                        f"Reduced target DPI from {self.target_dpi} to {test_dpi} "
                        f"to stay within limits ({new_width}x{new_height} = {total_pixels} pixels)"
                    )
                return test_dpi
        
        # Even 100 DPI is too large
        logger.warning(
            f"Image would exceed size limits even at 100 DPI. "
            f"Skipping resolution adjustment."
        )
        return None
    
    def preprocess(self, image_path: str, page_num: int = 1) -> Tuple[np.ndarray, int, int]:
        """
        Preprocess image for OCR using ImageMagick CLI.
        
        Pipeline order: Deskew -> Contrast -> Grayscale -> Remove Background -> 
                       Denoise -> TIFF -> Resolution
        
        Args:
            image_path: Path to image file
            page_num: Page number for debug output
            
        Returns:
            Tuple of (preprocessed image as numpy array (RGB), width, height)
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        # Create temp directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix="ocr_preprocess_")
        
        try:
            current_file = image_path
            step = 1
            
            # Step 1: Deskew (optional)
            if self.deskew:
                logger.info(f"Step {step}: Deskewing (threshold: {self.deskew_threshold}%)...")
                next_file = os.path.join(temp_dir, f"step{step}_deskew.jpg")
                if not self._run_imagemagick_cmd([
                    current_file, "-deskew", f"{self.deskew_threshold}%", "-background", "white", "+repage", next_file
                ]):
                    raise RuntimeError("Failed to deskew")
                current_file = next_file
                self._save_debug_image(next_file, "deskewed", page_num)
                step += 1
            
            
            # Step 3: Grayscale
            logger.info(f"Step {step}: Converting to grayscale...")
            next_file = os.path.join(temp_dir, f"step{step}_gray.jpg")
            if not self._run_imagemagick_cmd([
                current_file, "-colorspace", "Gray", next_file
            ]):
                raise RuntimeError("Failed to convert to grayscale")
            current_file = next_file
            self._save_debug_image(next_file, "grayscale", page_num)
            step += 1
            
            # Step 4: Remove background
            logger.info(f"Step {step}: Removing background (fuzz: {self.fuzz_percent}%)...")
            next_file = os.path.join(temp_dir, f"step{step}_nobg.jpg")
            if not self._run_imagemagick_cmd([
                current_file, "-fuzz", f"{self.fuzz_percent}%", "-transparent", "white",
                "-background", "white", "-alpha", "remove", "-auto-level", next_file
            ]):
                raise RuntimeError("Failed to remove background")
            current_file = next_file
            self._save_debug_image(next_file, "background_removed", page_num)
            step += 1

             # Step 5: Contrast enhancement (optional)
            if self.enhance_contrast and self.contrast_type != "none":
                logger.info(f"Step {step}: Enhancing contrast (type: {self.contrast_type})...")
                next_file = os.path.join(temp_dir, f"step{step}_contrast.jpg")
                
                if self.contrast_type == "sigmoidal":
                    # Sigmoidal contrast: -sigmoidal-contrast strength x midpoint%
                    contrast_arg = f"{self.contrast_strength}x{self.contrast_midpoint}%"
                    if not self._run_imagemagick_cmd([
                        current_file, "-auto-level", "-sigmoidal-contrast", contrast_arg, next_file
                    ]):
                        raise RuntimeError("Failed to enhance contrast")
                elif self.contrast_type == "linear":
                    # Linear contrast: just auto-level (histogram stretch)
                    if not self._run_imagemagick_cmd([
                        current_file, "-auto-level", next_file
                    ]):
                        raise RuntimeError("Failed to enhance contrast")
                
                current_file = next_file
                self._save_debug_image(next_file, "contrast_enhanced", page_num)
                step += 1

            
            # Step 5: Denoise (optional)
            if self.denoise:
                logger.info(f"Step {step}: Denoising...")
                next_file = os.path.join(temp_dir, f"step{step}_denoise.jpg")
                if not self._run_imagemagick_cmd([
                    current_file, "-enhance", next_file
                ]):
                    raise RuntimeError("Failed to denoise")
                current_file = next_file
                self._save_debug_image(next_file, "denoised", page_num)
                step += 1
            
            # Step 6: Convert to TIFF
            logger.info(f"Step {step}: Converting to TIFF...")
            next_file = os.path.join(temp_dir, f"step{step}_convert.tiff")
            if not self._run_imagemagick_cmd([current_file, "-compress", "lzw", next_file]):
                raise RuntimeError(f"Failed to convert to TIFF")
            current_file = next_file
            self._save_debug_image(next_file, "convert", page_num)
            step += 1
            
            # Step 7: Fix resolution (with size limit checking) - only if target_dpi is set
            if self.target_dpi is not None:
                logger.info(f"Step {step}: Checking resolution...")
                width, height, x_dpi, y_dpi = self._get_image_info(current_file)
                current_dpi = min(x_dpi, y_dpi) if x_dpi > 0 and y_dpi > 0 else 72.0
                
                logger.info(f"Current image: {width}x{height} at {current_dpi:.0f} DPI")
                
                # Find a safe DPI that won't exceed Tesseract/Pillow limits
                safe_dpi = self._find_safe_dpi(width, height, current_dpi)
                
                if safe_dpi is not None:
                    logger.info(f"Step {step}: Fixing resolution to {safe_dpi} DPI...")
                    next_file = os.path.join(temp_dir, f"step{step}_resolution.tiff")
                    if not self._run_imagemagick_cmd([
                        current_file, "-resample", str(safe_dpi), 
                        "-units", "PixelsPerInch", next_file
                    ]):
                        raise RuntimeError(f"Failed to fix resolution to {safe_dpi} DPI")
                    current_file = next_file
                    self._save_debug_image(next_file, "resolution_fixed", page_num)
                else:
                    logger.warning(
                        f"Skipping resolution adjustment - image at {current_dpi:.0f} DPI "
                        f"would exceed size limits at any higher DPI"
                    )
            else:
                logger.info("Skipping DPI resampling (preprocess-only mode)")
            
            # Load the final preprocessed image using ImageMagick to convert to RGB PNG
            # This avoids using Pillow for loading the TIFF
            logger.info("Finished preprocessing image, prepaering for OCR")
            # Now load the PNG as numpy array - we need Pillow here for numpy conversion
            # This is the minimal Pillow usage required for OCR engine compatibility
            from PIL import Image
            pil_img = Image.open(current_file)
            result = np.array(pil_img)
            
            # Get final dimensions
            final_height, final_width = result.shape[:2]
            
            # Ensure RGB format
            if len(result.shape) == 2:
                # Grayscale, convert to RGB
                result = np.stack([result, result, result], axis=-1)
            elif result.shape[2] == 4:
                # RGBA, drop alpha
                result = result[:, :, :3]
            
            # Save final preprocessed debug image
            if self.debug_manager:
                self.debug_manager.save_preprocessed_image(result, page_num)
            
            logger.info(f"Preprocessing complete: {final_width}x{final_height}")
            return result, final_width, final_height
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _save_debug_image(self, image_path: str, step_name: str, page_num: int):
        """Save a debug image for the current step.
        
        Copies the image file directly to the debug output job folder to preserve 
        the original format and avoid unintended conversions.
        """
        if not self.debug_manager:
            return
        try:
            # Get the original file extension to preserve format
            _, ext = os.path.splitext(image_path)
            if not ext:
                ext = ".tiff"
            
            # Use job_dir from debug manager (the proper job folder)
            debug_dir = self.debug_manager.job_dir if hasattr(self.debug_manager, 'job_dir') else (
                self.debug_manager.output_dir if hasattr(self.debug_manager, 'output_dir') else tempfile.gettempdir()
            )
            
            debug_path = os.path.join(
                str(debug_dir),
                f"page_{page_num}_{step_name}{ext}"
            )
            
            # Copy the file directly to preserve format (no conversion)
            shutil.copy2(image_path, debug_path)
            logger.debug(f"Saved debug image: {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug image for {step_name}: {e}")
