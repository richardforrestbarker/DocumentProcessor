"""
Debug output utilities for the Receipt OCR CLI.

Handles saving intermediary images and visualizations during processing pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Import cv2 at module level for performance
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)


class DebugOutputManager:
    """
    Manages debug output for the OCR processing pipeline.
    
    Saves intermediary images for each processing step:
    - Source image
    - Grayscale conversion
    - Denoised image
    - Deskewed image
    - Contrast enhanced image
    - OCR bounding boxes visualization
    - Result bounding boxes visualization
    """
    
    def __init__(self, output_dir: str = "./debug_output", job_id: Optional[str] = None):
        """
        Initialize debug output manager.
        
        Args:
            output_dir: Directory to save debug output files
            job_id: Optional job identifier for organizing output
        """
        self.output_dir = Path(output_dir)
        self.job_id = job_id or "debug"
        self.page_count = 0
        
        # Create output directory
        self.job_dir = self.output_dir / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Debug output directory: {self.job_dir}")
    
    def _save_image(self, image: np.ndarray, filename: str) -> str:
        """
        Save image to file.
        
        Args:
            image: Image as numpy array
            filename: Filename for the output
            
        Returns:
            Path to saved file
        """
        try:
            from PIL import Image
            
            # Handle different image formats
            if len(image.shape) == 2:
                # Grayscale - convert to RGB for saving
                pil_image = Image.fromarray(image, mode='L')
            elif image.shape[2] == 3:
                # RGB
                pil_image = Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                # RGBA
                pil_image = Image.fromarray(image, mode='RGBA')
            else:
                pil_image = Image.fromarray(image)
            
            filepath = self.job_dir / filename
            pil_image.save(filepath)
            logger.info(f"Debug: Saved {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"Failed to save debug image {filename}: {e}")
            return ""
    
    def save_source_image(self, image: np.ndarray, page_num: int = 1) -> str:
        """Save the original source image."""
        self.page_count = page_num
        filename = f"step_01_source_page{page_num:02d}.png"
        return self._save_image(image, filename)
    
    def save_grayscale_image(self, image: np.ndarray, page_num: int = 1) -> str:
        """Save the grayscale converted image."""
        filename = f"step_02_grayscale_page{page_num:02d}.png"
        return self._save_image(image, filename)
    
    def save_denoised_image(self, image: np.ndarray, page_num: int = 1) -> str:
        """Save the denoised image."""
        filename = f"step_03_denoised_page{page_num:02d}.png"
        return self._save_image(image, filename)
    
    def save_deskewed_image(self, image: np.ndarray, page_num: int = 1) -> str:
        """Save the deskewed image."""
        filename = f"step_04_deskewed_page{page_num:02d}.png"
        return self._save_image(image, filename)
    
    def save_contrast_enhanced_image(self, image: np.ndarray, page_num: int = 1) -> str:
        """Save the contrast enhanced image."""
        filename = f"step_05_contrast_enhanced_page{page_num:02d}.png"
        return self._save_image(image, filename)
    
    def save_preprocessed_image(self, image: np.ndarray, page_num: int = 1) -> str:
        """Save the final preprocessed image."""
        filename = f"step_06_preprocessed_final_page{page_num:02d}.png"
        return self._save_image(image, filename)
    
    def save_ocr_bounding_boxes(
        self,
        image: np.ndarray,
        words: List[Dict[str, Any]],
        page_num: int = 1,
        ocr_engine: str = "ocr"
    ) -> str:
        """
        Save image with OCR bounding boxes drawn.
        
        Args:
            image: Source image
            words: List of words with bounding boxes
            page_num: Page number
            ocr_engine: Name of OCR engine used (paddle or tesseract)
            
        Returns:
            Path to saved file
        """
        if not HAS_CV2:
            logger.warning("cv2 not available, cannot draw OCR bounding boxes")
            return ""
        
        try:
            # Make a copy to draw on
            vis_image = image.copy()
            
            # Ensure RGB format for drawing
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
            
            # Draw bounding boxes for each word
            for word in words:
                box = word.get('box', [])
                text = word.get('text', '')
                confidence = word.get('confidence', 0)
                
                if len(box) >= 4:
                    x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # Color based on confidence (green = high, red = low) in RGB format
                    color_intensity = int(confidence * 255) if isinstance(confidence, float) else 255
                    color = (0, color_intensity, 255 - color_intensity)
                    
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x0, y0), (x1, y1), color, 2)
                    
                    # Draw text label
                    label = f"{text} ({confidence:.2f})" if isinstance(confidence, float) else text
                    cv2.putText(vis_image, label[:20], (x0, y0 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            filename = f"step_07_{ocr_engine}_bboxes_page{page_num:02d}.png"
            return self._save_image(vis_image, filename)
            
        except Exception as e:
            logger.warning(f"Failed to save OCR bounding boxes: {e}")
            return ""
    
    def save_result_bounding_boxes(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        page_num: int = 1
    ) -> str:
        """
        Save image with result.json bounding boxes drawn.
        
        Draws bounding boxes for extracted fields from the final result.
        
        Args:
            image: Source image
            result: Result dictionary with extracted fields
            page_num: Page number
            
        Returns:
            Path to saved file
        """
        if not HAS_CV2:
            logger.warning("cv2 not available, cannot draw result bounding boxes")
            return ""
        
        try:
            # Make a copy to draw on
            vis_image = image.copy()
            
            # Ensure RGB format for drawing
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
            
            # Get image dimensions for scaling normalized boxes (0-1000 scale)
            img_height, img_width = vis_image.shape[:2]
            scale_x = img_width / 1000
            scale_y = img_height / 1000
            
            # Define colors for different field types
            field_colors = {
                'vendor_name': (255, 0, 0),      # Red
                'date': (0, 255, 0),              # Green
                'total_amount': (0, 0, 255),      # Blue
                'subtotal': (255, 165, 0),        # Orange
                'tax_amount': (128, 0, 128),      # Purple
                'line_items': (0, 255, 255),      # Cyan
            }
            
            # Draw bounding boxes for each extracted field
            fields_to_draw = ['vendor_name', 'date', 'total_amount', 'subtotal', 'tax_amount']
            
            for field_name in fields_to_draw:
                field_data = result.get(field_name)
                if field_data and isinstance(field_data, dict):
                    box = field_data.get('box')
                    value = field_data.get('value', '')
                    
                    if box:
                        # Scale from 0-1000 to actual image dimensions
                        x0 = int(box.get('x0', 0) * scale_x)
                        y0 = int(box.get('y0', 0) * scale_y)
                        x1 = int(box.get('x1', 0) * scale_x)
                        y1 = int(box.get('y1', 0) * scale_y)
                        
                        color = field_colors.get(field_name, (255, 255, 255))
                        
                        # Draw rectangle
                        cv2.rectangle(vis_image, (x0, y0), (x1, y1), color, 3)
                        
                        # Draw label
                        label = f"{field_name}: {str(value)[:20]}"
                        cv2.putText(vis_image, label, (x0, y0 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw line items
            line_items = result.get('line_items', [])
            for i, item in enumerate(line_items):
                box = item.get('box')
                description = item.get('description', f'Item {i+1}')
                
                if box:
                    x0 = int(box.get('x0', 0) * scale_x)
                    y0 = int(box.get('y0', 0) * scale_y)
                    x1 = int(box.get('x1', 0) * scale_x)
                    y1 = int(box.get('y1', 0) * scale_y)
                    
                    color = field_colors.get('line_items', (0, 255, 255))
                    
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x0, y0), (x1, y1), color, 2)
                    
                    # Draw label
                    label = f"Item: {description[:15]}"
                    cv2.putText(vis_image, label, (x0, y0 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Also draw page-level word bounding boxes if available
            pages = result.get('pages', [])
            for page in pages:
                if page.get('page_number') == page_num:
                    words = page.get('words', [])
                    for word in words:
                        box = word.get('box')
                        if box:
                            x0 = int(box.get('x0', 0) * scale_x)
                            y0 = int(box.get('y0', 0) * scale_y)
                            x1 = int(box.get('x1', 0) * scale_x)
                            y1 = int(box.get('y1', 0) * scale_y)
                            
                            # Light gray for word-level boxes
                            cv2.rectangle(vis_image, (x0, y0), (x1, y1), (200, 200, 200), 1)
            
            filename = f"step_08_result_bboxes_page{page_num:02d}.png"
            return self._save_image(vis_image, filename)
            
        except Exception as e:
            logger.warning(f"Failed to save result bounding boxes: {e}")
            return ""
    
    def save_debug_summary(self, result: Dict[str, Any]) -> str:
        """
        Save a summary of the debug output including paths to all generated files.
        
        Args:
            result: Final processing result
            
        Returns:
            Path to summary file
        """
        try:
            # List all files in the debug directory
            debug_files = sorted([f.name for f in self.job_dir.iterdir() if f.is_file()])
            
            summary = {
                "job_id": self.job_id,
                "debug_output_directory": str(self.job_dir),
                "files_generated": debug_files,
                "processing_steps": [
                    "step_01: Source image",
                    "step_02: Grayscale conversion",
                    "step_03: Denoising (if enabled)",
                    "step_04: Deskewing (if enabled)",
                    "step_05: Contrast enhancement",
                    "step_06: Final preprocessed image",
                    "step_07: OCR engine bounding boxes",
                    "step_08: Result bounding boxes"
                ],
                "result": result
            }
            
            filepath = self.job_dir / "debug_summary.json"
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Debug: Saved summary to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"Failed to save debug summary: {e}")
            return ""
