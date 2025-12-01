"""
CLI utility functions.

Common utilities for the Receipt OCR CLI.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def check_dependencies() -> None:
    """
    Check if required dependencies are available.
    
    Raises:
        ImportError: If required dependencies are missing
    """
    try:
        import numpy
    except ImportError:
        raise ImportError("NumPy is required. Install with: pip install numpy")
    
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")


def get_device(device_str: str) -> str:
    """
    Resolve device string to actual device.
    
    Args:
        device_str: Device string ('auto', 'cuda', or 'cpu')
        
    Returns:
        Resolved device string ('cuda' or 'cpu')
    """
    if device_str == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA is available, using GPU")
                return "cuda"
            else:
                logger.info("CUDA not available, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, defaulting to CPU")
            return "cpu"
    return device_str


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the CLI.
    
    Args:
        verbose: Whether to enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image path exists and is a valid image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, raises exception otherwise
    """
    from pathlib import Path
    
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported image format: {path.suffix}. Supported formats: {valid_extensions}")
    
    return True


def load_image(image_path: str) -> Any:
    """
    Load an image file and return as numpy array.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in RGB format
    """
    import numpy as np
    from PIL import Image
    
    validate_image_path(image_path)
    
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def get_image_dimensions(image) -> tuple:
    """
    Get image dimensions.
    
    Args:
        image: Numpy array image
        
    Returns:
        Tuple of (height, width)
    """
    if len(image.shape) == 3:
        return image.shape[:2]
    elif len(image.shape) == 2:
        return image.shape
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")
