"""
Configuration management for receipt OCR service.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    'model': {
        'name_or_path': 'naver-clova-ix/donut-base-finetuned-cord-v2',
        'type': 'donut',  # donut, idefics2, or layoutlmv3
        'device': 'auto',
        'num_labels': 13,  # Only used for layoutlmv3
    },
    'ocr': {
        'engine': 'paddle',
        'detection_mode': 'word',
        'lang': 'en',
        'use_gpu': True,
    },
    'preprocessing': {
        'target_dpi': 300,
        'denoise': True,
        'deskew': True,
        'enhance_contrast': True,
        # ImageMagick preprocessing parameters
        'fuzz_percent': 30,  # Fuzz percentage for background removal (0-100)
        'deskew_threshold': 40,  # Deskew threshold percentage (0-100)
        'contrast_type': 'sigmoidal',  # Contrast type: 'sigmoidal', 'linear', or 'none'
        'contrast_strength': 3,  # Contrast strength (for sigmoidal: 1-10 typical)
        'contrast_midpoint': 120,  # Contrast midpoint percentage (for sigmoidal: 0-200)
    },
    'postprocessing': {
        'min_confidence': 0.5,
        'verify_totals': True,
    },
    'storage': {
        'temp_dir': './temp',
        'model_cache_dir': './models',
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # Merge with defaults (user config overrides defaults)
                    config = _merge_configs(config, user_config)
                    logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
    
    return config


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {output_path}: {e}")


def get_device(device_str: str = 'auto') -> str:
    """
    Resolve device string to actual device.
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'cpu')
        
    Returns:
        Resolved device string
    """
    if device_str == 'auto':
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    
    return device_str
