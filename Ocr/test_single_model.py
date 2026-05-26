#!/usr/bin/env python3
"""
Quick single model test script.

Usage:
    python test_single_model.py donut
    python test_single_model.py idefics2
    python test_single_model.py phi3_vision
    python test_single_model.py internvl
    python test_single_model.py qwen2_vl
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

MODELS = {
    "donut": {
        "name": "Donut",
        "path": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "class": "DonutModel"
    },
    "idefics2": {
        "name": "IDEFICS2",
        "path": "HuggingFaceM4/idefics2-8b",
        "class": "IDEFICS2Model"
    },
    "phi3_vision": {
        "name": "Phi-3-Vision",
        "path": "microsoft/Phi-3-vision-128k-instruct",
        "class": "Phi3VisionModel"
    },
    "internvl": {
        "name": "InternVL",
        "path": "OpenGVLab/InternVL2-8B",
        "class": "InternVLModel"
    },
    "qwen2_vl": {
        "name": "Qwen2-VL",
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "class": "Qwen2VLModel"
    }
}


def test_model(model_id: str) -> bool:
    """Test loading a single model."""
    if model_id not in MODELS:
        logger.error(f"Unknown model: {model_id}")
        logger.error(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_id]
    logger.info(f"Testing {model_info['name']} model...")
    logger.info(f"Model path: {model_info['path']}")

    try:
        # Import the model class
        from src.models import (
            DonutModel, IDEFICS2Model, Phi3VisionModel,
            InternVLModel, Qwen2VLModel
        )

        model_classes = {
            "DonutModel": DonutModel,
            "IDEFICS2Model": IDEFICS2Model,
            "Phi3VisionModel": Phi3VisionModel,
            "InternVLModel": InternVLModel,
            "Qwen2VLModel": Qwen2VLModel
        }

        model_class = model_classes[model_info["class"]]

        # Initialize model
        logger.info("Initializing model...")
        model = model_class(
            model_name_or_path=model_info["path"],
            device="cpu"
        )

        # Load model
        logger.info("Loading model (this may take a while)...")
        model.load()

        logger.info(f"✓ {model_info['name']} loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to load {model_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_model.py <model_id>")
        print(f"Available models: {', '.join(MODELS.keys())}")
        sys.exit(1)

    model_id = sys.argv[1]
    success = test_model(model_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
