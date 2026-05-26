#!/usr/bin/env python3
"""
Smoke test for Document Processor OCR models.

This is a quick smoke test that verifies:
1. All dependencies are installed
2. At least one OCR engine is available
3. At least one AI model can be loaded

Usage:
    python smoke_test.py

Exit codes:
    0 - All tests passed
    1 - Critical failure (missing dependencies)
    2 - Warning (some models not available)
"""

import sys
import logging
from pathlib import Path

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check that critical dependencies are installed."""
    print("🔍 Checking dependencies...")

    missing = []

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not installed")
        missing.append("torch")

    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError:
        print("  ✗ Transformers not installed")
        missing.append("transformers")

    try:
        from PIL import Image
        import PIL
        print(f"  ✓ Pillow {PIL.__version__}")
    except ImportError:
        print("  ✗ Pillow not installed")
        missing.append("Pillow")

    if missing:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def check_ocr():
    """Check that at least one OCR engine is available."""
    print("\n🔍 Checking OCR engines...")

    has_ocr = False

    try:
        import paddleocr
        print("  ✓ PaddleOCR available")
        has_ocr = True
    except ImportError:
        print("  ⚠ PaddleOCR not installed")

    try:
        import pytesseract
        print("  ✓ Tesseract available")
        has_ocr = True
    except ImportError:
        print("  ⚠ Tesseract not installed")

    if not has_ocr:
        print("\n⚠️ No OCR engines available")
        print("Install with: pip install paddleocr pytesseract")
        return False

    return True


def test_model():
    """Test loading at least one model."""
    print("\n🔍 Testing model loading...")

    # Add src to path
    src_path = Path(__file__).parent
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    try:
        from src.models import DonutModel

        print("  Loading Donut model (this may take a moment)...")
        model = DonutModel(
            model_name_or_path="naver-clova-ix/donut-base-finetuned-cord-v2",
            device="cpu"
        )
        model.load()
        print("  ✓ Donut model loaded successfully")
        return True

    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False


def main():
    """Run smoke tests."""
    print("=" * 70)
    print("Document Processor OCR - Smoke Test")
    print("=" * 70)

    # Test 1: Dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n" + "=" * 70)
        print("❌ SMOKE TEST FAILED - Missing dependencies")
        print("=" * 70)
        return 1

    # Test 2: OCR engines
    ocr_ok = check_ocr()

    # Test 3: Model loading
    model_ok = test_model()

    # Summary
    print("\n" + "=" * 70)
    if deps_ok and model_ok:
        print("✅ SMOKE TEST PASSED")
        print("All critical components are working!")
        if not ocr_ok:
            print("\n⚠️ Note: No OCR engines installed (optional)")
            print("=" * 70)
            return 2
        print("=" * 70)
        return 0
    else:
        print("❌ SMOKE TEST FAILED")
        print("Some components are not working correctly")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
