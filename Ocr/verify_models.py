#!/usr/bin/env python3
"""
AI Model Verification Script

This script verifies that all AI models are installed correctly and can be loaded.
It also checks if PyTorch installation matches available hardware (GPU vs CPU) and
offers to install the correct version automatically.

Features:
- Detects CUDA hardware (NVIDIA GPU + CUDA Toolkit)
- Checks if PyTorch installation matches hardware
- Offers to install GPU or CPU version automatically
- Tests all 5 vision-language models (Donut, IDEFICS2, Phi-3-Vision, InternVL, Qwen2-VL)
- Verifies OCR engines (PaddleOCR, Tesseract)
- Checks required dependencies (PyTorch, Transformers, etc.)

Usage:
    python verify_models.py [options]

Options:
    --quick              Only verify dependencies and model availability (skip loading)
    --gpu                Verify GPU/CUDA support and load models on GPU
    --install-cuda       Force installation of PyTorch with CUDA support
    --skip-pytorch-check Skip automatic PyTorch version check/recommendation
    --json FILE          Save results to JSON file

Examples:
    # Interactive check with PyTorch installation recommendation
    python verify_models.py --quick

    # Force CUDA installation
    python verify_models.py --install-cuda

    # Full verification without installation prompts
    python verify_models.py --skip-pytorch-check

    # Test GPU performance
    python verify_models.py --gpu
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model definitions with their default configurations
MODELS = {
    "donut": {
        "name": "Donut",
        "path": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "license": "MIT",
        "class_name": "DonutModel"
    },
    "idefics2": {
        "name": "IDEFICS2",
        "path": "HuggingFaceM4/idefics2-8b",
        "license": "Apache 2.0",
        "class_name": "IDEFICS2Model"
    },
    "phi3_vision": {
        "name": "Phi-3-Vision",
        "path": "microsoft/Phi-3-vision-128k-instruct",
        "license": "MIT",
        "class_name": "Phi3VisionModel"
    },
    "internvl": {
        "name": "InternVL",
        "path": "OpenGVLab/InternVL2-8B",
        "license": "MIT",
        "class_name": "InternVLModel"
    },
    "qwen2_vl": {
        "name": "Qwen2-VL",
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "license": "Apache 2.0",
        "class_name": "Qwen2VLModel"
    }
}


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal."""
    # Handle encoding issues on Windows
    try:
        print(f"{color}{text}{Colors.ENDC}")
    except UnicodeEncodeError:
        # Fallback for terminals that don't support Unicode
        text_ascii = text.replace('✓', '+').replace('✗', 'X').replace('⚠', '!')
        print(f"{color}{text_ascii}{Colors.ENDC}")


def print_header(text: str):
    """Print section header."""
    print_colored(f"\n{'='*70}", Colors.HEADER)
    print_colored(f"{text}", Colors.HEADER + Colors.BOLD)
    print_colored(f"{'='*70}", Colors.HEADER)


def print_success(text: str):
    """Print success message."""
    print_colored(f"✓ {text}", Colors.OKGREEN)


def print_failure(text: str):
    """Print failure message."""
    print_colored(f"✗ {text}", Colors.FAIL)


def print_warning(text: str):
    """Print warning message."""
    print_colored(f"⚠ {text}", Colors.WARNING)


def print_info(text: str):
    """Print info message."""
    print_colored(f"  {text}", Colors.OKCYAN)


def check_cuda_hardware() -> Dict[str, Any]:
    """Check if CUDA hardware is available on the system."""
    import subprocess
    import shutil

    results = {
        "cuda_available": False,
        "cuda_version": None,
        "gpu_detected": False,
        "gpu_name": None
    }

    # Check for nvidia-smi (indicates NVIDIA GPU present)
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
            results["gpu_detected"] = True

            # Try to get GPU name
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5
                )
                results["gpu_name"] = output.stdout.strip().split('\n')[0]
            except:
                pass
        except:
            pass

    # Check for nvcc (indicates CUDA toolkit installed)
    if shutil.which("nvcc"):
        try:
            output = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            # Parse CUDA version from output
            import re
            match = re.search(r"release (\d+\.\d+)", output.stdout)
            if match:
                results["cuda_available"] = True
                results["cuda_version"] = match.group(1)
        except:
            pass

    return results


def check_pytorch_cuda_match() -> Dict[str, Any]:
    """Check if PyTorch installation matches CUDA hardware availability."""
    cuda_hw = check_cuda_hardware()

    results = {
        "cuda_hardware": cuda_hw,
        "pytorch_has_cuda": False,
        "mismatch": False,
        "recommendation": None
    }

    try:
        import torch
        results["pytorch_has_cuda"] = torch.cuda.is_available()
        results["pytorch_version"] = torch.__version__

        # Check for mismatch
        if cuda_hw["cuda_available"] and cuda_hw["gpu_detected"]:
            if not results["pytorch_has_cuda"]:
                results["mismatch"] = True
                results["recommendation"] = "install_cuda"
        elif not cuda_hw["cuda_available"] or not cuda_hw["gpu_detected"]:
            if "+cu" in torch.__version__:
                results["mismatch"] = True
                results["recommendation"] = "install_cpu"
    except ImportError:
        results["pytorch_installed"] = False
        if cuda_hw["cuda_available"] and cuda_hw["gpu_detected"]:
            results["recommendation"] = "install_cuda"
        else:
            results["recommendation"] = "install_cpu"

    return results


def offer_pytorch_installation():
    """Offer to install the correct PyTorch version based on hardware."""
    print_header("PyTorch Installation Check")

    match_result = check_pytorch_cuda_match()
    cuda_hw = match_result["cuda_hardware"]

    # Display hardware status
    if cuda_hw["gpu_detected"]:
        print_success(f"NVIDIA GPU detected: {cuda_hw.get('gpu_name', 'Unknown')}")
    else:
        print_info("No NVIDIA GPU detected")

    if cuda_hw["cuda_available"]:
        print_success(f"CUDA Toolkit detected: version {cuda_hw['cuda_version']}")
    else:
        print_info("CUDA Toolkit not detected")

    # Check for mismatch
    if match_result.get("mismatch"):
        print()
        if match_result["recommendation"] == "install_cuda":
            print_warning("PyTorch is installed with CPU-only support, but you have CUDA hardware!")
            print_info(f"GPU: {cuda_hw.get('gpu_name', 'Unknown')}")
            print_info(f"CUDA: {cuda_hw['cuda_version']}")
            print_info("Installing CUDA-enabled PyTorch will give you 2-4x faster performance.")
            print()

            response = input("Would you like to install PyTorch with CUDA support now? (y/N): ").strip().lower()
            if response == 'y':
                install_pytorch_cuda()
                return True
            else:
                print_warning("Continuing with CPU-only PyTorch. You can install CUDA support later with:")
                print_info("  python verify_models.py --install-cuda")
                print_info("  or run: .\\install_pytorch_cuda.ps1")

        elif match_result["recommendation"] == "install_cpu":
            print_warning("PyTorch is installed with CUDA support, but no CUDA hardware detected.")
            print_info("You may want to install the CPU-only version to save disk space.")

    elif not match_result.get("pytorch_installed", True):
        print_failure("PyTorch is not installed!")
        print()

        if cuda_hw["cuda_available"] and cuda_hw["gpu_detected"]:
            print_info("CUDA hardware detected. Recommending GPU-accelerated installation.")
            response = input("Install PyTorch with CUDA support? (Y/n): ").strip().lower()
            if response != 'n':
                install_pytorch_cuda()
                return True
        else:
            print_info("No CUDA hardware detected. Installing CPU-only version.")
            install_pytorch_cpu()
            return True

    else:
        # Everything matches
        if match_result["pytorch_has_cuda"]:
            print_success("PyTorch with CUDA support is correctly installed")
        else:
            print_success("PyTorch CPU-only version is correctly installed")

    return False


def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    import subprocess

    print()
    print_header("Installing PyTorch with CUDA Support")
    print_warning("This will uninstall the current PyTorch and install the CUDA version.")
    print_info("This may take several minutes and download ~2.5GB...")
    print()

    try:
        # Uninstall current version
        print_info("Uninstalling current PyTorch...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            check=False
        )

        # Install CUDA version
        print_info("Installing PyTorch with CUDA 12.4 support...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu124"],
            check=True
        )

        if result.returncode == 0:
            print_success("PyTorch with CUDA support installed successfully!")
            print()
            print_info("Verifying installation...")

            # Verify
            import importlib
            import torch
            importlib.reload(torch)

            if torch.cuda.is_available():
                print_success(f"CUDA is now available! GPU: {torch.cuda.get_device_name(0)}")
            else:
                print_warning("CUDA support installed but GPU not detected. Check drivers.")

    except subprocess.CalledProcessError as e:
        print_failure(f"Installation failed: {e}")
        print_info("You can try manually with:")
        print_info("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    except Exception as e:
        print_failure(f"Unexpected error: {e}")


def install_pytorch_cpu():
    """Install PyTorch CPU-only version."""
    import subprocess

    print()
    print_header("Installing PyTorch (CPU-only)")
    print_info("Installing CPU-only version...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            check=True
        )

        if result.returncode == 0:
            print_success("PyTorch CPU-only version installed successfully!")

    except subprocess.CalledProcessError as e:
        print_failure(f"Installation failed: {e}")
    except Exception as e:
        print_failure(f"Unexpected error: {e}")


def check_core_dependencies() -> Dict[str, Any]:
    """Check core Python dependencies."""
    print_header("Checking Core Dependencies")

    results = {
        "passed": True,
        "dependencies": {}
    }

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_info(f"Python version: {python_version}")
    if sys.version_info >= (3, 8):
        print_success("Python version >= 3.8")
        results["dependencies"]["python"] = {"status": "OK", "version": python_version}
    else:
        print_failure(f"Python version {python_version} < 3.8 (required)")
        results["passed"] = False
        results["dependencies"]["python"] = {"status": "FAIL", "version": python_version}

    # Check PyTorch
    try:
        import torch
        print_success(f"PyTorch: {torch.__version__}")
        results["dependencies"]["torch"] = {"status": "OK", "version": torch.__version__}

        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            print_success(f"CUDA: Available (version {cuda_version})")
            print_info(f"GPU devices: {device_count}")
            print_info(f"Primary GPU: {device_name}")
            results["dependencies"]["cuda"] = {
                "status": "OK",
                "version": cuda_version,
                "device_count": device_count,
                "device_name": device_name
            }
        else:
            print_warning("CUDA: Not available (CPU-only mode)")
            results["dependencies"]["cuda"] = {"status": "NOT_AVAILABLE"}
    except ImportError:
        print_failure("PyTorch: Not installed")
        results["passed"] = False
        results["dependencies"]["torch"] = {"status": "NOT_INSTALLED"}

    # Check Transformers
    try:
        import transformers
        print_success(f"Transformers: {transformers.__version__}")
        results["dependencies"]["transformers"] = {"status": "OK", "version": transformers.__version__}
    except ImportError:
        print_failure("Transformers: Not installed")
        results["passed"] = False
        results["dependencies"]["transformers"] = {"status": "NOT_INSTALLED"}

    # Check PIL/Pillow
    try:
        from PIL import Image
        import PIL
        print_success(f"Pillow: {PIL.__version__}")
        results["dependencies"]["pillow"] = {"status": "OK", "version": PIL.__version__}
    except ImportError:
        print_failure("Pillow: Not installed")
        results["passed"] = False
        results["dependencies"]["pillow"] = {"status": "NOT_INSTALLED"}

    # Check NumPy
    try:
        import numpy as np
        print_success(f"NumPy: {np.__version__}")
        results["dependencies"]["numpy"] = {"status": "OK", "version": np.__version__}
    except ImportError:
        print_failure("NumPy: Not installed")
        results["passed"] = False
        results["dependencies"]["numpy"] = {"status": "NOT_INSTALLED"}

    return results


def check_ocr_engines() -> Dict[str, Any]:
    """Check OCR engine availability."""
    print_header("Checking OCR Engines")

    results = {
        "passed": False,
        "engines": {}
    }

    # Check PaddleOCR
    try:
        import paddleocr
        print_success("PaddleOCR: Installed")
        results["engines"]["paddleocr"] = {"status": "OK"}
        results["passed"] = True
    except ImportError:
        print_warning("PaddleOCR: Not installed")
        results["engines"]["paddleocr"] = {"status": "NOT_INSTALLED"}

    # Check Tesseract
    try:
        import pytesseract
        print_success("Tesseract (pytesseract): Installed")
        results["engines"]["tesseract"] = {"status": "OK"}
        results["passed"] = True
    except ImportError:
        print_warning("Tesseract: Not installed")
        results["engines"]["tesseract"] = {"status": "NOT_INSTALLED"}

    if not results["passed"]:
        print_warning("No OCR engines installed. At least one OCR engine is recommended.")

    return results


def check_model_files() -> Dict[str, Any]:
    """Check if model files are available locally."""
    print_header("Checking Model Availability")

    results = {
        "models": {}
    }

    # Check Hugging Face cache
    try:
        from transformers import AutoConfig
        from huggingface_hub import scan_cache_dir

        print_info("Scanning Hugging Face cache...")
        try:
            cache_info = scan_cache_dir()
            cached_repos = {repo.repo_id.lower() for repo in cache_info.repos}

            for model_id, model_info in MODELS.items():
                model_path = model_info["path"].lower()
                is_cached = any(model_path in repo for repo in cached_repos)

                if is_cached:
                    print_success(f"{model_info['name']}: Found in cache")
                    results["models"][model_id] = {"status": "CACHED", "path": model_info["path"]}
                else:
                    print_warning(f"{model_info['name']}: Not cached (will download on first use)")
                    results["models"][model_id] = {"status": "NOT_CACHED", "path": model_info["path"]}
        except Exception as e:
            print_warning(f"Could not scan cache: {e}")
            for model_id, model_info in MODELS.items():
                results["models"][model_id] = {"status": "UNKNOWN", "path": model_info["path"]}
    except ImportError:
        print_warning("huggingface_hub not installed, cannot check cache")
        for model_id, model_info in MODELS.items():
            results["models"][model_id] = {"status": "UNKNOWN", "path": model_info["path"]}

    return results


def test_model_loading(quick: bool = False, use_gpu: bool = False) -> Dict[str, Any]:
    """Test loading each model."""
    print_header("Testing Model Loading")

    if quick:
        print_warning("Quick mode: Skipping actual model loading")
        return {"skipped": True}

    results = {
        "passed": True,
        "models": {}
    }

    device = "cuda" if use_gpu else "cpu"
    print_info(f"Using device: {device}")

    # Add src directory to path
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path.parent))

    try:
        from src.models import (
            DonutModel, IDEFICS2Model, Phi3VisionModel, 
            InternVLModel, Qwen2VLModel
        )

        model_classes = {
            "donut": DonutModel,
            "idefics2": IDEFICS2Model,
            "phi3_vision": Phi3VisionModel,
            "internvl": InternVLModel,
            "qwen2_vl": Qwen2VLModel
        }

        for model_id, model_class in model_classes.items():
            model_info = MODELS[model_id]
            print_info(f"\nTesting {model_info['name']}...")

            try:
                # Initialize model
                model = model_class(
                    model_name_or_path=model_info["path"],
                    device=device
                )
                print_success(f"{model_info['name']}: Initialized")

                # Try to load model
                print_info(f"Loading {model_info['name']} (this may take a while)...")
                model.load()
                print_success(f"{model_info['name']}: Loaded successfully")

                results["models"][model_id] = {
                    "status": "OK",
                    "name": model_info["name"],
                    "path": model_info["path"]
                }

                # Clean up to free memory
                del model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

            except Exception as e:
                print_failure(f"{model_info['name']}: Failed to load - {e}")
                results["passed"] = False
                results["models"][model_id] = {
                    "status": "FAIL",
                    "name": model_info["name"],
                    "error": str(e)
                }
    except ImportError as e:
        print_failure(f"Failed to import model classes: {e}")
        results["passed"] = False
        results["import_error"] = str(e)

    return results


def generate_report(all_results: Dict[str, Any]) -> None:
    """Generate a summary report."""
    print_header("Verification Summary")

    # Core dependencies
    deps = all_results.get("dependencies", {})
    if deps.get("passed", False):
        print_success("Core dependencies: OK")
    else:
        print_failure("Core dependencies: FAILED")

    # OCR engines
    ocr = all_results.get("ocr", {})
    if ocr.get("passed", False):
        print_success("OCR engines: OK (at least one available)")
    else:
        print_warning("OCR engines: No engines installed")

    # Model loading
    models = all_results.get("models", {})
    if models.get("skipped", False):
        print_warning("Model loading: SKIPPED (quick mode)")
    elif models.get("passed", False):
        print_success("Model loading: OK (all models loaded successfully)")
    else:
        print_failure("Model loading: FAILED (some models could not load)")

    # Overall status
    print()
    if deps.get("passed", False) and (models.get("passed", False) or models.get("skipped", False)):
        print_colored("=" * 70, Colors.OKGREEN)
        print_colored("✓ VERIFICATION PASSED", Colors.OKGREEN + Colors.BOLD)
        print_colored("All required components are working correctly!", Colors.OKGREEN)
        print_colored("=" * 70, Colors.OKGREEN)
    else:
        print_colored("=" * 70, Colors.FAIL)
        print_colored("✗ VERIFICATION FAILED", Colors.FAIL + Colors.BOLD)
        print_colored("Some components are missing or not working.", Colors.FAIL)
        print_colored("=" * 70, Colors.FAIL)


def main():
    """Main verification routine."""
    parser = argparse.ArgumentParser(
        description="Verify AI model installation and functionality"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only check dependencies, skip model loading"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Test GPU/CUDA support and load models on GPU"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--install-cuda",
        action="store_true",
        help="Install PyTorch with CUDA support (if CUDA hardware detected)"
    )
    parser.add_argument(
        "--skip-pytorch-check",
        action="store_true",
        help="Skip PyTorch installation check and recommendation"
    )

    args = parser.parse_args()

    print_colored("\n" + "="*70, Colors.HEADER + Colors.BOLD)
    print_colored("AI Model Verification Script", Colors.HEADER + Colors.BOLD)
    print_colored("="*70 + "\n", Colors.HEADER + Colors.BOLD)

    all_results = {}

    # Step 0: Check PyTorch installation matches hardware (unless --skip-pytorch-check)
    if args.install_cuda:
        # Force CUDA installation
        install_pytorch_cuda()
        print()
    elif not args.skip_pytorch_check:
        pytorch_reinstalled = offer_pytorch_installation()
        if pytorch_reinstalled:
            print()
            print_info("PyTorch has been reinstalled. Continuing with verification...")
            print()

    # Step 1: Check core dependencies
    all_results["dependencies"] = check_core_dependencies()

    # Step 2: Check OCR engines
    all_results["ocr"] = check_ocr_engines()

    # Step 3: Check model availability
    all_results["availability"] = check_model_files()

    # Step 4: Test model loading (unless quick mode)
    if not all_results["dependencies"].get("passed", False):
        print_warning("\nSkipping model loading tests due to missing dependencies")
        all_results["models"] = {"skipped": True, "reason": "missing_dependencies"}
    else:
        all_results["models"] = test_model_loading(quick=args.quick, use_gpu=args.gpu)

    # Generate summary report
    generate_report(all_results)

    # Save to JSON if requested
    if args.json:
        output_file = Path(args.json)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_info(f"\nResults saved to: {output_file}")

    # Exit with appropriate code
    if all_results["dependencies"].get("passed", False):
        if all_results["models"].get("passed", False) or all_results["models"].get("skipped", False):
            sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
