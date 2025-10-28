"""
Enhanced Attribution Engine Setup Guide

This script helps new users get started with the Enhanced Attribution Engine.
It verifies your environment and provides setup instructions.

Key features:
1. Environment verification
2. Dependencies check
3. Setup instructions
4. Usage examples
"""
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    required_version = (3, 9)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]} or higher required")
        print(f"   Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✓ Python version {sys.version.split()[0]} OK")
    return True


def check_ollama() -> bool:
    """Check if Ollama CLI is available."""
    if not shutil.which("ollama"):
        print("! Ollama CLI not found")
        print("  The pipeline will use fallback methods for LLM features")
        print("  To use Ollama, install from: https://ollama.ai")
        return False
    
    print("✓ Ollama CLI found")
    return True


def verify_project_structure() -> bool:
    """Verify critical project directories exist."""
    project_root = Path(__file__).resolve().parent
    
    required_paths = [
        "src",
        "config",
        "requirements.txt",
        "README.md",
        "bootstrap.py"
    ]
    
    missing = []
    for path in required_paths:
        if not (project_root / path).exists():
            missing.append(path)
    
    if missing:
        print("❌ Missing required files/directories:")
        for path in missing:
            print(f"   - {path}")
        return False
    
    print("✓ Project structure verified")
    return True


def check_gpu_support() -> None:
    """Check for GPU support via torch."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU support available ({torch.cuda.get_device_name(0)})")
        else:
            print("! No GPU detected - using CPU only")
    except ImportError:
        print("! torch not installed - skipping GPU check")


def print_setup_instructions() -> None:
    """Print instructions for setting up the environment."""
    print("\n=== Enhanced Attribution Engine Setup ===")
    print("\nFollow these steps to get started:\n")
    print("1. Create a virtual environment:")
    print("   python -m venv .venv")
    print("\n2. Activate the environment:")
    print("   Windows: .venv\\Scripts\\activate")
    print("   Unix:    source .venv/bin/activate")
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n4. Run the complete pipeline:")
    print("   python src/main.py")
    print("\nOr run individual components:")
    print("   python -m src.data_processing.clean_data")
    print("   python -m src.llm_engine.sentiment_analysis")
    print("   python -m src.attribution_model.grade_boost_model")
    print("\nData Directories:")
    print("   - Put raw data files in:        data/raw/")
    print("   - Processed data will be in:    data/processed/")
    print("   - Results will be saved to:     data/outputs/")
    print("   - Logs are written to:          logs/")
    print("\nFor more information, see README.md")


def main() -> None:
    """Run environment checks and print setup instructions."""
    checks_passed = all([
        check_python_version(),
        verify_project_structure()
    ])
    
    print("\n=== Optional Components ===")
    check_ollama()
    check_gpu_support()
    
    if not checks_passed:
        print("\n❌ Please fix the above issues before continuing")
        sys.exit(1)
    
    print_setup_instructions()


if __name__ == "__main__":
    main()